"""
Stage C of the MotionFix evaluation: bridge HumanML3D-263 edited motions back to SMPL-H
parameters, in the exact feature layout MotionFix's TMR evaluator consumes.

MotionFix's TMR features are *pose-only* — `[trans(3) | global_orient_6d(6) | body_pose_6d(126)]`
(see tmr_evaluator/motion2motion_retr.py::collect_gen_samples) — so what matters is recovering
correct joint *rotations*, not absolute positions. We do that by fitting SMPL-H (neutral) to the
edited 22-joint skeleton (SMPLify-style Adam optimisation), warm-started from the dataset's
*source* SMPL pose (frame-aligned, frame-rate-matched), which makes the fit fast and stable since
an edit only changes part of the body.

Frame handling (critical for the global_orient channel):
  - The editor works on HumanML3D-canonical joints (each clip rotated so frame-0 faces Z+, after a
    Z-up->Y-up swap). We undo that per-clip rotation (`*_canon_quat` from the manifest) + the swap so
    the recovered global_orient lives in the dataset's frame, matching how GT is loaded.
  - The editor runs at 20 fps; we resample joints to the clip's native 30 fps frame count so lengths
    line up with the dataset target the evaluator truncates against.
  - uniform-skeleton retargeting changes bone lengths (not angles); we pre-scale the target joints to
    SMPL-neutral leg length so positional fitting doesn't distort the recovered angles.

Runs in the `ma` env (smplx + torch + cuda). Examples:
    # Edited generations for one config:
    python src/eval/joints2smpl_fit.py --manifest data/motionfix_hml3d/test.jsonl \
        --edited_dir data/motionfix_edited/m2_only_s5 --out_dir data/motionfix_smpl/m2_only_s5
    # Bridge-validation gate (fit the GT target instead of an edit):
    python src/eval/joints2smpl_fit.py --manifest data/motionfix_hml3d/test.jsonl \
        --fit_gt_target --out_dir data/motionfix_smpl/_gt_target_fit
"""

import os
import sys
import json
import argparse

import numpy as np
import torch
from tqdm import tqdm

src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # -> src/
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import smplx
from smplx.lbs import batch_rodrigues

from utils.visualise import recover_from_ric, _qinv, _qrot
from data.motionfix_to_hml3d import resample, ZUP_TO_YUP, MOTIONFIX_DIR, SPLIT_FILE

SMPLH_PATH = "data/motionfix/data/body_models/smplh"
# HumanML3D/SMPL 22-joint indices for the left leg (pelvis->knee->ankle), for leg-length scaling.
_LEG = (0, 4, 7)


def aa_to_6d(aa: torch.Tensor) -> torch.Tensor:
    """axis-angle (..., 3) -> 6D rotation (first two rows of the matrix), (..., 6).
    Matches MotionFix's matrix_to_rotation_6d (pytorch3d convention)."""
    flat = aa.reshape(-1, 3)
    mat = batch_rodrigues(flat).reshape(*aa.shape[:-1], 3, 3)
    return mat[..., :2, :].reshape(*aa.shape[:-1], 6)


def uncanonicalize(joints_canon: np.ndarray, canon_quat: np.ndarray) -> np.ndarray:
    """HumanML3D-canonical (Y-up, faced-Z+) joints -> dataset frame (Z-up).
    Undo the per-clip face-Z+ rotation, then the Z-up<->Y-up swap (its own inverse)."""
    q_inv = _qinv(np.asarray(canon_quat, dtype=np.float64)[None])          # (1,4)
    j = _qrot(np.broadcast_to(q_inv, joints_canon.shape[:-1] + (4,)), joints_canon)
    return j @ ZUP_TO_YUP                                                  # swap is symmetric


def _leg_len(joints: torch.Tensor) -> torch.Tensor:
    p, k, a = _LEG
    return (joints[..., p, :] - joints[..., k, :]).norm(dim=-1).mean() + \
           (joints[..., k, :] - joints[..., a, :]).norm(dim=-1).mean()


def fit_smpl_to_joints(target_joints, init_aa, init_transl, body_model, device,
                       n_iter=300, lr=0.05, w_smooth=0.1, w_prior=1e-4, w_init=1.0):
    """
    Fit SMPL-H global_orient + body_pose + transl to `target_joints` (T,22,3).

    init_aa     : (T, 22, 3) axis-angle warm-start (joint 0 = global orient, 1..21 = body pose)
    init_transl : (T, 3) translation warm-start
    w_init      : pull the pose toward the warm-start. HumanML3D's IK-derived joints have already
                  lost the bone-twist DOF, so joint positions under-determine the SMPL rotation;
                  anchoring to the (twist-correct) source warm-start keeps the recovered pose in a
                  realistic region instead of drifting to a twist-equivalent but TMR-distant pose.
    Returns (global_orient_aa (T,3), body_pose_aa (T,21,3), transl (T,3), final_joint_rmse_m).
    """
    target = torch.as_tensor(target_joints, dtype=torch.float32, device=device)
    T = target.shape[0]
    # pre-scale target to SMPL-neutral leg length so positions are commensurate with the model
    with torch.no_grad():
        zero = torch.zeros(1, 22, 3, 3, device=device)
        zero[:] = torch.eye(3, device=device)
        smpl_leg = _leg_len(body_model(global_orient=zero[:, 0],
                                       body_pose=zero[:, 1:].expand(1, 21, 3, 3),
                                       transl=torch.zeros(1, 3, device=device)).joints[:, :22])
        scale = (smpl_leg / _leg_len(target.unsqueeze(0)).clamp(min=1e-6)).item()
    target = target * scale

    go0 = torch.as_tensor(init_aa[:, 0], dtype=torch.float32, device=device)
    bp0 = torch.as_tensor(init_aa[:, 1:], dtype=torch.float32, device=device)
    go = go0.clone().requires_grad_(True)
    bp = bp0.clone().requires_grad_(True)
    tr = (torch.as_tensor(init_transl, dtype=torch.float32, device=device) * scale).clone().requires_grad_(True)
    opt = torch.optim.Adam([go, bp, tr], lr=lr)

    for _ in range(n_iter):
        opt.zero_grad()
        go_m = batch_rodrigues(go)                                  # (T,3,3)
        bp_m = batch_rodrigues(bp.reshape(-1, 3)).reshape(T, 21, 3, 3)
        J = body_model(global_orient=go_m, body_pose=bp_m, transl=tr).joints[:, :22]
        loss_data = ((J - target) ** 2).sum(-1).mean()
        loss_smooth = ((go[1:] - go[:-1]) ** 2).mean() + ((bp[1:] - bp[:-1]) ** 2).mean()
        loss_prior = (bp ** 2).mean()
        loss_init = ((go - go0) ** 2).mean() + ((bp - bp0) ** 2).mean()
        (loss_data + w_smooth * loss_smooth + w_prior * loss_prior + w_init * loss_init).backward()
        opt.step()

    with torch.no_grad():
        go_m = batch_rodrigues(go)
        bp_m = batch_rodrigues(bp.reshape(-1, 3)).reshape(T, 21, 3, 3)
        J = body_model(global_orient=go_m, body_pose=bp_m, transl=tr).joints[:, :22]
        rmse = float((((J - target) ** 2).sum(-1).mean()).sqrt() / scale)   # back to metres
    # report transl in the original (un-scaled) frame
    return go.detach(), bp.detach().reshape(T, 21, 3), (tr.detach() / scale), rmse


def build_pose_feature(go_aa, bp_aa, transl):
    """-> (T,135) numpy: [trans(3) | global_orient_6d(6) | body_pose_6d(126)] (collect_gen_samples layout)."""
    orient6d = aa_to_6d(go_aa)                       # (T,6)
    pose6d = aa_to_6d(bp_aa).reshape(bp_aa.shape[0], -1)  # (T,126)
    return torch.cat([transl, orient6d, pose6d], dim=-1).cpu().numpy().astype(np.float32)


def load_warmstart(split):
    """id -> {'src_rots','src_trans'} from the original MotionFix dataset (30 fps SMPL params)."""
    import joblib
    ds = joblib.load(os.path.join(MOTIONFIX_DIR, SPLIT_FILE[split]))
    return {k: {"rots": np.asarray(v["motion_source"]["rots"], dtype=np.float32),
                "trans": np.asarray(v["motion_source"]["trans"], dtype=np.float32)}
            for k, v in ds.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="data/motionfix_hml3d/test.jsonl")
    ap.add_argument("--split", default="test", choices=list(SPLIT_FILE))
    ap.add_argument("--edited_dir", default=None,
                    help="Dir of edited (F,263) .npy from Stage B (one config). Omit with --fit_gt_target.")
    ap.add_argument("--fit_gt_target", action="store_true",
                    help="Gate mode: fit the GT *target* HumanML3D motion instead of an edit "
                         "(worst case — target pose is maximally far from the source warm-start).")
    ap.add_argument("--fit_source", action="store_true",
                    help="Reconstruction gate: fit the *source* itself (= a scale-0 edit). With the "
                         "source warm-start this is the realistic, matched-init bridge condition; "
                         "its generations should retrieve the true source at high R@1.")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_iter", type=int, default=300)
    ap.add_argument("--w_init", type=float, default=1.0,
                    help="Strength of the warm-start pose anchor (mitigates twist ambiguity).")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    if not (args.fit_gt_target or args.fit_source) and args.edited_dir is None:
        ap.error("provide --edited_dir, or use --fit_gt_target / --fit_source")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs(args.out_dir, exist_ok=True)
    body_model = smplx.SMPLHLayer(model_path=SMPLH_PATH, gender="neutral", ext="npz").to(device).eval()
    for p in body_model.parameters():
        p.requires_grad_(False)

    rows = [json.loads(l) for l in open(args.manifest) if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    warm = load_warmstart(args.split)

    rmses = []
    for row in tqdm(rows, desc="fit"):
        cid = row["id"]
        out_p = os.path.join(args.out_dir, f"{cid}.npy")
        if not args.overwrite and os.path.exists(out_p):
            continue

        if args.fit_gt_target:
            raw = np.load(row["target"]).astype(np.float32)
            canon_quat = row["tgt_canon_quat"]
            n30 = row["tgt_frames_30fps"]
        elif args.fit_source:
            raw = np.load(row["source"]).astype(np.float32)
            canon_quat = row["src_canon_quat"]
            n30 = row["src_frames_30fps"]
        else:
            edited_p = os.path.join(args.edited_dir, f"{cid}.npy")
            if not os.path.exists(edited_p):
                continue
            raw = np.load(edited_p).astype(np.float32)
            canon_quat = row["src_canon_quat"]
            n30 = row["src_frames_30fps"]

        joints_canon = recover_from_ric(raw, joints_num=22)          # (F20,22,3)
        joints_ds = uncanonicalize(joints_canon, canon_quat)         # dataset frame, 20 fps
        # resample directly to the recorded native (30 fps) length (src_fps=T, tgt_fps=L -> L frames)
        joints_30 = resample(joints_ds, src_fps=joints_ds.shape[0], tgt_fps=n30)
        T = joints_30.shape[0]

        # warm-start from source SMPL pose, frame-matched (resample 30fps source params to T)
        src_rots = warm[cid]["rots"][:, :66].reshape(-1, 22, 3)
        src_trans = warm[cid]["trans"]
        if src_rots.shape[0] != T:
            idx = np.linspace(0, src_rots.shape[0] - 1, T).round().astype(int)
            src_rots, src_trans = src_rots[idx], src_trans[idx]

        go_aa, bp_aa, tr, rmse = fit_smpl_to_joints(
            joints_30, src_rots, src_trans, body_model, device,
            n_iter=args.n_iter, w_init=args.w_init)
        feat = build_pose_feature(go_aa, bp_aa, tr)        # (T,135): [trans | orient6d | pose6d]
        # plain array (NOT a dict) so the .npy stays loadable across numpy 1.x/2.x; the Stage-D
        # wrapper loads these into a dict and hands them to retrieval().
        np.save(out_p, feat)
        rmses.append(rmse)

    if rmses:
        print(f"Fitted {len(rmses)} clips -> {args.out_dir} | "
              f"joint RMSE (m): mean={np.mean(rmses):.4f} median={np.median(rmses):.4f} max={np.max(rmses):.4f}")
    else:
        print("No clips fitted (all present? check --overwrite / --edited_dir).")


if __name__ == "__main__":
    main()
