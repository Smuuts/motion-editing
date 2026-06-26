# Data setup

Everything lives under `data/`. Two upstream repos are git **submodules** (code only — their
large/license-restricted data stays local and untracked); the rest is either downloaded datasets
or artifacts derived by scripts in this repo.

## `data/` layout

| path | what | source |
|---|---|---|
| `data/HumanML3D/` | **Submodule** — full HumanML3D repo (`amass_data/`, `index.csv`, processing notebooks), and home of the processed dataset + SMPL features below. | `github.com/EricGuo5513/HumanML3D` |
| `data/HumanML3D/HumanML3D/` | **Processed** HumanML3D the model trains on: `Mean.npy`, `Std.npy`, `new_joints/`, `new_joint_vecs/` (263-d), `texts/`, `{train,val,test}.txt`. This is `--data_root`. | produced by the HumanML3D repo |
| `data/HumanML3D/HumanML3D_smplh/` | **Derived** — HumanML3D in 135-d SMPL-H features for SMPL-rep training (`<id>.npy`, `M<id>.npy`, `Mean.npy`, `Std.npy`). | `src/data/amass_to_smplh.py` + `smplh_stats.py` |
| `data/motionfix/` | **Submodule** — MotionFix repo. Contains the dataset (`data/motionfix-dataset/*.pth.tar`), SMPL-H body models (`data/body_models/smplh/`), the TMR evaluator (`eval-deps/`), and its venv (`mfix-env/`). | `github.com/athn-nik/motionfix` |
| `data/t2m_evaluator/` | Guo et al. T2M evaluator (FID / R-precision) used by `src/evaluate.py`: `checkpoint/`, `glove/`, `t2m/`. | T2M / HumanML3D release |
| `data/motionfix/motionfix_smpl/` | **Derived** — SMPL fits for the MotionFix eval bridge / ceiling refs. | `src/eval/joints2smpl_fit.py` |

> The processed dataset and the SMPL features live **inside** the `data/HumanML3D` submodule
> (`data/HumanML3D/HumanML3D/` and `data/HumanML3D/HumanML3D_smplh/`) — they are untracked working
> data, not part of the submodule's committed history.
>
> The submodules' bulk data (AMASS ~23 GB, MotionFix dataset 5.1 GB, body models, venvs, TMR
> weights) is gitignored upstream, so it is **never tracked or pushed** by this repo — only the
> two upstream commit pointers in `.gitmodules` are.

## 0. Submodules

After cloning this repo:

```bash
git submodule update --init        # checks out data/HumanML3D and data/motionfix (code only)
```

The large data inside each submodule is acquired separately (below).

## 1. HumanML3D

**Acquire:** follow the instructions on the HumanML3D GitHub page
(`github.com/EricGuo5513/HumanML3D`) inside the `data/HumanML3D/` submodule — i.e. download
AMASS (SMPL-H, license-gated) into `data/HumanML3D/amass_data/`, get the body models, and run its
`raw_pose_processing` → `motion_representation` → `cal_mean_variance` notebooks, then extract
`texts.zip`.

**Use:** the processed output is the submodule's `HumanML3D/` subfolder, i.e.
`data/HumanML3D/HumanML3D/` (Mean/Std, `new_joint_vecs/`, `texts/`, splits). All training/eval
scripts take `--data_root data/HumanML3D/HumanML3D`.

## 2. MotionFix

**Acquire:** follow the instructions on the MotionFix GitHub page
(`github.com/athn-nik/motionfix`) inside the `data/motionfix/` submodule — run its `install.sh`
(creates `mfix-env/`) and `scripts/download_data.sh` (fetches the dataset, SMPL-H body models, the
`eval-deps/` TMR evaluator, and the TMED checkpoint).

You end up with:
- `data/motionfix/data/motionfix-dataset/{motionfix,motionfix_val,motionfix_test}.pth.tar` — joblib
  dicts of `id → {motion_source, motion_target, text}` (SMPL-H, 30 fps).
- `data/motionfix/data/body_models/smplh/SMPLH_NEUTRAL.npz` — used for SMPL fitting.
- `data/motionfix/eval-deps/` — TMR weights + `stats/humanml3d/amass_feats/{mean,std}.pt`.

**One required venv fix** (so the TMR evaluator imports — restores `pkg_resources`):

```bash
data/motionfix/mfix-env/bin/python -m pip install "setuptools<81"
```

(The venv's `pip` shebang is stale after relocation — always call it as `mfix-env/bin/python -m pip`.)

## 3. Derived: HumanML3D → SMPL-H features (for SMPL-rep training, Path A)

Builds a HumanML3D-aligned SMPL-H training set straight from AMASS rotations via `index.csv`:

```bash
HML=data/HumanML3D/HumanML3D                       # processed dataset (splits live here)
python src/data/amass_to_smplh.py \
    --hml3d_root data/HumanML3D \
    --out_dir data/HumanML3D/HumanML3D_smplh \
    --splits $HML/train.txt $HML/val.txt $HML/test.txt
python src/data/smplh_stats.py --feat_dir data/HumanML3D/HumanML3D_smplh --split $HML/train.txt
```
Produces 135-d `[trans_delta | body_pose_6d | global_orient_6d]` per clip (+ `M`-mirror) and
`Mean/Std.npy`. HumanAct12 clips (~1191) are skipped — no SMPL params. Filter the split files to
existing ids before training:

```bash
SMPLH=data/HumanML3D/HumanML3D_smplh
for s in train val test; do
  grep -Fxf <(ls $SMPLH | sed 's/\.npy$//') data/HumanML3D/HumanML3D/$s.txt > $SMPLH/$s.txt
done
```

## 4. T2M evaluator

`data/t2m_evaluator/` (Guo et al.) is needed by `src/evaluate.py` for FID / R-precision. Download
it from the HumanML3D / text-to-motion release into that folder (`checkpoint/`, `glove/`, `t2m/`).
