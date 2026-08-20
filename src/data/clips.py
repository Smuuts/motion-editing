"""
Reading individual clips out of a processed dataset root, for the scripts that work
on single clips rather than a DataLoader (edit/probe/generate/sample).

Every representation mirrors the same on-disk layout, so all of this is feature-dim
agnostic:
    <data_root>/new_joint_vecs/<id>.npy   (T, D) raw features
    <data_root>/texts/<id>.txt            HumanML3D annotations, "caption#word/POS…"
    <data_root>/<split>.txt               clip ids
"""

import os

import numpy as np


def split_ids(data_root: str, split: str) -> list[str]:
    with open(os.path.join(data_root, f"{split}.txt")) as f:
        return [l.strip() for l in f if l.strip()]


def feature_path(data_root: str, clip_id: str) -> str:
    return os.path.join(data_root, "new_joint_vecs", f"{clip_id}.npy")


def read_captions(data_root: str, clip_id: str) -> list[str]:
    """All annotations for a clip, stripped of HumanML3D's "#word/POS#start#end" suffix.
    Empty list if the clip has no text file."""
    path = os.path.join(data_root, "texts", f"{clip_id}.txt")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    return [c for c in (l.split("#")[0].strip() for l in lines) if c]


def read_caption(data_root: str, clip_id: str) -> str:
    """The clip's first annotation ("" if it has none)."""
    captions = read_captions(data_root, clip_id)
    return captions[0] if captions else ""


def read_tagged_caption(data_root: str, clip_id: str) -> tuple[str, list[str]]:
    """(caption, word/POS tokens) from the first annotation line.

    The pre-tagged tokens are what the T2M evaluator was trained on — re-tokenising
    here would shift its text embeddings, so evaluate.py reads them from disk.
    """
    path = os.path.join(data_root, "texts", f"{clip_id}.txt")
    if not os.path.exists(path):
        return "", []
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    if not lines:
        return "", []
    parts = lines[0].split("#")
    tokens = parts[1].strip().split() if len(parts) > 1 and parts[1].strip() else []
    return parts[0].strip(), tokens


def load_clip(data_root: str, clip_id: str, max_frames: int):
    """(raw_feat (T, D) float32, T, caption) for a clip named by id."""
    raw = np.load(feature_path(data_root, clip_id))[:max_frames].astype(np.float32)
    return raw, len(raw), read_caption(data_root, clip_id)


def load_source(source: str, data_root: str, split: str, max_frames: int):
    """Resolve a `--source` argument to (raw_feat (T, D) float32, clip_id, T, caption).

    `source` is either an index into <split>.txt or a path to a raw (T, D) .npy file
    (in which case there is no caption unless the id happens to exist in texts/).
    """
    if source.endswith(".npy") and os.path.exists(source):
        clip_id = os.path.splitext(os.path.basename(source))[0]
        raw = np.load(source)[:max_frames].astype(np.float32)
        return raw, clip_id, len(raw), read_caption(data_root, clip_id)
    clip_id = split_ids(data_root, split)[int(source)]
    raw, T, caption = load_clip(data_root, clip_id, max_frames)
    return raw, clip_id, T, caption


def iter_split_clips(data_root: str, split: str, max_frames: int, min_frames: int = 16,
                     max_clips: int | None = None, seed: int = 42,
                     with_text_emb: bool = False):
    """Every usable clip in a split as {id, text, vec_path, T, context_emb}, shuffled.

    A clip is skipped when its features or annotations are missing, or when it is
    shorter than `min_frames`. `context_emb` is the first precomputed text embedding
    from <data_root>/text_emb/ when `with_text_emb` and that directory exists, else None
    — generate.py uses it to skip loading the text encoder entirely.
    """
    emb_dir = os.path.join(data_root, "text_emb")
    has_emb = with_text_emb and os.path.isdir(emb_dir)

    # Cheap pass first (an existence check and an mmap'd header read per id), then
    # shuffle, then read captions/embeddings only for the clips actually kept —
    # `max_clips=3` should not open every caption file in the split (3.4 s on train).
    candidates = []
    for cid in split_ids(data_root, split):
        vec_path = feature_path(data_root, cid)
        if not os.path.exists(vec_path):
            continue
        T_raw = int(np.load(vec_path, mmap_mode="r").shape[0])
        if T_raw >= min_frames:
            candidates.append((cid, vec_path, min(T_raw, max_frames)))
    np.random.default_rng(seed).shuffle(candidates)

    clips = []
    for cid, vec_path, T in candidates:
        if max_clips is not None and len(clips) == max_clips:
            break
        captions = read_captions(data_root, cid)
        if not captions:
            continue

        context_emb = None
        if has_emb:
            emb_path = os.path.join(emb_dir, f"{cid}.npy")
            if os.path.exists(emb_path):
                context_emb = np.load(emb_path)[0].astype(np.float32)

        clips.append({"id": cid, "text": captions[0], "vec_path": vec_path,
                      "T": T, "context_emb": context_emb})
    return clips
