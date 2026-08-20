"""
Frozen text↔motion scorers behind one interface.

Taking an edit's localisation from an *external* frozen model, instead of from this
project's own denoiser whose attention is measured not to be body-part grounded, reduces
to one primitive: `s(motion, text)`. This module supplies it, so the probes never touch
an encoder directly and a second scorer can be added without editing them.

Only `t2m` exists today (the HumanML3D evaluator already in the repo). `tmr` — MotionFix's
retrieval model, the scorer the benchmark reports — is the intended next one; see
`build_scorer` for where it goes.

**The similarity lives here on purpose.** The T2M evaluator's joint space is metric, not
angular: `evaluate.py`'s R-precision and MM-Dist both rank by Euclidean distance, so this
scorer returns *negative distance* (higher = better match). TMR ranks by cosine. Putting
the metric behind the interface keeps that difference out of the probes.
"""

import os

import numpy as np


class T2MScorer:
    """The HumanML3D T2M evaluator (Guo et al. 2022) as a text↔motion scorer.

    Motion input is RAW 263-d HumanML3D features — what `data.clips.load_clip` returns —
    normalised here with the **evaluator's own** mean/std, which are not the training
    Mean/Std the model uses (`evaluate.py` makes the same swap; mixing them up silently
    invalidates every number).
    """

    name = "t2m"
    higher_is_better = True

    def __init__(self, evaluator_dir: str, device, eval_meta_dir: str | None = None):
        from model.t2m_eval import T2MEvaluator

        self.evaluator = T2MEvaluator(
            checkpoint_path=os.path.join(evaluator_dir, "checkpoint", "finest.tar"),
            glove_dir=os.path.join(evaluator_dir, "glove"),
            device=device,
        )
        meta = eval_meta_dir or os.path.join(evaluator_dir, "t2m", "Comp_v6_KLD01", "meta")
        self.mean = np.load(os.path.join(meta, "mean.npy"))
        self.std = np.load(os.path.join(meta, "std.npy"))

    def embed_motions(self, raw_motions) -> np.ndarray:
        """[(T, 263) raw features] → (N, 512). Clips shorter than 4 frames are invalid."""
        normed = []
        for m in raw_motions:
            if len(m) < self.evaluator.UNIT_LENGTH:
                raise ValueError(f"clip too short for the evaluator: {len(m)} frames")
            normed.append(((m - self.mean) / self.std).astype(np.float32))
        return self.evaluator.encode_motion(normed)

    def embed_texts(self, texts=None, tokens=None) -> np.ndarray:
        """(N, 512) from either pre-tagged `word/POS` token lists or plain sentences.

        Dataset captions ship tagged (`clips.read_tagged_caption`) and should be passed as
        `tokens` — re-tagging them would shift the embeddings away from what the evaluator
        was trained on. `texts` is for strings that have no tokens on disk, e.g. the
        editor's instructions; it is routed through `data.text_tags.tag_text`.
        """
        if (tokens is None) == (texts is None):
            raise ValueError("pass exactly one of texts= or tokens=")
        if tokens is None:
            from data.text_tags import tag_text
            tokens = [tag_text(t) for t in texts]
        return self.evaluator.encode_text(tokens)

    @staticmethod
    def similarity(motion_embs: np.ndarray, text_embs: np.ndarray) -> np.ndarray:
        """Row-wise score, higher = better match. Negative Euclidean distance."""
        return -np.sqrt(((motion_embs - text_embs) ** 2).sum(axis=-1))


def build_scorer(name: str, *, evaluator_dir: str = "data/t2m_evaluator", device="cpu",
                 **kwargs):
    """Scorer by name — the one place a new scorer gets registered."""
    if name == "t2m":
        return T2MScorer(evaluator_dir, device, **kwargs)
    if name == "tmr":
        raise NotImplementedError(
            "TMR is not wired up yet. It needs MotionFix's venv + hydra stack, in the "
            "manner of src/eval/run_motionfix_metrics.py, and a text-encoder entry point "
            "(that file only uses their motion↔motion retrieval). Add a TMRScorer with "
            "the same three methods and cosine similarity, then register it here.")
    raise SystemExit(f"unknown scorer '{name}' (known: t2m, tmr)")
