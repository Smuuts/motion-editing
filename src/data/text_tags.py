"""
POS-tag arbitrary text into HumanML3D's `word/POS` token format.

The T2M evaluator's text encoder does not take plain sentences: it looks each token up
in a GloVe vocabulary keyed by `word/POS` strings (`model/t2m_eval.py`, `WordVectorizer`
splits on "/"), so a bare string like "kick with the left leg" cannot be scored. Dataset
captions come pre-tagged on disk (`clips.read_tagged_caption`) — this module supplies the
same tagging for text that is *not* in the dataset, e.g. the editor's instructions.

The rules are a port of HumanML3D's own `data/HumanML3D/text_process.py::process_text`
(that file lives outside `src/`, is not importable, and does `spacy.load` at import time):
strip hyphens, drop non-alpha tokens, lemmatise NOUN/VERB *except* the literal word
"left" — the exception exists because spaCy lemmatises the adjective "left" to "leave",
which would silently destroy exactly the laterality signal the probes measure.

`verify_against_dataset` re-tags real captions and compares against the tokens on disk;
run it before trusting tagged text, since a drift here is invisible downstream.
"""

import os
import sys

# These scripts live one level below src/, so src/ is not on the path when they are run
# directly. Put it there before any project import.
_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from utils.logger import get_logger

log = get_logger(__name__)

_NLP = None


def _nlp():
    """Lazily load spaCy (~0.5 s) so importing this module stays cheap."""
    global _NLP
    if _NLP is None:
        try:
            import spacy
            _NLP = spacy.load("en_core_web_sm")
        except Exception as e:                                  # noqa: BLE001
            raise SystemExit(
                f"text_tags needs spaCy + en_core_web_sm ({type(e).__name__}: {e}).\n"
                "Install with: pip install spacy && python -m spacy download en_core_web_sm")
    return _NLP


def tag_text(sentence: str) -> list[str]:
    """Plain text → ['kick/VERB', 'with/ADP', …], the layout encode_text expects."""
    tokens = []
    for token in _nlp()(sentence.replace("-", "")):
        word = token.text
        if not word.isalpha():
            continue
        lemmatise = token.pos_ in ("NOUN", "VERB") and word != "left"
        tokens.append(f"{token.lemma_ if lemmatise else word}/{token.pos_}")
    return tokens


def verify_against_dataset(data_root: str, n: int = 200, split: str = "val") -> dict:
    """Re-tag `n` dataset captions and compare with the tokens stored on disk.

    Three rates, in increasing order of what actually matters downstream:
      `exact_match_rate` — token strings identical, POS included. Expect this to be low:
        the disk tokens came from an older spaCy, and the drift is almost entirely POS on
        function words ("their/DET" vs "their/PRON", "to/PART" vs "to/ADP").
      `word_match_rate`  — same words, POS ignored. This is the one that would signal a
        real bug (a lemma or a dropped token), since the word vector is looked up by word.
      `vip_match_rate`   — same words AND same POS on the tokens whose POS is *not*
        overridden by the evaluator's VIP lists. Body-part and laterality words
        ("left", "right", "arm", "leg", "kick") are all VIP, so their one-hot is fixed by
        the word alone — POS drift cannot touch the laterality signal.
    The decisive check is at the embedding level and lives in the probe, which has the
    encoder; this function is the cheap first look.
    """
    from data.clips import read_tagged_caption, split_ids
    from model.t2m_eval import VIP_dict

    vip_words = {w for words in VIP_dict.values() for w in words}
    ids = [c for c in split_ids(data_root, split) if not c.startswith("M")][:n]
    exact = words_ok = vip_ok = checked = 0
    examples = []
    for cid in ids:
        text, tokens = read_tagged_caption(data_root, cid)
        if not text or not tokens:
            continue
        checked += 1
        ours = tag_text(text)
        same_words = [t.rsplit("/", 1)[0] for t in ours] == [t.rsplit("/", 1)[0] for t in tokens]
        exact += ours == tokens
        words_ok += same_words
        vip_ok += same_words and all(
            a == b or a.rsplit("/", 1)[0] in vip_words
            for a, b in zip(ours, tokens))
        if ours != tokens and len(examples) < 5:
            examples.append({"clip": cid, "text": text,
                             "disk": " ".join(tokens), "ours": " ".join(ours)})
    d = max(checked, 1)
    return {"checked": checked,
            "exact_match_rate": exact / d,
            "word_match_rate": words_ok / d,
            "vip_match_rate": vip_ok / d,
            "examples": examples}


if __name__ == "__main__":                                      # quick manual check
    from utils.paths import repo_path
    root = sys.argv[1] if len(sys.argv) > 1 else repo_path("data/HumanML3D/HumanML3D")
    log.info(" ".join(tag_text("kick with the left leg")))
    res = verify_against_dataset(root)
    log.info(f"on {res['checked']} captions:  exact {res['exact_match_rate']:.3f}   "
          f"words {res['word_match_rate']:.3f}   vip {res['vip_match_rate']:.3f}")
    for e in res["examples"]:
        log.info(f"  {e['clip']}\n    disk: {e['disk']}\n    ours: {e['ours']}")
