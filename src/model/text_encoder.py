"""
Text encoder wrappers for motion diffusion.

Two backends:
  CLIPTextEncoder  — OpenAI CLIP, 77-token context, 512 (ViT-B) or 768 (ViT-L) dim
  T5TextEncoder    — HuggingFace T5EncoderModel (encoder only), configurable
                     max_length, 512/768/1024 dim depending on variant

Both return (B, L, dim) per-token embeddings compatible with the cross-attention
in MotionDiT / GroupDiT.  Both expose:
  encoder.dim         — embedding dimension (int)
  encoder.max_length  — fixed sequence length L (int)
  encoder.encode(texts)        -> (B, L, dim)
  encoder.token_info(text)     -> (positions, labels) of content tokens

token_info() is the tokenizer-agnostic entry point LEDITS++ Stage 2 uses to pick
which columns of the L_text attention dimension belong to real instruction words
(excluding BOS/EOS/padding) — see analyse_attention.py and the M1 mask.

The unconditional ("null") branch is NOT provided here: the model owns a learned
null_text_emb (used via context=None), which is what CFG dropout trains. A zero
embedding is deliberately not exposed so callers cannot accidentally substitute a
different null than the one the model was trained with.

Use build_text_encoder(config, device) to select the backend from a config dict.
Use get_encoder_dims(config) -> (dim, seq_len) to derive model build kwargs
without loading any weights.
"""

import torch
import torch.nn as nn


# Lookup table for T5 d_model without loading the model.
_T5_DIMS: dict[str, int] = {
    "t5-small":              512,
    "t5-base":               768,
    "t5-large":             1024,
    "t5-3b":                1024,
    "t5-11b":               1024,
    "google/flan-t5-small":  512,
    "google/flan-t5-base":   768,
    "google/flan-t5-large": 1024,
    "google/flan-t5-xl":    2048,
}


def get_encoder_dims(config: dict) -> tuple[int, int]:
    """Return (context_dim, text_seq_len) from a config dict without loading weights."""
    encoder_type = config.get("text_encoder", "clip")
    if encoder_type == "t5":
        dim = _T5_DIMS.get(config.get("t5_version", "t5-base"), 768)
        seq_len = int(config.get("t5_max_length", 128))
    else:
        dim = 768 if "L/14" in config.get("clip_version", "ViT-B/32") else 512
        seq_len = 77
    return dim, seq_len


class CLIPTextEncoder(nn.Module):
    def __init__(self, clip_version: str = "ViT-B/32", device="cpu"):
        super().__init__()
        try:
            import clip
        except ImportError:
            raise ImportError(
                "CLIP not installed. Run: pip install git+https://github.com/openai/CLIP.git"
            )
        self.device = device
        self.model, _ = clip.load(clip_version, device=device)
        self.model.eval()
        self.tokenize = clip.tokenize

        # CLIP special token ids (fixed by the BPE vocabulary)
        self._bos, self._eos, self._pad = 49406, 49407, 0
        try:
            from clip.simple_tokenizer import SimpleTokenizer
            decoder = SimpleTokenizer().decoder
            self._decode_tok = lambda tid: decoder.get(tid, f"[{tid}]").replace("</w>", "")
        except Exception:
            self._decode_tok = lambda tid: str(tid)

        # embedding dim: ViT-B/32 -> 512, ViT-L/14 -> 768
        self.dim = self.model.token_embedding.embedding_dim
        self.max_length = 77

        # freeze — we never train the text encoder
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode(self, texts: list[str]) -> torch.Tensor:
        """Encode B strings into per-token embeddings. Returns (B, 77, dim)."""
        tokens = self.tokenize(texts, truncate=True).to(self.device)

        # extract per-token features from the transformer
        x = self.model.token_embedding(tokens).type(self.model.dtype)
        x = x + self.model.positional_embedding.type(self.model.dtype)
        x = x.permute(1, 0, 2)
        x = self.model.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.model.ln_final(x).type(self.model.dtype)
        return x.float()  # (B, 77, dim)

    def token_info(self, text: str) -> tuple[list[int], list[str]]:
        """
        Return (positions, labels) of content tokens — excluding BOS, EOS and
        padding — as they appear in encode()'s L_text (=77) dimension.

        LEDITS++ Stage 2 mask M1 averages the stored cross-attention maps over
        exactly these column positions.
        """
        token_ids = self.tokenize([text], truncate=True)[0].tolist()  # (77,)
        idxs, labels = [], []
        for pos, tid in enumerate(token_ids):
            if tid == self._bos or tid == self._pad:
                continue
            if tid == self._eos:
                break
            idxs.append(pos)
            labels.append(self._decode_tok(tid))
        return idxs, labels


class T5TextEncoder(nn.Module):
    """
    T5 encoder-only text encoder (no decoder).

    Tokenises to a fixed max_length, runs the T5 encoder, and zeroes out
    padding positions so they contribute nothing to cross-attention.

    dim = model.config.d_model:
      t5-small: 512  |  t5-base: 768  |  t5-large / t5-3b: 1024

    Returns (B, max_length, dim) — same interface as CLIPTextEncoder.
    """

    def __init__(self, t5_version: str = "t5-base", max_length: int = 128,
                 device="cpu"):
        super().__init__()
        try:
            from transformers import T5EncoderModel, T5TokenizerFast
        except ImportError:
            raise ImportError(
                "transformers not installed. Run: pip install transformers"
            )
        self.device = device
        self.max_length = max_length

        self.tokenizer = T5TokenizerFast.from_pretrained(t5_version)
        self.model = T5EncoderModel.from_pretrained(t5_version).to(device)
        self.model.eval()
        self.dim = self.model.config.d_model

        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode(self, texts: list[str]) -> torch.Tensor:
        """Encode B strings into per-token embeddings. Returns (B, max_length, dim)."""
        enc = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        context = out.last_hidden_state.float()  # (B, max_length, dim)

        # zero out padding positions so they don't pollute cross-attention
        return context * attention_mask[:, :, None].float()

    def token_info(self, text: str) -> tuple[list[int], list[str]]:
        """
        Return (positions, labels) of content tokens — excluding the EOS sentinel
        and padding — as they appear in encode()'s L_text (=max_length) dimension.
        T5 has no BOS token. Sub-word pieces keep their own positions; the SentencePiece
        word-boundary marker (▁) is rendered as a leading space and stripped.
        """
        ids = self.tokenizer(
            [text], max_length=self.max_length, truncation=True
        )["input_ids"][0]
        idxs, labels = [], []
        for pos, tid in enumerate(ids):
            if tid == self.tokenizer.pad_token_id or tid == self.tokenizer.eos_token_id:
                continue
            idxs.append(pos)
            labels.append(self.tokenizer.convert_ids_to_tokens(tid).replace("▁", " ").strip())
        return idxs, labels


def build_text_encoder(config: dict, device="cpu") -> "CLIPTextEncoder | T5TextEncoder":
    """Instantiate a text encoder from a config dict."""
    encoder_type = config.get("text_encoder", "clip")
    if encoder_type == "t5":
        return T5TextEncoder(
            t5_version=config.get("t5_version", "t5-base"),
            max_length=int(config.get("t5_max_length", 128)),
            device=device,
        )
    return CLIPTextEncoder(
        clip_version=config.get("clip_version", "ViT-B/32"),
        device=device,
    )
