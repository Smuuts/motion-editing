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
  encoder.encode(texts, dropout_prob) -> (B, L, dim)
  encoder.null_embedding(batch_size)  -> (B, L, dim) all-zero

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

        # embedding dim: ViT-B/32 -> 512, ViT-L/14 -> 768
        self.dim = self.model.token_embedding.embedding_dim
        self.max_length = 77

        # freeze — we never train the text encoder
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode(self, texts: list[str], dropout_prob: float = 0.0) -> torch.Tensor:
        """
        Encode a list of strings into per-token embeddings.

        texts         : list of B strings
        dropout_prob  : fraction of items in the batch to replace with null
                        embeddings (classifier-free guidance training)

        Returns (B, 77, dim).
        """
        tokens = self.tokenize(texts, truncate=True).to(self.device)

        # extract per-token features from the transformer
        x = self.model.token_embedding(tokens).type(self.model.dtype)
        x = x + self.model.positional_embedding.type(self.model.dtype)
        x = x.permute(1, 0, 2)
        x = self.model.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.model.ln_final(x).type(self.model.dtype)
        context = x.float()  # (B, 77, dim)

        # classifier-free guidance: zero out some embeddings during training
        if dropout_prob > 0.0:
            mask = torch.rand(len(texts), device=self.device) < dropout_prob
            context[mask] = 0.0

        # LEDITS++ Stage 2 mask M1: to average attention only over edit-instruction
        # tokens (not BOS, EOS, or padding), call self.tokenize() on the edit text
        # and find the non-zero token IDs; their positions in [0, 77) are the indices
        # to select from the L_text dimension of the stored cross-attention maps.
        return context

    def null_embedding(self, batch_size: int) -> torch.Tensor:
        """Return all-zero context for unconditional forward passes."""
        return torch.zeros(batch_size, self.max_length, self.dim, device=self.device)


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
    def encode(self, texts: list[str], dropout_prob: float = 0.0) -> torch.Tensor:
        """
        Encode a list of strings into per-token embeddings.

        texts         : list of B strings
        dropout_prob  : fraction of items in the batch to replace with null
                        embeddings (classifier-free guidance training)

        Returns (B, max_length, dim).
        """
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
        context = context * attention_mask[:, :, None].float()

        if dropout_prob > 0.0:
            mask = torch.rand(len(texts), device=self.device) < dropout_prob
            context[mask] = 0.0

        return context

    def null_embedding(self, batch_size: int) -> torch.Tensor:
        """Return all-zero context for unconditional forward passes."""
        return torch.zeros(batch_size, self.max_length, self.dim, device=self.device)


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
