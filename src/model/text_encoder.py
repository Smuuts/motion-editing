"""
CLIP text encoder wrapper.

Encodes a list of strings into text embeddings that are passed as the
cross-attention context to MotionDiT. Handles batching, truncation,
and caching. Supports classifier-free guidance by optionally returning
null embeddings for a fraction of the batch.
"""

import torch
import torch.nn as nn


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

        Returns (B, L, dim) where L is the CLIP sequence length (77).
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

        return context

    def null_embedding(self, batch_size: int) -> torch.Tensor:
        """Return all-zero context for unconditional forward passes."""
        return torch.zeros(batch_size, 77, self.dim, device=self.device)
