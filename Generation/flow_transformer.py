import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def timestep_embedding(t, dim, max_period=10000):
    """Sinusoidal timestep embedding (same as in diffusion models)"""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half).to(t.device)
    args = t * freqs  # [B, 1] * [half] → [B, half]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        embedding = F.pad(embedding, (0, 1))  # zero pad
    return embedding  # [B, dim]


class TransformerFlow(nn.Module):
    def __init__(self, dim=1024, time_dim=128, hidden_dim=2048, num_heads=8, num_blocks=2):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.time_dim = time_dim

        self.input_proj = nn.Linear(dim, dim)
        self.attn_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                activation="gelu",
                batch_first=True,
                norm_first=True
            )
            for _ in range(num_blocks)
        ])

        self.output_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim)
        )

    def forward(self, x, t, **kwargs):
        # x: [B, D]
        if t.dim() == 0:
            t = t.expand(x.size(0), 1)
        elif t.dim() == 1:
            t = t.unsqueeze(1)            

        # Add timestep embedding
        t_emb = timestep_embedding(t, self.time_dim)
        t_proj = self.time_mlp(t_emb)  # [B, D]

        x = self.input_proj(x) + t_proj  # timestep-conditioned

        # Attention expects sequence — treat dim as seq
        x = x.unsqueeze(1)  # [B, 1, D]
        for block in self.attn_blocks:
            x = block(x)
        x = x.squeeze(1)  # [B, D]

        return self.output_proj(x)