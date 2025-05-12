import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
    )
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.linear2 = nn.Linear(dim, dim)

    def forward(self, x):
        residual = x
        x = self.linear1(F.silu(self.norm1(x)))
        x = self.linear2(F.silu(self.norm2(x)))
        return x + residual


class ResMLPFlow(nn.Module):
    def __init__(self, dim, hidden_dim=1024, num_blocks=4):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )

        self.input_proj = nn.Linear(dim, hidden_dim)
        self.resblocks = nn.Sequential(*[ResidualBlock(hidden_dim) for _ in range(num_blocks)])
        self.out_proj = nn.Linear(hidden_dim, dim)


    def forward(self, x, t, **kwargs):
        # Ensure t is [B, 1]
        if t.dim() == 0:
            t = t.expand(x.size(0), 1)
        elif t.dim() == 1:
            t = t.unsqueeze(1)

        t_emb = timestep_embedding(t.squeeze(1), x.size(-1))
        t_emb = self.time_mlp(t_emb)

        x = self.input_proj(x) + t_emb
        x = self.resblocks(x)
        x = self.out_proj(x)
        return x
