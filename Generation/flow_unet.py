import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from flow_matching.path import CondOTProbPath


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


class TimestepBlock(nn.Module):
    def forward(self, x, emb):
        raise NotImplementedError


class TimestepEmbedSequential(nn.Sequential):
    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class ResBlock1D(TimestepBlock):
    def __init__(self, in_channels, emb_channels, dropout, out_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        # self.norm1 = nn.GroupNorm(8, in_channels)
        self.norm1 = nn.LayerNorm(in_channels)
        self.conv1 = nn.Conv1d(in_channels, self.out_channels, 3, padding=1)

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, self.out_channels)
        )

        # self.norm2 = nn.GroupNorm(8, self.out_channels)
        self.norm2 = nn.LayerNorm(self.out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(self.out_channels, self.out_channels, 3, padding=1)

        if self.out_channels == in_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv1d(in_channels, self.out_channels, 1)

    def forward(self, x, emb):
        # Permute for LayerNorm: (B, C, L) → (B, L, C)
        h = x.permute(0, 2, 1)
        h = self.norm1(h)
        h = h.permute(0, 2, 1)  # Back to (B, C, L)
        
        h = self.conv1(F.silu(h))

        emb_out = self.emb_layers(emb).unsqueeze(-1)
        h = h + emb_out

        h = h.permute(0, 2, 1)
        h = self.norm2(h)
        h = h.permute(0, 2, 1)

        h = self.conv2(self.dropout(F.silu(h)))
        return h + self.skip_connection(x)



class FlowUNet1D(nn.Module):
    def __init__(self, embedding_dim=1024, model_channels=128, channel_mult=(1, 2, 4), num_res_blocks=2):
        super().__init__()
        self.model_channels = model_channels
        self.time_embed_dim = model_channels * 4

        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )

        self.input_proj = nn.Conv1d(1, model_channels, kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        in_channels = model_channels
        self.skip_channels = []

        # Down path
        for i, mult in enumerate(channel_mult):
            out_channels = model_channels * mult
            for _ in range(num_res_blocks):
                block = ResBlock1D(in_channels, self.time_embed_dim, 0.1, out_channels)
                self.down_blocks.append(block)
                self.skip_channels.append(out_channels)
                in_channels = out_channels
            if i != len(channel_mult) - 1:
                self.downsamples.append(nn.Conv1d(in_channels, in_channels, kernel_size=4, stride=2, padding=1))

        self.middle_block = ResBlock1D(in_channels, self.time_embed_dim, 0.1)

        # Up path
        for i, mult in reversed(list(enumerate(channel_mult))):
            out_channels = model_channels * mult
            if i != len(channel_mult) - 1:
                self.upsamples.append(nn.ConvTranspose1d(in_channels, in_channels, kernel_size=4, stride=2, padding=1))
            for _ in range(num_res_blocks):
                skip_channels = self.skip_channels.pop()
                block = ResBlock1D(in_channels + skip_channels, self.time_embed_dim, 0.1, out_channels)
                self.up_blocks.append(block)
                in_channels = out_channels

        # self.out_norm = nn.GroupNorm(8, in_channels)
        self.out_norm = nn.LayerNorm(in_channels)
        self.out = nn.Conv1d(in_channels, 1, kernel_size=3, padding=1)

        self.path_sampler = CondOTProbPath()

    def forward(self, x_t, t):
        t = t if t.ndim == 2 else t.unsqueeze(1)
        t_emb = timestep_embedding(t.squeeze(1), self.model_channels)
        emb = self.time_embed(t_emb)

        x = x_t.unsqueeze(1)  # Bx1xD
        x = self.input_proj(x)

        hs = []
        down_idx = 0
        for i, block in enumerate(self.down_blocks):
            x = block(x, emb)
            hs.append(x)
            if down_idx < len(self.downsamples) and (i + 1) % 2 == 0:
                x = self.downsamples[down_idx](x)
                down_idx += 1

        x = self.middle_block(x, emb)

        up_idx = 0
        for i, block in enumerate(self.up_blocks):
            if up_idx < len(self.upsamples) and (i % 2 == 0 and i > 0):
                x = self.upsamples[up_idx](x)
                up_idx += 1
            skip = hs.pop()
            x = torch.cat([x, skip], dim=1)
            x = block(x, emb)

        # x = self.out(F.silu(self.out_norm(x)))
        x = x.permute(0, 2, 1)  # (B, L, C) for LayerNorm
        x = self.out_norm(x)
        x = x.permute(0, 2, 1)  # back to (B, C, L)
        x = self.out(F.silu(x))

        return x.squeeze(1)