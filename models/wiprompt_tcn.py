from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TCNBlock(nn.Module):
    """
    Dilated causal Conv1D block with residual connection.

    Input:  [B, C, T]
    Output: [B, C', T]
    """
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.conv = nn.utils.weight_norm(conv)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.out_relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        # causal crop
        crop = self.conv.padding[0]
        if crop > 0:
            out = out[:, :, :-crop]
        out = self.relu(out)
        out = self.dropout(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.out_relu(out + res)


class TemporalAttentionPool(nn.Module):
    """
    Temporal attention pooling over time axis.

    Input:  [B, C, T]
    Output: [B, C]
    """
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xt = x.transpose(1, 2)           # [B, T, C]
        scores = torch.tanh(self.proj(xt))  # [B, T, 1]
        w = F.softmax(scores, dim=1)     # softmax over time
        out = (xt * w).sum(dim=1)        # [B, C]
        return out


@dataclass
class WiPromptTCNConfig:
    tcn_channels: int = 128
    tcn_layers: int = 4
    tcn_kernel: int = 5
    tcn_dropout: float = 0.1
    emb_dim: int = 256
    logit_scale: float = 1.0


class WiPromptTCN(nn.Module):
    """
    Wi-Prompt inspired encoder:
    - flatten (C,K) per frame
    - lazy projection to hidden
    - dilated TCN
    - temporal attention pooling
    - MLP head + classifier

    Forward returns:
      h: [B, emb_dim]
      logits: [B, n_classes]
    """
    def __init__(self, cfg: WiPromptTCNConfig, n_classes: int):
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.LazyLinear(cfg.tcn_channels)

        layers = []
        hidden = cfg.tcn_channels
        for i in range(cfg.tcn_layers):
            dilation = 2 ** i
            layers.append(TCNBlock(hidden, hidden, cfg.tcn_kernel, dilation, cfg.tcn_dropout))
        self.tcn = nn.Sequential(*layers)
        self.attn = TemporalAttentionPool(hidden)

        self.head = nn.Sequential(
            nn.Linear(hidden, cfg.emb_dim),
            nn.ReLU(inplace=True),
        )
        self.cls = nn.Linear(cfg.emb_dim, n_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, C, T, K]
        """
        B, C, T, K = x.shape
        x_flat = x.permute(0, 2, 1, 3).reshape(B, T, C * K)  # [B,T,C*K]
        feat = self.input_proj(x_flat)                        # [B,T,H]
        feat = feat.transpose(1, 2)                           # [B,H,T]

        feat_t = self.tcn(feat)                               # [B,H,T]
        pooled = self.attn(feat_t)                            # [B,H]

        h = self.head(pooled)                                 # [B,emb]
        logits = self.cls(h) * float(self.cfg.logit_scale)
        return h, logits
