"""
Bi-Level Graph Attention Network for Elderly Recommender (per paper)

Architecture:
  Level 1 — Social Attention: between user-user pairs in social graph
    α^(s)_{u,v} = softmax_v ( LeakyReLU ( a_s^T [W h_u || W h_v] ) )
  Level 2 — Item Content Attention: between user-item pairs
    α^(c)_{u,i} = softmax_i ( LeakyReLU ( a_c^T [W h_u || W h_i] ) )
  Aggregation:
    h_u^out = σ( Σ_v α^(s)_{u,v} W h_v + Σ_i α^(c)_{u,i} W h_i )

QoL-driven adaptive feedback:
  q_u^(t+1) = (1-β) q_u^(t) + β Δq_{u,i}
  h_u^out ← h_u^out + α · gate(q_u) · proj(q_u)

Rating prediction: r_hat = MLP([h_u^out || h_i])
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class SocialAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.W = nn.Linear(dim, dim, bias=False)
        self.a = nn.Parameter(torch.empty(2 * dim))
        nn.init.xavier_uniform_(self.a.unsqueeze(0))
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, h: torch.Tensor, social_adj: torch.Tensor) -> torch.Tensor:
        """
        h: [N_users, dim]
        social_adj: [N_users, N_users] binary
        Returns aggregated social signal: [N_users, dim]
        """
        Wh = self.W(h)                                    # [N, d]
        N, d = Wh.shape
        # all pairs concatenation
        a1 = (Wh @ self.a[:d]).unsqueeze(1)               # [N, 1]
        a2 = (Wh @ self.a[d:]).unsqueeze(0)               # [1, N]
        e = self.leaky(a1 + a2)                           # [N, N]
        # mask non-edges
        e = e.masked_fill(social_adj == 0, float('-inf'))
        # if a node has no neighbor, softmax of -inf -> nan. Replace with zeros
        no_neighbor = (social_adj.sum(dim=-1) == 0)
        alpha = F.softmax(e, dim=-1)
        alpha = torch.nan_to_num(alpha, nan=0.0)
        out = alpha @ Wh                                  # [N, d]
        out[no_neighbor] = 0.0
        return out, alpha


class ContentAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.W = nn.Linear(dim, dim, bias=False)
        self.a = nn.Parameter(torch.empty(2 * dim))
        nn.init.xavier_uniform_(self.a.unsqueeze(0))
        self.leaky = nn.LeakyReLU(0.2)

    def forward(
        self,
        h_u: torch.Tensor,
        h_i: torch.Tensor,
        ui_adj: torch.Tensor,
    ) -> torch.Tensor:
        """
        h_u: [N_u, d], h_i: [N_i, d], ui_adj: [N_u, N_i] binary
        Returns aggregated content signal per user: [N_u, d]
        """
        Wu = self.W(h_u)                                  # [N_u, d]
        Wi = self.W(h_i)                                  # [N_i, d]
        N_u, d = Wu.shape
        # for each user-item pair, compute attention
        a1 = (Wu @ self.a[:d]).unsqueeze(1)               # [N_u, 1]
        a2 = (Wi @ self.a[d:]).unsqueeze(0)               # [1, N_i]
        e = self.leaky(a1 + a2)                           # [N_u, N_i]
        e = e.masked_fill(ui_adj == 0, float('-inf'))
        no_item = (ui_adj.sum(dim=-1) == 0)
        alpha = F.softmax(e, dim=-1)
        alpha = torch.nan_to_num(alpha, nan=0.0)
        out = alpha @ Wi                                  # [N_u, d]
        out[no_item] = 0.0
        return out, alpha


class BiLevelGAT(nn.Module):
    """
    Inputs:
      x_u: user features [N_u, d_u_in]
      x_i: item features [N_i, d_i_in]
      social_adj: [N_u, N_u]
      ui_adj: [N_u, N_i]
    Outputs:
      r_hat: predicted rating for each (user, item) pair
    """
    def __init__(
        self,
        d_u_in: int,
        d_i_in: int,
        d_hidden: int = 64,
        d_qol: int = 4,
        use_qol_gate: bool = True,
    ):
        super().__init__()
        self.user_proj = nn.Linear(d_u_in, d_hidden)
        self.item_proj = nn.Linear(d_i_in, d_hidden)
        self.social = SocialAttention(d_hidden)
        self.content = ContentAttention(d_hidden)
        self.use_qol_gate = use_qol_gate
        if use_qol_gate:
            self.qol_gate = nn.Sequential(
                nn.Linear(d_qol, d_hidden), nn.Sigmoid()
            )
            self.qol_proj = nn.Linear(d_qol, d_hidden)
            self.alpha_q = nn.Parameter(torch.tensor(0.3))
        self.predictor = nn.Sequential(
            nn.Linear(2 * d_hidden, d_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_hidden, 1),
        )

    def encode(self, x_u, x_i, social_adj, ui_adj, q_u=None):
        h_u = self.user_proj(x_u)
        h_i = self.item_proj(x_i)
        soc_msg, alpha_s = self.social(h_u, social_adj)
        cnt_msg, alpha_c = self.content(h_u, h_i, ui_adj)
        h_u_out = F.elu(soc_msg + cnt_msg + h_u)          # +residual
        if self.use_qol_gate and q_u is not None:
            gate = self.qol_gate(q_u)
            qproj = self.qol_proj(q_u)
            h_u_out = h_u_out + self.alpha_q * gate * qproj
        return h_u_out, h_i, alpha_s, alpha_c

    def forward(self, x_u, x_i, social_adj, ui_adj, user_idx, item_idx, q_u=None):
        h_u_out, h_i, alpha_s, alpha_c = self.encode(
            x_u, x_i, social_adj, ui_adj, q_u
        )
        h_u_pair = h_u_out[user_idx]
        h_i_pair = h_i[item_idx]
        cat = torch.cat([h_u_pair, h_i_pair], dim=-1)
        r_hat = self.predictor(cat).squeeze(-1)
        # scale to [1, 5] via affine
        r_hat = 1.0 + 4.0 * torch.sigmoid(r_hat)
        return r_hat, alpha_s, alpha_c
