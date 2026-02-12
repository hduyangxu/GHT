import math
from typing import Dict, Optional, Tuple

import torch
from torch import nn


class GHTGraphBuilder(nn.Module):
    """Build a mutual-kNN graph based on cosine similarity."""

    def __init__(
        self,
        k: int = 8,
        metric: str = "cosine",
        mutual: bool = True,
        add_self: bool = False,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.k = k
        self.metric = metric
        self.mutual = mutual
        self.add_self = add_self
        self.eps = eps

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: [B, N, D]
        Returns:
            adjacency: [B, N, N] (float32, 0/1)
        """
        batch_size, num_tokens, _ = tokens.shape
        if num_tokens == 1:
            eye = torch.ones((batch_size, 1, 1), device=tokens.device, dtype=tokens.dtype)
            return eye

        if self.metric != "cosine":
            raise ValueError(f"Unsupported metric: {self.metric}")

        tokens_norm = tokens / (tokens.norm(dim=-1, keepdim=True) + self.eps)
        # TODO: replace dense similarity with sparse/topk-friendly implementation to reduce O(N^2) memory.
        sim = torch.matmul(tokens_norm, tokens_norm.transpose(1, 2))

        if not self.add_self:
            diag_mask = torch.eye(num_tokens, device=sim.device, dtype=torch.bool)
            sim = sim.masked_fill(diag_mask.unsqueeze(0), float("-inf"))

        k = min(self.k, num_tokens - (0 if self.add_self else 1))
        if k <= 0:
            eye = torch.eye(num_tokens, device=sim.device, dtype=sim.dtype).unsqueeze(0).repeat(batch_size, 1, 1)
            return eye

        topk_idx = torch.topk(sim, k=k, dim=-1).indices  # [B, N, k]
        adj = torch.zeros((batch_size, num_tokens, num_tokens), device=sim.device, dtype=torch.bool)
        adj.scatter_(dim=-1, index=topk_idx, value=True)

        if self.mutual:
            adj = adj & adj.transpose(1, 2)

        if self.add_self:
            eye = torch.eye(num_tokens, device=sim.device, dtype=torch.bool).unsqueeze(0)
            adj = adj | eye

        return adj.float()


class GHTSemanticPooling(nn.Module):
    """Graph-guided semantic pooling with soft assignment."""

    def __init__(
        self,
        num_tokens: int,
        hidden_size: int,
        ratio: float = 0.5,
        assign_hidden: int = 128,
        use_gnn: bool = True,
        local_window: Optional[int] = None,
        temperature: float = 1.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.hidden_size = hidden_size
        self.ratio = ratio
        self.num_pooled = max(1, int(math.ceil(num_tokens * ratio)))
        self.use_gnn = use_gnn
        self.local_window = local_window
        self.temperature = temperature
        self.eps = eps

        in_dim = hidden_size * 2 if use_gnn else hidden_size
        self.assignment = nn.Sequential(
            nn.Linear(in_dim, assign_hidden),
            nn.GELU(),
            nn.Linear(assign_hidden, self.num_pooled),
        )

        self.register_buffer("_local_mask", self._build_local_mask(), persistent=False)

    def _build_local_mask(self) -> Optional[torch.Tensor]:
        if self.local_window is None:
            return None
        anchors = torch.linspace(0, self.num_tokens - 1, steps=self.num_pooled)
        positions = torch.arange(self.num_tokens).unsqueeze(1).float()
        dist = (positions - anchors.unsqueeze(0)).abs()
        mask = dist > float(self.local_window)
        mask = mask.float() * float("-inf")
        return mask

    def forward(
        self,
        tokens: torch.Tensor,
        adjacency: torch.Tensor,
        timestamps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            tokens: [B, N, D]
            adjacency: [B, N, N]
            timestamps: [B, N]
        Returns:
            pooled_tokens: [B, N', D]
            pooled_timestamps: [B, N']
            stats: dict
        """
        if self.use_gnn:
            degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
            adj_norm = adjacency / degree
            agg = torch.matmul(adj_norm, tokens)
            features = torch.cat([tokens, agg], dim=-1)
        else:
            features = tokens

        logits = self.assignment(features)
        if self._local_mask is not None:
            logits = logits + self._local_mask.unsqueeze(0).to(logits.device)

        assign = torch.softmax(logits / self.temperature, dim=-1)  # [B, N, N']

        pooled_tokens = torch.matmul(assign.transpose(1, 2), tokens)  # [B, N', D]
        pooled_t = torch.matmul(assign.transpose(1, 2), timestamps.unsqueeze(-1)).squeeze(-1)  # [B, N']

        sort_idx = torch.argsort(pooled_t, dim=-1)
        pooled_tokens = _gather_by_index(pooled_tokens, sort_idx)
        pooled_t = torch.gather(pooled_t, dim=1, index=sort_idx)

        assign_st = torch.matmul(assign, assign.transpose(1, 2))
        link_loss = (assign_st - adjacency).pow(2).mean()
        entropy = -(assign * (assign + self.eps).log()).sum(dim=-1).mean()

        stats = {
            "ght_num_tokens": torch.tensor(self.num_tokens, device=tokens.device),
            "ght_num_pooled": torch.tensor(self.num_pooled, device=tokens.device),
            "ght_ratio": torch.tensor(float(self.num_pooled) / float(self.num_tokens), device=tokens.device),
            "ght_link_loss": link_loss,
            "ght_entropy_loss": entropy,
        }
        return pooled_tokens, pooled_t, stats


class GHTTokenizer(nn.Module):
    """GHT tokenizer: graph construction + semantic pooling."""

    def __init__(
        self,
        num_tokens: int,
        hidden_size: int,
        ratio: float = 0.5,
        k: int = 8,
        assign_hidden: int = 128,
        use_gnn: bool = True,
        local_window: Optional[int] = None,
        temperature: float = 1.0,
        add_self: bool = False,
        mutual: bool = True,
        graph_mode: str = "rule",
        ema_decay: float = 0.99,
        ema_update_every: int = 1,
    ) -> None:
        super().__init__()
        self.graph_builder = GHTGraphBuilder(k=k, mutual=mutual, add_self=add_self)
        self.pooling = GHTSemanticPooling(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            ratio=ratio,
            assign_hidden=assign_hidden,
            use_gnn=use_gnn,
            local_window=local_window,
            temperature=temperature,
        )
        self.graph_mode = graph_mode
        self.ema_decay = ema_decay
        self.ema_update_every = max(1, ema_update_every)
        self._ema_tokens: Optional[torch.Tensor] = None

    @property
    def num_pooled(self) -> int:
        return self.pooling.num_pooled

    def forward(
        self,
        tokens: torch.Tensor,
        timestamps: torch.Tensor,
        train: bool = False,
        step: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if self.graph_mode == "ema":
            teacher_tokens = self._get_ema_tokens(tokens, train=train, step=step)
            adjacency = self.graph_builder(teacher_tokens)
        else:
            adjacency = self.graph_builder(tokens)
        pooled_tokens, pooled_t, stats = self.pooling(tokens, adjacency, timestamps)
        stats["ght_pooled_timestamps"] = pooled_t
        stats["ght_graph_mode"] = torch.tensor(1 if self.graph_mode == "ema" else 0, device=tokens.device)
        return pooled_tokens, stats

    def _get_ema_tokens(
        self,
        tokens: torch.Tensor,
        train: bool,
        step: Optional[int],
    ) -> torch.Tensor:
        if self._ema_tokens is None or self._ema_tokens.shape != tokens.shape:
            self._ema_tokens = tokens.detach().clone()
            return self._ema_tokens

        if train and (step is None or step % self.ema_update_every == 0):
            with torch.no_grad():
                self._ema_tokens.mul_(self.ema_decay).add_(tokens.detach(), alpha=1.0 - self.ema_decay)
        return self._ema_tokens


class DualStreamGHT(nn.Module):
    """Dual-stream GHT tokenizer for trend/residual tokens."""

    def __init__(
        self,
        num_tokens: int,
        hidden_size: int,
        ratio_trend: float,
        ratio_residual: float,
        k: int = 8,
        assign_hidden: int = 128,
        use_gnn: bool = True,
        local_window: Optional[int] = None,
        temperature: float = 1.0,
        add_self: bool = False,
        mutual: bool = True,
        fuse: str = "concat",
        graph_mode: str = "rule",
        ema_decay: float = 0.99,
        ema_update_every: int = 1,
    ) -> None:
        super().__init__()
        self.fuse = fuse
        self.trend_tokenizer = GHTTokenizer(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            ratio=ratio_trend,
            k=k,
            assign_hidden=assign_hidden,
            use_gnn=use_gnn,
            local_window=local_window,
            temperature=temperature,
            add_self=add_self,
            mutual=mutual,
            graph_mode=graph_mode,
            ema_decay=ema_decay,
            ema_update_every=ema_update_every,
        )
        self.residual_tokenizer = GHTTokenizer(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            ratio=ratio_residual,
            k=k,
            assign_hidden=assign_hidden,
            use_gnn=use_gnn,
            local_window=local_window,
            temperature=temperature,
            add_self=add_self,
            mutual=mutual,
            graph_mode=graph_mode,
            ema_decay=ema_decay,
            ema_update_every=ema_update_every,
        )

    @property
    def num_pooled(self) -> int:
        if self.fuse == "concat":
            return self.trend_tokenizer.num_pooled + self.residual_tokenizer.num_pooled
        if self.fuse == "sum":
            if self.trend_tokenizer.num_pooled != self.residual_tokenizer.num_pooled:
                raise ValueError("DualStreamGHT sum requires equal pooled lengths.")
            return self.trend_tokenizer.num_pooled
        raise ValueError(f"Unknown fuse type: {self.fuse}")

    def forward(
        self,
        trend_tokens: torch.Tensor,
        residual_tokens: torch.Tensor,
        timestamps: torch.Tensor,
        train: bool = False,
        step: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        trend_pooled, trend_stats = self.trend_tokenizer(trend_tokens, timestamps, train=train, step=step)
        res_pooled, res_stats = self.residual_tokenizer(residual_tokens, timestamps, train=train, step=step)

        if self.fuse == "concat":
            fused_tokens = torch.cat([trend_pooled, res_pooled], dim=1)
            fused_t = torch.cat(
                [trend_stats["ght_pooled_timestamps"], res_stats["ght_pooled_timestamps"]], dim=1
            )
            sort_idx = torch.argsort(fused_t, dim=-1)
            fused_tokens = _gather_by_index(fused_tokens, sort_idx)
        elif self.fuse == "sum":
            if trend_pooled.shape[1] != res_pooled.shape[1]:
                raise ValueError("DualStreamGHT sum requires equal pooled lengths.")
            fused_tokens = trend_pooled + res_pooled
        else:
            raise ValueError(f"Unknown fuse type: {self.fuse}")

        stats = {f"trend_{k}": v for k, v in trend_stats.items()}
        stats.update({f"residual_{k}": v for k, v in res_stats.items()})
        return fused_tokens, stats


def _gather_by_index(values: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Gather on dim=1 with a [B, K] index."""
    batch_size, seq_len, dim = values.shape
    idx = index.unsqueeze(-1).expand(batch_size, seq_len, dim)
    return torch.gather(values, dim=1, index=idx)
