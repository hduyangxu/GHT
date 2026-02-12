import time
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from basicts.modules.decomposition import MovingAverageDecomposition
from basicts.modules.embed import PatchEmbedding
from basicts.modules.mlps import MLPLayer
from basicts.modules.norm import RevIN
from basicts.modules.transformer import Encoder, EncoderLayer, MultiHeadAttention
from basicts.modules.ght import DualStreamGHT, GHTTokenizer

from ..config.patchtst_config import PatchTSTConfig
from .patchtst_layers import PatchTSTBatchNorm, PatchTSTHead
from .patchtst_arch import PatchTSTBackbone


class PatchTSTBackboneGHT(nn.Module):
    """PatchTST backbone with GHT tokenizer."""

    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        self.num_features = config.num_features

        padding = (0, config.patch_stride) if config.padding else None
        self.patch_embedding = PatchEmbedding(
            config.hidden_size, config.patch_len, config.patch_stride, padding, config.fc_dropout
        )
        self.num_patches = int((config.input_len - config.patch_len) / config.patch_stride + 1)
        if config.padding:
            self.num_patches += 1

        self.patch_centers = self._build_patch_centers(
            config.input_len, config.patch_len, config.patch_stride, config.padding
        )

        self.ght_dual_stream = config.ght_dual_stream
        if self.ght_dual_stream:
            self.decomp_layer = MovingAverageDecomposition(config.moving_avg)
            self.ght = DualStreamGHT(
                num_tokens=self.num_patches,
                hidden_size=config.hidden_size,
                ratio_trend=config.ght_ratio_trend,
                ratio_residual=config.ght_ratio_residual,
                k=config.ght_k,
                assign_hidden=config.ght_assign_hidden,
                use_gnn=config.ght_use_gnn,
                local_window=config.ght_local_window,
                temperature=config.ght_temperature,
                add_self=config.ght_add_self,
                mutual=config.ght_mutual,
                fuse=config.ght_fuse,
                graph_mode=config.ght_graph_mode,
                ema_decay=config.ght_ema_decay,
                ema_update_every=config.ght_ema_update_every,
            )
        else:
            self.ght = GHTTokenizer(
                num_tokens=self.num_patches,
                hidden_size=config.hidden_size,
                ratio=config.ght_ratio,
                k=config.ght_k,
                assign_hidden=config.ght_assign_hidden,
                use_gnn=config.ght_use_gnn,
                local_window=config.ght_local_window,
                temperature=config.ght_temperature,
                add_self=config.ght_add_self,
                mutual=config.ght_mutual,
                graph_mode=config.ght_graph_mode,
                ema_decay=config.ght_ema_decay,
                ema_update_every=config.ght_ema_update_every,
            )
        self.num_patches = self.ght.num_pooled

        norm_type = nn.LayerNorm if config.norm_type == "layer_norm" else PatchTSTBatchNorm
        self.encoder = Encoder(
            nn.ModuleList([
                EncoderLayer(
                    MultiHeadAttention(config.hidden_size, config.n_heads, config.attn_dropout),
                    MLPLayer(
                        config.hidden_size,
                        config.intermediate_size,
                        hidden_act=config.hidden_act,
                        dropout=config.fc_dropout,
                    ),
                    layer_norm=(norm_type, config.hidden_size),
                    norm_position="post",
                )
                for _ in range(config.num_layers)
            ])
        )
        self.output_attentions = config.output_attentions

    @staticmethod
    def _build_patch_centers(
        input_len: int,
        patch_len: int,
        patch_stride: int,
        padding: bool,
    ) -> torch.Tensor:
        pad_right = patch_stride if padding else 0
        starts = list(range(0, input_len + pad_right - patch_len + 1, patch_stride))
        centers = [s + (patch_len - 1) / 2.0 for s in starts]
        return torch.tensor(centers, dtype=torch.float32)

    def forward(
        self,
        inputs: torch.Tensor,
        train: bool = False,
        step: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]], Dict[str, torch.Tensor]]:
        """
        Args:
            inputs: [B, L, C]
        Returns:
            hidden_states: [B, C, N', D]
            attn_weights: Optional[List[Tensor]]
            ght_stats: Dict
        """
        batch_size = inputs.shape[0]
        timestamps = self.patch_centers.to(inputs.device)

        if self.ght_dual_stream:
            seasonal, trend = self.decomp_layer(inputs)
            residual = seasonal
            trend_tokens = self.patch_embedding(trend)
            residual_tokens = self.patch_embedding(residual)
            ts_trend = timestamps.unsqueeze(0).repeat(trend_tokens.shape[0], 1)
            ts_residual = timestamps.unsqueeze(0).repeat(residual_tokens.shape[0], 1)
            pooled_tokens, ght_stats = self.ght(trend_tokens, residual_tokens, ts_trend, train=train, step=step)
        else:
            hidden_states = self.patch_embedding(inputs)  # [B*C, N, D]
            timestamps = timestamps.unsqueeze(0).repeat(hidden_states.shape[0], 1)
            pooled_tokens, ght_stats = self.ght(hidden_states, timestamps, train=train, step=step)
        pooled_tokens, attn_weights = self.encoder(pooled_tokens, output_attentions=self.output_attentions)
        pooled_tokens = pooled_tokens.reshape(
            batch_size, self.num_features, pooled_tokens.shape[-2], pooled_tokens.shape[-1]
        )
        return pooled_tokens, attn_weights, ght_stats


class PatchTSTForForecastingGHT(nn.Module):
    """PatchTST for forecasting with optional GHT tokenizer."""

    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        self.decomp = config.decomp
        if self.decomp:
            self.decomp_layer = MovingAverageDecomposition(config.moving_avg)
            self.seasonal_backbone = PatchTSTBackboneGHT(config) if config.use_ght else PatchTSTBackbone(config)
            self.trend_backbone = PatchTSTBackboneGHT(config) if config.use_ght else PatchTSTBackbone(config)
            self.num_patches = self.seasonal_backbone.num_patches
        else:
            self.backbone = PatchTSTBackboneGHT(config) if config.use_ght else PatchTSTBackbone(config)
            self.num_patches = self.backbone.num_patches

        self.flatten = nn.Flatten(start_dim=-2)
        self.forecasting_head = PatchTSTHead(
            self.num_patches * config.hidden_size,
            config.output_len,
            config.individual_head,
            config.num_features,
            config.head_dropout,
        )
        self.use_revin = config.use_revin
        if self.use_revin:
            self.revin = RevIN(
                config.num_features, affine=config.affine, subtract_last=config.subtract_last
            )
        self.output_attentions = config.output_attentions
        self.aux_link_weight = config.ght_link_weight
        self.aux_entropy_weight = config.ght_entropy_weight

    def forward(
        self,
        inputs: torch.Tensor,
        train: bool = False,
        step: Optional[int] = None,
    ) -> torch.Tensor:
        start_time = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        if self.use_revin:
            inputs = self.revin(inputs, "norm")

        ght_stats: Dict[str, torch.Tensor] = {}
        if self.decomp:
            seasonal_hidden_states, attn_weights_s, stats_s = self.seasonal_backbone(inputs, train=train, step=step)
            trend_hidden_states, attn_weights_t, stats_t = self.trend_backbone(inputs, train=train, step=step)
            hidden_states = seasonal_hidden_states + trend_hidden_states
            ght_stats.update({f"seasonal_{k}": v for k, v in stats_s.items()})
            ght_stats.update({f"trend_{k}": v for k, v in stats_t.items()})
            attn_weights = attn_weights_s
        else:
            if isinstance(self.backbone, PatchTSTBackboneGHT):
                hidden_states, attn_weights, ght_stats = self.backbone(inputs, train=train, step=step)
            else:
                hidden_states, attn_weights = self.backbone(inputs)

        hidden_states = self.flatten(hidden_states)
        prediction = self.forecasting_head(hidden_states).transpose(1, 2)

        if self.use_revin:
            prediction = self.revin(prediction, "denorm")

        forward_ms = (time.perf_counter() - start_time) * 1000.0
        peak_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0.0
        token_count = _get_token_count(ght_stats)
        ght_stats.update({
            "ght_forward_ms": torch.tensor(forward_ms, device=prediction.device),
            "ght_peak_mem": torch.tensor(peak_mem, device=prediction.device),
            "ght_tokens": torch.tensor(token_count, device=prediction.device),
        })

        if self.output_attentions:
            return {
                "prediction": prediction,
                "attn_weights": attn_weights,
                **ght_stats,
            }
        return {"prediction": prediction, **ght_stats}


class PatchTSTForClassificationGHT(nn.Module):
    """PatchTST for classification with optional GHT tokenizer."""

    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        self.num_classes = config.num_classes
        self.backbone = PatchTSTBackboneGHT(config) if config.use_ght else PatchTSTBackbone(config)
        self.flatten = nn.Flatten(start_dim=1)
        self.classification_head = PatchTSTHead(
            self.backbone.num_patches * config.hidden_size * config.num_features,
            config.num_classes,
            config.individual_head,
            config.num_features,
            config.head_dropout,
        )
        self.use_revin = config.use_revin
        if self.use_revin:
            self.revin = RevIN(
                config.num_features, affine=config.affine, subtract_last=config.subtract_last
            )
        self.output_attentions = config.output_attentions

    def forward(
        self,
        inputs: torch.Tensor,
        train: bool = False,
        step: Optional[int] = None,
    ) -> torch.Tensor:
        start_time = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        if self.use_revin:
            inputs = self.revin(inputs, "norm")

        if isinstance(self.backbone, PatchTSTBackboneGHT):
            hidden_states, attn_weights, ght_stats = self.backbone(inputs, train=train, step=step)
        else:
            hidden_states, attn_weights = self.backbone(inputs)
            ght_stats = {}

        hidden_states = self.flatten(hidden_states)
        prediction = self.classification_head(hidden_states)

        forward_ms = (time.perf_counter() - start_time) * 1000.0
        peak_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0.0
        token_count = _get_token_count(ght_stats)
        ght_stats.update({
            "ght_forward_ms": torch.tensor(forward_ms, device=prediction.device),
            "ght_peak_mem": torch.tensor(peak_mem, device=prediction.device),
            "ght_tokens": torch.tensor(token_count, device=prediction.device),
        })

        if self.output_attentions:
            return {
                "prediction": prediction,
                "attn_weights": attn_weights,
                **ght_stats,
            }
        return {"prediction": prediction, **ght_stats}


def _get_token_count(stats: Dict[str, torch.Tensor]) -> int:
    if "ght_num_pooled" in stats:
        return int(stats["ght_num_pooled"].item())
    trend_key = "trend_ght_num_pooled"
    residual_key = "residual_ght_num_pooled"
    if trend_key in stats and residual_key in stats:
        return int(stats[trend_key].item() + stats[residual_key].item())
    return 0
