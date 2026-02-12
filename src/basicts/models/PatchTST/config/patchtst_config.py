from dataclasses import dataclass, field
from typing import Literal, Optional

from basicts.configs import BasicTSModelConfig


@dataclass
class PatchTSTConfig(BasicTSModelConfig):

    """
    Config class for PatchTST model.
    """

    input_len: int = field(default=None, metadata={"help": "Input sequence length."})
    output_len: int = field(default=None, metadata={"help": "Output sequence length for forecasting task."})
    num_features: int = field(default=None, metadata={"help": "Number of features."})
    num_classes: int = field(default=None, metadata={"help": "Number of classes for classification task."})
    patch_len: int = field(default=16, metadata={"help": "Patch length."})
    patch_stride: int = field(default=8, metadata={"help": "Stride for patching."})
    padding: bool = field(default=True, metadata={"help": "Whether to pad the input sequence before patching."})
    hidden_size: int = field(default=256, metadata={"help": "Hidden size."})
    n_heads: int = field(default=1, metadata={"help": "Number of heads in multi-head attention."})
    intermediate_size: int = field(default=1024, metadata={"help": "Intermediate size of FFN layers."})
    hidden_act: str = field(default="gelu", metadata={"help": "Activation function."})
    num_layers: int = field(default=1, metadata={"help": "Number of encoder layers."})
    attn_dropout: float = field(default=0.1, metadata={"help": "Dropout rate for attention layers."})
    fc_dropout: float = field(default=0.1, metadata={"help": "Dropout rate for FC layers."})
    head_dropout: float = field(default=0.0, metadata={"help": "Dropout rate for head layers."})
    norm_type: Literal["layer_norm", "batch_norm"] = \
        field(default="layer_norm", metadata={"help": "Normalization type."})
    individual_head: bool = field(default=False, metadata={"help": "Whether to use individual head in PatchTSTHead."})
    use_revin: bool = field(default=True, metadata={"help": "Whether to use RevIN."})
    affine: bool = field(default=True, metadata={"help": "Whether to use affine transformation in RevIN."})
    subtract_last: bool = field(default=False, metadata={"help": "Whether to subtract the last element in RevIN."})
    decomp: bool = field(default=False, metadata={"help": "Whether to use decomposition."})
    moving_avg: int = field(default=25, metadata={"help": "Moving average window size for decomposition."})
    output_attentions: bool = field(default=False, metadata={"help": "Whether to output attention weights."})

    # GHT (Graph-Guided Hierarchical Tokenization)
    use_ght: bool = field(default=False, metadata={"help": "Whether to enable GHT tokenizer."})
    ght_ratio: float = field(default=1.0, metadata={"help": "Pooling ratio for GHT."})
    ght_k: int = field(default=8, metadata={"help": "k for mutual-kNN graph."})
    ght_assign_hidden: int = field(default=128, metadata={"help": "Hidden size for assignment MLP."})
    ght_use_gnn: bool = field(default=True, metadata={"help": "Whether to use neighborhood aggregation for assignment."})
    ght_local_window: Optional[int] = field(default=None, metadata={"help": "Local window size for assignment mask."})
    ght_temperature: float = field(default=1.0, metadata={"help": "Softmax temperature for assignment."})
    ght_add_self: bool = field(default=False, metadata={"help": "Whether to add self-loop in graph."})
    ght_mutual: bool = field(default=True, metadata={"help": "Whether to use mutual-kNN adjacency."})
    ght_link_weight: float = field(default=0.1, metadata={"help": "Link loss weight."})
    ght_entropy_weight: float = field(default=0.01, metadata={"help": "Entropy loss weight."})
    ght_graph_mode: Literal["rule", "ema"] = field(default="rule", metadata={"help": "Graph mode for GHT."})
    ght_ema_decay: float = field(default=0.99, metadata={"help": "EMA decay for teacher tokens."})
    ght_ema_update_every: int = field(default=1, metadata={"help": "EMA update interval (steps)."})
    ght_dual_stream: bool = field(default=False, metadata={"help": "Whether to enable dual-stream GHT."})
    ght_ratio_trend: float = field(default=0.125, metadata={"help": "Pooling ratio for trend stream."})
    ght_ratio_residual: float = field(default=0.5, metadata={"help": "Pooling ratio for residual stream."})
    ght_fuse: Literal["concat", "sum"] = field(default="concat", metadata={"help": "Fusion method for dual stream tokens."})
