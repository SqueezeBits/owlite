from typing import Literal

from .connect_inplace_ops_to_users import ConnectInplaceOpsToUsers
from .decompose_expm1 import DecomposeExpm1
from .decompose_in_projection import DecomposeInProjection
from .decompose_in_projection_packed import DecomposeInProjectionPacked
from .decompose_multi_head_attention_forward import DecomposeMultiHeadAttentionForward
from .decompose_multihead_attention import DecomposeMultiheadAttention
from .decompose_scaled_dot_product_attention import DecomposeScaledDotProductAttention
from .decompose_silu import DecomposeSiLU
from .decompose_transformer import DecomposeTransformer
from .decompose_transformer_decoder import DecomposeTransformerDecoder
from .decompose_transformer_decoder_layer import DecomposeTransformerDecoderLayer
from .decompose_transformer_encoder import DecomposeTransformerEncoder
from .decompose_transformer_encoder_layer import DecomposeTransformerEncoderLayer
from .eliminate_explicit_getitem import EliminateExplicitGetitem
from .eliminate_identity import EliminateIdentity
from .fix_hard_coded_devices import FixHardCodedDevice
from .fuse_consecutive_concats import FuseConsecutiveConcats
from .rewrite_layernorms_functional import RewriteLayerNormsFunctional

PassName = Literal[
    "ConnectSetitemToItsUsers",
    "DecomposeExpm1",
    "DecomposeInProjectionPacked",
    "DecomposeInProjection",
    "DecomposeMultiHeadAttentionForward",
    "DecomposeMultiheadAttention",
    "DecomposeScaledDotProductAttention",
    "DecomposeSiLU",
    "DecomposeTransformer",
    "DecomposeTransformerEncoder",
    "DecomposeTransformerEncoderLayer",
    "DecomposeTransformerDecoder",
    "DecomposeTransformerDecoderLayer",
    "EliminateExplicitGetitem",
    "EliminateIdentity",
    "FixHardCodedDevice",
    "FuseConsecutiveConcats",
    "RewriteLayerNormsFunctional",
]
