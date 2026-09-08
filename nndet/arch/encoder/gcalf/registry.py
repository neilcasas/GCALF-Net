"""Single string-to-module mapping for GCALF-Net's ablation surface.

modular.py calls only build_frequency_module(...) and build_fusion_module(...);
it never names a concrete FDSF/LFF or WAF/CAF class directly (ARCHITECTURE.md
Sec 4). LFF remains reserved for Phase 3; CAF is the implemented Phase 4
bidirectional cross-attention branch.
"""
from nndet.arch.encoder.gcalf.fdsf import FrequencyDomainSeparationAndShunting3D
from nndet.arch.encoder.gcalf.caf import build_caf
from nndet.arch.encoder.gcalf.waf import build_waf


def build_frequency_module(kind, in_channels, options=None):
    """Both branches return (x_low, x_high) and preserve in_channels."""
    options = options or {}
    if kind == "fdsf":
        return FrequencyDomainSeparationAndShunting3D(**options)
    if kind == "lff":
        raise NotImplementedError("frequency_filter_type 'lff' is built in Phase 3 (M6)")
    raise ValueError(f"Unknown frequency_filter_type: {kind}")


def build_fusion_module(kind, cnn_channels, transformer_channels, out_channels, options=None):
    options = options or {}
    if kind == "waf":
        return build_waf(cnn_channels, transformer_channels, out_channels, **options)
    if kind == "caf":
        return build_caf(cnn_channels, transformer_channels, out_channels, **options)
    raise ValueError(f"Unknown fusion_type: {kind}")
