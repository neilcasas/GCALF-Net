"""Single string-to-module mapping for GCALF-Net's ablation surface.

modular.py calls only build_frequency_module(...) and build_fusion_module(...);
it never names a concrete FDSF/LFF or WAF/CAF class directly (ARCHITECTURE.md
Sec 4). LFF (Phase 3, M6) and CAF (Phase 4, M7) are not built yet -- their kind
strings are reserved here so the config surface doesn't change again when they
land, but selecting them today raises rather than importing a module that
doesn't exist.
"""
from nndet.arch.encoder.gcalf.fdsf import FrequencyDomainSeparationAndShunting3D
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
        raise NotImplementedError("fusion_type 'caf' is built in Phase 4 (M7)")
    raise ValueError(f"Unknown fusion_type: {kind}")
