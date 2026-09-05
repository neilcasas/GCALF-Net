import torch
import torch.nn as nn
from typing import Callable, Tuple, Sequence, Union, List, Optional
from nndet.arch.encoder.abstract import AbstractEncoder
from nndet.arch.blocks.basic import AbstractBlock

from nndet.arch.encoder.swimTransformer import SwinTransformer3D
import torch.nn.functional as F
import logging
from nndet.arch.encoder.gcalf.registry import build_frequency_module, build_fusion_module

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

__all__ = ["Encoder"]


class Encoder(AbstractEncoder):
    def __init__(self,
                 conv: Callable[[], nn.Module],
                 conv_kernels: Sequence[Union[Tuple[int], int]],
                 strides: Sequence[Union[Tuple[int], int]],
                 block_cls: AbstractBlock,
                 in_channels: int,
                 start_channels: int,
                 stage_kwargs: Sequence[dict] = None,
                 out_stages: Sequence[int] = None,
                 max_channels: int = None,
                 first_block_cls: Optional[AbstractBlock] = None,
                 gcalf_cfg: Optional[dict] = None):
        """
        Args:
            gcalf_cfg: GCALF-Net's ablation config (ARCHITECTURE.md Sec 4):
                `frequency_filter_type` ("fdsf"|"lff"), `fusion_type` ("waf"|"caf"),
                `num_levels`, `fusion_levels`, and one options dict per frequency/fusion
                kind. When None, the encoder runs as a plain CNN (no Swin branch, no
                frequency separation, no fusion) -- this is the legacy/back-compat path
                used by callers that only exercise the base conv-stage mechanics.
        """
        super().__init__()
        self.num_stages = len(conv_kernels)
        self.dim = conv.dim
        if stage_kwargs is None:
            stage_kwargs = [{}] * self.num_stages
        elif isinstance(stage_kwargs, dict):
            stage_kwargs = [stage_kwargs] * self.num_stages
        assert len(stage_kwargs) == len(conv_kernels)

        if out_stages is None:
            self.out_stages = list(range(self.num_stages))
        else:
            self.out_stages = out_stages
        if first_block_cls is None:
            first_block_cls = block_cls

        self.gcalf_cfg = gcalf_cfg
        self.use_transformer = gcalf_cfg is not None
        if self.use_transformer:
            num_levels = gcalf_cfg.get("num_levels", 5)
            if self.num_stages != num_levels:
                raise ValueError(
                    f"GCALF-Net requires exactly {num_levels} encoder levels (ARCHITECTURE.md Sec 4); "
                    f"got {self.num_stages} levels from the resolved plan's conv_kernels/strides.")
            self.fusion_levels = list(gcalf_cfg.get("fusion_levels", list(range(num_levels))))
            if any(level < 0 or level >= num_levels for level in self.fusion_levels):
                raise ValueError(f"fusion_levels {self.fusion_levels} out of range for {num_levels} levels")

            frequency_kind = gcalf_cfg.get("frequency_filter_type", "fdsf")
            self.frequency_module = build_frequency_module(
                frequency_kind, in_channels, gcalf_cfg.get(frequency_kind, {}))

        stages = []
        self.out_channels = []
        in_ch = in_channels
        if isinstance(strides[0], int):
            strides = [tuple([s] * self.dim) for s in strides]
        self.strides = strides

        if self.use_transformer:
            self.depths = [2] * self.num_stages
            self.num_heads = [4 * (2 ** i) for i in range(self.num_stages)]
            self.transformer = SwinTransformer3D(
                in_chans=in_channels,
                embed_dim=start_channels,
                window_size=(2, 7, 7),
                patch_size=(2, 4, 4),
                depths=self.depths,
                num_heads=self.num_heads,
                mlp_ratio=4.,
                qkv_bias=True,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.2,
                norm_layer=nn.LayerNorm,
            )
            self.transformer_out_channels = [int(start_channels * 2 ** i) for i in range(self.num_stages)]
            self.fusion_modules = nn.ModuleDict()

        for stage_id in range(self.num_stages):
            current_in_channels = in_ch
            if stage_id == 0:
                _block = first_block_cls(
                    conv=conv,
                    in_channels=current_in_channels,
                    out_channels=start_channels,
                    conv_kernel=conv_kernels[stage_id],
                    stride=None,
                    max_out_channels=max_channels,
                    **stage_kwargs[stage_id],
                )
            else:
                _block = block_cls(
                    conv=conv,
                    in_channels=current_in_channels,
                    out_channels=None,
                    conv_kernel=conv_kernels[stage_id],
                    stride=strides[stage_id - 1],
                    max_out_channels=max_channels,
                    **stage_kwargs[stage_id],
                )
            in_ch = _block.get_output_channels()
            self.out_channels.append(in_ch)
            stages.append(_block)

            if self.use_transformer and stage_id in self.fusion_levels:
                fusion_kind = gcalf_cfg.get("fusion_type", "waf")
                self.fusion_modules[str(stage_id)] = build_fusion_module(
                    fusion_kind,
                    cnn_channels=in_ch,
                    transformer_channels=self.transformer_out_channels[stage_id],
                    out_channels=in_ch,
                    options=gcalf_cfg.get(fusion_kind, {}),
                )

        self.stages = torch.nn.ModuleList(stages)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outputs = []

        if self.use_transformer:
            # FDSF/LFF runs once at the input; low -> Swin branch, high -> CNN branch
            # (ARCHITECTURE.md Sec 5, fixed by the paper's hypotheses).
            x_low, x_high = self.frequency_module(x)
            cnn_x = x_high
            transformer_feats = self.transformer(x_low)
        else:
            cnn_x = x

        for stage_id, module in enumerate(self.stages):
            cnn_x = module(cnn_x)

            if self.use_transformer and stage_id in self.fusion_levels:
                transformer_feat = F.interpolate(
                    transformer_feats[stage_id], size=cnn_x.shape[2:],
                    mode='trilinear', align_corners=False,
                )
                cnn_x = self.fusion_modules[str(stage_id)](cnn_x, transformer_feat)

            if stage_id in self.out_stages:
                outputs.append(cnn_x)

        return outputs

    def get_channels(self) -> List[int]:
        """
        计算每个返回特征图的通道数
        """
        out_channels = []
        for stage_id in range(self.num_stages):
            if stage_id in self.out_stages:
                out_channels.append(self.out_channels[stage_id])
        return out_channels

    def get_strides(self) -> List[List[int]]:
        """
        计算每个输出特征图相对于输入大小的步幅
        """
        out_strides = []
        for stage_id in range(self.num_stages):
            if stage_id == 0:
                out_strides.append([1] * self.dim)
            else:
                new_stride = [prev_stride * s for prev_stride, s
                              in zip(out_strides[stage_id - 1], self.strides[stage_id - 1])]
                out_strides.append(new_stride)
        return out_strides
