# Initially pulled from https://github.com/black-forest-labs/flux and adapted for Chroma.
#
# Chroma (https://huggingface.co/lodestones/Chroma) is an 8.9B de-distilled finetune of
# FLUX.1-schnell. Architecturally it differs from FLUX.1 in that:
#   - It has no CLIP/vector input and no time/guidance embedding MLPs. Instead, all per-block
#     modulation vectors are produced by a single "distilled guidance" Approximator network.
#   - It uses only the T5 text encoder for conditioning (context_in_dim=4096).

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn

from invokeai.backend.chroma.layers import (
    Approximator,
    ChromaModulationOut,
    CustomDoubleStreamBlockProcessor,
    CustomSingleStreamBlockProcessor,
    DoubleStreamBlock,
    LastLayer,
    SingleStreamBlock,
)
from invokeai.backend.flux.extensions.regional_prompting_extension import RegionalPromptingExtension
from invokeai.backend.flux.modules.layers import EmbedND, timestep_embedding


@dataclass
class ChromaParams:
    in_channels: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    qkv_bias: bool
    out_channels: Optional[int]
    # Approximator (distilled guidance) params:
    in_dim: int
    out_dim: int
    hidden_dim: int
    n_layers: int


chroma_params = ChromaParams(
    in_channels=64,
    context_in_dim=4096,
    hidden_size=3072,
    mlp_ratio=4.0,
    num_heads=24,
    depth=19,
    depth_single_blocks=38,
    axes_dim=[16, 56, 56],
    theta=10_000,
    qkv_bias=True,
    out_channels=None,
    in_dim=64,
    out_dim=3072,
    hidden_dim=5120,
    n_layers=5,
)

# Total number of modulation vectors produced by the Approximator. Layout:
#   single     : depth_single_blocks * 3
#   double_img : depth * 6
#   double_txt : depth * 6
#   final      : 2
# For the default Chroma params: 38*3 + 19*6 + 19*6 + 2 = 344.
MOD_INDEX_LENGTH = 344


class Chroma(nn.Module):
    """Transformer model for flow matching on sequences (Chroma variant of FLUX)."""

    def __init__(self, params: ChromaParams):
        super().__init__()

        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = params.out_channels or self.in_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}")
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.in_dim = params.in_dim
        self.out_dim = params.out_dim
        self.hidden_dim = params.hidden_dim
        self.n_layers = params.n_layers
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)

        # Replaces FLUX's time_in / vector_in / guidance_in MLPs.
        self.distilled_guidance_layer = Approximator(
            params.in_dim,
            params.out_dim,
            params.hidden_dim,
            params.n_layers,
        )
        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                )
                for _ in range(params.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio)
                for _ in range(params.depth_single_blocks)
            ]
        )

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

    def get_modulations(self, tensor: torch.Tensor, block_type: str, *, idx: int = 0):
        # This function slices up the modulations tensor which has the following layout:
        #   single     : num_single_blocks * 3 elements
        #   double_img : num_double_blocks * 6 elements
        #   double_txt : num_double_blocks * 6 elements
        #   final      : 2 elements
        if block_type == "final":
            return (tensor[:, -2:-1, :], tensor[:, -1:, :])
        single_block_count = self.params.depth_single_blocks
        double_block_count = self.params.depth
        offset = 3 * idx
        if block_type == "single":
            return ChromaModulationOut.from_offset(tensor, offset)
        # Double block modulations are 6 elements so we double 3 * idx.
        offset *= 2
        if block_type in {"double_img", "double_txt"}:
            # Advance past the single block modulations.
            offset += 3 * single_block_count
            if block_type == "double_txt":
                # Advance past the double block img modulations.
                offset += 6 * double_block_count
            return (
                ChromaModulationOut.from_offset(tensor, offset),
                ChromaModulationOut.from_offset(tensor, offset + 3),
            )
        raise ValueError("Bad block_type")

    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        guidance: Tensor,
        regional_prompting_extension: RegionalPromptingExtension,
        controlnet_double_block_residuals: list[Tensor] | None = None,
        controlnet_single_block_residuals: list[Tensor] | None = None,
    ) -> Tensor:
        # ControlNet residuals are produced by FLUX-architecture ControlNets (InstantX / XLabs). Chroma's
        # double/single blocks share FLUX's hidden size and block counts, so the residuals are dimensionally
        # compatible and are injected additively exactly as in FLUX. The ControlNet itself runs independently
        # in the denoise loop (it sees Chroma's T5 conditioning with zeroed CLIP/guidance).
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        img = self.img_in(img)

        # distilled vector guidance
        mod_index_length = MOD_INDEX_LENGTH
        distill_timestep = timestep_embedding(timesteps.detach().clone(), 16).to(img.dtype)
        distil_guidance = timestep_embedding(guidance.detach().clone(), 16).to(img.dtype)

        # get all modulation index
        modulation_index = timestep_embedding(torch.arange(mod_index_length, device=img.device), 32).to(img.dtype)
        # we need to broadcast the modulation index here so each batch has all of the index
        modulation_index = modulation_index.unsqueeze(0).repeat(img.shape[0], 1, 1)
        # and we need to broadcast timestep and guidance along too
        timestep_guidance = (
            torch.cat([distill_timestep, distil_guidance], dim=1).unsqueeze(1).repeat(1, mod_index_length, 1)
        )
        # then and only then we could concatenate it together
        input_vec = torch.cat([timestep_guidance, modulation_index], dim=-1)

        mod_vectors = self.distilled_guidance_layer(input_vec)

        txt = self.txt_in(txt)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        if controlnet_double_block_residuals is not None:
            assert len(controlnet_double_block_residuals) == len(self.double_blocks)

        double_block: DoubleStreamBlock
        for block_index, double_block in enumerate(self.double_blocks):
            double_mod = (
                self.get_modulations(mod_vectors, "double_img", idx=block_index),
                self.get_modulations(mod_vectors, "double_txt", idx=block_index),
            )
            img, txt = CustomDoubleStreamBlockProcessor.custom_double_block_forward(
                block_index=block_index,
                block=double_block,
                img=img,
                txt=txt,
                vec=double_mod,
                pe=pe,
                regional_prompting_extension=regional_prompting_extension,
            )

            if controlnet_double_block_residuals is not None:
                img = img + controlnet_double_block_residuals[block_index]

        img = torch.cat((txt, img), 1)

        if controlnet_single_block_residuals is not None:
            assert len(controlnet_single_block_residuals) == len(self.single_blocks)

        single_block: SingleStreamBlock
        for block_index, single_block in enumerate(self.single_blocks):
            single_mod = self.get_modulations(mod_vectors, "single", idx=block_index)
            img = CustomSingleStreamBlockProcessor.custom_single_block_forward(
                block_index=block_index,
                block=single_block,
                img=img,
                vec=single_mod,
                pe=pe,
                regional_prompting_extension=regional_prompting_extension,
            )

            if controlnet_single_block_residuals is not None:
                img[:, txt.shape[1] :, ...] += controlnet_single_block_residuals[block_index]

        img = img[:, txt.shape[1] :, ...]

        final_mod = self.get_modulations(mod_vectors, "final")
        img = self.final_layer(img, vec=final_mod)  # (N, T, patch_size ** 2 * out_channels)
        return img
