# Initially pulled from https://github.com/black-forest-labs/flux and adapted for Chroma.
#
# Chroma is a modified, de-distilled FLUX.1-schnell architecture. It shares most of FLUX's
# double/single stream blocks, but replaces the per-block modulation layers with a single
# "distilled guidance" Approximator network whose output is sliced into per-block modulation
# vectors. See invokeai/backend/chroma/model.py for how the modulations are distributed.

import torch
from einops import rearrange
from torch import Tensor, nn

from invokeai.backend.flux.extensions.regional_prompting_extension import RegionalPromptingExtension
from invokeai.backend.flux.math import attention
from invokeai.backend.flux.modules.layers import (
    MLPEmbedder,
    ModulationOut,
    QKNorm,
    RMSNorm,
    SelfAttention,
)


class ChromaModulationOut(ModulationOut):
    @classmethod
    def from_offset(cls, tensor: torch.Tensor, offset: int = 0) -> ModulationOut:
        return cls(
            shift=tensor[:, offset : offset + 1, :],
            scale=tensor[:, offset + 1 : offset + 2, :],
            gate=tensor[:, offset + 2 : offset + 3, :],
        )


class Approximator(nn.Module):
    """The "distilled guidance" network. It maps a per-modulation input vector (built from the
    timestep embedding, the guidance embedding and a positional modulation index) to the full set
    of modulation vectors used by every block in the network."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        n_layers: int = 5,
        dtype=None,
        device=None,
    ):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim, bias=True, dtype=dtype, device=device)
        self.layers = nn.ModuleList([MLPEmbedder(hidden_dim, hidden_dim) for _ in range(n_layers)])
        self.norms = nn.ModuleList([RMSNorm(hidden_dim) for _ in range(n_layers)])
        self.out_proj = nn.Linear(hidden_dim, out_dim, dtype=dtype, device=device)

    @property
    def device(self):
        # Get the device of the module (assumes all parameters are on the same device)
        return next(self.parameters()).device

    def forward(self, x: Tensor) -> Tensor:
        x = self.in_proj(x)

        for layer, norms in zip(self.layers, self.norms, strict=False):
            x = x + layer(norms(x))

        x = self.out_proj(x)

        return x


class DoubleStreamBlock(nn.Module):
    """FLUX double stream block, but without its own modulation layers — the modulation vectors are
    provided externally (computed by the Approximator)."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False):
        super().__init__()

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

    def forward(
        self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor, attn_mask: Tensor | None
    ) -> tuple[Tensor, Tensor, Tensor]:
        (img_mod1, img_mod2), (txt_mod1, txt_mod2) = vec

        # prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "b n (qkv h d) -> qkv b h n d", qkv=3, h=self.num_heads)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "b n (qkv h d) -> qkv b h n d", qkv=3, h=self.num_heads)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        attn = attention(
            torch.cat((txt_q, img_q), dim=2),
            torch.cat((txt_k, img_k), dim=2),
            torch.cat((txt_v, img_v), dim=2),
            pe=pe,
            attn_mask=attn_mask,
        )

        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

        # calculate the img blocks
        img = img.addcmul(img_mod1.gate, self.img_attn.proj(img_attn))
        img = img.addcmul(img_mod2.gate, self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift))

        # calculate the txt blocks
        txt = txt.addcmul(txt_mod1.gate, self.txt_attn.proj(txt_attn))
        txt = txt.addcmul(txt_mod2.gate, self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift))

        if txt.dtype == torch.float16:
            txt = torch.nan_to_num(txt, nan=0.0, posinf=65504, neginf=-65504)

        return img, txt, img_q


class CustomDoubleStreamBlockProcessor:
    @staticmethod
    def custom_double_block_forward(
        block_index: int,
        block: DoubleStreamBlock,
        img: Tensor,
        txt: Tensor,
        vec,
        pe: Tensor,
        regional_prompting_extension: RegionalPromptingExtension,
    ) -> tuple[Tensor, Tensor]:
        attn_mask = regional_prompting_extension.get_double_stream_attn_mask(block_index)
        img, txt, _img_q = block(img, txt, vec, pe, attn_mask=attn_mask)
        return img, txt


class SingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface. The modulation vector is
    provided externally (computed by the Approximator).
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: float | None = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # qkv and mlp_in
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        # proj and mlp_out
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)

        self.norm = QKNorm(head_dim)

        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp_act = nn.GELU(approximate="tanh")

    def forward(self, x: Tensor, vec: ModulationOut, pe: Tensor, attn_mask: Tensor | None) -> Tensor:
        mod = vec
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)

        # compute attention
        attn = attention(q, k, v, pe=pe, attn_mask=attn_mask)
        # compute activation in mlp stream, cat again and run second linear layer
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        x = x.addcmul(mod.gate, output)
        if x.dtype == torch.float16:
            x = torch.nan_to_num(x, nan=0.0, posinf=65504, neginf=-65504)
        return x


class CustomSingleStreamBlockProcessor:
    @staticmethod
    def custom_single_block_forward(
        block_index: int,
        block: SingleStreamBlock,
        img: Tensor,
        vec,
        pe: Tensor,
        regional_prompting_extension: RegionalPromptingExtension,
    ):
        attn_mask = regional_prompting_extension.get_single_stream_attn_mask(block_index)
        return block(img, vec, pe, attn_mask=attn_mask)


class LastLayer(nn.Module):
    """Chroma final layer. Unlike FLUX, the (shift, scale) modulation is supplied externally rather
    than computed by an internal adaLN_modulation layer."""

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = vec
        shift = shift.squeeze(1)
        scale = scale.squeeze(1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x
