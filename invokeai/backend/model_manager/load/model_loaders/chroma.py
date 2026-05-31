# Copyright (c) 2024, the InvokeAI Development Team
"""Class for Chroma model loading in InvokeAI.

Chroma is a de-distilled FLUX.1-schnell variant. It reuses the FLUX VAE and T5 text encoder (both
already supported by InvokeAI), so only the main transformer needs a bespoke loader here.
"""

import contextlib
from pathlib import Path
from typing import Optional

import accelerate
import torch
from safetensors.torch import load_file

from invokeai.backend.chroma.model import Chroma, chroma_params
from invokeai.backend.model_manager.configs.base import Checkpoint_Config_Base
from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.configs.main import Main_Checkpoint_Chroma_Config, Main_GGUF_Chroma_Config
from invokeai.backend.model_manager.load.load_default import ModelLoader
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
from invokeai.backend.model_manager.taxonomy import AnyModel, BaseModelType, ModelFormat, ModelType, SubModelType
from invokeai.backend.quantization.gguf.loaders import gguf_sd_loader
from invokeai.backend.quantization.gguf.utils import TORCH_COMPATIBLE_QTYPES


@contextlib.contextmanager
def _default_dtype(dtype: torch.dtype):
    orig_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(orig_dtype)


def _strip_diffusion_model_prefix(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Some Chroma checkpoints store weights under a ``model.diffusion_model.`` prefix. Strip it so
    the keys match the Chroma module's state dict."""
    prefix = "model.diffusion_model."
    if any(k.startswith(prefix) for k in sd):
        return {k[len(prefix) :] if k.startswith(prefix) else k: v for k, v in sd.items()}
    return sd


def _update_ggml_tensor_ops_for_chroma() -> None:
    """Register the extra GGML tensor ops that the Chroma transformer's forward pass relies on.

    FLUX's GGUF path avoids these ops because partial-loading dequantizes weights before they reach
    the forward pass, but Chroma can hit ``aten.linear`` / ``aten.rms_norm`` directly on quantized
    tensors (e.g. when partial loading is disabled). These additions are no-ops for already-supported
    ops and are safe for other GGUF models, as they simply dequantize before running.
    """
    from invokeai.backend.quantization.gguf.ggml_tensor import (
        GGML_TENSOR_OP_TABLE,
        apply_to_quantized_tensor,
        dequantize_and_run,
    )

    def _ggml_no_shallow_copy(func, args, kwargs):  # pyright: ignore[reportUnusedFunction]
        return False

    more_ops = {
        torch.ops.aten.to.dtype_layout: apply_to_quantized_tensor,  # pyright: ignore
        torch.ops.aten.linear.default: dequantize_and_run,  # pyright: ignore
        torch.ops.aten.rms_norm.default: dequantize_and_run,  # pyright: ignore
        torch.ops.aten._has_compatible_shallow_copy_type.default: _ggml_no_shallow_copy,  # pyright: ignore
    }
    for k, v in more_ops.items():
        GGML_TENSOR_OP_TABLE.setdefault(k, v)


@ModelLoaderRegistry.register(base=BaseModelType.Chroma, type=ModelType.Main, format=ModelFormat.Checkpoint)
class ChromaCheckpointModel(ModelLoader):
    """Class to load Chroma main models from a single safetensors file."""

    def _load_model(self, config: AnyModelConfig, submodel_type: Optional[SubModelType] = None) -> AnyModel:
        if not isinstance(config, Checkpoint_Config_Base):
            raise ValueError("Only CheckpointConfigBase models are currently supported here.")
        if submodel_type is not SubModelType.Transformer:
            raise ValueError(
                f"Only Transformer submodels are currently supported. Received: "
                f"{submodel_type.value if submodel_type else 'None'}"
            )

        assert isinstance(config, Main_Checkpoint_Chroma_Config)
        model_path = Path(config.path)

        with _default_dtype(torch.bfloat16), accelerate.init_empty_weights():
            model = Chroma(chroma_params)

        sd = load_file(model_path)
        sd = _strip_diffusion_model_prefix(sd)
        new_sd_size = sum(ten.nelement() * torch.bfloat16.itemsize for ten in sd.values())
        self._ram_cache.make_room(new_sd_size)
        for k in sd.keys():
            # We need to cast to bfloat16 due to it being the only currently supported dtype for inference.
            sd[k] = sd[k].to(torch.bfloat16)
        model.load_state_dict(sd, assign=True)
        return model


@ModelLoaderRegistry.register(base=BaseModelType.Chroma, type=ModelType.Main, format=ModelFormat.GGUFQuantized)
class ChromaGGUFCheckpointModel(ModelLoader):
    """Class to load GGUF-quantized Chroma main models."""

    def _load_model(self, config: AnyModelConfig, submodel_type: Optional[SubModelType] = None) -> AnyModel:
        if not isinstance(config, Checkpoint_Config_Base):
            raise ValueError("Only CheckpointConfigBase models are currently supported here.")
        if submodel_type is not SubModelType.Transformer:
            raise ValueError(
                f"Only Transformer submodels are currently supported. Received: "
                f"{submodel_type.value if submodel_type else 'None'}"
            )

        assert isinstance(config, Main_GGUF_Chroma_Config)
        model_path = Path(config.path)

        _update_ggml_tensor_ops_for_chroma()

        with _default_dtype(torch.bfloat16), accelerate.init_empty_weights():
            model = Chroma(chroma_params)

        # HACK(ryand): We shouldn't be hard-coding the compute_dtype here.
        sd = gguf_sd_loader(model_path, compute_dtype=torch.bfloat16)
        sd = _strip_diffusion_model_prefix(sd)

        # Guard against broken GGUF models with the wrong shape for img_in.weight (see FLUX loader).
        img_in_weight = sd.get("img_in.weight", None)
        if img_in_weight is not None and img_in_weight._ggml_quantization_type in TORCH_COMPATIBLE_QTYPES:
            expected_img_in_weight_shape = model.img_in.weight.shape
            img_in_weight.quantized_data = img_in_weight.quantized_data.view(expected_img_in_weight_shape)
            img_in_weight.tensor_shape = expected_img_in_weight_shape

        model.load_state_dict(sd, assign=True)
        return model
