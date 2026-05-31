from contextlib import ExitStack
from typing import Iterator, Optional, Tuple

import torch
from transformers import T5EncoderModel, T5Tokenizer, T5TokenizerFast

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import (
    FieldDescriptions,
    FluxConditioningField,
    Input,
    InputField,
    TensorField,
    UIComponent,
)
from invokeai.app.invocations.model import T5EncoderField
from invokeai.app.invocations.primitives import FluxConditioningOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.model_manager.taxonomy import ModelFormat
from invokeai.backend.patches.layer_patcher import LayerPatcher
from invokeai.backend.patches.lora_conversions.flux_lora_constants import FLUX_LORA_T5_PREFIX
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import ConditioningFieldData, FLUXConditioningInfo
from invokeai.backend.util.devices import TorchDevice


@invocation(
    "chroma_text_encoder",
    title="Prompt - Chroma",
    tags=["prompt", "conditioning", "chroma"],
    category="conditioning",
    version="1.0.0",
)
class ChromaTextEncoderInvocation(BaseInvocation):
    """Encodes and preps a prompt for a Chroma image.

    Chroma uses only the T5 text encoder (no CLIP). Unlike FLUX, the prompt is not padded to a fixed
    max sequence length; the model was trained with minimal padding.
    """

    t5_encoder: T5EncoderField = InputField(
        title="T5Encoder",
        description=FieldDescriptions.t5_encoder,
        input=Input.Connection,
    )
    prompt: str = InputField(description="Text prompt to encode.", ui_component=UIComponent.Textarea)
    mask: Optional[TensorField] = InputField(
        default=None, description="A mask defining the region that this conditioning prompt applies to."
    )
    padding: int = InputField(
        default=1,
        ge=0,
        le=512,
        description="Number of padding tokens to add to the end of the prompt. Chroma was trained with padding=1.",
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> FluxConditioningOutput:
        t5_embeddings = self._t5_encode(context)

        # Chroma does not use CLIP. We store an empty CLIP embedding so the conditioning is compatible
        # with the shared FLUX conditioning data structure used downstream.
        clip_embeddings = torch.zeros((1, 0), dtype=t5_embeddings.dtype)

        t5_embeddings = t5_embeddings.detach().to("cpu")

        conditioning_data = ConditioningFieldData(
            conditionings=[FLUXConditioningInfo(clip_embeds=clip_embeddings, t5_embeds=t5_embeddings)]
        )

        conditioning_name = context.conditioning.save(conditioning_data)
        return FluxConditioningOutput(
            conditioning=FluxConditioningField(conditioning_name=conditioning_name, mask=self.mask)
        )

    def _tokenize(self, tokenizer, text: str) -> torch.Tensor:
        # Per the Chroma authors, T5 text embeddings should not be padded to a fixed max_length, though
        # adding a padding token or two at the end can help.
        results = tokenizer(
            text,
            truncation=False,
            max_length=None,
            return_length=False,
            return_overflowing_tokens=False,
            padding=False,
            return_tensors="pt",
        )
        input_ids = results.input_ids
        if self.padding > 0:
            pad = torch.full((1, self.padding), tokenizer.pad_token_id, dtype=input_ids.dtype)
            input_ids = torch.cat((input_ids, pad), dim=1)
        return input_ids

    def _t5_encode(self, context: InvocationContext) -> torch.Tensor:
        t5_encoder_info = context.models.load(self.t5_encoder.text_encoder)
        t5_encoder_config = t5_encoder_info.config
        assert t5_encoder_config is not None

        with (
            t5_encoder_info.model_on_device() as (cached_weights, t5_text_encoder),
            context.models.load(self.t5_encoder.tokenizer) as t5_tokenizer,
            ExitStack() as exit_stack,
        ):
            assert isinstance(t5_text_encoder, T5EncoderModel)
            assert isinstance(t5_tokenizer, (T5Tokenizer, T5TokenizerFast))

            # Determine if the model is quantized. Quantized models require sidecar LoRA patching.
            if t5_encoder_config.format in [ModelFormat.T5Encoder, ModelFormat.Diffusers]:
                model_is_quantized = False
            elif t5_encoder_config.format in [
                ModelFormat.BnbQuantizedLlmInt8b,
                ModelFormat.BnbQuantizednf4b,
                ModelFormat.GGUFQuantized,
            ]:
                model_is_quantized = True
            else:
                raise ValueError(f"Unsupported model format: {t5_encoder_config.format}")

            exit_stack.enter_context(
                LayerPatcher.apply_smart_model_patches(
                    model=t5_text_encoder,
                    patches=self._t5_lora_iterator(context),
                    prefix=FLUX_LORA_T5_PREFIX,
                    dtype=t5_text_encoder.dtype,
                    cached_weights=cached_weights,
                    force_sidecar_patching=model_is_quantized,
                )
            )

            input_ids = self._tokenize(t5_tokenizer, self.prompt).to(TorchDevice.choose_torch_device())

            context.util.signal_progress("Running T5 encoder")
            prompt_embeds = t5_text_encoder(
                input_ids=input_ids,
                # We don't pad to a fixed length, so an attention mask is unnecessary.
                attention_mask=None,
                output_hidden_states=False,
            ).last_hidden_state

        assert isinstance(prompt_embeds, torch.Tensor)
        return prompt_embeds

    def _t5_lora_iterator(self, context: InvocationContext) -> Iterator[Tuple[ModelPatchRaw, float]]:
        for lora in self.t5_encoder.loras:
            lora_info = context.models.load(lora.lora)
            assert isinstance(lora_info.model, ModelPatchRaw)
            yield (lora_info.model, lora.weight)
            del lora_info
