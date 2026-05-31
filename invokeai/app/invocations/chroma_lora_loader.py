from typing import Optional

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import FieldDescriptions, Input, InputField, OutputField
from invokeai.app.invocations.model import LoRAField, ModelIdentifierField, T5EncoderField, TransformerField
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType


@invocation_output("chroma_lora_loader_output")
class ChromaLoRALoaderOutput(BaseInvocationOutput):
    """Chroma LoRA Loader Output"""

    transformer: Optional[TransformerField] = OutputField(
        default=None, description=FieldDescriptions.transformer, title="Chroma Transformer"
    )
    t5_encoder: Optional[T5EncoderField] = OutputField(
        default=None, description=FieldDescriptions.t5_encoder, title="T5 Encoder"
    )


@invocation(
    "chroma_lora_loader",
    title="Apply LoRA - Chroma",
    tags=["lora", "model", "chroma"],
    category="model",
    version="1.0.0",
)
class ChromaLoRALoaderInvocation(BaseInvocation):
    """Apply a LoRA model to a Chroma transformer and/or T5 text encoder.

    Chroma uses the FLUX-format LoRAs (it shares FLUX's attention/MLP block structure). LoRA layers
    that target FLUX-only modulation layers are skipped at patch time, so generic FLUX LoRAs apply
    with tolerance, while Chroma-specific LoRAs (which may target the distilled guidance layer) apply
    fully. Chroma does not use CLIP, so there is no CLIP output.
    """

    lora: ModelIdentifierField = InputField(
        description=FieldDescriptions.lora_model,
        title="LoRA",
        ui_model_base=BaseModelType.Flux,
        ui_model_type=ModelType.LoRA,
    )
    weight: float = InputField(default=0.75, description=FieldDescriptions.lora_weight)
    transformer: TransformerField | None = InputField(
        default=None,
        description=FieldDescriptions.transformer,
        input=Input.Connection,
        title="Chroma Transformer",
    )
    t5_encoder: T5EncoderField | None = InputField(
        default=None,
        title="T5 Encoder",
        description=FieldDescriptions.t5_encoder,
        input=Input.Connection,
    )

    def invoke(self, context: InvocationContext) -> ChromaLoRALoaderOutput:
        lora_key = self.lora.key

        if not context.models.exists(lora_key):
            raise ValueError(f"Unknown lora: {lora_key}!")

        if self.transformer and any(lora.lora.key == lora_key for lora in self.transformer.loras):
            raise ValueError(f'LoRA "{lora_key}" already applied to transformer.')
        if self.t5_encoder and any(lora.lora.key == lora_key for lora in self.t5_encoder.loras):
            raise ValueError(f'LoRA "{lora_key}" already applied to T5 encoder.')

        output = ChromaLoRALoaderOutput()

        if self.transformer is not None:
            output.transformer = self.transformer.model_copy(deep=True)
            output.transformer.loras.append(LoRAField(lora=self.lora, weight=self.weight))
        if self.t5_encoder is not None:
            output.t5_encoder = self.t5_encoder.model_copy(deep=True)
            output.t5_encoder.loras.append(LoRAField(lora=self.lora, weight=self.weight))

        return output


@invocation(
    "chroma_lora_collection_loader",
    title="Apply LoRA Collection - Chroma",
    tags=["lora", "model", "chroma"],
    category="model",
    version="1.0.0",
)
class ChromaLoRACollectionLoader(BaseInvocation):
    """Applies a collection of LoRAs to a Chroma transformer and T5 text encoder."""

    loras: Optional[LoRAField | list[LoRAField]] = InputField(
        default=None, description="LoRA models and weights. May be a single LoRA or collection.", title="LoRAs"
    )
    transformer: Optional[TransformerField] = InputField(
        default=None,
        description=FieldDescriptions.transformer,
        input=Input.Connection,
        title="Transformer",
    )
    t5_encoder: T5EncoderField | None = InputField(
        default=None,
        title="T5 Encoder",
        description=FieldDescriptions.t5_encoder,
        input=Input.Connection,
    )

    def invoke(self, context: InvocationContext) -> ChromaLoRALoaderOutput:
        output = ChromaLoRALoaderOutput()
        loras = self.loras if isinstance(self.loras, list) else [self.loras]
        added_loras: list[str] = []

        if self.transformer is not None:
            output.transformer = self.transformer.model_copy(deep=True)
        if self.t5_encoder is not None:
            output.t5_encoder = self.t5_encoder.model_copy(deep=True)

        for lora in loras:
            if lora is None:
                continue
            if lora.lora.key in added_loras:
                continue
            if not context.models.exists(lora.lora.key):
                raise Exception(f"Unknown lora: {lora.lora.key}!")

            # Chroma uses FLUX-format LoRAs (it cannot be reliably distinguished from FLUX.1 at the
            # LoRA state-dict level, as it shares FLUX's hidden/mlp dimensions).
            assert lora.lora.base == BaseModelType.Flux

            added_loras.append(lora.lora.key)

            if self.transformer is not None and output.transformer is not None:
                output.transformer.loras.append(lora)
            if self.t5_encoder is not None and output.t5_encoder is not None:
                output.t5_encoder.loras.append(lora)

        return output
