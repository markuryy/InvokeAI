from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import FieldDescriptions, InputField, OutputField
from invokeai.app.invocations.model import ModelIdentifierField, T5EncoderField, TransformerField, VAEField
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.app.util.t5_model_identifier import (
    preprocess_t5_encoder_model_identifier,
    preprocess_t5_tokenizer_model_identifier,
)
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType, SubModelType


@invocation_output("chroma_model_loader_output")
class ChromaModelLoaderOutput(BaseInvocationOutput):
    """Chroma base model loader output."""

    transformer: TransformerField = OutputField(description=FieldDescriptions.transformer, title="Transformer")
    t5_encoder: T5EncoderField = OutputField(description=FieldDescriptions.t5_encoder, title="T5 Encoder")
    vae: VAEField = OutputField(description=FieldDescriptions.vae, title="VAE")


@invocation(
    "chroma_model_loader",
    title="Main Model - Chroma",
    tags=["model", "chroma"],
    category="model",
    version="1.0.0",
)
class ChromaModelLoaderInvocation(BaseInvocation):
    """Loads a Chroma base model, outputting its submodels.

    Chroma is a de-distilled FLUX.1-schnell variant. It uses only the T5 text encoder (no CLIP) and
    shares the FLUX VAE.
    """

    model: ModelIdentifierField = InputField(
        description=FieldDescriptions.flux_model,
        ui_model_base=BaseModelType.Chroma,
        ui_model_type=ModelType.Main,
        title="Model",
    )

    t5_encoder_model: ModelIdentifierField = InputField(
        description=FieldDescriptions.t5_encoder,
        title="T5 Encoder",
        ui_model_type=ModelType.T5Encoder,
    )

    vae_model: ModelIdentifierField = InputField(
        description=FieldDescriptions.vae_model,
        title="VAE",
        ui_model_base=BaseModelType.Flux,
        ui_model_type=ModelType.VAE,
    )

    def invoke(self, context: InvocationContext) -> ChromaModelLoaderOutput:
        for key in [self.model.key, self.t5_encoder_model.key, self.vae_model.key]:
            if not context.models.exists(key):
                raise ValueError(f"Unknown model: {key}")

        transformer = self.model.model_copy(update={"submodel_type": SubModelType.Transformer})
        vae = self.vae_model.model_copy(update={"submodel_type": SubModelType.VAE})

        tokenizer2 = preprocess_t5_tokenizer_model_identifier(self.t5_encoder_model)
        t5_encoder = preprocess_t5_encoder_model_identifier(self.t5_encoder_model)

        return ChromaModelLoaderOutput(
            transformer=TransformerField(transformer=transformer, loras=[]),
            t5_encoder=T5EncoderField(tokenizer=tokenizer2, text_encoder=t5_encoder, loras=[]),
            vae=VAEField(vae=vae),
        )
