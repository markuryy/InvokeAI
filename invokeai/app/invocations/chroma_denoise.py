import re
from contextlib import ExitStack
from typing import Callable, Iterator, Literal, Optional, Tuple

import torch

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import (
    DenoiseMaskField,
    FieldDescriptions,
    FluxConditioningField,
    Input,
    InputField,
    LatentsField,
    WithMetadata,
)
from invokeai.app.invocations.flux_controlnet import FluxControlNetField
from invokeai.app.invocations.flux_denoise import FluxDenoiseInvocation
from invokeai.app.invocations.model import TransformerField, VAEField
from invokeai.app.invocations.primitives import LatentsOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.chroma.denoise import denoise
from invokeai.backend.chroma.model import Chroma
from invokeai.backend.chroma.sampling_utils import get_schedule, get_schedule_sinelike
from invokeai.backend.flux.extensions.regional_prompting_extension import RegionalPromptingExtension
from invokeai.backend.flux.sampling_utils import (
    clip_timestep_schedule_fractional,
    generate_img_ids,
    get_noise,
    pack,
    unpack,
)
from invokeai.backend.flux.text_conditioning import FluxTextConditioning
from invokeai.backend.model_manager.taxonomy import ModelFormat
from invokeai.backend.patches.layer_patcher import LayerPatcher
from invokeai.backend.patches.lora_conversions.flux_lora_constants import FLUX_LORA_TRANSFORMER_PREFIX
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from invokeai.backend.rectified_flow.rectified_flow_inpaint_extension import RectifiedFlowInpaintExtension
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import FLUXConditioningInfo
from invokeai.backend.stable_diffusion.extensions.preview import PipelineIntermediateState
from invokeai.backend.util.devices import TorchDevice

ScheduleChoices = Literal["shifted", "linear", "sine"]

# Approximates the sigmoid/shifted curve for the "sine" schedule. No fundamental reason it must be
# this exact value.
DEFAULT_SINE_SMOOTHNESS_FACTOR = 0.7


@invocation(
    "chroma_denoise",
    title="Chroma Denoise",
    tags=["image", "chroma", "txt2img", "img2img"],
    category="image",
    version="1.0.0",
)
class ChromaDenoiseInvocation(BaseInvocation, WithMetadata):
    """Run the denoising process with a Chroma model."""

    # If latents is provided, this means we are doing image-to-image.
    latents: Optional[LatentsField] = InputField(
        default=None, description=FieldDescriptions.latents, input=Input.Connection
    )
    # denoise_mask is used for image-to-image inpainting. Only the masked region is modified.
    denoise_mask: Optional[DenoiseMaskField] = InputField(
        default=None, description=FieldDescriptions.denoise_mask, input=Input.Connection
    )
    denoising_start: float = InputField(default=0.0, ge=0, le=1, description=FieldDescriptions.denoising_start)
    denoising_end: float = InputField(default=1.0, ge=0, le=1, description=FieldDescriptions.denoising_end)
    add_noise: bool = InputField(default=True, description="Add noise based on denoising start.")

    transformer: TransformerField = InputField(
        description=FieldDescriptions.flux_model, input=Input.Connection, title="Transformer"
    )
    positive_text_conditioning: FluxConditioningField | list[FluxConditioningField] = InputField(
        description=FieldDescriptions.positive_cond, input=Input.Connection
    )
    negative_text_conditioning: FluxConditioningField | list[FluxConditioningField] | None = InputField(
        default=None,
        description="Negative conditioning tensor. Can be None if cfg_scale is 1.0.",
        input=Input.Connection,
    )
    control: FluxControlNetField | list[FluxControlNetField] | None = InputField(
        default=None,
        input=Input.Connection,
        description="ControlNet models. Chroma reuses FLUX-architecture ControlNets (e.g. InstantX/Union, XLabs); "
        "they generally need lower control weights (~0.3-0.5) on Chroma.",
    )
    controlnet_vae: VAEField | None = InputField(
        default=None,
        description=FieldDescriptions.vae,
        input=Input.Connection,
    )
    cfg_scale: float | list[float] = InputField(default=4.0, description=FieldDescriptions.cfg_scale, title="CFG Scale")
    cfg_scale_start_step: int = InputField(
        default=0,
        title="CFG Scale Start Step",
        description="Index of the first step to apply cfg_scale. Negative indices count backwards from the "
        + "last step (e.g. a value of -1 refers to the final step).",
    )
    cfg_scale_end_step: int = InputField(
        default=-1,
        title="CFG Scale End Step",
        description="Index of the last step to apply cfg_scale. Negative indices count backwards from the "
        + "last step (e.g. a value of -1 refers to the final step).",
    )
    width: int = InputField(default=1024, multiple_of=16, gt=0, description="Width of the generated image.")
    height: int = InputField(default=1024, multiple_of=16, gt=0, description="Height of the generated image.")
    num_steps: int = InputField(
        default=26, ge=1, description="Number of diffusion steps. Recommended: ~26 for Chroma."
    )
    schedule: ScheduleChoices = InputField(
        default="shifted",
        description="Timestep schedule. 'shifted' places more steps at high noise (resolution-dependent), "
        "'linear' spaces them evenly, 'sine' emphasizes the start and end.",
    )
    seed: int = InputField(default=0, description="Randomness seed for reproducibility.")

    # NOTE: Use no_grad (not inference_mode) to match FluxDenoiseInvocation. inference_mode marks newly
    # created tensors as "inference tensors", which cannot be wrapped as nn.Parameter (requires_grad=True).
    # The shared FLUX ControlNet loader does exactly that (load_state_dict(assign=True)) when a ControlNet is
    # loaded on a cold cache during this invocation, so inference_mode would raise.
    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> LatentsOutput:
        latents = self._run_diffusion(context)
        latents = latents.detach().to("cpu")
        name = context.tensors.save(tensor=latents)
        return LatentsOutput.build(latents_name=name, latents=latents, seed=None)

    def _run_diffusion(self, context: InvocationContext) -> torch.Tensor:
        inference_dtype = torch.bfloat16
        device = TorchDevice.choose_torch_device()

        # Load the conditioning data.
        transformer_info = context.models.load(self.transformer.transformer)

        # Prepare noise / dimensions.
        noise = get_noise(
            num_samples=1,
            height=self.height,
            width=self.width,
            device=device,
            dtype=inference_dtype,
            seed=self.seed,
        )
        b, _c, latent_h, latent_w = noise.shape
        packed_h = latent_h // 2
        packed_w = latent_w // 2
        img_seq_len = packed_h * packed_w

        # Load and prepare conditioning + regional prompting extensions.
        pos_text_conditionings = self._load_text_conditioning(
            context, self.positive_text_conditioning, packed_h, packed_w, inference_dtype, device
        )
        pos_regional_prompting_extension = RegionalPromptingExtension.from_text_conditioning(
            text_conditioning=pos_text_conditionings, redux_conditioning=[], img_seq_len=img_seq_len
        )

        neg_regional_prompting_extension: RegionalPromptingExtension | None = None
        if self.negative_text_conditioning is not None:
            neg_text_conditionings = self._load_text_conditioning(
                context, self.negative_text_conditioning, packed_h, packed_w, inference_dtype, device
            )
            neg_regional_prompting_extension = RegionalPromptingExtension.from_text_conditioning(
                text_conditioning=neg_text_conditionings, redux_conditioning=[], img_seq_len=img_seq_len
            )

        # Build the timestep schedule.
        timesteps = self._get_timesteps(img_seq_len)

        # Load the input latents, if provided (image-to-image).
        init_latents: torch.Tensor | None = None
        if self.latents is not None:
            init_latents = context.tensors.load(self.latents.latents_name).to(device=device, dtype=inference_dtype)

        # Prepare the initial latents.
        add_noise_factor = timesteps[0] if self.add_noise else 0.0
        if init_latents is not None:
            if add_noise_factor:
                x = torch.lerp(init_latents, noise, add_noise_factor)
            else:
                x = init_latents
        else:
            x = noise

        # Inpaint extension (image-to-image inpainting).
        inpaint_extension: RectifiedFlowInpaintExtension | None = None
        if init_latents is not None:
            inpaint_mask = FluxDenoiseInvocation._prep_inpaint_mask(self, context, init_latents)
            if inpaint_mask is not None:
                inpaint_extension = RectifiedFlowInpaintExtension(
                    init_latents=pack(init_latents),
                    inpaint_mask=pack(inpaint_mask),
                    noise=pack(noise),
                )

        x = pack(x)
        img_ids = generate_img_ids(h=latent_h, w=latent_w, batch_size=b, device=device, dtype=inference_dtype)

        cfg_scale = FluxDenoiseInvocation.prep_cfg_scale(
            self.cfg_scale, timesteps, self.cfg_scale_start_step, self.cfg_scale_end_step
        )

        with ExitStack() as exit_stack:
            # Prepare ControlNet extensions before loading the transformer to keep peak memory down. These reuse
            # FLUX's ControlNet model/extension stack (the residuals are dimensionally compatible with Chroma's
            # blocks); _prep_controlnet_extensions only reads self.control / self.controlnet_vae.
            controlnet_extensions = FluxDenoiseInvocation._prep_controlnet_extensions(
                self,  # type: ignore[arg-type]
                context=context,
                exit_stack=exit_stack,
                latent_height=latent_h,
                latent_width=latent_w,
                dtype=inference_dtype,
                device=device,
            )

            with transformer_info.model_on_device() as (cached_weights, transformer):
                assert isinstance(transformer, Chroma)

                if self.transformer.loras:
                    exit_stack.enter_context(
                        LayerPatcher.apply_smart_model_patches(
                            model=transformer,
                            patches=self._lora_iterator(context),
                            prefix=FLUX_LORA_TRANSFORMER_PREFIX,
                            dtype=inference_dtype,
                            cached_weights=cached_weights,
                            force_sidecar_patching=self._is_quantized(context),
                            # Chroma LoRAs may not include the modulation layers - suppress those warnings.
                            suppress_warning_layers=re.compile(r"(_mod|modulation)\.lin$"),
                        )
                    )
                del cached_weights

                x = denoise(
                    model=transformer,
                    img=x,
                    img_ids=img_ids,
                    pos_regional_prompting_extension=pos_regional_prompting_extension,
                    neg_regional_prompting_extension=neg_regional_prompting_extension,
                    timesteps=timesteps,
                    cfg_scale=cfg_scale,
                    inpaint_extension=inpaint_extension,
                    controlnet_extensions=controlnet_extensions,
                    step_callback=self._build_step_callback(context),
                )

        x = unpack(x.float(), self.height, self.width)
        return x

    def _get_timesteps(self, img_seq_len: int) -> list[float]:
        match self.schedule:
            case "shifted":
                timesteps = get_schedule(num_steps=self.num_steps, image_seq_len=img_seq_len, shift=True)
            case "linear":
                timesteps = get_schedule(num_steps=self.num_steps, image_seq_len=img_seq_len, shift=False)
            case "sine":
                return get_schedule_sinelike(self.num_steps, k=DEFAULT_SINE_SMOOTHNESS_FACTOR)
            case _:
                raise ValueError(f"Unknown schedule: {self.schedule}")
        return clip_timestep_schedule_fractional(timesteps, self.denoising_start, self.denoising_end)

    def _load_text_conditioning(
        self,
        context: InvocationContext,
        cond_field: FluxConditioningField | list[FluxConditioningField],
        packed_height: int,
        packed_width: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> list[FluxTextConditioning]:
        cond_list = [cond_field] if isinstance(cond_field, FluxConditioningField) else cond_field

        text_conditionings: list[FluxTextConditioning] = []
        for field in cond_list:
            cond_data = context.conditioning.load(field.conditioning_name)
            assert len(cond_data.conditionings) == 1
            chroma_conditioning = cond_data.conditionings[0]
            assert isinstance(chroma_conditioning, FLUXConditioningInfo)
            chroma_conditioning = chroma_conditioning.to(dtype=dtype, device=device)
            t5_embeddings = chroma_conditioning.t5_embeds
            clip_embeddings = chroma_conditioning.clip_embeds

            mask: Optional[torch.Tensor] = None
            if field.mask is not None:
                mask = context.tensors.load(field.mask.tensor_name)
                mask = mask.to(device=device)
                mask = RegionalPromptingExtension.preprocess_regional_prompt_mask(
                    mask, packed_height, packed_width, dtype, device
                )

            text_conditionings.append(FluxTextConditioning(t5_embeddings, clip_embeddings, mask))

        return text_conditionings

    def _is_quantized(self, context: InvocationContext) -> bool:
        transformer_config = context.models.get_config(self.transformer.transformer.key)
        if transformer_config.format in [ModelFormat.Checkpoint]:
            return False
        elif transformer_config.format in [
            ModelFormat.BnbQuantizedLlmInt8b,
            ModelFormat.BnbQuantizednf4b,
            ModelFormat.GGUFQuantized,
        ]:
            return True
        raise ValueError(f"Unsupported model format: {transformer_config.format}")

    def _lora_iterator(self, context: InvocationContext) -> Iterator[Tuple[ModelPatchRaw, float]]:
        for lora in self.transformer.loras:
            lora_info = context.models.load(lora.lora)
            assert isinstance(lora_info.model, ModelPatchRaw)
            yield (lora_info.model, lora.weight)
            del lora_info

    def _build_step_callback(self, context: InvocationContext) -> Callable[[PipelineIntermediateState], None]:
        def step_callback(state: PipelineIntermediateState) -> None:
            state.latents = unpack(state.latents.float(), self.height, self.width).squeeze()
            context.util.flux_step_callback(state)

        return step_callback
