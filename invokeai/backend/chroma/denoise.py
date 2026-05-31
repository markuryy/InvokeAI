# Chroma denoising loop.
#
# Chroma uses rectified-flow sampling like FLUX. Unlike FLUX.1-schnell (which is distilled and
# typically run without CFG), Chroma is de-distilled and benefits from classifier-free guidance, so
# the loop supports an optional negative conditioning pass.

import math
from itertools import pairwise
from typing import Callable

import torch
from torch import Tensor
from tqdm import tqdm

from invokeai.backend.chroma.model import Chroma
from invokeai.backend.flux.extensions.regional_prompting_extension import RegionalPromptingExtension
from invokeai.backend.rectified_flow.rectified_flow_inpaint_extension import RectifiedFlowInpaintExtension
from invokeai.backend.stable_diffusion.extensions.preview import PipelineIntermediateState


def denoise(
    model: Chroma,
    # model input
    img: Tensor,
    img_ids: Tensor,
    # conditioning
    pos_regional_prompting_extension: RegionalPromptingExtension,
    neg_regional_prompting_extension: RegionalPromptingExtension | None,
    # sampling parameters
    timesteps: list[float],
    cfg_scale: list[float],
    inpaint_extension: RectifiedFlowInpaintExtension | None = None,
    step_callback: Callable[[PipelineIntermediateState], None] | None = None,
) -> Tensor:
    """Run the Chroma denoising loop, returning the final (packed) latents.

    When ``cfg_scale`` is > 1 and a negative conditioning is provided, classifier-free guidance is
    applied at each step.
    """
    txt = pos_regional_prompting_extension.regional_text_conditioning.t5_embeddings
    txt_ids = pos_regional_prompting_extension.regional_text_conditioning.t5_txt_ids

    # The guidance vector is inert for Chroma (de-distilled from schnell), but the distilled guidance
    # layer still expects an embedding, so we pass zeros.
    guidance_vec = torch.zeros((img.shape[0],), device=img.device, dtype=img.dtype)

    for step_index, (t_curr, t_prev) in tqdm(list(enumerate(pairwise(timesteps))), desc="denoising", smoothing=1):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        delta_t = t_prev - t_curr
        cfg_t = cfg_scale[step_index]

        pred = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            timesteps=t_vec,
            guidance=guidance_vec,
            regional_prompting_extension=pos_regional_prompting_extension,
        )

        if math.isclose(cfg_t, 1.0) or neg_regional_prompting_extension is None:
            preview_img = img - pred
            img = img + pred * delta_t
        else:
            pred_neg = model(
                img=img,
                img_ids=img_ids,
                txt=neg_regional_prompting_extension.regional_text_conditioning.t5_embeddings,
                txt_ids=neg_regional_prompting_extension.regional_text_conditioning.t5_txt_ids,
                timesteps=t_vec,
                guidance=guidance_vec,
                regional_prompting_extension=neg_regional_prompting_extension,
            )
            pred_cfg = pred_neg + (pred - pred_neg) * cfg_t
            preview_img = img - pred_cfg
            img = img + pred_cfg * delta_t

        if inpaint_extension is not None:
            img = inpaint_extension.merge_intermediate_latents_with_init_latents(img, t_prev)
            preview_img = inpaint_extension.merge_intermediate_latents_with_init_latents(preview_img, 0.0)

        if step_callback is not None:
            step_callback(
                PipelineIntermediateState(
                    step=step_index,
                    order=1,
                    total_steps=len(timesteps),
                    timestep=int(1000 * t_curr),
                    latents=preview_img,
                )
            )

    return img
