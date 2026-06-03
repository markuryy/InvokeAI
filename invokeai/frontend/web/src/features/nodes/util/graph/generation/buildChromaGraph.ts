import { logger } from 'app/logging/logger';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { selectMainModelConfig, selectParamsSlice } from 'features/controlLayers/store/paramsSlice';
import { selectCanvasMetadata, selectCanvasSlice } from 'features/controlLayers/store/selectors';
import { addControlNets } from 'features/nodes/util/graph/generation/addControlAdapters';
import { addImageToImage } from 'features/nodes/util/graph/generation/addImageToImage';
import { addInpaint } from 'features/nodes/util/graph/generation/addInpaint';
import { addNSFWChecker } from 'features/nodes/util/graph/generation/addNSFWChecker';
import { addOutpaint } from 'features/nodes/util/graph/generation/addOutpaint';
import { addRegions } from 'features/nodes/util/graph/generation/addRegions';
import { addTextToImage } from 'features/nodes/util/graph/generation/addTextToImage';
import { addWatermarker } from 'features/nodes/util/graph/generation/addWatermarker';
import { Graph } from 'features/nodes/util/graph/generation/Graph';
import { selectCanvasOutputFields, selectPresetModifiedPrompts } from 'features/nodes/util/graph/graphBuilderUtils';
import type { GraphBuilderArg, GraphBuilderReturn, ImageOutputNodes } from 'features/nodes/util/graph/types';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import type { Invocation } from 'services/api/types';
import type { Equals } from 'tsafe';
import { assert } from 'tsafe';

import { addChromaLoRAs } from './addChromaLoRAs';

const log = logger('system');

export const buildChromaGraph = async (arg: GraphBuilderArg): Promise<GraphBuilderReturn> => {
  const { generationMode, state, manager } = arg;

  log.debug({ generationMode, manager: manager?.id }, 'Building Chroma graph');

  const model = selectMainModelConfig(state);
  assert(model, 'No model found in state');
  assert(model.base === 'chroma', 'Selected model is not a Chroma model');

  const params = selectParamsSlice(state);
  const canvas = selectCanvasSlice(state);

  const { cfgScale: cfg_scale, steps, fluxVAE, t5EncoderModel, chromaSchedule } = params;

  assert(t5EncoderModel, 'No T5 Encoder model found in state');
  assert(fluxVAE, 'No FLUX VAE model found in state');

  const prompts = selectPresetModifiedPrompts(state);

  const g = new Graph(getPrefixedId('chroma_graph'));

  const modelLoader = g.addNode({
    type: 'chroma_model_loader',
    id: getPrefixedId('chroma_model_loader'),
    model,
    t5_encoder_model: t5EncoderModel,
    vae_model: fluxVAE,
  });

  const positivePrompt = g.addNode({
    id: getPrefixedId('positive_prompt'),
    type: 'string',
  });
  const posCond = g.addNode({
    type: 'chroma_text_encoder',
    id: getPrefixedId('pos_cond'),
  });
  const negCond = g.addNode({
    type: 'chroma_text_encoder',
    id: getPrefixedId('neg_cond'),
    prompt: prompts.negative,
  });

  const seed = g.addNode({
    id: getPrefixedId('seed'),
    type: 'integer',
  });
  const denoise = g.addNode({
    type: 'chroma_denoise',
    id: getPrefixedId('chroma_denoise'),
    cfg_scale,
    num_steps: steps,
    schedule: chromaSchedule,
    denoising_start: 0,
    denoising_end: 1,
  });
  const l2i = g.addNode({
    type: 'flux_vae_decode',
    id: getPrefixedId('flux_vae_decode'),
  });

  g.addEdge(modelLoader, 'transformer', denoise, 'transformer');
  g.addEdge(modelLoader, 't5_encoder', posCond, 't5_encoder');
  g.addEdge(modelLoader, 't5_encoder', negCond, 't5_encoder');
  g.addEdge(modelLoader, 'vae', l2i, 'vae');

  // Route conditioning through collectors so regional guidance can append per-region conditioning. With no
  // regions, each collector simply carries the single base conditioning (chroma_denoise accepts a list).
  const posCondCollect = g.addNode({
    type: 'collect',
    id: getPrefixedId('pos_cond_collect'),
  });
  const negCondCollect = g.addNode({
    type: 'collect',
    id: getPrefixedId('neg_cond_collect'),
  });

  g.addEdge(positivePrompt, 'value', posCond, 'prompt');
  g.addEdge(posCond, 'conditioning', posCondCollect, 'item');
  g.addEdge(negCond, 'conditioning', negCondCollect, 'item');
  g.addEdge(posCondCollect, 'collection', denoise, 'positive_text_conditioning');
  g.addEdge(negCondCollect, 'collection', denoise, 'negative_text_conditioning');

  g.addEdge(seed, 'value', denoise, 'seed');
  g.addEdge(denoise, 'latents', l2i, 'latents');

  addChromaLoRAs(state, g, denoise, modelLoader, posCond, negCond);

  // ControlNet. Chroma reuses FLUX-architecture ControlNets (their block residuals are dimensionally
  // compatible with Chroma's transformer). Requires the canvas manager to rasterize control layers.
  if (manager !== null) {
    const controlNetCollector = g.addNode({
      type: 'collect',
      id: getPrefixedId('control_net_collector'),
    });
    const controlNetResult = await addControlNets({
      manager,
      entities: canvas.controlLayers.entities,
      g,
      rect: canvas.bbox.rect,
      collector: controlNetCollector,
      model,
    });
    if (controlNetResult.addedControlNets > 0) {
      g.addEdge(controlNetCollector, 'collection', denoise, 'control');
      // InstantX/Union ControlNets VAE-encode their control image; reuse the FLUX VAE from the loader.
      g.addEdge(modelLoader, 'vae', denoise, 'controlnet_vae');
    } else {
      g.deleteNode(controlNetCollector.id);
    }

    // Regional guidance. Unlike FLUX, Chroma uses CFG, so regional negative prompts and auto-negative all work.
    // Chroma has no IP-Adapter/Redux support, so the ip-adapter collector is only a placeholder (regions with
    // reference images are rejected by the validator) and is removed if unused.
    const ipAdapterCollect = g.addNode({
      type: 'collect',
      id: getPrefixedId('ip_adapter_collect'),
    });
    const regionsResults = await addRegions({
      manager,
      regions: canvas.regionalGuidance.entities,
      g,
      bbox: canvas.bbox.rect,
      model,
      posCond,
      negCond,
      posCondCollect,
      negCondCollect,
      ipAdapterCollect,
      fluxReduxCollect: null,
    });
    const addedIPAdapters = regionsResults.reduce((count, r) => count + r.addedIPAdapters, 0);
    if (addedIPAdapters === 0) {
      g.deleteNode(ipAdapterCollect.id);
    }
  }

  g.upsertMetadata({
    cfg_scale,
    negative_prompt: prompts.negative,
    model: Graph.getModelMetadataField(model),
    steps,
    chroma_schedule: chromaSchedule,
    vae: fluxVAE,
    t5_encoder: t5EncoderModel,
  });
  g.addEdgeToMetadata(seed, 'value', 'seed');
  g.addEdgeToMetadata(positivePrompt, 'value', 'positive_prompt');

  let canvasOutput: Invocation<ImageOutputNodes> = l2i;

  if (generationMode === 'txt2img') {
    canvasOutput = addTextToImage({
      g,
      state,
      denoise,
      l2i,
    });
    g.upsertMetadata({ generation_mode: 'chroma_txt2img' });
  } else if (generationMode === 'img2img') {
    assert(manager !== null);
    const i2l = g.addNode({
      type: 'flux_vae_encode',
      id: getPrefixedId('flux_vae_encode'),
    });
    canvasOutput = await addImageToImage({
      g,
      state,
      manager,
      l2i,
      i2l,
      denoise,
      vaeSource: modelLoader,
    });
    g.upsertMetadata({ generation_mode: 'chroma_img2img' });
  } else if (generationMode === 'inpaint') {
    assert(manager !== null);
    const i2l = g.addNode({
      type: 'flux_vae_encode',
      id: getPrefixedId('flux_vae_encode'),
    });
    canvasOutput = await addInpaint({
      g,
      state,
      manager,
      l2i,
      i2l,
      denoise,
      vaeSource: modelLoader,
      modelLoader,
      seed,
    });
    g.upsertMetadata({ generation_mode: 'chroma_inpaint' });
  } else if (generationMode === 'outpaint') {
    assert(manager !== null);
    const i2l = g.addNode({
      type: 'flux_vae_encode',
      id: getPrefixedId('flux_vae_encode'),
    });
    canvasOutput = await addOutpaint({
      g,
      state,
      manager,
      l2i,
      i2l,
      denoise,
      vaeSource: modelLoader,
      modelLoader,
      seed,
    });
    g.upsertMetadata({ generation_mode: 'chroma_outpaint' });
  } else {
    assert<Equals<typeof generationMode, never>>(false);
  }

  if (state.system.shouldUseNSFWChecker) {
    canvasOutput = addNSFWChecker(g, canvasOutput);
  }

  if (state.system.shouldUseWatermarker) {
    canvasOutput = addWatermarker(g, canvasOutput);
  }

  g.updateNode(canvasOutput, selectCanvasOutputFields(state));

  if (selectActiveTab(state) === 'canvas') {
    g.upsertMetadata(selectCanvasMetadata(state));
  }

  g.setMetadataReceivingNode(canvasOutput);
  return {
    g,
    seed,
    positivePrompt,
  };
};
