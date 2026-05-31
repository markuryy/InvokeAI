import type { RootState } from 'app/store/store';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { zModelIdentifierField } from 'features/nodes/types/common';
import type { Graph } from 'features/nodes/util/graph/generation/Graph';
import type { Invocation, S } from 'services/api/types';

/**
 * Adds LoRAs to a Chroma graph.
 *
 * Chroma shares FLUX's attention/MLP block structure, so it uses FLUX-format LoRAs (which probe as
 * `base: 'flux'`). They are applied to the Chroma transformer and T5 encoder via the Chroma-specific
 * LoRA collection loader (Chroma has no CLIP). LoRA layers that target FLUX-only modulation layers
 * are skipped at patch time, so generic FLUX LoRAs apply with tolerance while Chroma-specific LoRAs
 * apply fully.
 */
export const addChromaLoRAs = (
  state: RootState,
  g: Graph,
  denoise: Invocation<'chroma_denoise'>,
  modelLoader: Invocation<'chroma_model_loader'>,
  posCond: Invocation<'chroma_text_encoder'>,
  negCond: Invocation<'chroma_text_encoder'>
): void => {
  const enabledLoRAs = state.loras.loras.filter((l) => l.isEnabled && l.model.base === 'flux');
  const loraCount = enabledLoRAs.length;

  if (loraCount === 0) {
    return;
  }

  const loraMetadata: S['LoRAMetadataField'][] = [];

  // Collect LoRAs into a single collection node, then pass them to the Chroma LoRA collection loader,
  // which applies each LoRA to the transformer and T5 encoder.
  const loraCollector = g.addNode({
    id: getPrefixedId('lora_collector'),
    type: 'collect',
  });
  const loraCollectionLoader = g.addNode({
    type: 'chroma_lora_collection_loader',
    id: getPrefixedId('chroma_lora_collection_loader'),
  });

  g.addEdge(loraCollector, 'collection', loraCollectionLoader, 'loras');
  // Feed the model loader's transformer + T5 encoder through the LoRA collection loader.
  g.addEdge(modelLoader, 'transformer', loraCollectionLoader, 'transformer');
  g.addEdge(modelLoader, 't5_encoder', loraCollectionLoader, 't5_encoder');
  // Reroute model connections through the LoRA collection loader.
  g.deleteEdgesTo(denoise, ['transformer']);
  g.deleteEdgesTo(posCond, ['t5_encoder']);
  g.deleteEdgesTo(negCond, ['t5_encoder']);
  g.addEdge(loraCollectionLoader, 'transformer', denoise, 'transformer');
  g.addEdge(loraCollectionLoader, 't5_encoder', posCond, 't5_encoder');
  g.addEdge(loraCollectionLoader, 't5_encoder', negCond, 't5_encoder');

  for (const lora of enabledLoRAs) {
    const { weight } = lora;
    const parsedModel = zModelIdentifierField.parse(lora.model);

    const loraSelector = g.addNode({
      type: 'lora_selector',
      id: getPrefixedId('lora_selector'),
      lora: parsedModel,
      weight,
    });

    loraMetadata.push({
      model: parsedModel,
      weight,
    });

    g.addEdge(loraSelector, 'lora', loraCollector, 'item');
  }

  g.upsertMetadata({ loras: loraMetadata });
};
