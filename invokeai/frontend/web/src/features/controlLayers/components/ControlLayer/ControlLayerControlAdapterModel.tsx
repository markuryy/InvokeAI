import { Combobox, FormControl, Tooltip } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { useGroupedModelCombobox } from 'common/hooks/useGroupedModelCombobox';
import { selectBase } from 'features/controlLayers/store/paramsSlice';
import { isControlNetCompatibleWithMainModelBase } from 'features/modelManagerV2/models';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useControlLayerModels } from 'services/api/hooks/modelsByType';
import type {
  AnyModelConfig,
  ControlLoRAModelConfig,
  ControlNetModelConfig,
  T2IAdapterModelConfig,
} from 'services/api/types';

type Props = {
  modelKey: string | null;
  onChange: (modelConfig: ControlNetModelConfig | T2IAdapterModelConfig | ControlLoRAModelConfig) => void;
};

export const ControlLayerControlAdapterModel = memo(({ modelKey, onChange: onChangeModel }: Props) => {
  const { t } = useTranslation();
  const currentBaseModel = useAppSelector(selectBase);
  const [modelConfigs, { isLoading }] = useControlLayerModels();
  const selectedModel = useMemo(() => modelConfigs.find((m) => m.key === modelKey), [modelConfigs, modelKey]);

  const _onChange = useCallback(
    (modelConfig: ControlNetModelConfig | T2IAdapterModelConfig | ControlLoRAModelConfig | null) => {
      if (!modelConfig) {
        return;
      }
      onChangeModel(modelConfig);
    },
    [onChangeModel]
  );

  const getIsDisabled = useCallback(
    (model: AnyModelConfig): boolean => {
      if (!currentBaseModel) {
        return true;
      }
      // Chroma reuses FLUX ControlNets, but FLUX Control LoRA has no Chroma support.
      if (currentBaseModel === 'chroma' && model.type === 'control_lora') {
        return true;
      }
      return !isControlNetCompatibleWithMainModelBase(currentBaseModel, model.base);
    },
    [currentBaseModel]
  );

  const isSelectedModelCompatible = useMemo(() => {
    if (!currentBaseModel || !selectedModel) {
      return false;
    }
    if (currentBaseModel === 'chroma' && selectedModel.type === 'control_lora') {
      return false;
    }
    return isControlNetCompatibleWithMainModelBase(currentBaseModel, selectedModel.base);
  }, [currentBaseModel, selectedModel]);

  const { options, value, onChange, noOptionsMessage } = useGroupedModelCombobox({
    modelConfigs,
    onChange: _onChange,
    selectedModel,
    getIsDisabled,
    isLoading,
    groupByType: true,
  });

  return (
    <Tooltip label={selectedModel?.description}>
      <FormControl isInvalid={!value || !isSelectedModelCompatible} w="full">
        <Combobox
          options={options}
          placeholder={t('common.placeholderSelectAModel')}
          value={value}
          onChange={onChange}
          noOptionsMessage={noOptionsMessage}
        />
      </FormControl>
    </Tooltip>
  );
});

ControlLayerControlAdapterModel.displayName = 'ControlLayerControlAdapterModel';
