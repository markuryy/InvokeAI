import type { ComboboxOnChange, ComboboxOption } from '@invoke-ai/ui-library';
import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { selectChromaSchedule, setChromaSchedule } from 'features/controlLayers/store/paramsSlice';
import { isParameterChromaSchedule } from 'features/parameters/types/parameterSchemas';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

// Chroma timestep schedule options. These control where steps are placed on the noise timeline.
const CHROMA_SCHEDULE_OPTIONS: ComboboxOption[] = [
  { value: 'shifted', label: 'Shifted (Flux shift)' },
  { value: 'linear', label: 'Linear (shift = 1)' },
  { value: 'sine', label: 'Sine' },
];

const ParamChromaScheduler = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const chromaSchedule = useAppSelector(selectChromaSchedule);

  const onChange = useCallback<ComboboxOnChange>(
    (v) => {
      if (!isParameterChromaSchedule(v?.value)) {
        return;
      }
      dispatch(setChromaSchedule(v.value));
    },
    [dispatch]
  );

  const value = useMemo(() => CHROMA_SCHEDULE_OPTIONS.find((o) => o.value === chromaSchedule), [chromaSchedule]);

  return (
    <FormControl>
      <InformationalPopover feature="paramScheduler">
        <FormLabel>{t('parameters.scheduler')}</FormLabel>
      </InformationalPopover>
      <Combobox value={value} options={CHROMA_SCHEDULE_OPTIONS} onChange={onChange} />
    </FormControl>
  );
};

export default memo(ParamChromaScheduler);
