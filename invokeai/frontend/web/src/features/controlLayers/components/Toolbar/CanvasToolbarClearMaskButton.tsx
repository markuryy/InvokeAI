import { Button } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useEntityAdapterSafe } from 'features/controlLayers/contexts/EntityAdapterContext';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { useEntityIsLocked } from 'features/controlLayers/hooks/useEntityIsLocked';
import { entityReset } from 'features/controlLayers/store/canvasSlice';
import { selectSelectedEntityIdentifier } from 'features/controlLayers/store/selectors';
import { isMaskEntityIdentifier } from 'features/controlLayers/store/types';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiEraserBold } from 'react-icons/pi';

/**
 * Clears the contents of the currently-selected mask (inpaint mask / regional guidance), leaving an empty mask
 * layer - it does NOT delete the layer. Mirrors the "resetSelected" hotkey (see useCanvasResetLayerHotkey) and is
 * only shown when a mask entity is being edited, next to the brush-size picker in the canvas toolbar.
 */
export const CanvasToolbarClearMaskButton = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const entityIdentifier = useAppSelector(selectSelectedEntityIdentifier);
  const isBusy = useCanvasIsBusy();
  const adapter = useEntityAdapterSafe(entityIdentifier);
  const isLocked = useEntityIsLocked(entityIdentifier);

  const onClick = useCallback(() => {
    if (entityIdentifier === null || adapter === null || !isMaskEntityIdentifier(entityIdentifier)) {
      return;
    }
    adapter.bufferRenderer.clearBuffer();
    dispatch(entityReset({ entityIdentifier }));
  }, [adapter, dispatch, entityIdentifier]);

  if (entityIdentifier === null || !isMaskEntityIdentifier(entityIdentifier)) {
    return null;
  }

  return (
    <Button
      onClick={onClick}
      isDisabled={isBusy || isLocked}
      size="sm"
      variant="ghost"
      leftIcon={<PiEraserBold />}
      flexShrink={0}
    >
      {t('controlLayers.clearMask')}
    </Button>
  );
});

CanvasToolbarClearMaskButton.displayName = 'CanvasToolbarClearMaskButton';
