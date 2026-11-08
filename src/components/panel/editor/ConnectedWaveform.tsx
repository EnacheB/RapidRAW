import { useShallow } from 'zustand/react/shallow';
import Waveform from './Waveform';
import { useEditorStore } from '../../../store/useEditorStore';

interface ConnectedWaveformProps {
  setDisplayMode(mode: string): void;
  theme?: string;
}

export default function ConnectedWaveform({ setDisplayMode, theme }: ConnectedWaveformProps) {
  const { hasWaveform, hasHistogram, activeWaveformChannel, showClipping } = useEditorStore(
    useShallow((state) => ({
      hasWaveform: !!state.waveform,
      hasHistogram: !!state.histogram?.red,
      activeWaveformChannel: state.activeWaveformChannel,
      showClipping: state.viewOptions.showClipping,
    })),
  );

  const setEditor = useEditorStore((state) => state.setEditor);
  const onToggleClipping = () =>
    setEditor((state) => ({ viewOptions: { ...state.viewOptions, showClipping: !state.viewOptions.showClipping } }));

  return (
    <Waveform
      hasWaveform={hasWaveform}
      hasHistogram={hasHistogram}
      displayMode={activeWaveformChannel || 'luma'}
      setDisplayMode={setDisplayMode}
      showClipping={showClipping}
      onToggleClipping={onToggleClipping}
      theme={theme}
    />
  );
}
