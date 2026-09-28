import { useEffect, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { Invokes } from '../components/ui/AppProperties';
import { useSettingsStore } from '../store/useSettingsStore';

// Name of the camera profile the photo at `path` is decoded with; null without one
// (not a RAW, no profile for the camera, or Camera Profiles off in settings).
export function useCameraProfileName(path: string | undefined): string | null {
  const useCameraProfiles = useSettingsStore((s) => !!s.appSettings?.useCameraProfiles);
  const [name, setName] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setName(null);
    if (path && useCameraProfiles) {
      invoke<string | null>(Invokes.GetCameraProfileName, { path })
        .then((result) => !cancelled && setName(result))
        .catch(() => {});
    }
    return () => {
      cancelled = true;
    };
  }, [path, useCameraProfiles]);

  return name;
}
