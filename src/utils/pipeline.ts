import { Panel } from '../components/ui/AppProperties';
import { Adjustments, INITIAL_ADJUSTMENTS, INITIAL_MASK_ADJUSTMENTS, SectionVisibility } from './adjustments';

// Every step a photo goes through, in the order the backend runs them:
// decode in src-tauri/src/raw_processing.rs and image_loader.rs, geometry in
// lib.rs (compute_patched_and_warped), render in the main() of shaders/shader.wgsl.
// Keep this list in step with those when the pipeline changes.

export type PipelinePhase = 'decode' | 'geometry' | 'render';

export interface PipelineStage {
  id: string;
  phase: PipelinePhase;
  // Adjustment keys the stage reads. The stage is on when any of them differs from
  // its default, globally or in a visible mask. Bypassing resets them to default.
  keys?: string[];
  // Overrides the key comparison, for stages whose keys don't say whether they run.
  isActive?: (adj: Adjustments) => boolean;
  // Adjustments section whose eye icon also switches the stage off.
  section?: keyof SectionVisibility;
  // Where the stage's controls live.
  panel?: Panel;
  // Decode stages and the crop run regardless of the preview; they can't be bypassed.
  fixed?: boolean;
}

const TRANSFORM_KEYS = [
  'transformDistortion',
  'transformVertical',
  'transformHorizontal',
  'transformRotate',
  'transformAspect',
  'transformScale',
  'transformXOffset',
  'transformYOffset',
];

export const PIPELINE_STAGES: PipelineStage[] = [
  { id: 'decodeImage', phase: 'decode', fixed: true },
  { id: 'demosaic', phase: 'decode', fixed: true },
  { id: 'highlightRecovery', phase: 'decode', fixed: true },
  { id: 'cameraColor', phase: 'decode', fixed: true },
  { id: 'embeddedLens', phase: 'decode', fixed: true },
  { id: 'rawPreprocessing', phase: 'decode', fixed: true },

  {
    id: 'heal',
    phase: 'geometry',
    keys: ['aiPatches'],
    isActive: (adj) => (adj.aiPatches || []).some((p) => p.visible),
    panel: Panel.Ai,
  },
  {
    id: 'lensCorrection',
    phase: 'geometry',
    keys: ['lensDistortionParams'],
    isActive: (adj) =>
      !!adj.lensDistortionParams &&
      ((adj.lensDistortionEnabled && adj.lensDistortionAmount !== 0) ||
        (adj.lensTcaEnabled && adj.lensTcaAmount !== 0) ||
        (adj.lensVignetteEnabled && adj.lensVignetteAmount !== 0)),
    panel: Panel.Crop,
  },
  {
    id: 'transform',
    phase: 'geometry',
    keys: [...TRANSFORM_KEYS, 'guidedPerspective'],
    isActive: (adj) =>
      TRANSFORM_KEYS.some((k) => isChanged(adj[k], INITIAL_ADJUSTMENTS[k])) || !!adj.guidedPerspective?.enabled,
    panel: Panel.Crop,
  },
  // Relight and fog add light to the scene, so white balance and exposure act on it later.
  {
    id: 'relight',
    phase: 'geometry',
    keys: ['relightEnabled'],
    isActive: (adj) => adj.relightEnabled && !!adj.relightNormalMap,
    section: 'effects',
    panel: Panel.Adjustments,
  },
  {
    id: 'fog',
    phase: 'geometry',
    keys: ['fogEnabled'],
    isActive: (adj) => adj.fogEnabled && !!adj.fogDepthMap && adj.fogAmount > 0,
    section: 'effects',
    panel: Panel.Adjustments,
  },
  {
    id: 'lensBlur',
    phase: 'geometry',
    keys: ['lensBlurEnabled'],
    isActive: (adj) => adj.lensBlurEnabled && !!adj.lensBlurDepthMap,
    section: 'effects',
    panel: Panel.Adjustments,
  },
  {
    id: 'crop',
    phase: 'geometry',
    keys: ['crop', 'rotation', 'flipHorizontal', 'flipVertical', 'orientationSteps'],
    panel: Panel.Crop,
    fixed: true,
  },

  {
    id: 'chromaticAberration',
    phase: 'render',
    keys: ['chromaticAberrationRedCyan', 'chromaticAberrationBlueYellow'],
    section: 'details',
    panel: Panel.Adjustments,
  },
  {
    id: 'noiseReduction',
    phase: 'render',
    keys: ['lumaNoiseReduction', 'colorNoiseReduction'],
    section: 'details',
    panel: Panel.Adjustments,
  },
  { id: 'sharpening', phase: 'render', keys: ['sharpness'], section: 'details', panel: Panel.Adjustments },
  { id: 'whiteBalance', phase: 'render', keys: ['temperature', 'tint'], section: 'color', panel: Panel.Adjustments },
  {
    id: 'localContrast',
    phase: 'render',
    keys: ['clarity', 'structure', 'centré'],
    section: 'details',
    panel: Panel.Adjustments,
  },
  { id: 'exposure', phase: 'render', keys: ['exposure'], section: 'basic', panel: Panel.Adjustments },
  { id: 'glow', phase: 'render', keys: ['glowAmount'], section: 'effects', panel: Panel.Adjustments },
  { id: 'halation', phase: 'render', keys: ['halationAmount'], section: 'effects', panel: Panel.Adjustments },
  { id: 'flare', phase: 'render', keys: ['flareAmount'], section: 'effects', panel: Panel.Adjustments },
  { id: 'dehaze', phase: 'render', keys: ['dehaze'], section: 'details', panel: Panel.Adjustments },
  {
    id: 'tone',
    phase: 'render',
    keys: ['contrast', 'shadows', 'whites', 'blacks'],
    section: 'basic',
    panel: Panel.Adjustments,
  },
  { id: 'highlights', phase: 'render', keys: ['highlights'], section: 'basic', panel: Panel.Adjustments },
  {
    id: 'colorCalibration',
    phase: 'render',
    keys: ['colorCalibration'],
    section: 'color',
    panel: Panel.Adjustments,
  },
  { id: 'hsl', phase: 'render', keys: ['hsl'], section: 'color', panel: Panel.Adjustments },
  { id: 'hue', phase: 'render', keys: ['hue'], section: 'color', panel: Panel.Adjustments },
  {
    id: 'saturation',
    phase: 'render',
    keys: ['saturation', 'vibrance'],
    section: 'color',
    panel: Panel.Adjustments,
  },
  { id: 'colorGrading', phase: 'render', keys: ['colorGrading'], section: 'color', panel: Panel.Adjustments },
  { id: 'vignette', phase: 'render', keys: ['vignetteAmount'], section: 'effects', panel: Panel.Adjustments },
  // Curves act on scene light, on an axis in stops, just before the rendering.
  { id: 'curves', phase: 'render', keys: ['curves'], section: 'curves', panel: Panel.Adjustments },
  { id: 'toneMapper', phase: 'render', isActive: () => true, panel: Panel.Adjustments, fixed: true },
  // A scene-referred LUT reads linear light and blends with the tone mapper's output;
  // a display LUT acts on the finished picture.
  {
    id: 'sceneLut',
    phase: 'render',
    keys: ['lutPath'],
    isActive: (adj) => !!adj.lutPath && !!adj.lutIsSceneReferred,
    section: 'effects',
    panel: Panel.Adjustments,
  },
  { id: 'brightness', phase: 'render', keys: ['brightness'], section: 'basic', panel: Panel.Adjustments },
  {
    id: 'lut',
    phase: 'render',
    keys: ['lutPath'],
    isActive: (adj) => !!adj.lutPath && !adj.lutIsSceneReferred,
    section: 'effects',
    panel: Panel.Adjustments,
  },
  { id: 'grain', phase: 'render', keys: ['grainAmount'], section: 'effects', panel: Panel.Adjustments },
  {
    id: 'masks',
    phase: 'render',
    keys: ['masks'],
    isActive: (adj) => (adj.masks || []).some((m) => m.visible),
    panel: Panel.Masks,
  },
];

export function isStageActive(stage: PipelineStage, adj: Adjustments): boolean {
  if (stage.isActive) return stage.isActive(adj);
  if (!stage.keys) return true;
  const keys = stage.keys;
  if (keys.some((k) => isChanged(adj[k], INITIAL_ADJUSTMENTS[k]))) return true;
  return (adj.masks || []).some(
    (m) =>
      m.visible &&
      keys.some((k) => k in INITIAL_MASK_ADJUSTMENTS && isChanged(m.adjustments?.[k], INITIAL_MASK_ADJUSTMENTS[k])),
  );
}

// Resets the bypassed stages' keys to their defaults, in place, on a copy of the
// adjustments that is only sent to the preview renderer.
export function bypassStages(adj: Adjustments, stageIds: string[]): Adjustments {
  for (const stage of PIPELINE_STAGES) {
    if (!stageIds.includes(stage.id) || !stage.keys) continue;
    for (const key of stage.keys) {
      adj[key] = structuredClone(INITIAL_ADJUSTMENTS[key]);
      for (const mask of adj.masks || []) {
        if (key in INITIAL_MASK_ADJUSTMENTS) mask.adjustments[key] = structuredClone(INITIAL_MASK_ADJUSTMENTS[key]);
      }
    }
  }
  return adj;
}

// A missing key reads as its default everywhere, so it isn't a change.
function isChanged(value: unknown, defaultValue: unknown): boolean {
  return value !== undefined && !isEqual(value, defaultValue);
}

function isEqual(a: unknown, b: unknown): boolean {
  if (a === b) return true;
  if (a == null || b == null || typeof a !== 'object' || typeof b !== 'object') return a == b;
  if (Array.isArray(a) !== Array.isArray(b)) return false;
  const ra = a as Record<string, unknown>;
  const rb = b as Record<string, unknown>;
  const keys = new Set([...Object.keys(ra), ...Object.keys(rb)]);
  for (const k of keys) {
    if (!isEqual(ra[k], rb[k])) return false;
  }
  return true;
}
