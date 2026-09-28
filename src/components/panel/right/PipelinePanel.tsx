import { useEffect } from 'react';
import { Eye, EyeOff, RotateCcw } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import type { ParseKeys, TFunction } from 'i18next';
import clsx from 'clsx';
import Text from '../../ui/Text';
import { TextColors, TextVariants, TextWeights } from '../../../types/typography';
import { useEditorStore } from '../../../store/useEditorStore';
import { useSettingsStore } from '../../../store/useSettingsStore';
import { useUIStore } from '../../../store/useUIStore';
import { useCameraProfileName } from '../../../hooks/useCameraProfileName';
import { Adjustments } from '../../../utils/adjustments';
import { PIPELINE_STAGES, PipelinePhase, PipelineStage, isStageActive } from '../../../utils/pipeline';

type StageStatus = 'on' | 'off' | 'hidden' | 'bypassed';

const PHASES: PipelinePhase[] = ['decode', 'geometry', 'render'];

// Decode stages that only exist for one kind of file.
const RAW_ONLY = ['demosaic', 'cameraColor', 'highlightRecovery', 'embeddedLens'];

export default function PipelinePanel() {
  const { t } = useTranslation();
  const adjustments = useEditorStore((s) => s.adjustments);
  const selectedImage = useEditorStore((s) => s.selectedImage);
  const bypassedStages = useEditorStore((s) => s.bypassedStages);
  const setEditor = useEditorStore((s) => s.setEditor);
  const appSettings = useSettingsStore((s) => s.appSettings);
  const setPanel = useUIStore((s) => s.setPanel);

  const imagePath = selectedImage?.path;
  const isRaw = !!selectedImage?.isRaw;
  const profileName = useCameraProfileName(imagePath);

  // Bypassing is an inspection aid: it lasts while this panel is open on one photo.
  useEffect(() => {
    const clearBypass = () => setEditor((s) => (s.bypassedStages.length ? { bypassedStages: [] } : {}));
    clearBypass();
    return clearBypass;
  }, [imagePath, setEditor]);

  if (!selectedImage) {
    return (
      <div className="flex items-center justify-center h-full">
        <Text variant={TextVariants.heading} color={TextColors.secondary} weight={TextWeights.normal}>
          {t('editor.ai.noImageSelected')}
        </Text>
      </div>
    );
  }

  const colorNr = appSettings?.rawPreprocessingColorNr ?? 0.5;
  const sharpening = appSettings?.rawPreprocessingSharpening ?? 0.35;
  const preprocesses = (isRaw || !!appSettings?.applyPreprocessingToNonRaws) && (colorNr > 0 || sharpening > 0);

  const statusOf = (stage: PipelineStage): StageStatus => {
    if (bypassedStages.includes(stage.id)) return 'bypassed';
    if (stage.section && adjustments.sectionVisibility?.[stage.section] === false) return 'hidden';
    if (stage.id === 'rawPreprocessing') return preprocesses ? 'on' : 'off';
    return isStageActive(stage, adjustments) ? 'on' : 'off';
  };

  const detailOf = (stage: PipelineStage, status: StageStatus): string | null => {
    if (status === 'hidden') return t('editor.pipeline.hiddenInSection');
    if (status === 'bypassed') return t('editor.pipeline.bypassed');
    return stageDetail(stage.id, adjustments, {
      t,
      profileName,
      useCameraProfiles: !!appSettings?.useCameraProfiles,
      colorNr,
      sharpening,
      preprocesses,
    });
  };

  const toggleBypass = (id: string) =>
    setEditor((s) => ({
      bypassedStages: s.bypassedStages.includes(id)
        ? s.bypassedStages.filter((b) => b !== id)
        : [...s.bypassedStages, id],
    }));

  const stages = PIPELINE_STAGES.filter((stage) => {
    if (RAW_ONLY.includes(stage.id)) return isRaw;
    if (stage.id === 'decodeImage') return !isRaw;
    return true;
  });

  return (
    <div className="flex flex-col h-full">
      <div className="p-3 flex justify-between items-center shrink-0 border-b border-surface">
        <Text variant={TextVariants.title}>{t('editor.pipeline.title')}</Text>
        <button
          className="p-2 rounded-full hover:bg-surface disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          disabled={bypassedStages.length === 0}
          onClick={() => setEditor({ bypassedStages: [] })}
          data-tooltip={t('editor.pipeline.restoreAll')}
        >
          <RotateCcw size={18} />
        </button>
      </div>

      <div className="grow overflow-y-auto p-3 flex flex-col gap-4">
        {PHASES.map((phase) => (
          <div key={phase} className="flex flex-col gap-0.5">
            <div className="px-2 pb-1">
              <Text variant={TextVariants.label} weight={TextWeights.semibold}>
                {t(`editor.pipeline.phases.${phase}` as ParseKeys)}
              </Text>
              <Text variant={TextVariants.small} color={TextColors.secondary}>
                {t(`editor.pipeline.phases.${phase}Desc` as ParseKeys)}
              </Text>
            </div>
            {stages
              .filter((stage) => stage.phase === phase)
              .map((stage) => {
                const status = statusOf(stage);
                const detail = detailOf(stage, status);
                const canBypass = !stage.fixed && (status === 'on' || status === 'bypassed');
                return (
                  <div
                    key={stage.id}
                    className={clsx(
                      'flex items-center gap-2.5 px-2 py-1.5 rounded-md',
                      stage.panel && 'cursor-pointer hover:bg-surface',
                    )}
                    onClick={() => stage.panel && setPanel(stage.panel)}
                  >
                    <span
                      className={clsx(
                        'w-2 h-2 rounded-full shrink-0',
                        status === 'on' && 'bg-accent',
                        status === 'bypassed' && 'bg-yellow-400',
                        (status === 'off' || status === 'hidden') && 'border border-text-secondary opacity-50',
                      )}
                    />
                    <div className={clsx('flex-1 min-w-0', status !== 'on' && 'opacity-60')}>
                      <Text variant={TextVariants.body} className={clsx(status === 'bypassed' && 'line-through')}>
                        {t(`editor.pipeline.stages.${stage.id}` as ParseKeys)}
                      </Text>
                      {detail && (
                        <Text variant={TextVariants.small} color={TextColors.secondary} className="truncate">
                          {detail}
                        </Text>
                      )}
                    </div>
                    {canBypass && (
                      <button
                        className="p-1 rounded-full hover:bg-card-active transition-colors shrink-0"
                        onClick={(e) => {
                          e.stopPropagation();
                          toggleBypass(stage.id);
                        }}
                        data-tooltip={t(status === 'bypassed' ? 'editor.pipeline.enable' : 'editor.pipeline.bypass')}
                      >
                        {status === 'bypassed' ? <EyeOff size={16} /> : <Eye size={16} />}
                      </button>
                    )}
                  </div>
                );
              })}
          </div>
        ))}
      </div>

      {bypassedStages.length > 0 && (
        <div className="px-3 py-2 shrink-0 border-t border-surface bg-accent/10">
          <Text variant={TextVariants.small} color={TextColors.accent}>
            {t('editor.pipeline.bypassNotice', { count: bypassedStages.length })}
          </Text>
        </div>
      )}
    </div>
  );
}

interface DetailContext {
  t: TFunction;
  profileName: string | null;
  useCameraProfiles: boolean;
  colorNr: number;
  sharpening: number;
  preprocesses: boolean;
}

function stageDetail(id: string, adj: Adjustments, ctx: DetailContext): string | null {
  const { t } = ctx;
  switch (id) {
    case 'demosaic':
      return t('editor.pipeline.details.demosaic');
    case 'cameraColor':
      if (ctx.profileName) return ctx.profileName;
      return t(ctx.useCameraProfiles ? 'editor.pipeline.details.noProfile' : 'editor.pipeline.details.profilesOff');
    case 'highlightRecovery':
      return t('editor.pipeline.details.highlightRecovery');
    case 'embeddedLens':
      return t('editor.pipeline.details.embeddedLens');
    case 'decodeImage':
      return t('editor.pipeline.details.decodeImage');
    case 'rawPreprocessing':
      return ctx.preprocesses
        ? t('editor.pipeline.details.rawPreprocessing', {
            colorNr: Math.round(ctx.colorNr * 100),
            sharpening: Math.round(ctx.sharpening * 100),
          })
        : null;
    case 'heal': {
      const count = (adj.aiPatches || []).filter((p) => p.visible).length;
      return count ? t('editor.pipeline.details.count', { count }) : null;
    }
    case 'lensCorrection':
      return [adj.lensMaker, adj.lensModel].filter(Boolean).join(' ') || null;
    case 'toneMapper': {
      const name = t(`adjustments.basic.mappers.${adj.toneMapper || 'basic'}` as ParseKeys);
      return adj.toneMapper === 'profile' && ctx.profileName ? `${name}: ${ctx.profileName}` : name;
    }
    case 'lut':
    case 'sceneLut':
      return adj.lutPath ? adj.lutName || null : null;
    case 'masks': {
      const count = (adj.masks || []).filter((m) => m.visible).length;
      return count ? t('editor.pipeline.details.count', { count }) : null;
    }
    default:
      return null;
  }
}
