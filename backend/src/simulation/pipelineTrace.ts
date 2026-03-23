import type { MetricSnapshot, PipelineStageInfo, PipelineTrace } from "@gpu-sim/shared";
import { PIPELINE_METRIC_LINKS } from "@gpu-sim/shared";

export function snapshotSnippet(
  s: MetricSnapshot,
  keys: string[],
  extra?: Record<string, number | string | undefined>
): Record<string, number | string | undefined> {
  const out: Record<string, number | string | undefined> = { ...extra };
  for (const k of keys) {
    const v = (s as unknown as Record<string, unknown>)[k];
    if (v !== undefined) out[k] = v as number | string;
  }
  return out;
}

export function finalizePipelineTrace(tick: number, stages: PipelineStageInfo[]): PipelineTrace {
  return { tick, stages, links: PIPELINE_METRIC_LINKS };
}
