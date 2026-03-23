import type { MetricSnapshot } from "@gpu-sim/shared";

/**
 * Derives a bounded multiplier from model/gpu/workflow so dropdown selections
 * have a visible effect on chained module outputs.
 */
export function stackSelectionTune(snapshot: MetricSnapshot): number {
  const mh = [...snapshot.modelId].reduce((a, c) => a + c.charCodeAt(0), 0);
  const gh = [...snapshot.gpuId].reduce((a, c) => a + c.charCodeAt(0), 0);
  const wf = snapshot.workflow ?? "training";
  let w = 0;
  if (wf === "inference") w = -0.035;
  if (wf === "throughput_benchmark") w = 0.045;
  return 1 + (mh % 23) / 220 - (gh % 19) / 240 + w;
}
