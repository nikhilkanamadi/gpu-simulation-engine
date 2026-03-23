/** Minimal fields per tick for tradeoff scatter plots (derived from MetricSnapshot). */
export interface TradeoffSample {
  tick: number;
  mfu: number;
  hbmBandwidthUtil: number;
  smOccupancy: number;
  warpStallRate: number;
  tokensPerSec: number;
  tensorCoreUtil: number;
  tflopsAchieved?: number;
  loss?: number;
}
