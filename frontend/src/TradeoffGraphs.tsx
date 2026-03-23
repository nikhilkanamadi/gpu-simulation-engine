import { useMemo } from "react";
import { ResponsiveScatterPlot } from "@nivo/scatterplot";

import type { TradeoffSample } from "./tradeoffTypes";

type ScatterDatum = { x: number; y: number; tick: number };

const nivoTheme = {
  background: "transparent",
  text: { fill: "#e6ebff", fontSize: 11 },
  axis: {
    domain: {
      line: { stroke: "#27345f", strokeWidth: 1 }
    },
    ticks: {
      line: { stroke: "#27345f" },
      text: { fill: "#9aa7cf", fontSize: 10 }
    },
    legend: {
      text: { fill: "#9aa7cf", fontSize: 11 },
      fontWeight: 600
    }
  },
  grid: {
    line: { stroke: "#27345f", strokeWidth: 1, strokeOpacity: 0.6 }
  },
  tooltip: {
    container: {
      background: "#18213b",
      color: "#e6ebff",
      fontSize: 12,
      borderRadius: 8,
      border: "1px solid #27345f",
      padding: "8px 10px"
    }
  }
};

const ACCENT = "#7aa2ff";

const chartMargin = { top: 12, right: 16, bottom: 48, left: 52 };

function buildTradeoffData(samples: TradeoffSample[]) {
  const roofline: ScatterDatum[] = [];
  const throughputLatency: ScatterDatum[] = [];
  const hbmMfu: ScatterDatum[] = [];
  const occStall: ScatterDatum[] = [];
  const speedQuality: ScatterDatum[] = [];

  for (const s of samples) {
    const tflops = s.tflopsAchieved ?? 100 + s.mfu * 250;
    const intensity = (s.mfu + 0.05) / (s.hbmBandwidthUtil + 0.08);
    const perfNorm = Math.min(1.25, tflops / 450);
    roofline.push({ x: intensity, y: perfNorm, tick: s.tick });

    const latencyProxy = 35 + s.warpStallRate * 100 + (1 - s.smOccupancy) * 75;
    throughputLatency.push({ x: s.tokensPerSec, y: latencyProxy, tick: s.tick });

    hbmMfu.push({ x: s.hbmBandwidthUtil, y: s.mfu, tick: s.tick });

    occStall.push({ x: s.smOccupancy, y: s.warpStallRate, tick: s.tick });

    if (s.loss != null) {
      speedQuality.push({ x: s.tensorCoreUtil, y: s.loss, tick: s.tick });
    }
  }

  return { roofline, throughputLatency, hbmMfu, occStall, speedQuality };
}

function toNivoSeries(points: ScatterDatum[]) {
  if (points.length === 0) return [];
  return [{ id: "ticks", data: points.map((p) => ({ ...p })) }];
}

export function TradeoffGraphs(props: { samples: TradeoffSample[] }) {
  const { samples } = props;

  const data = useMemo(() => buildTradeoffData(samples), [samples]);

  const empty = samples.length === 0;

  return (
    <div className="tradeoff-section">
      <h2 className="section-title">Tradeoff graphs</h2>
      <p className="subtitle tradeoff-subtitle">
        <strong>Why scatter?</strong> Tradeoffs are easiest to read when two competing quantities are plotted
        together: each point is one tick, and the <em>shape</em> of the cloud (diagonal, hook, or cluster) shows
        how the engine couples metrics. A <strong>line over time</strong> is better for trends (see the loss curve
        below); scatter is better for <strong>cause–effect tension</strong> between axes.
      </p>
      <p className="subtitle tradeoff-subtitle tradeoff-subtitle-second">
        Rendered with{" "}
        <a href="https://nivo.rocks" target="_blank" rel="noreferrer">
          Nivo
        </a>{" "}
        scatter plots from live <code>MetricSnapshot</code> streams.
      </p>

      {empty ? (
        <div className="tradeoff-empty">Start a simulation to see tradeoff scatter plots.</div>
      ) : (
        <div className="tradeoff-grid">
          <TradeoffScatter
            title="1 — Roofline proxy"
            subtitle="Arithmetic intensity (mfu·hbm⁻¹) vs normalized TFLOP/s (÷450)"
            explain="During simulation, the pipeline alternates between math-heavy steps (higher MFU) and memory-heavy traffic (higher HBM use). This chart compresses that tension: points drifting up-left mean you are getting useful compute without proportionally higher memory intensity; a tight vertical cluster suggests you are memory-bound at many ticks. As ticks advance, watch whether the cloud moves toward higher normalized TFLOP/s or stays flat—that is the engine’s idea of ‘roofline headroom’ from your model and GPU settings."
            points={data.roofline}
            xLabel="Intensity (proxy)"
            yLabel="Relative TFLOP/s"
            formatX={(v) => v.toFixed(2)}
            formatY={(v) => v.toFixed(3)}
          />

          <TradeoffScatter
            title="2 — Throughput vs latency proxy"
            subtitle="tokens/s vs stall+occupancy delay (ms)"
            explain="The simulator reports tokens per second as throughput while warp stalls and low occupancy inflate a synthetic ‘latency proxy.’ When you see points slide right (more tokens/s) but also rise (higher latency proxy), that is the classic serving tradeoff: batching and parallelism push throughput until memory or scheduling delays dominate. Pausing or changing workflow in the UI should shift this cloud: throughput_benchmark-style runs typically push the right side; inference-oriented settings often trade raw tokens/s for lower stall-driven delay in this proxy."
            points={data.throughputLatency}
            xLabel="Tokens/s"
            yLabel="Latency proxy (ms)"
            formatX={(v) => v.toFixed(0)}
            formatY={(v) => v.toFixed(0)}
          />

          <TradeoffScatter
            title="3 — Memory vs compute pressure"
            subtitle="HBM bandwidth util vs MFU"
            explain="HBM utilization tracks how saturated off-chip bandwidth is; MFU tracks how much of the model’s nominal FLOPs you are converting into useful work. In the chained simulation, memory stage outputs feed scheduler inputs, so these two rarely move independently. A diagonal band means raising MFU also pulls more traffic—typical of large matmuls with big activations. If points hug high HBM with middling MFU, read that as pressure on the memory subsystem; the opposite suggests compute-bound phases where caches and bandwidth are less contested."
            points={data.hbmMfu}
            xLabel="HBM util"
            yLabel="MFU"
            formatX={(v) => `${(v * 100).toFixed(0)}%`}
            formatY={(v) => `${(v * 100).toFixed(0)}%`}
          />

          <TradeoffScatter
            title="4 — Occupancy vs warp stalls"
            subtitle="SM occupancy vs warp stall rate"
            explain="High occupancy means enough resident warps to hide latency; stalls mean warps are waiting on loads or barriers. The tradeoff appears because you cannot always increase both: more runnable warps may collide with the same memory queues, or kernels may be register-heavy and limit occupancy. While the simulation runs, a downward-right shape (higher occupancy, fewer stalls) is the ‘healthy’ direction. Sudden upward moves on the stall axis often coincide with bottleneck alerts in the panel—those ticks are when the scheduler is starved despite reasonable occupancy."
            points={data.occStall}
            xLabel="SM occupancy"
            yLabel="Warp stall"
            formatX={(v) => `${(v * 100).toFixed(0)}%`}
            formatY={(v) => `${(v * 100).toFixed(0)}%`}
          />

          <TradeoffScatter
            title="5 — Speed vs quality (training)"
            subtitle="Tensor core util vs loss (lower = better)"
            explain="Tensor-core utilization is a stand-in for ‘fast math path’ usage; loss is the optimization target. In real training you might trade precision or batch size for speed; here the model stage drives loss down over ticks while compute stages set tensor utilization. During simulation, expect loss to fall along the vertical axis as training progresses while tensor utilization jitters horizontally with MFU-like variation. If loss stalls but tensor util stays high, the story is optimization difficulty, not under-use of hardware; the opposite suggests the run is not exercising tensor kernels as strongly."
            points={data.speedQuality}
            xLabel="Tensor core util"
            yLabel="Loss"
            formatX={(v) => `${(v * 100).toFixed(0)}%`}
            formatY={(v) => v.toFixed(3)}
            emptyMessage="No loss samples yet."
          />
        </div>
      )}
    </div>
  );
}

function TradeoffScatter({
  title,
  subtitle,
  explain,
  points,
  xLabel,
  yLabel,
  formatX,
  formatY,
  emptyMessage
}: {
  title: string;
  subtitle: string;
  explain: string;
  points: ScatterDatum[];
  xLabel: string;
  yLabel: string;
  formatX: (v: number) => string;
  formatY: (v: number) => string;
  emptyMessage?: string;
}) {
  const series = toNivoSeries(points);
  const noData = points.length === 0;

  return (
    <figure className="tradeoff-card">
      <figcaption className="tradeoff-caption">{title}</figcaption>
      <p className="tradeoff-sub">{subtitle}</p>
      <div className="tradeoff-chart">
        {noData ? (
          <div className="tradeoff-chart-empty">{emptyMessage ?? "No data."}</div>
        ) : (
          <ResponsiveScatterPlot
            data={series}
            theme={nivoTheme}
            margin={chartMargin}
            xScale={{ type: "linear", min: "auto", max: "auto" }}
            yScale={{ type: "linear", min: "auto", max: "auto" }}
            axisBottom={{
              legend: xLabel,
              legendOffset: 36,
              legendPosition: "middle"
            }}
            axisLeft={{
              legend: yLabel,
              legendOffset: -44,
              legendPosition: "middle"
            }}
            colors={[ACCENT]}
            nodeSize={8}
            blendMode="normal"
            animate={false}
            tooltip={({ node }) => (
              <div>
                tick {node.data.tick}
                <br />
                {formatX(node.xValue as number)} · {formatY(node.yValue as number)}
              </div>
            )}
          />
        )}
      </div>
      <p className="tradeoff-explain">{explain}</p>
    </figure>
  );
}
