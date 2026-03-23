import { useEffect, useMemo, useRef, useState } from "react";
import { Line, LineChart, ResponsiveContainer, XAxis, YAxis } from "recharts";

import type { BottleneckAlert, DataSource, MetricSnapshot, SimulationWorkflow } from "@gpu-sim/shared";
import { METRIC_REFERENCE } from "./metricReference";
import { buildDerivedArchitectureMetrics } from "./architectureMetrics";
import { PipelineSidebar } from "./PipelineSidebar";
import { TradeoffGraphs } from "./TradeoffGraphs";
import type { TradeoffSample } from "./tradeoffTypes";

function formatPct(x: number | undefined) {
  if (x == null || Number.isNaN(x)) return "—";
  return `${(x * 100).toFixed(1)}%`;
}

/** Deterministic seed from selections so different dropdowns produce different synthetic streams. */
function seedFromSelections(base: number, modelId: string, gpuId: string, workflow: string): number {
  let h = base >>> 0;
  const mix = (s: string) => {
    for (let i = 0; i < s.length; i++) {
      h = Math.imul(h ^ s.charCodeAt(i), 0x5bd1e995);
    }
  };
  mix(modelId);
  mix(gpuId);
  mix(workflow);
  return (h >>> 0) % 2_000_000_000 || 1;
}

export default function App() {
  const [modelId, setModelId] = useState("llama-3.1-8b");
  const [gpuId, setGpuId] = useState("h100-80gb");
  const [source, setSource] = useState<DataSource>("synthetic");
  const [workflow, setWorkflow] = useState<SimulationWorkflow>("training");
  const [runId, setRunId] = useState<string | null>(null);
  const [latest, setLatest] = useState<MetricSnapshot | null>(null);
  const [alerts, setAlerts] = useState<BottleneckAlert[]>([]);
  const [lossPoints, setLossPoints] = useState<Array<{ tick: number; loss: number }>>([]);
  const [startError, setStartError] = useState<string | null>(null);
  const [tradeoffSamples, setTradeoffSamples] = useState<TradeoffSample[]>([]);

  const wsRef = useRef<WebSocket | null>(null);

  const canControl = useMemo(() => Boolean(runId), [runId]);
  const architectureMetrics = useMemo(() => buildDerivedArchitectureMetrics(modelId), [modelId]);

  async function startRun() {
    setLatest(null);
    setAlerts([]);
    setLossPoints([]);
    setTradeoffSamples([]);
    setStartError(null);

    const res = await fetch("/api/runs/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        modelId,
        gpuId,
        source,
        options: {
          durationTicks: 800,
          tickMs: 40,
          flashAttention: true,
          seed: seedFromSelections(42, modelId, gpuId, workflow),
          workflow
        }
      })
    });

    const data = (await res.json()) as { runId?: string; error?: string };
    if (!res.ok) {
      setStartError(data.error ?? `Start failed (${res.status})`);
      return;
    }
    if (!data.runId) {
      setStartError("Start failed: no run id returned.");
      return;
    }
    setRunId(data.runId);
  }

  function postControl(endpoint: string) {
    if (!runId) return Promise.resolve();
    return fetch(endpoint, { method: "POST" }).catch(() => undefined);
  }

  useEffect(() => {
    if (!runId) return;

    const proto = window.location.protocol === "https:" ? "wss" : "ws";
    const wsUrl = `${proto}://${window.location.host}/ws/live?runId=${encodeURIComponent(runId)}`;
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data as string) as
          | { type: "snapshot"; runId: string; snapshot: MetricSnapshot }
          | { type: "alert"; runId: string; alert: BottleneckAlert };

        if (msg.type === "snapshot") {
          setLatest(msg.snapshot);
          const s = msg.snapshot;
          setTradeoffSamples((prev) => {
            const row: TradeoffSample = {
              tick: s.tick,
              mfu: s.mfu,
              hbmBandwidthUtil: s.hbmBandwidthUtil,
              smOccupancy: s.smOccupancy,
              warpStallRate: s.warpStallRate ?? 0,
              tokensPerSec: s.tokensPerSec ?? 0,
              tensorCoreUtil: s.tensorCoreUtil ?? s.mfu,
              tflopsAchieved: s.tflopsAchieved,
              loss: s.loss
            };
            return [...prev, row].slice(-500);
          });
          if (msg.snapshot.loss != null) {
            setLossPoints((prev) => {
              const next = [...prev, { tick: msg.snapshot.tick, loss: msg.snapshot.loss! }];
              // Keep chart lightweight for MVP.
              return next.slice(-250);
            });
          }
        } else if (msg.type === "alert") {
          setAlerts((prev) => {
            const next = [...prev, msg.alert];
            return next.slice(-20);
          });
        }
      } catch {
        // Ignore malformed messages in MVP.
      }
    };

    ws.onerror = () => {
      // eslint-disable-next-line no-console
      console.warn("WebSocket error");
    };

    return () => {
      wsRef.current = null;
      ws.close();
    };
  }, [runId]);

  return (
    <div className="app-layout">
      <PipelineSidebar trace={latest?.pipelineTrace} />
      <div className="app-shell">
      <section className="panel">
        <div className="header">
          <div>
            <h1 className="title">GPU Algorithm Simulation Engine</h1>
            <p className="subtitle">Live metrics from orchestrated simulation ticks.</p>
          </div>
          <div className="status-pill">{runId ? "Run Active" : "Idle"}</div>
        </div>

        <div className="controls-grid">
          <div className="field">
            <label htmlFor="model-id">Model ID</label>
            <input id="model-id" value={modelId} onChange={(e) => setModelId(e.target.value)} />
          </div>

          <div className="field">
            <label htmlFor="gpu-id">GPU ID</label>
            <input id="gpu-id" value={gpuId} onChange={(e) => setGpuId(e.target.value)} />
          </div>

          <div className="field">
            <label htmlFor="source">Source</label>
            <select id="source" value={source} onChange={(e) => setSource(e.target.value as DataSource)}>
              <option value="synthetic">synthetic</option>
              <option value="mlperf">mlperf</option>
              <option value="kineto">kineto</option>
              <option value="hta">hta</option>
              <option value="wandb">wandb</option>
              <option value="lambda">lambda</option>
            </select>
          </div>
          <div className="field">
            <label htmlFor="workflow">Workflow</label>
            <select
              id="workflow"
              value={workflow}
              onChange={(e) => setWorkflow(e.target.value as SimulationWorkflow)}
            >
              <option value="training">training</option>
              <option value="inference">inference</option>
              <option value="throughput_benchmark">throughput_benchmark</option>
            </select>
          </div>
        </div>

        {startError ? (
          <div className="start-error" role="alert">
            {startError}
          </div>
        ) : null}

        <div className="btn-row" style={{ marginTop: 12 }}>
          <button className="btn btn-primary" onClick={startRun}>
            Start simulation
          </button>
          <button className="btn" disabled={!canControl} onClick={() => postControl(`/api/runs/${runId}/pause`)}>
            Pause
          </button>
          <button className="btn" disabled={!canControl} onClick={() => postControl(`/api/runs/${runId}/resume`)}>
            Resume
          </button>
          <button className="btn" disabled={!canControl} onClick={() => postControl(`/api/runs/${runId}/step`)}>
            Step
          </button>
        </div>

        <div className="run-id">
          {runId ? (
            <>
              Run ID: <code>{runId}</code>
            </>
          ) : (
            "Start a simulation to view live metrics."
          )}
        </div>
      </section>

      <section className="panel">
        <div className="cards">
          <MetricCard title="MFU (Model FLOP Utilization)" value={latest ? formatPct(latest.mfu) : "—"} />
          <MetricCard title="HBM bandwidth util" value={latest ? formatPct(latest.hbmBandwidthUtil) : "—"} />
          <MetricCard title="SM occupancy" value={latest ? formatPct(latest.smOccupancy) : "—"} />
          <MetricCard title="Loss" value={latest?.loss != null ? latest.loss.toFixed(3) : "—"} />
          <MetricCard title="Tensor core util" value={latest ? formatPct(latest.tensorCoreUtil) : "—"} />
          <MetricCard title="L2 cache hit rate" value={latest ? formatPct(latest.l2HitRate) : "—"} />
          <MetricCard title="KV cache util" value={latest ? formatPct(latest.kvCacheUtil) : "—"} />
          <MetricCard title="Energy (watts)" value={latest?.energyWatts != null ? latest.energyWatts.toFixed(0) : "—"} />
          <MetricCard title="CPU input util" value={latest ? formatPct(latest.cpuInputUtil) : "—"} />
          <MetricCard
            title="Workflow"
            value={latest?.workflow != null ? latest.workflow.replace("_", " ") : workflow.replace("_", " ")}
          />
        </div>
      </section>

      <section className="panel">
        <TradeoffGraphs samples={tradeoffSamples} />
      </section>

      <section className="panel">
        <h2 className="section-title">Loss curve (latest 250 points)</h2>
        <div className="chart-wrap">
          <ResponsiveContainer>
            <LineChart data={lossPoints}>
              <XAxis dataKey="tick" stroke="#93a2cf" />
              <YAxis stroke="#93a2cf" />
              <Line type="monotone" dataKey="loss" stroke="#7aa2ff" dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </section>

      <section className="panel">
        <h2 className="section-title">Bottleneck alerts</h2>
        {alerts.length === 0 ? (
          <div className="empty">No alerts yet.</div>
        ) : (
          <div className="alerts">
            {alerts
              .slice()
              .reverse()
              .map((a, idx) => (
                <div key={`${a.tick}-${idx}`} className="alert">
                  <div className="alert-type">{a.type}</div>
                  <div className="alert-meta">tick: {a.tick}</div>
                  {"mfu" in a ? <div className="alert-meta">mfu: {a.mfu.toFixed(3)}</div> : null}
                  {"hbmUtil" in a ? <div className="alert-meta">hbm: {a.hbmUtil.toFixed(3)}</div> : null}
                  {"smOccupancy" in a ? <div className="alert-meta">sm: {a.smOccupancy.toFixed(3)}</div> : null}
                  {"gradNorm" in a ? <div className="alert-meta">gradNorm: {a.gradNorm.toFixed(3)}</div> : null}
                  {"recentLossDelta" in a ? (
                    <div className="alert-meta">recentLossDelta: {a.recentLossDelta.toFixed(4)}</div>
                  ) : null}
                </div>
              ))}
          </div>
        )}
      </section>

      <section className="panel">
        <h2 className="section-title">Metrics Reference</h2>
        <p className="subtitle" style={{ marginTop: 0 }}>
          Definitions and healthy ranges from the project metric documentation.
        </p>
        <div className="metric-ref-grid">
          {METRIC_REFERENCE.map((metric) => (
            <article key={metric.key} className="metric-ref-card">
              <h3 className="metric-ref-title">{metric.name}</h3>
              <p className="metric-ref-line">
                <strong>Definition:</strong> {metric.definition}
              </p>
              <p className="metric-ref-line">
                <strong>Healthy range/target:</strong> {metric.healthy}
              </p>
              <p className="metric-ref-line">
                <strong>How to read it:</strong> {metric.interpretation}
              </p>
            </article>
          ))}
        </div>
      </section>

      <section className="panel">
        <h2 className="section-title">Architecture-Aware Metrics</h2>
        <p className="subtitle" style={{ marginTop: 0 }}>
          Derived from selected model family traits (decoder type, attention, context, and active-vs-total params).
        </p>
        <div className="metric-ref-grid">
          {architectureMetrics.map((metric) => (
            <article key={metric.name} className="metric-ref-card">
              <h3 className="metric-ref-title">{metric.name}</h3>
              <p className="metric-ref-value">{metric.value}</p>
              <p className="metric-ref-line">{metric.meaning}</p>
            </article>
          ))}
        </div>
      </section>
      </div>
    </div>
  );
}

function MetricCard({ title, value }: { title: string; value: string }) {
  return (
    <div className="metric-card">
      <div className="metric-title">{title}</div>
      <div className="metric-value">{value}</div>
    </div>
  );
}

