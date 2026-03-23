import type { PipelineTrace } from "@gpu-sim/shared";
import { PIPELINE_METRIC_LINKS } from "./pipelineLinks";

const METRIC_LABELS: Record<string, string> = {
  loss: "Loss",
  gradNorm: "Grad norm",
  mfu: "MFU",
  hbmBandwidthUtil: "HBM util",
  smOccupancy: "SM occupancy",
  tflopsAchieved: "TFLOPs",
  warpStallRate: "Warp stall",
  tensorCoreUtil: "Tensor core util",
  l2HitRate: "L2 hit rate",
  flashAttnSavings: "Flash-attn savings",
  activeWarps: "Active warps"
};

function labelFor(key: string) {
  return METRIC_LABELS[key] ?? key;
}

function formatValue(key: string, v: number | string | undefined): string {
  if (v === undefined) return "—";
  if (typeof v === "string") return v;
  const n = Number(v);
  if (key === "loss" || key === "gradNorm") return n.toFixed(3);
  if (key === "tflopsAchieved" || key === "activeWarps") return n.toFixed(0);
  if (
    [
      "mfu",
      "hbmBandwidthUtil",
      "smOccupancy",
      "tensorCoreUtil",
      "l2HitRate",
      "flashAttnSavings",
      "warpStallRate"
    ].includes(key)
  ) {
    return `${(n * 100).toFixed(1)}%`;
  }
  return String(n);
}

export function PipelineSidebar(props: { trace: PipelineTrace | null | undefined }) {
  const { trace } = props;
  const links = trace?.links ?? PIPELINE_METRIC_LINKS;

  return (
    <aside className="pipeline-sidebar" aria-label="Simulation pipeline">
      <h2 className="pipeline-sidebar-title">Execution order</h2>
      <p className="pipeline-sidebar-lead">
        Each tick runs stages top to bottom; later stages read metrics produced by earlier ones.
      </p>

      {trace?.stages?.length ? (
        <ol className="pipeline-stages">
          {trace.stages.map((s) => (
            <li key={s.moduleId} className="pipeline-stage">
              <div className="pipeline-stage-head">
                <span className="pipeline-stage-order">{s.order}</span>
                <span className="pipeline-stage-title">{s.title}</span>
              </div>
              <p className="pipeline-stage-summary">{s.summary}</p>
              <div className="pipeline-io">
                {s.readsMetrics.length > 0 ? (
                  <div className="pipeline-io-row">
                    <span className="pipeline-io-label">Reads</span>
                    <span className="pipeline-io-keys">
                      {s.readsMetrics.map((k) => (
                        <code key={k} className="metric-key">
                          {labelFor(k)}
                        </code>
                      ))}
                    </span>
                  </div>
                ) : null}
                <div className="pipeline-io-row">
                  <span className="pipeline-io-label">Produces</span>
                  <span className="pipeline-io-keys">
                    {s.producesMetrics.map((k) => (
                      <code key={k} className="metric-key metric-key-out">
                        {labelFor(k)}
                      </code>
                    ))}
                  </span>
                </div>
              </div>
              <dl className="pipeline-values">
                {Object.entries(s.valuesAtStage).map(([k, v]) => (
                  <div key={k} className="pipeline-value-row">
                    <dt>{labelFor(k)}</dt>
                    <dd>{formatValue(k, v)}</dd>
                  </div>
                ))}
              </dl>
            </li>
          ))}
        </ol>
      ) : (
        <p className="pipeline-placeholder">Start a simulation to see per-tick stage values.</p>
      )}

      <h3 className="pipeline-links-title">How metrics connect</h3>
      <ul className="pipeline-links">
        {links.map((link, i) => (
          <li key={`${link.fromKey}-${link.toKey}-${i}`} className="pipeline-link-item">
            <div className="pipeline-link-edge">
              <code>{labelFor(link.fromKey)}</code>
              <span className="pipeline-arrow" aria-hidden>
                →
              </span>
              <code>{labelFor(link.toKey)}</code>
            </div>
            <p className="pipeline-link-desc">{link.description}</p>
          </li>
        ))}
      </ul>
    </aside>
  );
}
