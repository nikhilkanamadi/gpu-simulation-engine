import fs from "node:fs";
import path from "node:path";
import initSqlJs from "sql.js";

import type { DataSource, MetricSnapshot, PipelineTrace } from "@gpu-sim/shared";

const SCHEMA_SQL = `
CREATE TABLE IF NOT EXISTS metric_snapshots (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  tick INTEGER NOT NULL,
  run_id TEXT NOT NULL,
  model_id TEXT NOT NULL,
  gpu_id TEXT NOT NULL,
  mfu REAL NOT NULL,
  hbm_bw_util REAL NOT NULL,
  sm_occupancy REAL NOT NULL,
  tflops_achieved REAL,
  loss REAL,
  grad_norm REAL,
  tokens_per_sec REAL,
  warp_stall_rate REAL,
  tensor_core_util REAL,
  l2_hit_rate REAL,
  flash_attn_savings REAL,
  kv_cache_util REAL,
  energy_watts REAL,
  cpu_input_util REAL,
  workflow TEXT,
  source TEXT NOT NULL,
  timestamp_ms INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_run_tick ON metric_snapshots (run_id, tick);
CREATE INDEX IF NOT EXISTS idx_model_tick ON metric_snapshots (model_id, tick);
`;

type SqlValue = string | number | null;

function toSnapshot(row: Record<string, SqlValue>): MetricSnapshot {
  // `sql.js` returns everything as string/number; coerce to expected types.
  const source = row["source"] as DataSource;
  return {
    tick: Number(row["tick"]),
    modelId: String(row["model_id"]),
    gpuId: String(row["gpu_id"]),
    mfu: Number(row["mfu"]),
    hbmBandwidthUtil: Number(row["hbm_bw_util"]),
    smOccupancy: Number(row["sm_occupancy"]),
    tflopsAchieved: row["tflops_achieved"] == null ? undefined : Number(row["tflops_achieved"]),
    loss: row["loss"] == null ? undefined : Number(row["loss"]),
    gradNorm: row["grad_norm"] == null ? undefined : Number(row["grad_norm"]),
    tokensPerSec: row["tokens_per_sec"] == null ? undefined : Number(row["tokens_per_sec"]),
    warpStallRate: row["warp_stall_rate"] == null ? undefined : Number(row["warp_stall_rate"]),
    tensorCoreUtil: row["tensor_core_util"] == null ? undefined : Number(row["tensor_core_util"]),
    l2HitRate: row["l2_hit_rate"] == null ? undefined : Number(row["l2_hit_rate"]),
    flashAttnSavings: row["flash_attn_savings"] == null ? undefined : Number(row["flash_attn_savings"]),
    kvCacheUtil: row["kv_cache_util"] == null ? undefined : Number(row["kv_cache_util"]),
    energyWatts: row["energy_watts"] == null ? undefined : Number(row["energy_watts"]),
    cpuInputUtil: row["cpu_input_util"] == null ? undefined : Number(row["cpu_input_util"]),
    workflow: row["workflow"] == null ? undefined : (String(row["workflow"]) as any),
    source,
    timestampMs: Number(row["timestamp_ms"]),
    pipelineTrace: parsePipelineTrace(row["pipeline_trace_json"])
  };
}

function parsePipelineTrace(raw: SqlValue | undefined): PipelineTrace | undefined {
  if (raw == null || raw === "") return undefined;
  try {
    return JSON.parse(String(raw)) as PipelineTrace;
  } catch {
    return undefined;
  }
}

export class TimeSeriesStore {
  private constructor(
    private db: any,
    private readonly dbFile: string
  ) {}

  static async create(dbFile: string): Promise<TimeSeriesStore> {
    const SQL = await initSqlJs({
      // `sql.js` loads its WASM via this locator. Using the package's dist folder keeps it simple.
      locateFile: (file: string) => path.join(__dirname, "../../../node_modules/sql.js/dist", file)
    });

    let db: any;
    if (fs.existsSync(dbFile)) {
      const bytes = fs.readFileSync(dbFile);
      db = new SQL.Database(new Uint8Array(bytes));
    } else {
      db = new SQL.Database();
    }

    const store = new TimeSeriesStore(db, dbFile);
    store.db.exec(SCHEMA_SQL);
    store.ensureOptionalColumns();
    return store;
  }

  private ensureOptionalColumns() {
    const pragma = this.db.exec("PRAGMA table_info(metric_snapshots);");
    const existing = new Set<string>();
    if (pragma.length > 0 && pragma[0].values) {
      for (const row of pragma[0].values as any[]) existing.add(String(row[1]));
    }

    const add = (name: string, type: string) => {
      if (!existing.has(name)) this.db.exec(`ALTER TABLE metric_snapshots ADD COLUMN ${name} ${type};`);
    };
    add("tensor_core_util", "REAL");
    add("l2_hit_rate", "REAL");
    add("flash_attn_savings", "REAL");
    add("kv_cache_util", "REAL");
    add("energy_watts", "REAL");
    add("cpu_input_util", "REAL");
    add("workflow", "TEXT");
    add("pipeline_trace_json", "TEXT");
  }

  insertSnapshot(runId: string, snapshot: MetricSnapshot) {
    const stmt = this.db.prepare(
      `INSERT INTO metric_snapshots (
        tick, run_id, model_id, gpu_id,
        mfu, hbm_bw_util, sm_occupancy,
        tflops_achieved, loss, grad_norm,
        tokens_per_sec, warp_stall_rate, tensor_core_util,
        l2_hit_rate, flash_attn_savings, kv_cache_util, energy_watts, cpu_input_util, workflow,
        source, timestamp_ms, pipeline_trace_json
      ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)`
    );

    const values: SqlValue[] = [
      snapshot.tick,
      runId,
      snapshot.modelId,
      snapshot.gpuId,
      snapshot.mfu,
      snapshot.hbmBandwidthUtil,
      snapshot.smOccupancy,
      snapshot.tflopsAchieved ?? null,
      snapshot.loss ?? null,
      snapshot.gradNorm ?? null,
      snapshot.tokensPerSec ?? null,
      snapshot.warpStallRate ?? null,
      snapshot.tensorCoreUtil ?? null,
      snapshot.l2HitRate ?? null,
      snapshot.flashAttnSavings ?? null,
      snapshot.kvCacheUtil ?? null,
      snapshot.energyWatts ?? null,
      snapshot.cpuInputUtil ?? null,
      snapshot.workflow ?? null,
      snapshot.source,
      snapshot.timestampMs,
      snapshot.pipelineTrace != null ? JSON.stringify(snapshot.pipelineTrace) : null
    ];

    stmt.bind(values);
    stmt.step();
    stmt.free();
  }

  // Persist DB contents to disk. Call sparingly for MVP speed.
  persist() {
    const dir = path.dirname(this.dbFile);
    fs.mkdirSync(dir, { recursive: true });
    const data = this.db.export();
    fs.writeFileSync(this.dbFile, Buffer.from(data));
  }

  getSnapshots(runId: string, offset: number, limit: number): MetricSnapshot[] {
    const stmt = this.db.prepare(
      `SELECT
        tick, model_id, gpu_id,
        mfu, hbm_bw_util, sm_occupancy,
        tflops_achieved, loss, grad_norm,
        tokens_per_sec, warp_stall_rate, tensor_core_util,
        l2_hit_rate, flash_attn_savings, kv_cache_util, energy_watts, cpu_input_util, workflow,
        source, timestamp_ms, pipeline_trace_json
      FROM metric_snapshots
      WHERE run_id = ?
      ORDER BY tick
      LIMIT ? OFFSET ?`
    );

    stmt.bind([runId, limit, offset]);

    const out: MetricSnapshot[] = [];
    while (stmt.step()) {
      const row = stmt.getAsObject() as Record<string, SqlValue>;
      out.push(toSnapshot(row));
    }
    stmt.free();
    return out;
  }

  getSummary(runId: string) {
    const stmt = this.db.prepare(
      `SELECT
        MIN(tick) AS min_tick,
        MAX(tick) AS max_tick,
        AVG(mfu) AS avg_mfu,
        AVG(hbm_bw_util) AS avg_hbm_util,
        AVG(sm_occupancy) AS avg_sm_occupancy,
        MAX(loss) AS peak_loss,
        MIN(loss) AS min_loss
      FROM metric_snapshots
      WHERE run_id = ?`
    );
    stmt.bind([runId]);
    stmt.step();
    const row = stmt.getAsObject() as Record<string, SqlValue>;
    stmt.free();

    return {
      runId,
      tickWindow: { startTick: Number(row["min_tick"]), endTick: Number(row["max_tick"]) },
      avgMfu: Number(row["avg_mfu"]),
      avgHbmUtil: Number(row["avg_hbm_util"]),
      avgSmOccupancy: Number(row["avg_sm_occupancy"]),
      peakLoss: row["peak_loss"] == null ? undefined : Number(row["peak_loss"]),
      minLoss: row["min_loss"] == null ? undefined : Number(row["min_loss"])
    };
  }
}

