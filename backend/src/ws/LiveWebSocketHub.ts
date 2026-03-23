import type { BottleneckAlert, LiveMessage, MetricSnapshot } from "@gpu-sim/shared";
import type WebSocket from "ws";

export class LiveWebSocketHub {
  private readonly clientsByRunId = new Map<string, Set<WebSocket>>();

  addClient(runId: string, ws: WebSocket) {
    const set = this.clientsByRunId.get(runId) ?? new Set<WebSocket>();
    set.add(ws);
    this.clientsByRunId.set(runId, set);
  }

  removeClient(runId: string, ws: WebSocket) {
    const set = this.clientsByRunId.get(runId);
    if (!set) return;
    set.delete(ws);
    if (set.size === 0) this.clientsByRunId.delete(runId);
  }

  broadcastSnapshot(runId: string, snapshot: MetricSnapshot) {
    const message: LiveMessage = { type: "snapshot", runId, snapshot };
    this.broadcast(runId, message);
  }

  broadcastAlert(runId: string, alert: BottleneckAlert) {
    const message: LiveMessage = { type: "alert", runId, alert };
    this.broadcast(runId, message);
  }

  private broadcast(runId: string, message: LiveMessage) {
    const set = this.clientsByRunId.get(runId);
    if (!set) return;
    const payload = JSON.stringify(message);
    for (const ws of set) {
      // `ws.OPEN` is a static WebSocket constant; instances don't expose it.
      if (ws.readyState === 1 /* OPEN */) ws.send(payload);
    }
  }
}

