export interface LlmArchitectureProfile {
  id: string;
  aliases: string[];
  decoderType: "dense" | "sparse_moe" | "sparse_hybrid" | "recurrent";
  attention: string;
  contextTokens: number;
  totalParamsB: number;
  activeParamsB?: number;
}

export interface DerivedArchitectureMetric {
  name: string;
  value: string;
  meaning: string;
}

// Lightweight profile set for the models most likely used in this simulator.
// Values are sourced from the public LLM Architecture Gallery fact sheets.
const PROFILES: LlmArchitectureProfile[] = [
  {
    id: "llama-3-8b",
    aliases: ["llama-3", "llama 3", "llama-3.1-8b", "llama 3 8b"],
    decoderType: "dense",
    attention: "GQA with RoPE",
    contextTokens: 8192,
    totalParamsB: 8
  },
  {
    id: "deepseek-v3",
    aliases: ["deepseek-v3", "deepseek v3"],
    decoderType: "sparse_moe",
    attention: "MLA",
    contextTokens: 128000,
    totalParamsB: 671,
    activeParamsB: 37
  },
  {
    id: "qwen3-32b",
    aliases: ["qwen3", "qwen 3", "qwen3 32b", "qwen3-32b"],
    decoderType: "dense",
    attention: "GQA with QK-Norm",
    contextTokens: 128000,
    totalParamsB: 32
  },
  {
    id: "gpt-oss-20b",
    aliases: ["gpt-oss", "gpt oss", "gpt-oss-20b", "gpt-oss 20b"],
    decoderType: "sparse_moe",
    attention: "GQA with alternating sliding-window/global",
    contextTokens: 128000,
    totalParamsB: 20,
    activeParamsB: 3.6
  },
  {
    id: "mistral-small-3.1",
    aliases: ["mistral small 3.1", "mistral-small-3.1", "mistral 24b"],
    decoderType: "dense",
    attention: "Standard GQA",
    contextTokens: 128000,
    totalParamsB: 24
  }
];

export function resolveArchitectureProfile(modelId: string): LlmArchitectureProfile | null {
  const needle = modelId.toLowerCase().trim();
  const found = PROFILES.find((p) => p.aliases.some((alias) => needle.includes(alias)));
  return found ?? null;
}

export function buildDerivedArchitectureMetrics(modelId: string): DerivedArchitectureMetric[] {
  const p = resolveArchitectureProfile(modelId);
  if (!p) {
    return [
      {
        name: "Architecture profile",
        value: "Unknown",
        meaning:
          "No gallery profile matched this model ID. Use a known model family (e.g., Llama 3, DeepSeek V3, Qwen3)."
      }
    ];
  }

  const activeRatio = p.activeParamsB ? p.activeParamsB / p.totalParamsB : 1;
  const sparsityGain = p.activeParamsB ? p.totalParamsB / p.activeParamsB : 1;
  const contextClass =
    p.contextTokens >= 256000 ? "Ultra-long" : p.contextTokens >= 128000 ? "Long-context" : "Standard-context";

  return [
    {
      name: "Decoder family",
      value: humanDecoderType(p.decoderType),
      meaning: "Indicates whether inference compute is dense, sparsely routed (MoE), hybrid, or recurrent."
    },
    {
      name: "Attention stack",
      value: p.attention,
      meaning: "Attention design impacts memory pressure, KV-cache growth, and long-context behavior."
    },
    {
      name: "Context capacity",
      value: `${p.contextTokens.toLocaleString()} tokens (${contextClass})`,
      meaning: "Upper bound for prompt length before truncation/windowing strategies are required."
    },
    {
      name: "Active parameter ratio",
      value: `${(activeRatio * 100).toFixed(1)}%`,
      meaning: "Percent of total parameters active per token. Lower values usually indicate MoE sparsity."
    },
    {
      name: "Theoretical sparsity gain",
      value: `${sparsityGain.toFixed(1)}x`,
      meaning: "Approximate total/active parameter ratio; useful as a rough efficiency indicator for sparse models."
    }
  ];
}

function humanDecoderType(type: LlmArchitectureProfile["decoderType"]): string {
  if (type === "dense") return "Dense Transformer";
  if (type === "sparse_moe") return "Sparse MoE Transformer";
  if (type === "sparse_hybrid") return "Sparse Hybrid";
  return "Recurrent";
}

