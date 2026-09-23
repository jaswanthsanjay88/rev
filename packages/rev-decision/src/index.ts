import {
  SystemOneRequest,
  SystemOneResponse,
  RevClientOptions,
  ApiInfo,
  ModelsResponse,
  Question,
  ChoiceQuestion,
  NoulQuestion,
  ScoreQuestion,
} from "./types";

export * from "./types";

declare const process: { env: Record<string, string | undefined> } | undefined;

const getEnv = (key: string): string | undefined => {
  try {
    return typeof process !== "undefined" ? process?.env?.[key] : undefined;
  } catch {
    return undefined;
  }
};

export class RevClient {
  private baseUrl: string;
  private apiKey?: string;
  private timeoutMs: number;
  private defaultModel: string;
  private fetchFn: typeof fetch;

  constructor(options: RevClientOptions = {}) {
    this.baseUrl = (options.baseUrl || getEnv("REV_BASE_URL") || "http://127.0.0.1:8000").replace(/\/+$/, "");
    this.apiKey = options.apiKey || getEnv("REV_API_KEY");
    this.timeoutMs = options.timeoutMs ?? 30000;
    this.defaultModel = options.defaultModel || "rev-latest";
    this.fetchFn = options.fetch || globalThis.fetch;

    if (!this.fetchFn) {
      throw new Error(
        "No fetch implementation found. Ensure you are running on Node 18+, Bun, Deno, or provide a fetch ponyfill in options."
      );
    }
  }

  /**
   * Evaluates multiple typed questions in a single prefill pass with prefix KV-caching.
   */
  async predict(request: SystemOneRequest): Promise<SystemOneResponse> {
    const payload = {
      model: request.model || this.defaultModel,
      state: request.state,
      questions: request.questions,
    };

    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };
    if (this.apiKey) {
      headers["Authorization"] = `Bearer ${this.apiKey}`;
    }

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);

    try {
      const res = await this.fetchFn(`${this.baseUrl}/v1/systemone`, {
        method: "POST",
        headers,
        body: JSON.stringify(payload),
        signal: controller.signal,
      });

      if (!res.ok) {
        const errText = await res.text().catch(() => "");
        throw new Error(`rev API error (${res.status} ${res.statusText}): ${errText}`);
      }

      return (await res.json()) as SystemOneResponse;
    } finally {
      clearTimeout(timer);
    }
  }

  /**
   * Evaluates questions separately without inter-option interaction.
   */
  async separate(request: SystemOneRequest): Promise<SystemOneResponse> {
    const payload = {
      model: request.model || this.defaultModel,
      state: request.state,
      questions: request.questions,
    };

    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };
    if (this.apiKey) {
      headers["Authorization"] = `Bearer ${this.apiKey}`;
    }

    const res = await this.fetchFn(`${this.baseUrl}/v1/systemone/separate`, {
      method: "POST",
      headers,
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const errText = await res.text().catch(() => "");
      throw new Error(`rev API error (${res.status} ${res.statusText}): ${errText}`);
    }

    return (await res.json()) as SystemOneResponse;
  }

  /**
   * Retrieves server runtime information, active backbone, and device.
   */
  async getInfo(): Promise<ApiInfo> {
    const res = await this.fetchFn(`${this.baseUrl}/api/info`);
    if (!res.ok) {
      throw new Error(`Failed to fetch info (${res.status})`);
    }
    return (await res.json()) as ApiInfo;
  }

  /**
   * Lists available model backbones.
   */
  async getModels(): Promise<ModelsResponse> {
    const res = await this.fetchFn(`${this.baseUrl}/v1/models`);
    if (!res.ok) {
      throw new Error(`Failed to fetch models (${res.status})`);
    }
    return (await res.json()) as ModelsResponse;
  }
}

/**
 * Default global rev singleton client configured for http://127.0.0.1:8000.
 */
export const rev = new RevClient();

/**
 * Built-in Question Constructors and Presets.
 */
export const presets = {
  choice: (instructions: string, criteria: Record<string, string>): ChoiceQuestion => ({
    type: "choice",
    instructions,
    criteria,
  }),
  noul: (instructions: string, criteria?: { false?: string; true?: string }): NoulQuestion => ({
    type: "noul",
    instructions,
    criteria,
  }),
  score: (instructions: string, criteria: string[]): ScoreQuestion => ({
    type: "score",
    instructions,
    criteria,
  }),
  triageQuestions: (): Record<string, Question> => ({
    intent: {
      type: "choice",
      instructions: "Classify the primary user intent",
      criteria: {
        billing: "Billing, invoices, charges, refunds, payment issues",
        technical: "Bugs, errors, integration, technical failure",
        sales: "Purchasing, upgrades, plans, enterprise quote",
        general: "Other informational inquiries",
      },
    },
    urgency: {
      type: "noul",
      instructions: "Does this require urgent (<1 hour) human escalation?",
    },
    severity: {
      type: "score",
      instructions: "Assess severity on a 4-level scale",
      criteria: [
        "P3: Low priority / question",
        "P2: Normal issue with workaround",
        "P1: High business impact",
        "P0: Critical outage / security emergency",
      ],
    },
  }),
};
