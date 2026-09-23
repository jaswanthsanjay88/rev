export type JSONPrimitive = string | number | boolean | null;
export type JSONValue = JSONPrimitive | JSONObject | JSONArray;
export type JSONObject = { [key: string]: JSONValue };
export type JSONArray = JSONValue[];

export interface ChoiceQuestion {
  type: "choice";
  instructions: JSONValue;
  criteria: Record<string, JSONValue>;
}

export interface NoulQuestion {
  type: "noul";
  instructions: JSONValue;
  criteria?: {
    false?: JSONValue;
    true?: JSONValue;
  };
}

export interface ScoreQuestion {
  type: "score";
  instructions: JSONValue;
  criteria: JSONValue[];
}

export type Question = ChoiceQuestion | NoulQuestion | ScoreQuestion;

export interface SystemOneRequest {
  state?: JSONValue;
  image?: string;
  image_b64?: string;
  model?: string;
  questions: Record<string, Question>;
}

export interface ChoiceAnswer {
  type: "choice";
  choice: string;
  confidence: number;
  probabilities: Record<string, number>;
}

export interface NoulAnswer {
  type: "noul";
  noul: number;
}

export interface ScoreAnswer {
  type: "score";
  score: number;
  legend: JSONValue[];
  probabilities: number[];
}

export type Answer = ChoiceAnswer | NoulAnswer | ScoreAnswer;

export interface SystemOneResponse {
  model: string;
  answers: Record<string, Answer>;
  usage: {
    input_tokens: number;
    output_tokens: number;
  };
  latency_ms: number;
  cached: boolean;
}

export interface RevClientOptions {
  baseUrl?: string;
  apiKey?: string;
  timeoutMs?: number;
  defaultModel?: string;
  fetch?: typeof fetch;
}

export interface ApiInfo {
  run: string;
  device: string;
  base: string;
  mode: string;
  unified_engine?: string;
}

export interface ModelEntry {
  id: string;
  base?: string;
  run?: string;
  type?: string;
  routing?: string;
  parameters?: string;
  context?: number;
  language?: string;
}

export interface ModelsResponse {
  models: ModelEntry[];
}
