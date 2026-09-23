# rev-decision

> **TypeScript / JavaScript Client for `rev`: Fast System 1 Decision Engine with Calibrated Probabilities.**

[![npm version](https://img.shields.io/npm/v/rev-decision.svg?style=flat-square&color=000000)](https://www.npmjs.com/package/rev-decision)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg?style=flat-square)](LICENSE)
[![Zero Dependencies](https://img.shields.io/badge/dependencies-0-brightgreen.svg?style=flat-square)](package.json)

`rev-decision` is the official, zero-dependency TypeScript/JavaScript client for **rev**. It connects to your local or hosted `rev-server` to execute typed decisions (`choice`, `noul`, `score`) in a single forward prefill pass with prefix KV-caching.

---

## Installation

```bash
npm install rev-decision
# or
pnpm add rev-decision
# or
yarn add rev-decision
```

Works out of the box in **Node.js (18+)**, **Bun**, **Deno**, **Next.js**, **Vite**, and all modern browsers.

---

## Quickstart

### 1. Basic Decision

```typescript
import { rev } from "rev-decision";

const res = await rev.predict({
  state: "Customer received broken parcel and demands an immediate refund.",
  questions: {
    intent: {
      type: "choice",
      instructions: "Classify primary customer intent",
      criteria: {
        refund: "Customer demands money back",
        support: "General support assistance",
        feedback: "Product feedback",
      },
    },
    urgent: {
      type: "noul",
      instructions: "Does this require urgent human escalation?",
    },
  },
});

console.log(res.answers.intent.choice);      // "refund"
console.log(res.answers.intent.confidence);  // 0.984
console.log(res.answers.urgent.noul);        // 0.96 (probability of true)
console.log(`Evaluated in ${res.latency_ms} ms (cached: ${res.cached})`);
```

---

### 2. Custom Client Configuration

```typescript
import { RevClient } from "rev-decision";

const client = new RevClient({
  baseUrl: "http://127.0.0.1:8000", // Default or hosted instance
  apiKey: process.env.REV_API_KEY,   // Optional
  timeoutMs: 15000,
});

const info = await client.getInfo();
console.log(info.run, info.device, info.mode);
```

---

### 3. Using Built-in Presets

```typescript
import { rev, presets } from "rev-decision";

const res = await rev.predict({
  state: "System database connections exhausted on production-us-east-1",
  questions: presets.triageQuestions(),
});

console.log(res.answers);
```

---

## Question Types

| Question Type | Output Shape | Description |
| :--- | :--- | :--- |
| **`choice`** | `{ choice, confidence, probabilities }` | Multi-class selection over criteria mapping. |
| **`noul`** | `{ noul }` | Calibrated continuous probability $\in [0, 1]$ ($p(\text{true})$). |
| **`score`** | `{ score, legend, probabilities }` | Expected ordinal level across graded tiers. |

---

## License

Apache-2.0 © Jaswanth Sanjay
