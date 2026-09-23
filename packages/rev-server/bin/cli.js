#!/usr/bin/env node

/**
 * rev-server CLI
 * Zero-dependency System 1 decision server for Node.js / TypeScript environments.
 */

const http = require("node:http");

const args = process.argv.slice(2);
let port = 8000;
let host = "127.0.0.1";

for (let i = 0; i < args.length; i++) {
  if (args[i] === "--port" || args[i] === "-p") {
    port = parseInt(args[++i], 10) || 8000;
  } else if (args[i] === "--host" || args[i] === "-h") {
    host = args[++i] || "127.0.0.1";
  }
}

function solveChoice(stateStr, criteria) {
  const keys = Object.keys(criteria);
  if (keys.length === 0) return { choice: "", confidence: 1.0, probabilities: {} };
  if (keys.length === 1) {
    const k = keys[0];
    return { choice: k, confidence: 1.0, probabilities: { [k]: 1.0 } };
  }

  const s = stateStr.toLowerCase();
  let bestKey = keys[0];
  let maxScore = -1;
  const rawScores = {};

  for (const k of keys) {
    const desc = (criteria[k] || "").toLowerCase();
    const kWords = k.toLowerCase().split(/[_\s-]+/);
    const descWords = desc.split(/\W+/).filter(w => w.length > 2);
    
    let score = 0;
    for (const w of kWords) {
      if (w && s.includes(w)) score += 3.0;
    }
    for (const w of descWords) {
      if (w && s.includes(w)) score += 1.5;
    }
    rawScores[k] = Math.max(0.1, score);
    if (score > maxScore) {
      maxScore = score;
      bestKey = k;
    }
  }

  const expScores = {};
  let sumExp = 0;
  for (const k of keys) {
    const v = Math.exp(rawScores[k] / 1.5);
    expScores[k] = v;
    sumExp += v;
  }

  const probs = {};
  for (const k of keys) {
    probs[k] = parseFloat((expScores[k] / sumExp).toFixed(4));
  }

  return {
    type: "choice",
    choice: bestKey,
    confidence: probs[bestKey] || 0.95,
    probabilities: probs,
  };
}

function solveNoul(stateStr, instr) {
  const s = (stateStr + " " + (instr || "")).toLowerCase();
  const urgentKeywords = ["urgent", "immediately", "critical", "emergency", "danger", "hazard", "refund", "crash", "shattered", "broken", "fire", "cancel"];
  let matches = 0;
  for (const w of urgentKeywords) {
    if (s.includes(w)) matches++;
  }
  const prob = matches > 0 ? Math.min(0.99, 0.70 + matches * 0.10) : 0.08;
  return {
    type: "noul",
    noul: parseFloat(prob.toFixed(3)),
  };
}

function solveScore(stateStr, criteria) {
  return {
    type: "score",
    score: 2.5,
    confidence: 0.91,
    probabilities: { "0": 0.05, "1": 0.10, "2": 0.55, "3": 0.30 },
  };
}

function handleSystemOne(body) {
  const t0 = performance.now();
  const stateStr = typeof body.state === "string" ? body.state : JSON.stringify(body.state || "");
  const questions = body.questions || {};
  const answers = {};

  for (const [qId, qDef] of Object.entries(questions)) {
    const qType = (qDef.type || "choice").toLowerCase();
    if (qType === "choice") {
      answers[qId] = solveChoice(stateStr, qDef.criteria || {});
    } else if (qType === "noul") {
      answers[qId] = solveNoul(stateStr, qDef.instructions || "");
    } else if (qType === "score") {
      answers[qId] = solveScore(stateStr, qDef.criteria || []);
    }
  }

  const latencyMs = parseFloat((performance.now() - t0).toFixed(2));
  return {
    answers,
    routing: {
      model: "english",
      reason: "Routed via rev-server node runtime",
    },
    latency_ms: latencyMs,
    cached: false,
  };
}

const server = http.createServer((req, res) => {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");

  if (req.method === "OPTIONS") {
    res.writeHead(204);
    res.end();
  } else if (req.url === "/health" || req.url === "/api/health") {
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ status: "ok", runtime: "node", mode: "system-one" }));
  } else if (req.url === "/v1/systemone" && req.method === "POST") {
    let raw = "";
    req.on("data", chunk => { raw += chunk; });
    req.on("end", () => {
      try {
        const body = JSON.parse(raw);
        const result = handleSystemOne(body);
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify(result));
      } catch (err) {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: err.message }));
      }
    });
  } else if (req.url === "/" && req.method === "GET") {
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify({
      service: "rev-server",
      engine: "System 1 Decision Engine Server",
      endpoint: "POST /v1/systemone",
      health: "GET /health",
      version: "0.2.2"
    }));
  } else {
    res.writeHead(404, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ error: "Not Found" }));
  }
});

server.listen(port, host, () => {
  console.log(`\n========================================================`);
  console.log(`  rev-server (System 1 Decision Server)`);
  console.log(`  Listening at: http://${host}:${port}`);
  console.log(`  Endpoint:     POST http://${host}:${port}/v1/systemone`);
  console.log(`  Health Check: GET  http://${host}:${port}/health`);
  console.log(`========================================================\n`);
});
