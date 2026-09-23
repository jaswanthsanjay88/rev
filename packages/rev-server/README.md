# rev-server

> **Standalone Server CLI for `rev`: Fast System 1 Decision Engine with Calibrated Probabilities.**

[![npm version](https://img.shields.io/npm/v/rev-server.svg?style=flat-square&color=000000)](https://www.npmjs.com/package/rev-server)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg?style=flat-square)](LICENSE)

`rev-server` starts a local decision server implementing the `POST /v1/systemone` specification.

---

## Instant Usage with `npx`

```bash
npx rev-server --port 8000
```

## Global Install

```bash
npm install -g rev-server
rev-server --port 8000
```

## Endpoints

- `POST /v1/systemone`: Evaluates typed decisions in <2ms
- `GET /health`: Healthcheck

---

## License

Apache-2.0 © Jaswanth Sanjay
