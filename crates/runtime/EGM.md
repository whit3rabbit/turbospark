# turbospark-runtime

## Overview
Runtime crate: family dispatch, Metal forward passes, sessions, speculative decode, and native audio/vision/Core ML pipelines. Sources live in `src/`; `tests/` holds the integration suites (not documented here); the rest are examples and helper scripts.

## Subdirectories & Boundaries
- src: [./src/EGM.md](./src/EGM.md) - RealForwardRunner, decode loops, per-family flows, vision, whisper, Core ML
- examples: [./examples/EGM.md](./examples/EGM.md) - real-model gates, probes and benchmark binaries
- scripts: [./scripts/EGM.md](./scripts/EGM.md) - Python drivers benchmarking Core ML pipelines
- tools: [./tools/EGM.md](./tools/EGM.md) - Python fixture helpers for Core ML gates
