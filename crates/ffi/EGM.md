# turbospark-ffi

## Overview
C ABI over the inference engine for the Swift app: hand-written header plus Rust entry points, sessions, in-process server and model management. Package: turbospark-ffi.

## Subdirectories & Boundaries
- include: [./include/EGM.md](./include/EGM.md) - turbospark.h contract, status codes, ABI version.
- src: [./src/EGM.md](./src/EGM.md) - ABI guard, sessions, wire shapes, server, STT, vision; children api, generate, models.
