# turbospark-coreml

## Overview
Apple Core ML runner with a non-macOS stub, plus Python tools that generate ANE programs and fixtures. Package: turbospark-coreml.

## Subdirectories & Boundaries
- src: [./src/EGM.md](./src/EGM.md) - CoreMlModel, tensors, compile cache, macOS sys and stub backends.
- examples: [./examples/EGM.md](./examples/EGM.md) - coreml_describe and coreml_plan inspection examples.
- tests and testdata: no EGM.md. [tests/fixtures.rs] (macOS only) round-trips the fixture packages in `testdata/`.
- tools: [./tools/EGM.md](./tools/EGM.md) - Python generators and bakers for ANE prefill, vision and sweep programs, plus the fixture generator.
