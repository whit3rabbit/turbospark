# turbospark-server Blueprint

## System Overview
Crate `turbospark-server` (lib `turbospark_server`, bin `turbospark-server`): an OpenAI-, Anthropic- and Ollama-compatible local generation server. Source lives under `src/`; `tests/` holds integration tests.

## Subdirectories & System Boundaries
- src: [./src/EGM.md](./src/EGM.md) - router, registry, queue, auth, backends, and endpoint modules, with the audio, guardrails, handler, and responses subpackages documented beneath it.

## Global Invariants
- The Host check and auth layer are applied in [src/lib.rs:build_router_with_options]. The rule that a Tailnet bind requires a key, and the `TURBOSPARK_API_KEY` fallback, live in [src/main.rs:resolve_api_key].
- `GET /health` is routed outside the auth layer ([src/lib.rs:build_router_with_options]).

## Global Side Effects
- Network listener, model loading, and GPU generation; see [./src/EGM.md](./src/EGM.md).

## System Verification
- Run: `cargo test -p turbospark-server`
