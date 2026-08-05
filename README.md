# Inference engine (Rust port)

A behavior-compatible Rust port of the Mference local inference engine and
its command-line/server surface. See `ROADMAP.md` for phase-by-phase scope
and `DEVIATIONS.md` for what is fully wired versus scaffolded.

This workspace is built and tested with cargo.

## Layout

- `crates/core`: shared primitives and the public runtime configuration.
- `crates/compute`: CPU reference kernels (RmsNorm, RoPE, attention,
  int4/int8 quant, MoE, sampling) and the destination compute strategy.
- `crates/invocation`: command-line argument translation, usage text, and
  process-status/stream routing decisions (pure data, no I/O).
- `crates/selection`: candidate selection under a validated shaping
  configuration.
- `crates/window-fit`: conversation-window fitting (turn dropping).
- `crates/tokenizer`: tokenizer wrapper, chat templates, streaming
  detokenizer, tool-call parsing.
- `crates/model-io`: manifest/arch validation, packed-expert layout,
  resident tensor index, SHA-256 verification, install receipt.
- `crates/streaming`: routed-expert `pread` streamer and its LFU/LRU cache
  policy.
- `crates/gpu`: Metal pipeline cache and kernel dispatch (macOS only).
- `crates/runtime`: the raw-completion prefill+decode loop.
- `crates/cli`: the `mference-check` process entry point.
- `crates/repack`: safetensors header parsing, ranged-download planning,
  and quantization repack.
- `crates/server`: OpenAI-compatible Chat Completions server on loopback.

## Build and test

```sh
cargo build --workspace
cargo test --workspace
cargo fmt --check
cargo clippy --workspace --tests
```

## License

MIT.
