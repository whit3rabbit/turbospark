# Inference engine (clean-room Rust port)

A behavior-compatible Rust implementation of an approved local inference
engine and its command-line surface.

This workspace is built and tested with cargo. The shared foundation lives in
the `core` crate (runtime configuration and primitives). The destination
compute strategy lives in the `compute` crate. Command-line argument
translation lives in the `invocation` crate, and candidate selection lives in
the `selection` crate. Further generation crates are added by later work
items.

## Layout

- `crates/core`: shared primitives and the public runtime configuration.
- `crates/compute`: destination-selected compute strategy (skeleton).
- `crates/invocation`: command-line argument translation, usage text, and
  process-status/stream routing decisions (pure data, no I/O).
- `crates/selection`: candidate selection under a validated shaping
  configuration.

## Build and test

```sh
cargo build --workspace
cargo test --workspace
```

## License

MIT.
