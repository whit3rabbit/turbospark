# turbospark-ffi

The C ABI used by the Swift package and macOS app.

## Read first

- [Detailed module guide](../../.claude/docs/modules/ffi.md)
- [Swift binding contract](../../docs/SWIFT_BINDINGS.md)

## Rules

- Unsafe code is expected here because the ABI receives raw pointers. Every
  exported entry point must stay inside the ABI guard and must not unwind across
  `extern "C"`.
- Keep the handwritten `turbospark.h` synchronized with the Rust wire
  contract. Rust tests alone do not validate the copied Swift header.
- Trace enum strings and capability predicates through both Rust and Swift.
- `make swift-lib` stages the static library and canonical header before
  SwiftPM tests.

## Checks

```sh
cargo test -p turbospark-ffi
make swift-lib
make swift-test
```

Real binding tests need a pinned install and the second blocked install
described in the verification reference.
