# turbospark-vision-io

Portable vision preprocessing, image tensors, and position metadata.

## Read first

- [Detailed module guide](../../.claude/docs/modules/vision-io.md)
- [Vision pipeline](../../docs/VISION.md)
- [Vision checkpoint](../../docs/VISION_PHASE0.md)
- [Testing rules](../../docs/TESTING.md)

## Rules

- Keep this crate buildable off macOS. Its portable status is intentional.
- Preserve resize, channel, normalization, crop, and positional encoding
  contracts. Visual output does not prove tensor parity.
- Checked-in fixtures establish shape and arithmetic behavior. Real tower
  parity requires the pinned reference and the vision gates.
- Keep large image and model artifacts outside the repository.

## Checks

```sh
cargo test -p turbospark-vision-io
cargo check --target x86_64-unknown-linux-gnu -p turbospark-vision-io
```
