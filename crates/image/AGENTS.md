# turbospark-image

Portable image preprocessing and macOS image execution boundaries.

## Read first

- [Detailed module guide](../../.claude/docs/modules/image.md)
- [Image model guide](../../docs/ZIMAGE_TURBO.md)
- [Vision pipeline](../../docs/VISION.md)
- [Testing rules](../../docs/TESTING.md)

## Rules

- Keep preprocessing portable and keep Metal-specific code behind its target
  gate.
- Preserve image dimensions, channel order, normalization, and position-table
  contracts exactly. A visual smoke is not a tensor parity check.
- Real image claims require the pinned resource and quality gates. Synthetic
  fixtures establish structure only.
- Keep image artifacts and large model payloads out of the repository.

## Checks

```sh
cargo test -p turbospark-image
cargo check --target x86_64-unknown-linux-gnu -p turbospark-image
```
