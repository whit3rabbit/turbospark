# StatusLabel

Status as a glyph plus a word, never color alone, so it survives color-vision deficiency and Differentiate Without Color.

- Success: checkmark in `success`. Warning: triangle in `warning`. Danger: octagon in `danger`. Each word sits beside its glyph.
- As a pill, the background is the matching `-tint` and the text keeps the status color, never `ink`.
- Permission rows show status as glyph plus text and offer one next step: Allow while macOS can still prompt, Open System Settings after.
- `success` is a bluish green, off the red-green axis, so it stays distinct from `danger`.
- Intentional addition: the app draws status with system green, orange, and red today. Light-mode system red is 3.6:1 on white. These tokens are 4.5:1 or better on every surface.

The consumer supplies the status, a localized word, and any next-step action.
