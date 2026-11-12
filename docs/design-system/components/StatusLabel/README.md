# StatusLabel

Status as a glyph plus a word, never color alone, so it survives color-vision deficiency and Differentiate Without Color.

- Success: checkmark in `success`. Warning: triangle in `warning`. Danger: octagon in `danger`. Each word sits beside its glyph.
- As a pill, the background is the matching `-tint` and the text keeps the status color, never `ink`.
- Permission rows show status as glyph plus text and offer one next step: Allow while macOS can still prompt, Open System Settings after.
- `success` is a bluish green, off the red-green axis, so it stays distinct from `danger`.
- The app draws these as `.appSuccess`, `.appWarning` and `.appDanger` (`ResolvedAppTheme`). System red was 3.6:1 on white. See the README's known gaps for the two themes where `danger` is 4.3:1.

The consumer supplies the status, a localized word, and any next-step action.
