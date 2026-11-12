# SearchField

The one search field: a magnifying glass, a plain text field, and a clear button that appears once there is text.

```swift
AppSearchField(text: $query, prompt: Text("Filter chats...", bundle: .module))
```

- The prompt is a `Text`, so the bundled literal stays at the call site and doubles as the field's accessible name.
- Width is min 140, ideal 220, max 360 (`search-min-width` to `search-max-width`). Never a fixed width.
- `surface` fill, 1px `border`, `radius-control`. Field text is `base`, the glyph is `small` in `ink-secondary`.
- Pass `focus:` for a field that takes focus on open (a picker sheet) or from a shortcut.
- The clear button is named "Clear search" for VoiceOver and the tooltip.

The consumer supplies the text binding, a localized prompt, and optionally a focus binding.
