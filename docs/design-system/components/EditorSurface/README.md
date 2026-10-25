# EditorSurface

The shared surface for a multi-line `TextEditor` or code area: the raised theme surface, the theme hairline, and a continuous corner.

```swift
TextEditor(text: $prompt)
    .appEditorSurface(minHeight: 120)
    .appFocusGlow(isFocused)
```

- `surface` fill, 1px `border`, `radius-control`, 8px inset. Code and prompts set in `code`.
- `TextEditor` has no native focus ring, so add `.appFocusGlow(_:)`: the accent edge plus `glow-focus`. A `TextField` keeps its system ring instead.
- `TextEditor` has no intrinsic height. Measure with a hidden `Text` (`PromptComposerEditor` is the reference) rather than picking a maximum.
- Under Increase Contrast the glow becomes a solid 2.5px accent ring.

The consumer supplies the text binding, a minimum height, and the focus state.
