# EmptyState

The empty state every list and workspace uses: a symbol, a title, an explanation, and the action that fills the space.

```swift
AppEmptyState(
    systemImage: "square.stack.3d.up",
    title: Text("No models installed yet", bundle: .module),
    message: Text(...)) {
    Button { ... } label: { Text("Discover Models...", bundle: .module) }.appAction(.primary)
}
```

- Symbol at `display` in `ink-secondary`, hidden from VoiceOver. Title in `title3` semibold with `.isHeader`. Message in `base` `ink-secondary`, at most 440pt wide times the text scale.
- Say what goes here and offer the action that fills it, in user terms. Never name an internal tool or type.
- One primary action at most. A second path (Add Local Folder) is secondary.

The consumer supplies an SF Symbol, a localized title and message, and the actions.
