# IconButton

An icon-only action that cannot ship without a name: its label is both the tooltip and the VoiceOver name, so the two never drift.

```swift
AppIconButton("sidebar.left", label: Text("Toggle Sidebar", bundle: .module)) {
    toggleSidebar()
}
```

- Use it for every icon-only control. A bare `Image` inside a `Button` has no name.
- The glyph is an SF Symbol in `ink-secondary`, `ink` on hover, with a `hover` fill.
- Pass `role: .destructive` for a destructive icon action, and confirm it like any other.
- If the action has a shortcut, list the chord in `KeyboardShortcutCatalog` and show it in the menu item, not in the tooltip.

The consumer supplies the SF Symbol name, a localized `Text` label, an optional role, and the action.
