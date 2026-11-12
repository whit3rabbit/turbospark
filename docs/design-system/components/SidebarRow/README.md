# SidebarRow

A row in the sidebar panel beside the rail: an 18pt glyph and a `small` label that wraps to two lines.

```swift
SidebarPanelRow(title: Text("Chat", bundle: .module), systemImage: "bubble.left", isSelected: true) { ... }
```

- 8pt by 4pt padding, `radius-control`, 9pt between glyph and label.
- Resting: `ink-secondary` glyph and label. Hover: `hover` fill. Selected: `selection` fill, `ink` semibold label, `accent-ink` glyph, and `.isSelected`.
- The label is the tooltip and the VoiceOver name. Hover actions on a row are also accessibility actions and reachable by keyboard.
- Keyboard chords are verbatim, in `code`, trailing.
- Use `SidebarModePanel` or `SidebarPanelList` rather than building rows by hand.

The consumer supplies a localized title, an SF Symbol, the selection, and the action.
