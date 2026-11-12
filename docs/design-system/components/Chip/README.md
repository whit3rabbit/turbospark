# Chip

The composer-style toggle chip used by the chat composer and Humanize: a faint fill that deepens on hover, and an accent tint while on.

```swift
AppChip(Text("Approve for me", bundle: .module), systemImage: "checkmark.shield", isOn: approves) {
    approves.toggle()
}
```

- Off: `ink-secondary` label on `chip-fill`, `chip-hover` on hover. On: `ink` label on `accent-tint` with a 0.5px `accent-edge`.
- The label is the visible title, the tooltip, and the VoiceOver name. An `iconOnly` chip still announces it.
- On state is exposed as `.isSelected`, not only as color.
- Disabled chips drop to 45% opacity and say why in the tooltip.
- Press scales to 0.96 with `TSMotion.press`. Hover never changes the chip's frame.

The consumer supplies a localized `Text`, an SF Symbol, the on state, and the action.
