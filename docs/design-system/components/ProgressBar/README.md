# ProgressBar

The themed progress bar: an accent fill with a soft glow and a sheen across the completed part, or an indeterminate sweep when the work has no total.

```swift
AppProgressBar(
    value: fraction,
    label: Text("Downloading model files", bundle: .module),
    valueText: Text(verbatim: "12.4 / 20.4 GB"),
    size: .regular,
    animated: !isPaused)
```

- `.regular` (8pt) for the main progress of a card, `.compact` (4pt) in a status strip or row. Both scale with the text size.
- Track is `track`. Fill is `accent` fading from 70% at the leading edge. Under Increase Contrast the fill is solid and the track gets a `border` stroke.
- VoiceOver reads a native progress indicator: role, label, and value. Pass `valueText` for a counter or time instead of a percent.
- Paused work holds still (`animated: false`), because a moving sheen reads as progress. Under Reduce Motion nothing moves and indeterminate becomes a static half bar.
- Mirrors as a whole drawing in right-to-left languages.
- Size with min, ideal, and max widths. Never a fixed width.

The consumer supplies the fraction or nil, a localized label, an optional value text, the size, and whether it animates.
