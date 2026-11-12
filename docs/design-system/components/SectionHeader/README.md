# SectionHeader

A heading inside a scrolling pane: a semibold title marked as a heading for VoiceOver's rotor, with an optional secondary line.

```swift
AppSectionHeader(
    title: Text("Thinking level", bundle: .module),
    subtitle: Text("No internal reasoning (fastest)", bundle: .module))
```

- Title in `base` semibold with `.isHeader`, subtitle in `small` `ink-secondary`, 3px apart.
- The subtitle wraps. It never truncates.
- A Settings pane never repeats its own title: the shell already shows it. Start with a one-line purpose.

The consumer supplies a localized title and an optional subtitle.
