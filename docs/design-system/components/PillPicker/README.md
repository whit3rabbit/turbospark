# PillPicker

The capsule mode switcher at the top of a pane that does a few different jobs, such as Speech-to-Text's Notes, Meeting, and Any Audio. One selection capsule slides between the options, so the move reads as one mark changing place.

```swift
AppPillPicker(selection: $mode, items: SpeechToTextMode.allCases.map { mode in
    AppPillItem(value: mode, label: Text(LocalizedStringKey(mode.titleKey), bundle: .module),
                systemImage: mode.symbol, liveLabel: mode == liveMode ? Text("Recording", bundle: .module) : nil)
}, label: Text("Speech-to-Text mode", bundle: .module))
```

- Track: `surface` capsule with a 1px `border`. Selected: `ink` label and `accent-ink` glyph on a `selection` capsule, with an `accent` stroke under Increase Contrast. Others: `ink-secondary`, with a `hover` fill on hover.
- The segments are native buttons, not a drawn replacement: Tab, Space, and VoiceOver work as they do for any button. The selected one carries `.isSelected`, and the group is named by `label`.
- Left and Right arrows move the selection and are mirrored under right-to-left. Focus draws a 2px `accent` ring, because custom-styled buttons draw none on macOS.
- The capsule slides with `TSMotion.select`, hover uses `TSMotion.hover`, and press scales to 0.96. Under Reduce Motion the capsule moves at once.
- An option with live work, such as a running recording, shows a breathing `accent` dot (`.tsPulse`) and speaks its `liveLabel` as the value, so it is never color alone.
- Every option uses the same weight in every state, so selecting never changes a width. The control does not shrink: put it in an `AppAdaptiveHeader`, which moves it to its own row when the actions do not fit. Do not add a menu fallback through `ViewThatFits`. Measuring the menu's popup that way re-measured the window every frame.
- Use it for two to four modes of one pane. A section's destinations belong in the sidebar panel, and a form choice belongs in a native `Picker`.

The consumer supplies the selection, a localized `Text` and SF Symbol per option, the live state, and a group label.
