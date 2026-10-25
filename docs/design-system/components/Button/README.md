# Button

The four action treatments, picked by consequence: one primary per surface, secondary for the rest, quiet for in-row and repeated actions, and destructive for anything that deletes or discards.

```swift
Button { load() } label: { Text("Load Model", bundle: .module) }
    .appAction(.primary)
    .appPrimaryGlow(canLoad)
```

- **Primary:** the thing the pane exists to do. `accent` fill, `on-accent` label, `glow-primary` while it can run. One per surface.
- **Secondary:** every other action. `elevated` fill with a `border` edge.
- **Quiet:** in-row and repeated actions (Cancel, row actions). No fill until hover.
- **Destructive:** `danger` label on the secondary shape. Always `Button(role: .destructive)` behind a `confirmationDialog` with a localized title, message, and button, so the role reaches VoiceOver and not only sighted users.
- Header actions use `.regular`. Actions inside rows, cards, and inspectors use `.small`. Never mix sizes in one row.
- A disabled button says why, in its tooltip or in text beside it. The chat Send button is the reference.
- Crowded headers fold the rest into `Menu { ... } label: { ... }.appMenuAction()`, a labeled pull-down drawn like a secondary action.

The consumer supplies a localized `Text` label, an optional SF Symbol, the role, and the size.
