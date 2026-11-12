# NavigationRail

The 52pt icon rail at the leading edge: where am I. One item per section, selected by accent, with Settings and the profile at the foot.

- `rail` fill, a 0.5px `hairline` on its trailing edge. Translucent at 90% unless Reduce Transparency is on.
- Items are 34pt squares (up to 42pt with the sidebar scale), `radius-chip`, glyph in `ink-secondary`, `hover` fill on hover.
- Selected: `accent-ink` glyph on `accent-tint`, plus a 2.5pt `accent` bar on the leading edge at 55% of the item height. The item carries `.isSelected`.
- Every item is an `AppIconButton`-style control: the section name is the tooltip and the VoiceOver name.
- Sections come from `AppNavigationSection`, the same source as the View menu and the Keyboard Shortcuts pane.

The consumer supplies the sections and the selection.
