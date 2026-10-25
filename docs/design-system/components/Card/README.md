# Card

A grouped card: the raised theme surface, the theme border, and the card radius. Flat, with no shadow.

```swift
VStack(alignment: .leading) { ... }
    .appCardSurface()          // padding 18
```

- `surface` fill, 1px `border`, `radius-card` (10), `space-18` padding.
- Cards separate from the page by fill step and border, never by drop shadow.
- Start a card with an `AppSectionHeader` when it has a title. Card titles are `callout` or the section header's `base` semibold.
- Machine facts inside a card (sizes, token counts, ids) are set in `code`.
- Do not nest cards. Group with a section header and spacing instead.

The consumer supplies the content and, rarely, a padding override.
