# AdaptiveHeader

A pane or workspace header that reflows instead of truncating: one row, then actions under the title, then a compact set that keeps the primary action and folds the rest into a labeled menu.

```swift
AppAdaptiveHeader {
    Text("Installed Models", bundle: .module).themedFont(.title2, weight: .semibold)
} actions: {
    AppSearchField(text: $q, prompt: Text("Search", bundle: .module))
    importButton.appAction(.secondary)
    discoverButton.appAction(.primary)
} compactActions: {
    discoverButton.appAction(.primary)
    Menu { importButton } label: { Text("More", bundle: .module) }.appMenuAction()
}
```

- `ViewThatFits` tries the three layouts in order. The last always takes the proposal, so German and Extra Large text still lay out.
- No fixed `.frame(width:)` on a picker, search field, or label in a header. Use min, ideal, and max.
- Extension-style panes (MCP, Skills, Agents, Plugins, Hooks) share one order: scope picker, search, spacer, overflow menu, primary action.
- Actions are `.regular`, 8px apart (`space-8`).

The consumer supplies the leading content, the full actions, and the compact actions.
