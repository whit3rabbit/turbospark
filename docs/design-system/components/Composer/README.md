# Composer

The prompt box: the app's hero surface, with the most raised fill, the largest radius, and the only drop shadow.

```swift
VStack { editor; toolbar }
    .appComposerSurface(focused: isFocused, halo: false)
```

- `composer` fill, `radius-composer` (20), `shadow-composer`, a 1px `composer-edge`. On focus the edge becomes 1.5px `focus-edge` and the shadow deepens. The chat and Humanize composers pass `halo: false` and keep the edge-only look.
- `focus-edge` is 2.7:1 dark and 2.1:1 light, under the 3:1 focus floor. Increase Contrast swaps it for a solid 2.5px accent ring. Consider that ring at all contrast levels.
- Toolbar: attachments on the leading side, mode chips, then dictation and Send on the trailing side.
- Send is the surface's one primary action: `accent` circle, `on-accent` arrow, `glow-primary` while there is something to send. Disabled Send says why in its tooltip.
- The placeholder is "Ask anything" in `ink-secondary`. The editor measures its own height up to a cap, then scrolls.

The consumer supplies the editor, the chips, and the send action.
