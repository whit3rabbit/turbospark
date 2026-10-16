# TurboSpark design system

The condensed, machine-readable reference for TurboSparkApp UI: tokens for
every theme, the shared control vocabulary as component guides, and the brand
book. It condenses [DESIGN.md](../../DESIGN.md) and the SwiftUI theme in
`swift/TurboSparkApp/Sources/TurboSparkApp/Theme`. Read it together with
DESIGN.md before any UI or design change.

## Using this folder

Precedence, when two sources disagree:

1. [DESIGN.md](../../DESIGN.md) decides the rules.
2. The Swift theme code decides the values. If `tokens.json` disagrees with
   it, `tokens.json` is stale: regenerate it (below) instead of copying the
   stale value into Swift.
3. This folder is the derived reference.

For a UI change, read in this order:

1. DESIGN.md.
2. The matching section of this README (Color, Type, Space, Motion,
   Accessibility, Localization, Iconography).
3. `components/<Name>/README.md` for the control you are touching. Each one
   names its SwiftUI call, what the caller supplies, and its do and don't
   rules.
4. `tokens.json` for an exact value or a contrast figure.

What is here:

| Path | What it is |
| --- | --- |
| `tokens.json` | Colors for 8 themes, type scale, spacing, radii, shadows, and layout sizes. Every token has a `usage` note that names where it is used and, for text colors, the worst contrast it holds. |
| `tokens.css` | `tokens.json` compiled to CSS custom properties, one `[data-theme]` block per theme. Generated. |
| `components/<Name>/README.md` | One guide per control in the DESIGN.md vocabulary. |
| `components/<Name>/preview.html`, `components/bundle.css` | Static web renditions of those controls. The app draws native macOS controls, not these. |
| `components/Cover/preview.html` | The cover art for the rendered copy. |
| [assets.md](assets.md) | Where the logos, mascot art, tool icons, and fonts live in the repo. Nothing is duplicated here. |
| `gallery.html` | A local viewer for every preview in every theme. |
| `tools/derive_tokens.py` | Regenerates the color and shadow tokens and `tokens.css` from the shipped theme values. |

Rules for using it:

- Never paste a hex value from `tokens.json` into Swift. Use the theme
  accessor from the SwiftUI mapping table at the end of this file.
- A token whose note starts with "Intentional addition" does not exist in
  Swift yet. Adding it to the theme is a code change, with its own review and
  tests, not a lookup.
- Theme ids: `dark` and `light` are Spark Blue, the default. `codex-*` and
  `emerald-*` are shipped presets. `hc-*` is the High Contrast preset under
  Increase Contrast.

Commands, from the repository root:

```sh
# Regenerate color and shadow tokens and tokens.css after a theme value changes
python3 docs/design-system/tools/derive_tokens.py
# Fail if tokens.json or tokens.css is stale, or a preset hex no longer matches Swift
python3 docs/design-system/tools/derive_tokens.py --check
# Browse every preview in every theme
python3 -m http.server 8000
# then open http://localhost:8000/docs/design-system/gallery.html
```

A rendered copy of this system lives on claude.ai as the "TurboSpark" Design
System artifact: https://claude.ai/artifact/9vc8M9xMSUaWt5yymosrG2. It is
private, so ask the owner for access. This folder is the source and the
artifact mirrors it, so update the folder first. An agent with the Artifact
tool can read the artifact, but it should read this folder instead.

---

TurboSpark is a macOS app that runs local models for text, audio, images, and agent decisions on the user's own Mac. The interface is a developer tool first: a dark, layered night-sky canvas, one blue accent, machine facts in monospace, and native macOS controls that carry keyboard, focus, and VoiceOver behavior for free. It ships in 21 languages and must survive Extra Large text, Increase Contrast, Reduce Motion, Reduce Transparency, and right-to-left layout as normal conditions, not edge cases.

## Principles

- **Native first:** Build on `Button`, `Menu`, `Toggle`, `Picker`, `TextField`, `Form`. Share the treatment through the vocabulary (`.appAction`, `AppChip`, `.appCardSurface` and the rest), never by drawing a replacement control.
- **The theme decides color and size:** No view picks its own color, font family, or point size. Read `@Environment(\.appTheme)` or the `.app*` shape styles. Size text by role (`.themedFont(.small)`), never by number.
- **One accent, spent on purpose:** `accent` marks the one primary action, the current location, focus, and live work. Everything else is ink on layered surfaces.
- **Machine facts in mono:** Model ids, file paths, token counts, context sizes, telemetry, and keyboard chords are set in the `code` styles. Prose and labels stay in the interface face.
- **Layouts adapt instead of clipping:** A header reflows to two rows, then folds actions into a labeled menu. Text is never shrunk to fit.
- **Every control has a name, and every name is localized.**

## Voice and copy

TurboSpark speaks plainly about what the machine is doing. Short, literal, local.

- Lead with the user's outcome or the machine's state: "Everything here runs on your Mac. Nothing leaves it." "No tasks yet". "Eject the loaded model and free its memory".
- Title Case for buttons, menu items, settings rows, and pane and window titles: "Choose Model Folder...", "Reload Model", "Unload Model", "Power Profile". Sentence case for descriptions, placeholders, empty-state messages, and tooltips: "Ask anything", "No internal reasoning (fastest)".
- Name the target when a label could mean several things: "Load Chat Model", "Unload Image Model". A label that reads true only in context is wrong in a menu.
- Say what to do next, in user terms. An empty state never names an internal tool or type.
- ASCII only: three periods for an ellipsis ("Search..."), "--" or a colon where an em dash would go. No emoji anywhere in the interface.
- Second person, no "I". The assistant's own voice lives in the transcript, not in chrome.
- Units and numbers are verbatim and grouped: "262,144 ctx", "12.16 GB", "2.6% CPU".
- Never localize prompts sent to a model, persisted values, identifiers, or log lines.

## Color

The palette is three values per mode (background, foreground, accent), and every surface is a fixed step off the background. That is what lets any custom theme keep the same hierarchy. Spark Blue is the default preset: `page` #090D16 with `accent` #3B9EFF in dark, white with #1D6FE0 in light.

- **Surfaces, lowest to highest:** `page` (canvas and transcript), `rail`, `sidebar`, `bar` (top bar and status strip), `surface` (cards, fields, editors), `elevated` (menus, popovers), and `composer` (the prompt box). Separate them by fill step and a 1px `border`, not by shadow.
- **Seams:** chrome bands meet at a 0.5px `hairline` that is a lighter line of the same palette. Never `Color(nsColor: .separatorColor)` or any system color on a themed surface.
- **Text:** `ink` for content, `ink-secondary` for subtitles, metadata, and placeholders, `ink-disabled` only for disabled controls.
- **Accent:** `accent` fills the primary action and progress. `accent-ink` is accent as text or a small glyph. `on-accent` is the label on an accent fill, never literal white (white on the Spark Blue dark accent is 2.7:1). Selected rows use `selection`. On chips and the selected rail item use `accent-tint` with `accent-edge`.
- **Status:** `success`, `warning`, and `danger`, each with a `-tint` background, always with a glyph and a word: checkmark.circle.fill "Allowed", exclamationmark.triangle.fill "Degraded", xmark.octagon.fill "Failed". Status is never color alone, so it survives color-vision deficiency and Differentiate Without Color. Red is the destructive convention and also carries `role: .destructive`.
- `night` is the constant brand navy for the cover, marketing art, and icons. It is not a UI surface.

### Theme presets

Users pick a preset or build their own. Each is a background, foreground, and accent per mode, and the surface ladder above is derived from them.

| Preset | Light accent / background / ink | Dark accent / background / ink |
| --- | --- | --- |
| Spark Blue (default) | #1D6FE0 / #FFFFFF / #141A24 | #3B9EFF / #090D16 / #F2F5FB |
| Codex | #111827 / #FFFFFF / #1A1C1F | #F3F4F6 / #181818 / #FFFFFF |
| TurboSpark Emerald | #237D32 / #FFFFFF / #1A1C1F | #6ABA71 / #151816 / #F0FDF4 |
| Midnight Blue | #2563EB / #F8FAFC / #0F172A | #38BDF8 / #0B0F19 / #F1F5F9 |
| Charcoal Mono | #18181B / #FAFAFA / #18181B | #E4E4E7 / #09090B / #FAFAFA |
| High Contrast | #000000 / #FFFFFF / #000000 | #FFFFFF / #000000 / #FFFFFF |
| Sandstone | #98602B / #FAF6EE / #302A24 | #E6B77C / #211C17 / #F3E9D8 |
| Orchid | #7B3DA5 / #FAF5FC / #2C2034 | #D3A6F0 / #201626 / #F4EAF9 |
| Forest | #176B5D / #F0F8F4 / #18342C | #76D3B3 / #10251F / #E2F5EB |

The eight themes in this system are Spark Blue, Codex, and TurboSpark Emerald, each in dark and light, plus the High Contrast preset under Increase Contrast in dark and light. Codex is the neutral near-black look. Emerald is the original brand green.

## Type

One canonical scale, sized by role. Each role is a multiplier of the user's base size (16pt UI, 12pt code by default), so Text Size (Default 1.0, Large 1.15, Extra Large 1.30) and Dynamic Type move every surface together.

- Body and transcript: `base`. Control labels, card and sheet titles: `callout`. Secondary rows, sidebar text, subtitles: `small`. Timestamps, dense metadata: `tiny`. Overline badges and chart ticks: `micro`, never for sentences.
- Headings: `title3` for sections and empty states, `title2` for panes, `title` for windows. Mark every heading with `.accessibilityAddTraits(.isHeader)`.
- Code and machine facts: `code` (12pt) for code blocks, terminal output, model ids, paths, token counts, and chords. `code-callout` is for telemetry in the top bar and status strip.
- The welcome greeting is the one serif line in the app: `greeting`, the system serif at display size, regular weight. Use it nowhere else.
- The interface face is the system font (SF Pro) unless the user picks another. The code face is SF Mono. Geist and Geist Mono are bundled options and the web fallback in this system. Line heights here are the web rendition. SwiftUI uses each face's natural leading.
- The Appearance options type group in `tokens.json` shows the bundled families a user can choose: Inter for the interface, and JetBrains Mono, Fira Code, Cascadia Code, Hack, and Source Code Pro for code. They are specimens. A view never names a family, it reads the theme. Iosevka also ships but is not carried here (11 MB per file).
- A size the scale lacks is a new `AppFontStep` case with its pinned expectation in `FontPropagationTests`, never a literal.

## Space, shape, and layout

- Spacing is the real, irregular set the views use: `space-8` between header actions, `space-18` card padding, `space-5` chip and field padding, `space-12` between a title and its actions. Do not snap these to a grid.
- Corners are continuous: `radius-control` (7) for fields, editors, and sidebar rows, `radius-chip` (8) for chips and rail items, `radius-card` (10) for cards, `radius-composer` (20) for the composer, and `radius-pill` for progress and status pills.
- The main window is four bands, each answering one question: the rail and sidebar (where am I), the `top-bar-height` bar (what is the machine doing), the working panes, and the `status-bar-height` strip. Do not add a fifth band or move a question between bands.
- No window or header control has a fixed width. Use min, ideal and max (`search-min-width` to `search-max-width`), and adjustable split views for list and detail.
- A frame that holds text, and any width breakpoint, is multiplied by `AppMetrics.textScale(theme)` so Extra Large text changes layout instead of clipping.
- Long paths are selectable, truncate in the middle on one line, and carry the full value in the tooltip.

## Depth and effects

- Cards are flat: fill step plus border. The composer is the only surface with a drop shadow (`shadow-composer`, `shadow-composer-focus`).
- Glows are accent shadows from the vocabulary only: `glow-focus` around an editor that has no native ring, `glow-primary` under the one primary action while it can run. Under Increase Contrast every glow becomes a solid ring or nothing, so a glow is never the only signal of focus or readiness.
- Blur and translucency only on navigation chrome (rail at 90%, sidebar at 94%), and fully opaque under Reduce Transparency.

## Motion

Motion is fast, springy, and named. Every animation goes through `TSMotion` and honors Reduce Motion (the app setting first, the system setting as its fallback).

| Curve | Value | Use |
| --- | --- | --- |
| hover | ease-out 0.13s | hover, chip highlight, focus edge |
| select | spring response 0.32, damping 0.82 | selection moving, segments, disclosure |
| press | spring response 0.22, damping 0.7 | button press-down (scale 0.96 to 0.98) |
| pane | spring response 0.44, damping 0.86 | panes sliding, welcome entrance |
| sweep | ease-in-out 1.4s | one pass of a progress sheen or indeterminate sweep |
| breathe | ease-in-out 0.85s | one breath of a live dot (opacity 0.45 to 1) |
| snap | linear 0.01s | the invisible reset at the end of a sweep |

Hover lifts by scale (1.012), never by padding, so a row never re-lays-out its neighbors. A looping animation runs only while its state is live and stops for paused work. Under Reduce Motion an indeterminate bar becomes a static half bar.

## Accessibility

Every pair below was computed from the tokens, as the worst case across every surface in that theme, rounded down. Status is the lowest of `success`, `warning`, and `danger`, on any surface or its own tint.

| Theme | `ink` | `ink-secondary` | `accent-ink` | `on-accent` | status (worst) | `focus-ring` | `border-strong` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Spark Blue dark | 14.4 | 8.6 | 5.6 | 6.5 | 5.5 | 5.6 | 3.1 |
| Spark Blue light | 15.6 | 7.2 | 5.0 | 4.7 | 4.8 | 4.2 | 3.0 |
| Codex dark | 13.8 | 8.4 | 12.5 | 16.7 | 4.8 | 12.5 | 3.0 |
| Codex light | 15.2 | 7.0 | 15.8 | 17.7 | 4.8 | 15.8 | 3.0 |
| Emerald dark | 13.3 | 8.1 | 5.8 | 7.7 | 4.9 | 5.8 | 3.0 |
| Emerald light | 15.2 | 7.0 | 4.6 | 5.1 | 4.8 | 4.6 | 3.0 |
| High contrast dark | 17.9 | 16.5 | 17.9 | 18.3 | 8.6 | 17.9 | 10.3 |
| High contrast light | 18.8 | 17.7 | 18.8 | 21.0 | 7.3 | 18.8 | 9.7 |

Known gaps in the shipped values, kept exact and flagged:

- `border` is 1.5:1 to 2.2:1 across the preset themes (Spark Blue 2.1:1 dark, 1.6:1 light). Fine for decorative card edges. Use `border-strong` where an edge alone identifies a control.
- `focus-edge`, the composer's 55% accent focus edge, is 2.6:1 dark and 2.1:1 light in Spark Blue, and 2.8:1 and 2.1:1 in Emerald. Only Codex clears 3:1 (4.9:1 and 3.8:1). Pair it with the solid `focus-ring` or raise it.
- Spark Blue's light `accent` as text drops to 4.2:1 on `composer`: set text in `accent-ink`. Its `focus-ring` is 4.2:1 there, still above the 3:1 focus floor.
- The app still draws status with system green, orange, and red. Light-mode system red is 3.5:1 on white. The status tokens here are the replacement.

Rules:

- Every control has an accessible name. Icon-only actions use `AppIconButton`, whose label is both tooltip and VoiceOver name. A toggle with a separate row label keeps a real label hidden with `.labelsHidden()`.
- Sliders and steppers expose `.accessibilityValue`. Selected rows, tabs, and on chips carry `.isSelected`. A row that reads as one item is one element.
- Hover-only actions are also an `.accessibilityAction` and reachable by keyboard. Long lists offer a rotor.
- Every action is reachable by keyboard. Sheets bind Save or Done to `.defaultAction` and Cancel to `.cancelAction`. A sheet focuses its first field and returns focus on close.
- Asynchronous completion and failure are announced (`showToast` posts an announcement).
- A disabled control says why, in its tooltip or adjacent text.
- macOS privacy prompts appear only after the user presses Allow. Declining one never blocks a feature that does not need it.

## Localization

The string catalog is the source of truth for 21 languages: English, Spanish, French, German, Italian, Portuguese (Brazil), Russian, Japanese, Korean, Chinese Simplified, Chinese Traditional, Arabic, Hindi, Dutch, Polish, Turkish, Ukrainian, Swedish, Vietnamese, Indonesian, and Hebrew.

- A new string ships with all 21 translations in the same change.
- Every visible string, placeholder, tooltip, dialog title, and VoiceOver name goes through `Text("...", bundle: .module)` or `AppStrings.localized("Key")`. A bare literal renders English under every language.
- Components take `Text`, not `String`, so the literal stays visible to the localization gates.
- Leading and trailing, never left and right. Arabic and Hebrew mirror the whole layout, and progress mirrors as a whole drawing.
- Budget for expansion: German runs about a third longer than English, and Extra Large text adds 30% more. Headers reflow. Labels wrap or truncate with the full text in the tooltip.
- Glyphs, units, numbers, key names, code, file names, model ids, and URLs are verbatim. Language names stay in their own script.
- Check every UI change in English, German, and Arabic or Hebrew, at Default and Extra Large text, and in light, dark, and one custom theme.

## Iconography

- **Interface glyphs are SF Symbols**, named in code (`magnifyingglass`, `xmark.circle.fill`, `arrow.up`, `sparkles`). They inherit the themed font size and `ink-secondary` or `accent-ink`. Previews in this system draw stand-ins with stroke `currentColor`.
- **Tool icons are Lucide**, pinned to revision 266fd1c (ISC), shipped as 24px single-ink SVGs with stroke `currentColor` and stroke width 2. All 51 are listed in [assets.md](assets.md). They mark tool calls and agent capabilities in the transcript.
- **Status glyphs** are fixed: checkmark for success, triangle for warning, octagon for danger.
- No emoji, anywhere.
- **The mascot** is a cyan flame with a face. It appears on the welcome screen, task progress, and the AGENTS.md, MEMORY.md, and SOUL.md file screens, one character per screen at most, and never as a status indicator. The art is raster. Never redraw or recolor it.
- **The app icon** is the mascot riding a rocket on the night sky. The lockup sets it beside the TurboSpark wordmark.

## SwiftUI mapping

| Token | SwiftUI |
| --- | --- |
| `page`, `surface`, `elevated` | `.appPage`, `.appSurface`, `.appElevated` |
| `ink`, `ink-secondary`, `border`, `accent` | `.appText`, `.appSecondary`, `.appBorder`, `.appAccent` |
| `rail`, `sidebar`, `bar`, `hairline` | `TurboSparkTheme.railBackgroundColor(isDark:)` and siblings (non-view code only) |
| `composer` | `theme.composerBackground` |
| `selection` | `theme.selection` |
| `on-accent` | `Color.contrastForeground` |
| type styles | `.themedFont(.<role>)`, `.themedCode(.<role>)`, `theme.ui(.display, systemDesign: .serif)` |
| `radius-control`, `radius-card`, `radius-composer` | `AppMetrics.controlRadius`, `.cardRadius`, `.composerRadius` |
| `space-8` | `AppMetrics.actionSpacing` |
| motion | `TSMotion.hover / select / press / pane / sweep / breathe / snap` |

Every scene injects the theme, locale, and layout direction itself with `.appThemed()`. A new scene that skips it renders the light fallback in English.
