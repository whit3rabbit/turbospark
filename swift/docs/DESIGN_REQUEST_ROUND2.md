# Drop-in request, round 2

What round 1 (`swift-dropins`, 7 files) did not cover, written as a brief for
whoever draws the next set. The boards depict roughly twelve surfaces; round 1
replaced five of them. The five landed and are live. The rest still render the
pre-refresh design, which is why the running app does not match the board.

Repo: `swift/TurboSparkApp/Sources/TurboSparkApp/`. Everything below is a path
relative to that.

## Already landed, do not redraw

`Chrome/TopBarView.swift` (telemetry centre), `Chrome/NavigationRailView.swift`,
`Components/WelcomeCharacterView.swift`, `Components/GenerateControl.swift`,
`Generation/ChatSidebarChatRowView.swift`, `Diagnostics/InspectorEssentialsSection.swift`,
`Theme/TurboSparkMotion.swift`. Spark Blue is the default theme and is pinned by
`AppearanceSettingsTests.testAFreshArchiveDefaultsToSparkBlue`.

One open question on a landed file: the board's suggestion cards read "Explain
this codebase" and "Write a Rust function"; the delivered
`WelcomeCharacterView.swift` says "Explain a codebase" and "Write a function".
If the board copy is the intended one, send the corrected file.

## Wanted, in priority order

### 1. `Chrome/ModelLoaderControl.swift` (231 lines today)

Board: a pill carrying a status dot, the alias `gptoss-20b`, a secondary line
`Ready - 64K ctx`, and an `Eject` button with a pause glyph.
Today: `gptoss-20b [play] Load`, no status line, no eject affordance.

Needs to cover all of: no model selected, selected but not open, opening
(progress), installing (stage text), open, and open-with-error. The resolved
context comes from `AppModel.resolvedContextTokens`; format it with the same
`.formatted()` the ladder uses rather than inventing a `64K` abbreviation, or
say explicitly that an abbreviation is wanted and where it should live.

### 2. `Generation/ChatSidebarView.swift` (391 lines today)

Board: a `Chats | Projects` segmented control in a rounded container; a
prominent `New chat` button showing its own shortcut; a `Search chats` field
showing its shortcut; `TODAY` / `EARLIER` group headers over the rows.
Today: plain tabs, a plain `New chat` row, a `Filter chats...` field, no
grouping.

This file owns the ONLY reachable Chat/Projects switch in the app now, and
Ctrl+Cmd+S can hide the whole sidebar, so a View-menu picker was added as the
fallback. Keep the segmented control here; do not assume the top bar has one.
At 391 lines it is over the 400-line guideline once grown, so split by subview.

### 3. `Generation/PromptComposerView.swift` (288) + `PromptComposerControls.swift` (161)

Board: placeholder `Ask anything, or press / for skills` with the `/` drawn as
an inline key cap; an `Ask before tools` pill; an accent focus ring on the
container.
Today: `Ask anything`, and the pill reads `Approve for me`.

`Approve for me` is the current permission-mode label. If `Ask before tools` is
a rename rather than a different control, say so explicitly, because the label
is user-facing state and there is a settings surface that also names it.

### 4. `Diagnostics/InspectorView.swift` (57) and a new advanced section

Board: header reads `Model` with a `LOADED` badge; below it a card with the
provider logo tile, `GPT-OSS 20B (MXFP4) - 12.12 GB on disk`, and the install
path in a monospace box with a copy button. Then `ESSENTIALS`. Then a single
collapsed row: `Advanced > Cache slots, expert residency, speculation, rate cap,
seed`.
Today: header reads `Model Settings`; the model block is a plain `Form` of
Model / Path / Choose Model Folder / Load Model / Installed size / Family; the
advanced knobs are a flat `Open & Architecture Options` section
(`Diagnostics/InspectorOptionsSection.swift`, 363 lines).

`InspectorEssentialsSection`'s doc comment already promises an
`AdvancedDisclosure`. No such type was delivered and none exists; the comment
now says so. Either send that type, or confirm the flat section is the intent.

Reuse `Installation/ModelLogoView.swift` for the logo tile. It template-tints
monochrome marks now, so a black-fill mark is legible on a dark tile.

### 5. `Chrome/StatusBarView.swift` (372 lines today)

Board: `Idle - nominal` plus a `Details` button, and nothing else.
Today: context fill, fans, throughput, tokens, thermal, memory pressure, and a
graph-mode toggle.

This is a large deletion, so it needs a decision rather than a drawing: where do
the six readouts go when the strip collapses to two? `Details` implies a popover
or a pane. Memory and CPU already moved up to the top bar in round 1; tok/s and
the token count were deliberately kept OUT of the top bar because a per-token
number in the chrome pulls the eye.

## Constraints every file must satisfy

These are gates in this repo, not preferences. Round 1 tripped three of them.

- **macOS 14 SDK.** `Package.swift` is `.macOS(.v14)` and the release job builds
  against it. A newer symbol does not compile even inside
  `if #available(macOS 15.0, *)`, because `#available` gates execution and not
  symbol resolution. `Color.mix(with:by:)` is the known trap; use the AppKit
  equivalent.
- **`@MainActor` on the TYPE**, not just on `body`. On the 14 SDK only `body` is
  isolated by the protocol, so private helpers that call each other need the
  type annotated.
- **One property per `Section`.** The 14 SDK type checker's budget is smaller; a
  `Form` holding five inline `Section`s exceeded the solver there while
  compiling locally.
- **No hardcoded `.font(`, and no hardcoded size.** `FontPropagationTests`
  reddens per CALL SITE on any `.font(` line with none of `themedFont` /
  `themedCode` / `theme.` / `uiFont` / `codeFont`, and on any raw size
  literal. Size text BY ROLE: `.themedFont(.small)` / `theme.code(.callout)`
  -- `AppFontStep`'s doc table is the canonical scale, and the old
  `points:` spellings were deleted. The one numeric spelling,
  `.themedFont(fitting:)`, is for layout-derived glyph sizes (a letterform
  at `size * 0.45`), never a constant.
- **Every `Text("literal"` takes `bundle: .module`**, and a new key must be
  translated into all 21 catalog languages in the same change. **List the new
  user-facing strings separately at the top of the delivery** so they can be
  added to `Localization/Localizable.xcstrings` without being reverse-engineered
  from the diff. The key IS the format string:
  `Text("\(x) on this Mac")` keys as `%@ on this Mac`.
- **ASCII only.** No emojis, no em dashes. The Swift tree has zero of the latter
  and uses ` -- ` where a dash is wanted. Round 1 introduced 16.
- **Under about 400 lines per file.** Split by subview past that.
- **Reduce motion** through `AppearanceManager.shouldReduceMotion(systemReduceMotion:)`,
  never `\.accessibilityReduceMotion` alone.
- **Read `theme.accent`, never a literal colour.** Round 1's `GenerateControl`
  fixed the one site that hardcoded a green send button.

## Two rules round 1 broke, stated as requirements

**Ship every definition the file references.** The `TopBarView.swift` drop-in
kept `GenerationPhaseIndicator(model: model)` at line 35 and dropped the
definition, which had lived at the bottom of the file it replaced. If a
replacement retires a type that other files use, name it in the README.

**Do not tell us to delete a call site without naming what the callee uniquely
does.** Round 1's README said to delete `ContextWindowOptionsView` from the
options section. That control is the only expression of `Auto` (window resolved
from the checkpoint and this machine's memory) and of a custom window size,
neither of which is a rung on the ladder that replaced it; following the
instruction would have made picking any rung a one-way door out of Auto. Same
shape with `PromptInteractionModeSegment`, whose removal left the Chat/Projects
switch reachable only inside a sidebar that can be hidden.

## Reuse rather than reinvent

- `Theme/TurboSparkMotion.swift`: `TSMotion.hover/.select/.press/.pane`,
  `.buttonStyle(TSPressScaleStyle(scale:))`, `.tsHoverLift()`, `.tsEntrance()`,
  `TSIdlingSparkView`.
- `Installation/ContextLadderView.swift` and `Installation/ModelFitPresentation.swift`
  for anything that prices a context window or shows a fit verdict.
- `Diagnostics/MetricFormat.swift` for bytes, rates, percentages and durations.
- `Theme/AppChromePresentation.swift` (`AppChromeLayout`) for bar heights, rail
  width, and `trafficLightClearance`. The window is `.hiddenTitleBar`, so the
  system draws traffic lights over the content at roughly x = 13 to 66.
- `Components/TaskProgressIndicatorView.swift` (`TaskProgressFlameIcon`) for the
  running indicator.

## Delivery shape that works

One folder of complete `.swift` files, plus a README that states, per file: what
it replaces, which types it adds, which types it REMOVES, any call site outside
the folder that must change, and the list of new user-facing strings. Round 1's
README did all of this except the removals and the strings.
