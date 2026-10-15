# DESIGN.md

Design rules for the TurboSpark macOS app (`swift/TurboSparkApp`). Read this
before ANY UI or design change: a new view, a new control, a copy change, a
color, a font, a layout, a menu item, a shortcut, an animation.

Every rule here exists because breaking it already shipped a defect. The
2026-10-11 UI audit found truncated headers, controls with no accessible
name, English under every language, duplicate menus, and a different button
style on every pane. Earlier cleanup passes found raw font sizes, system
colors on themed surfaces and scenes drawn with the wrong theme. Following
this file is how those stay fixed.

Code lives in `swift/TurboSparkApp/Sources/TurboSparkApp` (below: `Sources/`).

## Contents

1. Principles
2. The shared control vocabulary
3. Layout
4. Theming and type
5. Accessibility
6. Localization
7. Menus, commands, and shortcuts
8. Platform constraints
9. Verification

## 1. Principles

- **Native first.** Build on `Button`, `Menu`, `Toggle`, `Picker`,
  `TextField`, and `Form`. They carry keyboard, focus, and VoiceOver behavior
  for free. Share the TREATMENT through the vocabulary below, never by
  replacing a native control with a drawn one. Do not add a UI dependency
  without asking first.
- **One source of truth.** A setting, a label, a command name, or a set of
  picker options is defined once and reused. Two spellings of the same thing
  drift (Installed vs Installed Models, three different Text Size pickers).
- **Every control has a name, and every name is localized.** Visible text,
  placeholders, tooltips, dialog titles, and VoiceOver names all go through
  the string catalog.
- **The theme decides color and size.** No view picks its own color, font
  family, or point size.
- **Layouts adapt instead of clipping.** Long translations, the Extra Large
  text size and a narrow window are normal conditions, not edge cases.
- **Access is asked for, never taken.** The app shows a macOS privacy prompt
  only after the user chooses to allow access, and declining one permission
  never blocks a feature that does not need it.

## 2. The shared control vocabulary

Build from these instead of local variants. They live in
`Sources/Theme/AppControls.swift`, `Sources/Theme/AppPreferenceOptions.swift`,
`Sources/Theme/AppStrings.swift` and `Sources/Theme/TurboSparkMotion.swift`.

| Need | Use |
| :--- | :--- |
| An action | `.appAction(.primary / .secondary / .quiet / .destructive, size:)` |
| Overflow of a crowded header | `Menu { ... } label: { ... }.appMenuAction()` |
| A pane or workspace header | `AppAdaptiveHeader(leading:actions:compactActions:)` |
| Search | `AppSearchField(text:prompt:)`, with `focus:` for a field that takes focus |
| An icon-only action | `AppIconButton(symbol, label: Text(...)) { ... }` |
| A composer-style toggle chip | `AppChip(Text(...), systemImage:isOn:) { ... }` |
| A multi-line editor | `TextEditor(...).appEditorSurface(minHeight:)` |
| A heading in a scrolling pane | `AppSectionHeader(title:subtitle:)` |
| An empty list or workspace | `AppEmptyState(systemImage:title:message:) { actions }` |
| Mode, Text Size, Language pickers | `AppearanceModeOptions()`, `TextSizeOptions()`, `LanguageOptions()` |
| A localized `String` value | `AppStrings.localized("Key")` |
| An animation from an action | `TSMotion.animate(TSMotion.pane) { ... }` |
| An animation on a value | `.tsAnimation(TSMotion.hover, value:)` |
| Radii | `AppMetrics.controlRadius` (fields, chips), `AppMetrics.cardRadius` (cards) |

If the vocabulary lacks something you need twice, add it to the vocabulary in
the same change, documented here. A second local copy is how the drift starts.

### Action hierarchy

- **One primary action per surface**: the thing the pane exists to do. The
  rest are secondary, quiet (in-row or repeated actions), or destructive.
- Header actions use `.regular`. Actions inside rows, cards, and inspectors
  use `.small`. Do not mix sizes within one row.
- Destructive actions use `Button(role: .destructive)` and go through a
  `confirmationDialog` with a localized title, message, and button. The role
  reaches VoiceOver and the dialog, the red label only reaches sighted users.
- A disabled control says why, in its tooltip or in adjacent text. The chat
  Send button is the reference.

### Asking for macOS access

Microphone and screen recording (which ScreenCaptureKit needs even for audio
only) are macOS privacy grants. A prompt the user did not ask for reads as
the app taking access, and macOS shows each prompt only once.

- Never request at launch, on appear, on a toggle, or as a side effect.
  Reading status never prompts, so show it freely. Requesting does, and so
  does any ScreenCaptureKit content query while access is missing.
- Only `RecordingPermissions` calls the prompt APIs, from an Allow button the
  user pressed after seeing what the access is for. The capture layer checks
  and throws an error that points to the access sheet. It never asks.
  `RecordingPermissionsTests` scans `Sources/` to hold this.
- A control that starts a capture goes through
  `.recordingAccessGate($request, permissions: RecordingPermissions.shared)`:
  start directly when nothing is missing, otherwise set a
  `RecordingAccessRequest`. The Record panel is the reference.
- Each permission row shows status as glyph plus text and offers one next
  step: Allow while macOS can still prompt, Open System Settings after.
- Declining is a supported path. Offer to continue with the allowed sources
  (Record microphone only), keep Not now one click away, and say what still
  works without access. Import, transcription, and editing need none.

## 3. Layout

### Window structure

- The main window is four chrome bands, each answering one question:
  sidebar (where am I), top bar (what is the machine doing), working panes,
  status strip. Do not add a fifth band or move a question between bands
  (`.claude/docs/modules/swift.md` Gotcha 17).
- No window has a fixed frame. The main window enforces its minimum through
  `MainWindowMinimumSize`. The Settings window is resizable above
  `AppSettingsView.minimumSize` and opens at its default size through
  `SettingsWindowFrame`, because the Settings scene ignores ideal sizes.
- Each Settings pane is shown under a shell that ALREADY renders the pane
  title. A pane never repeats its own title. It starts with a one-line
  purpose and its actions. Insets come from `SettingsLayout`.

### Headers and rows

- A header never truncates. Use `AppAdaptiveHeader`: one row, then actions
  under the leading content, then a compact set that keeps the primary action
  visible and folds the rest into a labeled menu.
- No fixed `.frame(width:)` on a picker, search field, or label in a header or
  form row. Use `minWidth`/`idealWidth`/`maxWidth`. A fixed width is what
  wrapped "Installed model type" into several lines and clipped the Agents
  scope picker.
- Extension-style panes (MCP, Skills, Agents, Plugins, Hooks) share one
  header layout: scope picker if any, search, spacer, secondary or overflow
  menu, and primary action.
- List and detail splits are adjustable: `HSplitView`, or a list column with
  min, ideal, and max widths. Never a fixed list width.

### Text in layouts

- Never shrink text to make it fit. Let it wrap, or truncate with the full
  value in a tooltip.
- Long filesystem paths are selectable (`.textSelection(.enabled)`), use
  `.lineLimit(1)` with `.truncationMode(.middle)`, and carry the full path in
  `.help`.
- `TextEditor` has no intrinsic height, and both obvious ways to give it one
  pick the maximum. Measure with a hidden `Text` instead
  (`PromptComposerView.editor`, Gotcha 18).

### Forms and files

- Settings forms use `Form` with `.formStyle(.grouped)`,
  `.scrollContentBackground(.hidden)` and `.background(.appPage)`. A grouped
  `Section` draws AppKit chrome no SwiftUI background can override, so do
  not fight it with custom fills.
- One computed property per `Section`. A `Form` holding several sections in
  one expression exceeded the macOS 14 SDK type-checker budget in CI while
  compiling fine locally.
- Keep view files near 400 lines. Split by subview into a sibling file
  rather than growing one (Gotcha 15).

### Empty states and placement

- Every list and workspace has an `AppEmptyState` that says what goes here
  and offers the action that fills it, in user terms. Never name an internal
  tool or type (the Scheduled Tasks empty state used to say "CronCreate").
- A setting has ONE canonical location. If it must be reachable from a
  second place, reuse the same component and the same label there, and point
  the settings search entry at the canonical pane.

## 4. Theming and type

### Color

- Read the theme from the environment: `@Environment(\.appTheme)` or the
  semantic shape styles `.appPage`, `.appSurface`, `.appElevated`, `.appText`,
  `.appSecondary`, `.appBorder`, `.appAccent`. These resolve per mode and per
  custom theme.
- Never use system colors on a themed surface (`Color(nsColor:
  .separatorColor)`, `.controlBackgroundColor` and friends). They ignore the
  palette, which is how a dark-navy theme drew neutral gray boxes
  (Gotcha 62).
- `TurboSparkTheme`'s static accessors are for non-view code only.
  `NSApp.effectiveAppearance` does not move with `.preferredColorScheme`, so
  a static read in a view draws the wrong mode (Gotcha 36).
- Every scene injects the theme, locale, and layout direction itself:
  `Window`, `Settings`, and `MenuBarExtra` each call `.appThemed()` and set
  `\.locale` and `\.layoutDirection`. A new scene must too, or it renders the
  light fallback theme in English.
- Red is the destructive convention. Status is never color alone: pair it
  with a glyph or text (checkmark, triangle, octagon), so it survives color
  blindness and Differentiate Without Color.

### Type

- Size text by role, never by number: `.themedFont(.small)`,
  `.themedCode(.base)`, or `theme.ui(.title3, weight:)`. The roles are
  `AppFontStep`: micro, tiny, small, callout, base, large, title3, title2,
  title, hero, display. Never write `.font(.caption)` or a point size.
- `.themedFont(fitting:)` is only for glyphs whose size a frame dictates
  (`size * 0.45`), never a constant.
- A size the scale lacks is a new `AppFontStep` case, documented in its doc
  table, with the pinned expectation in `FontPropagationTests` updated
  (Gotcha 63).
- Font families come from Appearance settings. A site that must look serif
  or monospaced at the default passes `systemDesign:`.

### Surfaces and effects

- Surfaces go through the theme so Reduce Transparency and Increase Contrast
  apply: `surfaceOpacity` forces opaque structural surfaces, and
  `isHighContrast` raises borders and secondary text (Gotcha 79). Do not
  hard-code a lower opacity for borders or secondary text than the theme
  gives.
- Blur and glass only on floating or navigation chrome where they help
  hierarchy, availability-gated, and opaque under Reduce Transparency.
- Corners are continuous (`style: .continuous`) and use `AppMetrics` radii.

## 5. Accessibility

### Names, values, and structure

- Every control has an accessible name. A toggle, slider, or picker whose
  visible row label is separate keeps a REAL label and hides it with
  `.labelsHidden()`. `Toggle("")` is a control with no name, and the
  localization gate rejects it.
- Icon-only actions use `AppIconButton`, which makes the label both the
  tooltip and the VoiceOver name so the two cannot drift.
- Sliders and steppers expose their formatted value with
  `.accessibilityValue`.
- Titles and section headings carry `.accessibilityAddTraits(.isHeader)`.
  Selected rows, tabs, and on chips carry `.isSelected`.
- A row that reads as one item is one element
  (`.accessibilityElement(children: .combine)`).
- Hover-only actions are also exposed with `.accessibilityAction` and are
  reachable by keyboard. Chat messages and sidebar rows are the reference
  (`swift/docs/KEYBOARD_SHORTCUTS.md`).
- Long lists that users navigate by item offer a rotor (the transcript's
  Messages rotor, Audio's Markers and Speakers rotors).
- Asynchronous completion and failure are announced. `showToast` already
  posts an announcement, so prefer it over a silent status change.

### Keyboard

- Every action is reachable by keyboard. Sheets bind Save or Done to
  `.keyboardShortcut(.defaultAction)` and Cancel to `.cancelAction`.
- Focus moves predictably: a sheet focuses its first field, and closing a
  sheet returns focus to where it was.

### Motion and display settings

- Use the `TSMotion` curves (hover, select, press, pane). No inline
  durations.
- Animations honor Reduce Motion through `TSMotion.animate`,
  `.tsAnimation(_:value:)` or `TSMotionContext`. A bare `withAnimation` or
  `.animation(_:value:)` animates even when the user turned motion off
  (Humanize did).
- Layouts survive the Extra Large text size, Increase Contrast, and Reduce
  Transparency. These are checked, not assumed (section 9).

## 6. Localization

The string catalog (`swift/TurboSparkApp/Localization/Localizable.xcstrings`)
is the source of truth for 21 languages. The full pipeline and its ten
gates are in `swift/docs/SWIFT_LOCALIZATION.md`. The rules that matter on
every change:

- **A new string ships with all 21 languages in the same change.** Add the
  key (the English text is the key), translate it, run
  `make compile-strings`. "Translate later" fails the parity gate.
- **Never pass a bare literal to a key-only API.** Those overloads resolve
  against `Bundle.main`, which is empty in a SwiftPM build, so the text
  renders in English under every language. Use the bundled spellings:

  | Instead of | Write |
  | :--- | :--- |
  | Text with a bare literal | `Text("Save", bundle: .module)` |
  | Button with a title | `Button { ... } label: { Text("Save", bundle: .module) }` |
  | Label with a title | `Label { Text("Add", bundle: .module) } icon: { Image(systemName: "plus") }` |
  | Toggle with a title | `Toggle(isOn: $x) { Text("Enabled", bundle: .module) }` |
  | Picker with a title | `Picker(selection: $x) { ... } label: { Text("Mode", bundle: .module) }` |
  | TextField with a placeholder | `TextField(text: $q, prompt: Text("Search...", bundle: .module)) { Text("Search...", bundle: .module) }` |
  | Tooltip | `.help(Text("Tip", bundle: .module))` |
  | VoiceOver name or hint | `.accessibilityLabel(Text("Name", bundle: .module))` |
  | Dialog title | `.confirmationDialog(Text("Delete Skill", bundle: .module), isPresented: ...)` |

- **Fixed English held in a variable** (an enum's `title` or `label`) renders
  through `Text(LocalizedStringKey(value), bundle: .module)`, and every value
  it can take is a catalog key. No source scan can see these, so add the
  enum to `testEnumBackedDisplayStringsAreCatalogKeys` in
  `LocalizationParityTests` in the same change.
- **A `String` that reaches the UI without a `Text`** (toast, alert, or panel
  title, a status string stored in a model, a seeded chat draft) uses
  `AppStrings.localized("Key")`. `String(localized:bundle:)` follows the
  system language and ignores the in-app language picker.
- **Interpolation becomes a format key.** A String interpolates as `%@`, an
  Int as `%lld`. Never interpolate a Double, CGFloat, or Date into a localized
  key: format it to a String first. Translations keep the same specifiers
  (positional `%1$@` when word order changes).
- **Nothing to translate means verbatim.** Glyphs, units, numbers, key
  names, code, file names, model ids, and URLs use `Text(verbatim:)`. Language
  names stay in their own script.
- **Ternaries hide from the gates.** Write `cond ? Text("A", bundle: .module)
  : Text("B", bundle: .module)`, never a ternary inside one `Text`.
- **Components take `Text`, not `String` or a key**, so the bundled literal
  stays at the call site where the gates can see it.
- **Comments are scanned too.** Do not quote a literal call shape like a
  `Button` with a string argument in a comment. Paraphrase it.
- **Settings search is generated from source.** Keep
  `.settingsControl("Title", pane: ...)` and `preferenceRow(title: "...")`
  with a string literal first argument, and run
  `scripts/generate-settings-catalog.py` after changing them.
- **Right-to-left works by construction.** Use leading and trailing, never
  left and right, and let `AppLanguage` set the layout direction.
- **Do not localize** prompts sent to a model, persisted values that are
  compared or decoded later, identifiers, or log lines.

## 7. Menus, commands, and shortcuts

- Menu labels use `Text(_, bundle: .module)` like every other label.
- App navigation joins the SYSTEM View menu (`CommandGroup`), generated from
  `AppNavigationSection` like the sidebar and the Keyboard Shortcuts pane.
  Never declare a second menu named after a system menu.
- Display preference pickers (Mode, Text Size, Language) use the shared
  option views everywhere they appear: the Appearance menu, the sidebar
  profile menu, and General and Appearance settings.
- A command acts on the section whose UI it changes. If a workspace binds a
  chord to its own button, the menu item for that chord is disabled there.
  An enabled menu item takes the chord from a disabled in-window button and
  acts on something the user cannot see. The Generation menu sent hidden
  chat drafts from Audio and Humanize this way.
- Name the target when a menu could mean several things (Load Chat Model,
  Unload Image Model). A label that reads true only in context is wrong in a
  menu.
- A menu item carries one chord. Alternate chords are hidden buttons in
  `AlternateShortcutBridge` that mirror the menu item's disabled state, and
  every chord is listed in `KeyboardShortcutCatalog`, which
  `KeyboardShortcutCatalogTests` holds against the app.

## 8. Platform constraints

- The floor is macOS 14 and CI compiles with that SDK. `#available` gates
  execution, not symbol resolution: a newer API inside an availability check
  still fails to compile there. Use an AppKit equivalent.
- On the macOS 14 SDK only `body` is main-actor isolated. A view type whose
  helpers call each other needs an explicit `@MainActor` on the TYPE.
- Keep expressions small for the older type checker: split large `Form`s and
  view builders into computed properties.

## 9. Verification

Before calling a UI change done:

```sh
# from the repository root
make compile-strings
(cd swift/TurboSparkApp && swift build -Xbuild-tools-swiftc -suppress-warnings)
(cd swift/TurboSparkApp && swift test --filter 'LocalizationParityTests|LocalizationTests|LocalizationAndAccessibilityTests|FontPropagationTests|KeyboardShortcutCatalogTests|UIAuditFixesTests|SettingsWindowLayoutTests')
python3 scripts/generate-settings-catalog.py --check
make swift-test-app
```

Then look at the change in the running app. A green build proves none of the
following, and a UI change is not verified until the touched screens have
been checked against them:

- Default and Extra Large text size.
- The narrowest window the view allows.
- Light, dark, and one custom theme.
- Keyboard only, including sheets and menus.
- VoiceOver: every control announces a name, a value where it has one, and
  its role. Headings navigate.
- Reduce Motion, Reduce Transparency, and Increase Contrast.
- English, a long translation (German), and a right-to-left language (Arabic
  or Hebrew).

Report which of these you checked. Do not claim a visual or accessibility
result from compilation alone.

## Related docs

- `swift/AGENTS.md`: Swift package, build order, and app rules.
- `swift/docs/SWIFT_LOCALIZATION.md`: the catalog pipeline and its gates.
- `swift/docs/KEYBOARD_SHORTCUTS.md`: shortcuts, menus, and VoiceOver
  contracts.
- `swift/docs/SWIFT_SETTINGS_SCOPE_AND_THEMES.md`: settings scope, themes and
  the settings search catalog.
- `.claude/docs/modules/swift.md`: the numbered Swift gotchas cited above.
