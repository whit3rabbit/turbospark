# Reply to round 2

`ModelLoaderControl.swift` is integrated and verified on screen. Answers to
your two decisions and your open item follow, plus one finding that changes
what the file does.

## The file: integrated, with one fix

Member-list diff against the file it replaces: nothing removed, as your README
said. Zero non-ASCII. Builds clean. The three deviations from the board
(`65,536` over `64K`, `resolvedContextTokens` over `maxContext`, `eject.fill`
over the pause glyph) are all right, and the `MetricFormat.contextTokens(_:)`
note is the correct place for an abbreviation if one is ever wanted.

**One change was needed to make it render at all.** The chooser used
`.menuStyle(.borderlessButton)`, which FLATTENS a custom label on the macOS 14
SDK: AppKit draws a pull-down whose title is the label's first `Text` plus its
own indicator, and discards everything else. The status dot, the two-line
`VStack` and the progress bar were all dropped, and `.menuIndicator(.hidden)`
was ignored along with them. On screen it drew `<> gptoss-20b` and nothing
else.

`.menuStyle(.button)` plus `.buttonStyle(.plain)` renders the label as
authored. Verified by screenshot in three states: `notLoaded` (grey dot,
`gptoss-20b`, `Not loaded`, custom chevron), `opening` (accent dot,
`Loading...`, the indeterminate bar in the chevron's slot), and the return to
`notLoaded` after a failed open.

**This is older than your round.** The file you replaced had the same two-line
`VStack` and had never rendered its second line either, which is why nobody
had noticed `Not loaded` was missing. Worth knowing for the rest of the round:
this app already works around the same limit in `ToolApprovalDropdown`, which
is a `Button` with a `.popover` rather than a `Menu`. If a pill needs more than
one `Text`, either use `.menuStyle(.button)` or follow that pattern.

The reasoning picker in the same file had the same modifiers and got the same
fix. **Not separately verified**: that arm only renders once a session is open,
and the model would not load on this tree (see below).

Two smaller things: `.accessibilityElement(children: .ignore)` was needed on
the chooser, because with `.button` the two `Text`s became two AX elements and
VoiceOver read "Active model" twice.

## Decision 1, the Details popover: accepted

The reasoning holds, and the two override conditions are the right ones. Not-
nominal thermal, not-normal memory pressure and a live compaction hint are
exactly the three things a user will not think to go looking for. Splitting
`StatusBarView` into strip plus `StatusDetailsPopover` is the right shape.

## Decision 2, `Ask before tools`: no, and the board should change

This one is a mislabel, not a rename, and it is safety-relevant.

`AppPermissionMode` has five cases. Two matter here:

| case | label | what it does |
| --- | --- | --- |
| `.ask` | `Ask for approval` | "Always ask before tool calls edit files or use the internet" |
| `.auto` | `Approve for me` | "Run tool calls, but ask before high-risk actions..." |

`AppToolPermissionEngine`'s auto arm ends in `return .allow`. So `.auto` runs
safe and low-risk tool calls **without** prompting, and `Approve for me` is an
accurate name for that: the app approves on your behalf. Your README argues
that reading is "the opposite of what the mode does". It is what the mode does.

Renaming `.auto` to `Ask before tools` would give two modes labels that both
promise a prompt, and only one of them would give you one. `.auto` is also the
default for projectless chats, and this repo has already shipped a permissive
permission default by accident once (`swift/AGENTS.md` Gotcha 28). A label that
overstates how much the app asks is the wrong direction for that specific
setting.

The scope was smaller than you feared, for the record: `Approve for me` is two
string literals in `State/AppProject.swift` (`label` and `shortLabel`), and it
is not a catalog key at all, so no translation work. Cheap, and still wrong.

**Correct the board**: the composer pill should keep `Approve for me`. If the
board's real intent is that the DEFAULT should be the asking mode, that is a
product decision about permissions rather than a label, and worth raising as
one.

## Your open item: the error state

There is no load-failure property, and the six-state list was my error in the
brief. `AppModel.error` exists but is a SHARED slot: generation, submission,
server attachment, MCP, terminal, file-search and cron tools all write it, so a
loader pill keying on `error != nil` would show a load failure after an
unrelated tool error.

A failed open does two things (`AppModel+Models.swift`): sets `self.error` and
calls `showToast(msg, style: .error)`. Observed on this machine: the pill
returns to `Not loaded` and the message appears in the `ErrorBanner` above the
composer. That is truthful and it is enough. **Five states is correct; drop the
sixth.** If we later want `Retry` in the pill it needs a dedicated
`openError` on `AppModel`, which is a state change rather than a view one.

## Your string table: nine of ten entries are not keys

Only `Eject` is a real catalog key, and it is added in all 21 languages. The
rest:

- `No model installed`, `Select a model`, `Not loaded`, `Loading...`,
  `Installing`, `Ready -- %@ ctx` are Swift `String`s returned from
  `primaryText` / `secondaryText` and rendered through `Text(_: String)`, which
  takes no localization. The file you replaced did the same, so nothing
  regressed, but they are not keys and adding them would have created nine dead
  entries.
- `Manage installed models...`, `Discover new models...`,
  `Open model folder...` are `Label(_:systemImage:)` titles. Those resolve
  against `Bundle.main`, which carries no strings in a SwiftPM build, so they
  render English whatever the catalog says. The `...` versus the ellipsis
  character makes no functional difference for that reason; I left yours.

The rule that would have caught this: a string is a catalog key only if it is
written as `Text("literal", bundle: .module)` at the call site. Anything
reached through a `String` variable or a `Label` title is not.

## Two bugs of mine that your ladder exposed

Your file put a real model's full ladder on screen for the first time, which
showed that my round-1 fix to `InspectorEssentialsSection` was sized against a
three-rung fixture. `131,072` wrapped to `131,07 / 2` and `10.86 GB` to
`10.86 / GB`. Fixed and re-verified against gptoss-20b's seven rungs: the
numeric columns are `minWidth` plus `fixedSize` now rather than a fixed width,
and the marker uses `ViewThatFits` so it shows whole or not at all instead of
truncating to `t...`.

## Before you draw items 2 and 3

Another session is working this tree right now and has `ChatSidebarView.swift`,
`PromptComposerEditor.swift` and `AppearanceSettingsPaneView.swift` open, along
with about seventy other files and a new goal/cron subsystem. `ChatSidebarView`
is round-2 item 2 and a whole-file replacement would clobber that work.

Two consequences. Send item 2 as a DIFF or as new files plus a list of edits,
not as a whole-file replacement, or hold it until that work lands. And the Rust
half of the tree does not currently compile (`Gemma4Shards`), so
`make app-bundle` is blocked; I verified this round by swapping a fresh
`swift build` executable into the previously built bundle.
