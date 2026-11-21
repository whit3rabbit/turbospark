# Chat components audit, group B (GradientDeck, React/shadcn vs TurboSpark SwiftUI)

Paths below are relative to `swift/TurboSparkApp/Sources/TurboSparkApp` (Sources/). Evidence is from reading the component descriptions/FAQ/source and skimming the named Swift files (heads of files, `st` hits). Nothing was built or run.

## Verdict table

| slug | verdict | value/effort | one line |
|---|---|---|---|
| model-picker | IMPROVE | M/M | Loader menu is a flat installed list; add search, grouping by local/remote, per-row context/RAM facts, reasoning-effort control |
| panel-dock | SKIP | - | Right-column single-slot claimant model already exists and is richer |
| point-and-ask | IMPROVE | M/M | Image annotation layer (box/arrow/text) exists; add a "pin a question" tool that attaches a region to the composer |
| prompt-input | SKIP | - | Composer already has plus menu, autocomplete, pills, attachments; the web variants are visual-only |
| reasoning | IMPROVE | M/S | Reasoning is a plain monospace box, no "Thought for Ns", no fade-preview rail; add duration and 3-line preview |
| report-output | SKIP | - | Chat export (md/docx/pdf) exists; sectioned report layout is a SaaS deep-research shape |
| scroll-rail | ADOPT | M/M | No turn navigation in long chats; a tick rail with hover preview is new and cheap |
| session-tabs | SKIP | - | SessionTabStripView already does it (recently closed, All Tabs menu) |
| slash-command-menu | IMPROVE | L/S | Inline `/` autocomplete exists; only missing "command becomes a chip" in the composer |
| source-viewer | IMPROVE | L/M | Sources chip + popover exist; no quote highlight (needs citation anchors the app does not produce) |
| sources-list | SKIP | - | SourcesChipView/SourcesPopoverView (grouped by domain, copy, open in pane) cover it |
| streaming-text | IMPROVE | L/S | Markdown streams already; only worth taking: hold back dangling markers, Stop/Continue affordance check |
| suggestion-chips | IMPROVE | L/S | Empty-state PromptExamplesView exists; follow-up chips under a reply are new but need model-generated text |
| text-shimmer | IMPROVE | L/S | Status line is static text plus flame icon; a Reduce-Motion-aware shimmer on "Thinking..." is a 30-line modifier |
| thread-header | IMPROVE | L/S | No in-conversation title bar; only add if chat rename and model/context facts are not already in the top bar |
| thread-list | SKIP | - | ChatSidebarView has pinned, date buckets, archive, search, projects; palette search matches message text |

---

## model-picker
- Component: 4 variants. Effort slider plus model list, searchable list grouped by provider with a separate effort menu, detailed dropdown, three-way segmented switch. Cmd+1..9 row shortcuts.
- Existing: `Chrome/ModelLoaderControl.swift` (pill with status dot, alias, "Ready -- 65,536 ctx", separate load/eject targets; `Menu` with `installedModelsSection`, Manage, Discover, Open folder, Reload). Also `Generation/RemoteModelPill.swift`, `Components/EmbeddingModelPicker.swift`.
- Gap: the menu is a native `Menu`, so no search field, no grouping by family or local vs remote, no per-row size/quant/RAM-fit/ctx facts. No reasoning-effort control in the picker (thinking settings live in Engine settings and `Diagnostics/GenerationSamplingSection.swift`; I did not find a composer-level effort control, treat as unverified).
- Verdict: IMPROVE. Value M, effort M.
- Approach: keep the existing pill and its load/eject split (it is deliberately built for the warm-session cost). Replace the chooser `Menu` with a `Button` + `.popover` containing a `TextField` filter (use the existing search field style from `ChatSidebarView`), `List` sections (Loaded, Installed by family, Remote), and a trailing facts line per row (size on disk, quant, fits-in-memory badge via StatusDot). Add a small `AppPillPicker` (Theme/AppPillPicker.swift) for thinking effort only when the selected family exposes it; data from the existing capability predicates, do not hardcode. Do not copy the 3-way "fast/balanced/smart" segmented variant: it implies a model swap the app cannot do instantly.
- Maps to: Menu/Popover, AppPillPicker, StatusDot, Chip. Note the file doc comment forbids abbreviating ctx ("65,536"), keep that.

## panel-dock
- Component: resizable side/bottom dock with tabs (preview, sources, files, terminal, steps) that collapses.
- Existing: `Theme/AppRightColumnClaimant.swift` (one-slot right column: artifact, html/file preview, plugin page, browser, whiteboard, inspector, project summary) plus `Generation/ArtifactPanelView.swift`, `Generation/OutputPaneView.swift`, `Browser/BrowserPaneView.swift`, `Chrome/WorkspaceTabStripView.swift`.
- Verdict: SKIP. The app has a richer, arbitrated right column; a generic dock would add a second layout system. The only idea worth noting (bottom dock for terminal/steps) is covered by `BackgroundShellsStripView` and `TrajectoryPaneView`.

## point-and-ask
- Component: click a spot on an image or page, pin a question to it, model answers about that place. Pin stores percent coordinates; page variant supports threaded replies.
- Existing: `Files/ImageAnnotation.swift`, `Files/AnnotationLayerView.swift` (box, arrow, text tools, pixel-space model, exporter) in `ImagePreviewView`. `Browser/ElementPickerController.swift` (pick a DOM element, attach to composer as `AppPromptAttachment`). So the "page" variant is effectively built.
- Gap: image annotations do not become a question to the model. No "pin and ask" tool. Needs a vision-capable model to be useful.
- Verdict: IMPROVE (image side only). Value M, effort M.
- Approach: add an `AnnotationTool.ask` (a numbered pin + inline `TextField` popover) to `AnnotationLayerView`. On send, use `AnnotationExporter` to flatten the pinned image and attach it plus a text block "Pin 1 at (x%, y%): question" via the same attachment sink the element picker uses (`AttachmentSink`). Gate the tool on the loaded model's vision capability so it is not offered for text-only models. Reuse existing stroke style, do not add percent coordinates (pixel space is the repo rule).
- Maps to: Button (tool palette), Popover, Composer attachment.

## prompt-input
- Component: auto-growing input, Enter sends, stop button while streaming; 6 layouts. FAQ admits paperclip/model/search buttons are visual only.
- Existing: `Generation/PromptComposerView.swift`, `PromptComposerEditor.swift`, `PromptComposerFooter.swift`, `PromptComposerPlusMenu.swift`, `PromptAttachmentsView.swift`, `PromptContextPills.swift`, `PromptSafetyPills.swift`, `ComposerAutocompleteController.swift`.
- Verdict: SKIP. Everything functional is already there and wired. The decorative variants (hero, floating dock) fight DESIGN.md's Composer rules. The IME check (`isComposing`) is already a native concern of `NSTextView`; verify once in `PromptComposerEditor` but no new work.

## reasoning
- Component: Collapsible ("Thinking..." shimmer, auto-folds 800 ms after stream ends into "Thought for 12s", capped scroll body with left rule) and Rail (2px animated rail, folded preview showing the newest ~72px of text with a fade mask, "Reasoned for 9s").
- Existing: `Generation/ChatActivityLine.swift` `ReasoningTraceView` (plain `Text`, code font, elevated rounded box, no cap, no duration), toggled from the activity line in `ActiveStreamingRowView` (starts closed, previews latest line) and `MessageRowView` (history). `AppChatMessage.reasoning` is a String; no duration is stored (search for "Thought for"/duration found nothing).
- Gap: (1) no elapsed-thinking duration shown after the fact, (2) expanded body is unbounded height, (3) no fold-with-fade preview, (4) the box uses `.appElevated` card styling, heavier than the rail look.
- Verdict: IMPROVE. Value M, effort S.
- Approach: store `reasoningSeconds: Double?` on `AppChatMessage` (additive, decode with `decodeIfPresent`, matching the file's tolerant-decode pattern), set from the existing `startedAt` in `ActiveStreamingRowView` when the first answer token arrives. Show "Thought for Ns" in the activity line label. Restyle `ReasoningTraceView` to the Rail variant: 2pt `Rectangle` in `theme.border`, `Text` in `metadataForeground`, `.frame(maxHeight: 72)` folded with a `.mask(LinearGradient)` fade and bottom-anchored text, full height on tap. Animate with `TurboSparkMotion` and honor the app's two-source Reduce Motion check. Skip the moving gradient rail while streaming (decorative, costs a TimelineView).
- Maps to: Disclosure/activity line, theme.border, metadataForeground. No new control.

## report-output
- Component: long answer as report: sections + contents list, figures, numbered sources, short brief that expands, export PDF/Word/Markdown.
- Existing: `State/AppChatExport.swift` (markdown, json, docx formats), `Presentation/DocumentWriter`, `Generation/ChatShareButton.swift`, `State/AppChatShareDocument.swift`, `Generation/ArtifactPanelView.swift` for rendered outputs.
- Verdict: SKIP. A sectioned "report" is a product feature of a deep-research agent that the app does not have; export already exists. If research-style output is added later, render it through `ChatMessageMarkdownView` with heading anchors rather than a bespoke report view.

## scroll-rail
- Component: column of ticks, one per prompt; hover swells ticks like a dock, shows a preview card (prompt, reply title, excerpt), click jumps, optional bookmark, keyboard arrows/Home/End.
- Existing: nothing equivalent. `Generation/ChatTranscriptView.swift` has follow-the-stream logic (`TranscriptFollow.swift`) and a `JumpToLatestButton`, nothing for mid-conversation navigation. No bookmark concept for messages (the `bookmark` hits are audio and browser).
- Verdict: ADOPT. Value M (long agent sessions with many tool turns are hard to navigate), effort M.
- Approach: new `Generation/TranscriptScrollRail.swift` overlaid on the leading or trailing edge of `ChatTranscriptView` (outside the 760pt reading column, hide below a window width threshold). Items = user messages (id, first line of prompt, first heading or sentence of the following reply). Ticks: `VStack` of `Capsule` 2pt tall in `theme.border`, active one in `.appAccentText`; hover preview via `.popover` or a small overlay using `onContinuousHover` for the pointer-to-slot mapping. Active tick from the transcript's visible anchor (`ScrollViewReader` is already used; add `.onScrollTargetVisibilityChange` or `scrollPosition(id:)` if the deployment target allows, else keep geometry preferences). Click calls `proxy.scrollTo(id, anchor: .top)` and must unpin follow-mode via the existing `TranscriptFollowRules`. Cap to the last N or collapse to buckets over ~60 turns. Accessibility: each tick a `Button` with the prompt as label, `.accessibilityElement(children: .contain)`, arrow-key navigation; localize every string. Skip bookmarks in v1 (needs persistence in `AppChat`).
- Maps to: new small component; uses StatusDot-like tokens, Card for the preview. Must pass `make design-check` (no raw colors).

## session-tabs
- Existing: `Generation/SessionTabStripView.swift` (OC-04): tab per open chat, draft glyph, plus button, All Tabs menu, Recently Closed stack, close dimmed not hidden, Cmd+Shift+W, running state ("Generating" label, line ~181). State in `State/SessionTabStore.swift`, `AppModel+SessionTabs.swift`.
- Verdict: SKIP. Already a superset (reopen closed, keyboard/VoiceOver menu). The "side tabs" variant duplicates `ChatSidebarView`.

## slash-command-menu
- Component: typing `/` opens a filterable list; picking inserts a removable chip before the text; Backspace on empty removes chip; chip alone can be sent.
- Existing: `Generation/ComposerAutocompleteController.swift`, `ComposerAutocompleteEngine.swift`, `ComposerSuggestionPopup.swift` (inline list, keyboard selection, scroll-to-selected, hint row), `SlashCommandMenuGroup.swift`, `State/AppModel+SlashParity.swift`, `Generation/SteeringComposerMenu.swift`.
- Gap: the app inserts text, the component converts a command to a structured chip. That only matters if commands carry state beyond text; for most slash commands inserting text is simpler and works with history/edit.
- Verdict: IMPROVE, low priority. Value L, effort S.
- Approach: only for commands that map to a mode (e.g. plan mode, steering preset, agent). Render a chip with `AppChip` (Components in Theme/AppControls) in `PromptContextPills` instead of leaving `/plan` text in the editor. Do not copy the Enter-hint `kbd` badge unless DESIGN.md's shortcut hint pattern already has one.
- Maps to: Chip, Composer.

## source-viewer
- Component: open a cited source beside the answer with the quoted passage highlighted; previous/next quote stepping; document and web variants.
- Existing: `Generation/SourcesPopoverView.swift` (domain-grouped, opens page in the browser pane, copy), `State/AppChatTurnSources.swift` (`TurnSource` url/title/domain, `fetchedPage` vs `searchResult`, origin message and call id for "reveal in tool card"), `Files/FilePreviewView.swift`, `Browser/BrowserPaneView.swift`.
- Gap: no quote-level highlight. That needs the model to emit citation spans mapped to source text, which the app's turn-sources model does not record (only URL-level provenance).
- Verdict: IMPROVE, but defer. Value L, effort M-L. A cheap partial: when the source is a local file the agent read (Read tool call), open `FilePreviewView` scrolled to the line range the tool returned. That uses data already in tool-call records and no citation protocol. Skip the web-page highlighter (WKWebView text-fragment highlight is possible but not worth it for on-device models that rarely cite precisely).

## sources-list
- Component: six variants (favicon stack, cards, carousel, footnotes, side panel, research log).
- Existing: `SourcesChipView` + `SourcesPopoverView` (OC-09), `AppChatTurnSources`, `AppModel+TurnSources`.
- Verdict: SKIP. The popover grouped by domain with Cited/kind distinction already covers the function. Favicons would need network fetches, which conflicts with the local-first posture. The "research log" (read vs skipped) maps to tool cards (`ToolGroupView`, `BrowserToolCardView`) already.

## streaming-text
- Component: word fade-in, markdown that forms cleanly mid-stream (hold back lone `#`, `-`, backticks, a trailing `*`), soft-focus blur edge, caret with Stop/Continue/Skip and speed, table that fills row by row.
- Existing: `Generation/ActiveStreamingRowView.swift` renders `ChatMessageMarkdownView(output)` live; `Presentation/ResponseMarkdownRenderer.swift`, `MarkdownTableGrid.swift`, `Generation/StreamingStatusFooterView.swift` (elapsed, tokens), `TranscriptFollow.swift`, `State/AppModel+Cancellation.swift`, `StallRecoveryBannerView`.
- Gap/useful bits: (1) dangling-marker holdback to avoid flicker (a heading flashing as `#`): check whether `ResponseMarkdownRenderer` already handles a tail line; I did not verify. (2) Open code fence shows chrome early with Copy disabled. (3) Word fade-in and blur are decoration and re-layout every token on a large SwiftUI `Text` tree; skip. (4) "Speed" and "Skip to end" are fake-typewriter controls, meaningless when tokens arrive at real GPU speed; skip. Stop already exists.
- Verdict: IMPROVE (parser hygiene only). Value L, effort S. Add a pure function `streamingTail(_ s: String) -> String` that trims a final line matching `^(`{1,2}|#{1,3}|-|\*)$` and test it; unit-testable per TESTING rules. No animation work.

## suggestion-chips
- Component: chips, starter cards, follow-up list.
- Existing: `Generation/PromptExamplesView.swift` (empty-state card grid of `AppPromptPreset`, "more examples"), `Generation/NewChatSuggestionBanner.swift`, `Generation/LoadingPhrases.swift` is unrelated.
- Gap: no follow-up suggestions after a reply. Generating them costs an extra on-device pass (or prompt tail) per turn, which costs tokens, KV and latency on the same GPU the user is waiting on.
- Verdict: IMPROVE (empty state is done; follow-ups are optional). Value L, effort M. If wanted, make it opt-in and generate only when idle, from a tiny templated set (e.g. "Explain", "Shorter", "Show the code") that needs no model call: render as `AppChip` row under the last reply in `MessageActionBarView`. Do not auto-generate with the loaded LLM by default.
- Maps to: Chip, EmptyState (already).

## text-shimmer
- Component: gradient sweep over status text, letter wave, spectral variant.
- Existing: only a `Humanize/HumanizeChrome.swift` shimmer placeholder (line ~331). Chat status uses `TaskProgressFlameIcon` and static text in `ChatActivityLine` / `StreamingStatusFooterView`; `LoadingPhrases` rotates witty text every 15 s.
- Verdict: IMPROVE. Value L, effort S.
- Approach: extract the Humanize shimmer into `Theme/AppShimmer.swift` as `.appShimmer(active:)` (mask with a moving `LinearGradient` driven by `TimelineView(.animation)` only while active). Apply to the "Thinking..." label. Must read the app's Reduce Motion two-source check (`TurboSparkMotion.swift`) and render static when reduced; never use the spectral gradient (design-system tokens only; no raw hex). Make sure it pauses when the window is not key or the chat is not visible to avoid wasting power on a machine that is already running inference.
- Maps to: motion tokens in Theme; no component change.

## thread-header
- Component: title + model + share/options menu; project breadcrumb; agent task status bar with progress and Stop/Resume; scroll-aware collapsing header.
- Existing: `Chrome/TopBarView.swift` (sidebar controls, right-column toggles), `Generation/ChatShareButton.swift`, `Generation/GoalBannerView.swift`, `Generation/ProjectChatSummaryView.swift`, `Generation/ContextUsageRingView.swift`, `Chrome/StatusBarView.swift`, `Chrome/WorkspaceTabStripView.swift`, tab strip in `SessionTabStripView`. A `.title3` in `ConversationPaneView` line ~62 suggests a title is drawn there (not read).
- Verdict: IMPROVE (small). Value L, effort S. The agent-task status bar is already covered by `GoalBannerView` and `BackgroundAgentsStripView`. Only worth borrowing the scroll-aware collapse if the conversation title row costs vertical space on a 13-inch window; otherwise leave. Do not add a breadcrumb; the project is already in the sidebar.

## thread-list
- Component: dated sidebar, projects, search palette matching inside messages, bulk-manage page, hover rail.
- Existing: `Generation/ChatSidebarView.swift` (search field, pinned, time buckets, archived section), `ChatSidebarChatRowView.swift` (pin/unpin), `ChatSidebarProjectsSectionView.swift`, `ChatSidebarGroupedProjectsView.swift`, `ChatSidebarInboxRows.swift`, `ChatSearch.swift` + `ChatSearchResultRowView.swift` + `CommandPaletteView.swift` (snippets from message content, match counts).
- Verdict: SKIP. Equivalent or better. Only unmet idea is a bulk "All chats" management page; it is a Settings-sized feature, not a chat UI one, and not requested by anything in the app's own docs I read.

---
## Cross-cutting notes
- Take: scroll-rail (ADOPT), reasoning duration + rail fold, shimmer modifier, model-picker search/facts. These four give visible quality gains without new data flows beyond one optional `reasoningSeconds` field.
- Reject on purpose: fake-typewriter speed controls, word blur/fade animations, spectral rainbow gradients, favicon fetching, auto-generated follow-ups. They either cost GPU/CPU while inference is running, require the network, or break the theme-token rule.
- Any UI change here needs `make design-check`, localized keys in the string catalog (ten parity gates), and the accessibility name for every new control per DESIGN.md.
- Not verified: composer-level reasoning-effort control existence; whether `ResponseMarkdownRenderer` already holds back partial markers; conversation title row implementation.
