# Audit: output & artifacts (gradientdeck AI components vs TurboSpark SwiftUI app)

Paths below are relative to swift/TurboSparkApp/Sources/TurboSparkApp. I read the component descriptions and variant source, and the Swift files cited. Design-system docs were only skimmed (README plus the components folder list); I did not open DESIGN.md or per-control guides, so mappings use names from docs/design-system/components (Card, Chip, IconButton, ProgressBar, StatusDot, StatusLabel, SectionHeader, EmptyState, PillPicker, EditorSurface).

## Verdict table

| Slug | Verdict | Value | Effort | One line |
|---|---|---|---|---|
| artifact-card | IMPROVE | M | S | Existing chip in MessageRowView lacks in-flight state and kind/revision subtitle |
| artifact-panel | IMPROVE | M | L | Panel exists; revision is only a counter, no version switcher or history |
| calendar-event | SKIP | - | - | Needs calendar write integration the local app does not have |
| chart | IMPROVE | M | M | Chart fences go through a CDN-loading web page; native Swift Charts renderer would be offline and themed |
| code-block | IMPROVE | L | S | CodeBlockContainer already has language, copy, highlight; only marked lines and file name are missing |
| data-table | SKIP | - | - | InteractiveMarkdownTableView already has sort, copy, CSV export |
| diff | IMPROVE | M | M | WorktreeDiffView and edit-call diffs exist; per-hunk accept/reject is the gap |
| email-draft | SKIP | - | - | Needs mail send integration; out of scope |
| file-tree | ADOPT | M | S | No tree of files touched in a turn; build from tool calls and worktree status |
| itinerary | SKIP | - | - | Travel planning, web/cloud domain |
| product-card | SKIP | - | - | Shopping recommendation UI; GenUi table/bars cover generic cases |
| test-results | ADOPT | M | M | No structured view for cargo/swift test output; parse tool output into a summary card |

## artifact-card
- Existing: `Generation/MessageRowView.swift` `artifactChips` (one Button chip per artifact: symbol, fileName, formatLabel, sidebar glyph; opens panel via `model.openArtifact`). Also `Generation/ProjectChatSummaryView.swift:126`. Model: `State/AppArtifact.swift` (`revision`, `formatLabel`, `symbolName`, `toolCallID`).
- Gap: no state while the tool call is still writing (card disables Open until progress is 1 and says "being written"), no revision badge in the chip (it exists on the model: `revision > 1`), no "open" pressed state when the panel shows this artifact, no hover peek.
- Verdict: IMPROVE. Value M, effort S.
- Approach: in the same ViewBuilder, derive state from the pending tool call (call present, no result yet) and show a `ProgressView().controlSize(.small)` plus "Writing" text instead of the sidebar glyph; disable the Button. Add `v\(revision)` as a small tabular `themedCode(.tiny)` badge (reuse `AppArtifact.contentKey`/revision). Bind pressed state to `model` panel selection (`.artifact(id)` case in ArtifactPanelView) with accent border via `.appAccent`. Skip the fake block thumbnail: the component itself admits it is a decorative sketch; a real QuickLook/web thumbnail would be a separate, costly feature. Maps to Card/Chip, StatusLabel. Localize any new strings (ten parity gates).

## artifact-panel
- Existing: `Generation/ArtifactPanelView.swift` (493 lines): Picker view mode (preview/source), copy source, save a copy, reveal in Finder, larger preview, close; `State/AppModel+ArtifactPanel.swift`; sandboxed `ArtifactWebView`.
- Gap: version switching. `AppArtifact.revision` is bumped on update (`merged.revision += 1`) but the file is overwritten in place; I saw no snapshot store, so old revisions appear unrecoverable (not fully verified; check AppArtifact archive code before scoping). Also no live streaming of source while the file is being written, and no regenerate action.
- Verdict: IMPROVE, but only the version history. Value M, effort L (needs storage: keep prior revision bytes under the chat's artifact dir, cap count/size, cleanup on chat delete, profile encryption rules, persistence migration). Streaming source view is L and adds little because artifacts are written by a tool call, not token-streamed.
- Approach: `[AppArtifactRevision]` in AppArtifact (Codable, optional decode with default to stay compatible), a compact `Picker` or segmented control in the panel header next to view mode (PillPicker), "Restore this version" as a Button. Copy/download already exist; do not duplicate. Maps to PillPicker, IconButton, EditorSurface.
- Do not port: Regenerate (agent loop owns that), the pricing-specific preview.

## calendar-event
- Existing: none in chat. Calendar hits in the repo are unrelated (audio, cron). 
- Verdict: SKIP. Adding/proposing meetings implies EventKit or an online calendar account and attendee availability lookups; no local-inference story, permission prompt cost (DESIGN rule: privacy prompts only after user chooses Allow). If ever wanted, GenUi (`Plugins/UI/GenUiSpec.swift`) could render it from a plugin.

## chart
- Existing: three paths. (1) Chat fences `echarts`/`chartjs`/`chart` in `Presentation/ChatMessageMarkdownView.swift` `CodeBlockContainer.chartHTML`: opens a web preview that loads ECharts or Chart.js from cdn.jsdelivr.net and asks the user to grant network. (2) GenUi `bars` component (`Plugins/UI/GenUiSpec.swift`, `GenUiCardView.swift`). (3) Native Swift Charts only in `Diagnostics/UsageDashboardSection.swift` and `Server/ServerChartsView.swift`.
- Gap: for a local on-device app, a chart in chat that needs a CDN fetch and a network grant is the wrong default. No native line, stacked bar or donut from model data; no keyboard or VoiceOver access to values in the web path.
- Verdict: IMPROVE. Value M, effort M.
- Approach: add a small `ChatChartSpec` (type bar/line/donut, labels, series, optional unit) decoded from a new fence language (for example `chart-json`) or from the existing `chart` fence when it matches the simple schema; render with Swift Charts (`BarMark`, `LineMark`, `SectorMark`, macOS 14+; confirm deployment target first) inside a Card. Keep the ECharts/Chart.js fallback for complex options. Colors from theme accessors only (never the component's hex gradient); use `chartForegroundStyleScale` with theme series colors plus an opt-in pattern or label for colorblind safety; `chartOverlay` hover rule plus `.accessibilityChartDescriptor` for VoiceOver (this is the equivalent of the component's keyboard arrow navigation). Extend GenUi with `line`/`donut` instead if plugin-authored cards are the main source. Maps to Card, SectionHeader. Copy the summary-string idea (aria-label with all values).

## code-block
- Existing: `Presentation/ChatMessageMarkdownView.swift` `CodeBlockContainer` (language tag, Copy with Copied state and accessibility label, Preview button for html/chart fences, native highlighter `Presentation/CodeSyntaxHighlighter.swift`).
- Gap: no file name header, no marked-line highlighting, no line numbers, no tabs. All cosmetic; the markdown fence has no standard syntax for filename or highlights.
- Verdict: IMPROVE, low value. Effort S.
- Approach: parse optional fence info string `lang title="path" {6-9}` in the markdown block renderer (ResponseMarkdownRenderer), show title in the header via `themedCode(.tiny)`, tint marked lines with `.appAccent.opacity(...)` on a per-line background; line numbers as an opt-in toggle only, since they break copy fidelity and local models rarely emit these hints. Skip tabs: multi-file output is better served by artifact chips. Do not copy the component's regex highlighter; the native one exists.

## data-table
- Existing: `Generation/InteractiveMarkdownTableView.swift` and `Presentation/MarkdownTableGrid.swift`: sortable headers with chevron state, row selection, copy as TSV/CSV/markdown, CSV export. Matches or exceeds the component (the component's third-click-clears sort is the only difference; the app cycles asc/desc, see lines 175-179).
- Verdict: SKIP. Optional micro-IMPROVE: add the third-click-clear-sort and an `aria-sort` equivalent (`accessibilityValue`), about 10 lines, not worth a card.

## diff
- Existing: `Generation/WorktreeDiffView.swift` (parsed hunks, collapsible context, line numbers, color-or-marker mode via `appearance.diffMarkers`), used in `Generation/WorktreeView.swift`; inline previews of edit, multiedit and apply_patch calls in `Generation/ToolCallCardView.swift` (patchPreview, ToolApprovalPreviewSummary).
- Gap: no per-hunk accept/reject. Hunks are display only.
- Verdict: IMPROVE. Value M, effort M to L. Valuable where it matters: the approval flow for `apply_patch`/`multiedit` (safety-relevant, user decides before write) and worktree review.
- Approach: add `Set<Int>` of rejected hunk ids to `WorktreeDiffView` state (parseDiff already yields hunk blocks), a per-hunk header with two IconButtons (Accept/Reject, labelled, not color only), and an "Apply N of M hunks" Button. Applying a subset needs a rebuilt patch: regenerate unified diff from kept hunks and run through the existing apply path (git apply --recached or the patch tool) with a dry-run check first; a hunk that fails to apply must report, not silently drop. For tool approvals, returning a partial edit changes tool semantics (the model asked for the whole patch): feed back "user kept hunks 1,3" in the tool result. Worktree-side only is the cheaper first slice. Maps to Button, IconButton, StatusLabel. Observe Safety invariants in DESIGN.md before touching approvals (security-sensitive: ask first).

## email-draft
- Existing: none (no mail composer). 
- Verdict: SKIP. Sending or even drafting through Mail needs account access; the chat already lets the model write the text and the user copy it. Not worth a bespoke card.

## file-tree
- Existing: `Files/FilesSectionView.swift` and `Files/ProjectFileIndex.swift` (project file browsing), `Generation/WorktreeView.swift` (git changes list), artifact chips per tool call. None summarizes "the files this turn touched" as a tree with added/modified/deleted marks. `Generation/ToolCallCardView.swift` shows each write/edit as a separate card.
- Verdict: ADOPT. Value M (agent mode: one glance at the blast radius of a turn), effort S.
- Approach: at the end of an assistant turn that had write/edit/patch/multiedit calls, a collapsed Card "N files changed" built from the turn's `message.toolCalls` (path argument and tool name map via `ToolPresentation.registry` to added/modified; deletes only if a shell `rm` is detectable, otherwise omit that state rather than guess). Render with SwiftUI `DisclosureGroup`/`OutlineGroup` over a small tree node struct, path-compressed single-child folders, tabular +/- counts when a diff is available. Row click opens the file via existing file preview or `WorktreeDiffView`. Status shown with symbol plus text (not color alone). Maps to Card, SectionHeader, StatusDot/StatusLabel, SidebarRow density. New strings go through the string catalog.

## itinerary
- Existing: none. Verdict: SKIP. Travel/route planning with a map is a web assistant feature, off-mission for local inference.

## product-card
- Existing: none; closest is GenUi (`Plugins/UI/GenUiSpec.swift` text, table, bars, follow_up) and the image gallery cards. Verdict: SKIP. Shopping recommendations need live prices and product data, which a local model does not have; comparison tables are already covered by markdown tables and GenUi `table`.
- Note: the comparison variant's "show only differences" toggle is a tidy idea if GenUi `table` ever gets a filter, not a reason to port this component.

## test-results
- Existing: none structured. Shell output goes through `Presentation/ToolOutputFormatter.swift` and `Presentation/ANSIColorizer.swift` as text. `project_diagnostics` and Report Findings are different (LSP, review).
- Verdict: ADOPT. Value M for agent workflows (this app's agent runs cargo/swift test constantly), effort M.
- Approach: pure parsers in Presentation/ for `cargo test` (test result lines, failures section), `swift test`/xctest, `xcodebuild`, pytest, jest (start with the first two, add others later behind tests with real captured fixtures); produce `TestRunSummary { passed, failed, ignored, duration, suites[{name, status, failures[{name, message}]}] }` and return nil when unsure so the raw output still shows. Render as a Card above the raw output in `ToolCallCardView`: SectionHeader with counts, ProgressBar of pass ratio, `DisclosureGroup` per failed suite with the error in monospace; failed rows get an "Ask to fix" `follow_up` Button that composes a prompt (existing follow-up pattern in GenUi). Status via StatusDot plus word. Do not hide raw output; never claim pass when the parser saw no summary line. Tests: fixtures per runner; mutation-check the failure counting.

## Priority order if only some get built
1. test-results (ADOPT, M/M) and file-tree (ADOPT, M/S): new, purely presentational, derived from data already in the transcript.
2. artifact-card in-flight and revision badge (S) and code-block file title and marked lines (S): quick wins.
3. chart native renderer (M): removes the CDN and network-grant dependency for the common case.
4. diff hunk accept/reject (M to L, security-adjacent, ask first) and artifact version history (L, storage design).
