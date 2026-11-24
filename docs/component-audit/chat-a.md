# Chat components audit, group A (16 slugs)

Paths below are relative to swift/TurboSparkApp/Sources/TurboSparkApp. Component descriptions were read from the registry metadata only (variant source skimmed for the two small ones: assistant-launcher, branch-picker; the other variants' source was not read, so verdicts rest on the descriptions plus what the app already does).

## Verdict table

| Slug | Verdict | Value / Effort | One line |
| --- | --- | --- | --- |
| assistant-launcher | SKIP | - | Floating help widget for web pages; the app is the chat. |
| attachments-tray | IMPROVE | M / S | Chips exist; add token weight vs context and an over-budget warning. |
| branch-picker | SKIP | - | `MessageVariantSwitcherView` already does n / m (and branch/retry/edit exist). |
| chain-of-thought | IMPROVE | L / M | Reasoning trace + tool groups exist; only a step summary line is new. |
| context-meter | IMPROVE | M / S | Ring + stacked breakdown popover exist; add a "what to do" action row. |
| conversation | SKIP | - | Follow-bottom, split panes, agent transcript all exist (`TranscriptFollow`, `SplitChatPanes`). |
| edit-message | SKIP | - | `MessageEditingViews.swift` + `AppModel+MessageEditing.swift` cover it. |
| empty-state | IMPROVE | M / M | `PromptExamplesView` is a bare grid; add greeting + resume-work + project home. |
| image-prompt-composer | IMPROVE | M / L | Image composer has prompt/style/size/count; missing refs, negatives, rewrite. |
| inline-citation | ADOPT | M / M | Sources are a chip + popover per turn; no in-text markers. Needs model support. |
| knowledge-sources | IMPROVE | M / M | `@` references and project index exist; no per-question source scoping UI. |
| link-preview | SKIP | - | Needs network fetch of page metadata; off-brand for a local, offline-first app. |
| loader | SKIP | - | `ChatActivityLine` + `StreamingStatusFooterView` already cover it, better. |
| mention-menu | IMPROVE | M / L | `@` autocomplete exists as plain text tokens; no inline pills, no grouping. |
| message | SKIP | - | `MessageRowView` is richer (hover bar, collapse, sources, memory rows). |
| message-actions | IMPROVE | L / S | Hover bar exists; skip thumbs/share-feedback, only fix discoverability. |
| message-error | IMPROVE | H / M | Failures only surface in a global `ErrorBanner`; no in-transcript failed/stopped row with Retry. |

---

## assistant-launcher
- Existing: none needed. The app is a full chat window (`App/RootView.swift`, `Generation/ConversationPaneView.swift`); `Generation/GhostChatButton.swift` and `Generation/CommandPaletteView.swift` cover quick entry.
- Verdict: SKIP. A corner FAB that opens a help panel on a web page. A native macOS window has a sidebar, a palette and menu commands for this. A floating launcher would also fight DESIGN.md's "native first" rule.

## attachments-tray
- Existing: `Generation/PromptAttachmentsView.swift` (horizontal chips with icon, name, detail text, preview, remove, disabled while running), `Files/AttachmentImporter.swift` (per-file failure lines), `State/AppChat.swift` (`detailText`, `wasTruncatedDuringExtraction`, "truncated" suffix). `Files/AttachmentThumbnailStore.swift` for images.
- Gap: the registry component adds upload progress and retry (not applicable: local files, extraction is near instant), a drop zone that validates types (drop handling is in the importer, worth a quick check for a visible reject reason), reorder (low value), and "files weighed against the context limit" (the one that maps).
- Verdict: IMPROVE (small).
- Value M, effort S. Show estimated tokens per chip (chars/4 or the tokenizer count if cheap) and a combined "attachments use N% of context" line in the tray, turning `AppAttention` warning color when over the remaining window. Reuse the same fraction source as `ContextUsageRingView` (`estimatedContextTokens` / `resolvedContextTokens`) so surfaces agree. Data: add a computed `estimatedTokens` next to `detailText` on the attachment model. Maps to: Chip (`AppChip` styling), ProgressBar (`AppProgressBar` for the aggregate), StatusLabel for the over-budget text.

## branch-picker
- Existing: `Generation/MessageEditingViews.swift` `MessageVariantSwitcherView` ("< n / m >", older/newer chevrons, only when count > 1), wired into `MessageActionBarView.swift`; model in `State/AppModel+MessageEditing.swift` (retry, edit, branch, regenerate).
- Verdict: SKIP. Same pattern, already shipped. The only novelty is a 0.28 s opacity crossfade between versions; optional polish, gated on `accessibilityReduceMotion`, not worth a task.

## chain-of-thought
- Existing: `Generation/ActiveStreamingRowView.swift` (live reasoning, collapsed, one-line preview via `ChatActivityLine`), `ReasoningTraceView`, `MessageRowView.swift` (reasoning reveal on completed messages), `Generation/ToolGroupView.swift` and `ToolCallCardView.swift` (coalesced tool steps, auto-expand on approval), `TodoChecklistViews.swift` (status list).
- Gap: the component's value is structuring thinking as discrete steps (status list, "options weighed and ruled out", steps woven into the answer). Local models emit one raw reasoning blob; there is no step structure to render without a second model pass.
- Verdict: IMPROVE, low value. Only worthwhile bit: a collapsed summary header such as "Thought for 12 s, 3 tool calls" on the completed row (elapsed is already computed for the activity line). Value L, effort M with the existing views. Do not invent steps by parsing prose. Maps to: StatusLabel + collapsible disclosure already in `ChatActivityLine`.

## context-meter
- Existing: `Generation/ContextUsageRingView.swift` (circular ring, green/yellow at 70/red at 90, popover with a stacked capsule per component and "Free space", compaction and auto-compact-off states, remote variant with "Estimate" vs "Reported" provenance), plus `Chrome/StatusBarView.swift` meter and `Diagnostics/ContextSelectionPopover.swift`.
- Gap: this is already a superset of the web component. What is missing is action: at 70+ percent the popover only reports. 
- Verdict: IMPROVE. Value M, effort S. Add a row of quiet buttons to the popover: "Compact now" (`AppChatCompaction`), "Microcompact" (`AppChatMicrocompact`), and "Start new chat with summary" (`NewChatSuggestionBanner` already exists for this). Pure view change calling existing model methods. Maps to: Button (`.appAction(.quiet, size: .small)`), StatusDot semantics already used.

## conversation
- Existing: `Generation/ChatTranscriptView.swift` + `TranscriptFollow.swift` (follow stream unless the reader scrolled away, with tolerance), `SplitChatPanes.swift` (secondary chats side by side), branching in `AppModel+MessageEditing.swift`, `MidTurnTranscriptRowPresentation.swift`, `ToolGroupView`, `CompactionDividerView`.
- Verdict: SKIP. Every variant listed (stick-to-bottom, branches, side-by-side, agent transcript) exists. One possible gap, "two models answering side by side" for the same prompt (compare mode), is a feature rather than a view, and a local app with 36 GB cannot hold two large models resident; out of scope. A "side thread for a follow-up" is also new but a product decision, not a UI mimic.

## edit-message
- Existing: `MessageEditingViews.swift` (in-place edit composer replacing the prompt row, branch sheet), `MessageActionBarView.swift` edit/branch actions, `AppModel+MessageEditing.swift` (`editApplied`, `branchedChat`, `retryApplied` pure cores).
- Verdict: SKIP. Done, and it has pure tested cores. Edit is limited to the last real user prompt, with branch for earlier ones: a deliberate rule, do not change.

## empty-state
- Existing: `Generation/PromptExamplesView.swift` ("Try an example" grid of presets, 2-line cards, fills the composer), `Generation/GreetingProvider.swift` in Presentation, `AppEmptyState` in DESIGN.md for lists, `Generation/ProjectChatSummaryView.swift`, `Generation/NewChatSuggestionBanner.swift`.
- Gap: the web version offers greeting, project home (files + recent chats), return screen with unfinished work, agent landing page. The app has the greeting provider and an examples grid but no "continue where you left off" (unfinished goal, queued prompts, last chat) and no project home inside an empty chat.
- Verdict: IMPROVE. Value M, effort M. Compose an empty-chat view: greeting (existing provider) + model/readiness line (is a model loaded? offer Load) + up to 3 recents/unfinished goals (`AppModel+Goal.swift`, `ChatInboxModel.swift`) + project files count when in a project (`ProjectFileIndex`) + the existing examples grid. Use `LazyVGrid` of `Card`-style rows, `AppEmptyState` for the no-model case. Localize every string (ten parity gates). Maps to: EmptyState, Card, Chip.

## image-prompt-composer
- Existing: `Images/ImageComposerView.swift` (axis-vertical prompt field, settings popover for model/quant/size/count, `ImageStyleMenu` as an `appMenuAction`, Generate/Stop/Unload, unavailable-in-temporary-chat note), `Images/ImageStylePicker.swift`, `ImageStyleFavoriteStore.swift`, `ImageResolutionPreset.swift`.
- Gap: reference images with a role per image (img2img needs engine support; check `crates/image` first, none of the greps for reference/negative/enhance hit Swift), a "leave out" negative list (only useful if the model family supports negative prompts or CFG; Z-Image Turbo and other distilled models ignore it), and "rewrite my prompt in more detail".
- Verdict: IMPROVE, split in three:
  - Prompt enhance: value M, effort S-M. A local LLM is already in the app; add an "Enhance" quiet button that runs a one-shot rewrite through the loaded text model and shows a diff-like accept/undo (`Humanize/HumanizeDiff.swift` has a diff view to borrow). Caveat: the image model and text model compete for memory on 36 GB; load-gate via `AppModel+LoadGate.swift` and say so.
  - Reference images: value M, effort L, blocked on engine support. Do not build UI first.
  - Negative prompt: SKIP unless a supported family honors it (see `docs/` model gates), else it is a dead control.
- Maps to: Composer, Chip (`AppChip`), Button (`.quiet`).

## inline-citation
- Existing: `Generation/SourcesPopoverView.swift` (`SourcesChipView` count chip on the reply row, popover of deduplicated sources grouped by domain, copy action, open in browser pane), `State/AppChatTurnSources.swift` (page fetched beats search result), `State/AppModel+TurnSources.swift`. Per-turn only.
- Gap: no marker inside the prose tying a claim to a source. For local models this only works if the model is instructed to emit `[1]` style markers, which small models do unreliably and sometimes fabricate.
- Verdict: ADOPT, scoped narrowly. Value M, effort M. Phase 1: when a turn has `TurnSources`, render a trailing numbered chip row ("1 example.com 2 ...") under the reply and make `[n]` tokens in the text, if present and n is in range, render as small superscript links in `ResponseMarkdownRenderer.swift` (it already carries `link:` runs), hovering or clicking opens the existing popover scrolled to row n. Validate n against the known source count so a hallucinated marker degrades to plain text. Do not claim the marker proves the claim came from that page. Maps to: Chip (compact variant), StatusLabel for the number badge; popover stays a Card. Needs a system-prompt nudge for web-search turns only (`SystemReminders.swift`), measured on a real model before shipping.

## knowledge-sources
- Existing: `@file:`, `@folder:`, `@git:`, `@url:`, `@diff`, `@staged` references (`Files/ContextReferenceParser.swift`, `MentionResolver.swift`, `ProjectFileIndex.swift`), `Generation/PromptContextPills.swift` `PromptProjectContextPill`, project files in `Generation/ProjectSettingsSheet.swift`, conversation recall (`docs/SYNTEXT_CONTEXT.md`, `AppModel+ConversationRecall.swift`).
- Gap: the registry pattern shows what the project can draw on and lets the user narrow it for one question ("only these sources this turn"). The app has the pieces but no single view showing what is in scope for the next message (project files, memory, recall, pinned instructions, MCP servers) with per-turn toggles.
- Verdict: IMPROVE. Value M, effort M. Extend `PromptProjectContextPill` into a popover (list of scoped sources with `Toggle` rows, token cost per row from the same estimator as the attachments tray). Needs a per-turn override flag in `AppModel+Submission.swift`. Maps to: Chip, Card, StatusDot (loaded/indexed). Skip "connected apps" (cloud connectors): the app's equivalent is MCP, already in `ToolsStatusChip.swift`.

## link-preview
- Existing: none. Links render as `.link` attributes (`Presentation/ChatMessageMarkdownView.swift`); `Browser/` pane opens pages.
- Verdict: SKIP. Previews require fetching third-party pages on render (LinkPresentation/`LPMetadataProvider`), which leaks every cited URL to the network without user action. That contradicts the privacy posture (opt-in network, `AppNetworkPolicy.swift`). If ever wanted, only for URLs the agent already fetched (the turn's `TurnSources` has title and domain), at which point the sources popover already is the preview.

## loader
- Existing: `Generation/ChatActivityLine.swift` (priority-ordered state: prefill bar with real token progress, thinking with last-line preview, running tools, elapsed), `StreamingStatusFooterView.swift` (breathing logo, tok/s, elapsed), `LoadingPhrases.swift`, `Generation/ToolsStatusChip.swift`.
- Verdict: SKIP. A three-dot or orb indicator is strictly less informative than the existing determinate prefill progress. A pure spinner would be a regression for a local model where prefill can take many seconds. Keep.

## mention-menu
- Existing: `Generation/ComposerAutocompleteController.swift` / `ComposerAutocompleteEngine.swift` (trailing-token trigger for `/` and `@`, quoted paths, typed references), `ComposerSuggestionPopup.swift` (inline list inside the composer, keyboard nav, accessibility hint), `Files/MentionResolver.swift`.
- Gap: the menu is a flat list; the accepted mention is plain text in `PromptComposerEditor`, not an inline pill, so the user cannot see what resolved (a bad path just stays text). No grouping by kind (file, folder, git, url, agent, skill).
- Verdict: IMPROVE. Value M, effort L for pills (needs `NSTextAttachment` or attributed-range styling inside the AppKit editor, plus delete-as-unit and copy/paste behavior: easy to get wrong), effort S for grouping. Do the cheap half first: section headers by `ComposerSuggestion` kind in `ComposerSuggestionPopup`, and a resolved/unresolved tint on `@` tokens via attributed-range styling (green-ish `.appSuccess` vs `.appWarning`, never color alone: add a glyph in the popup). Defer true pills. Maps to: Chip, Composer.

## message
- Existing: `Generation/MessageRowView.swift` (user bubble, assistant plain layout, tool cards, memory rows, command rows, collapse for long content in `CollapsibleMessageContentView.swift`, sources chip, timestamp via `MessageTimestampFormatter.swift`, hover bar).
- Verdict: SKIP. The three layouts (bubbles, flat, compact) are a density preference; `ConversationLayout.swift` already exists. No gap worth a task.

## message-actions
- Existing: `Generation/MessageActionBarView.swift` (hover row: copy, share, read aloud, edit, branch, retry/regenerate, pin as instruction, timestamp, variant switcher).
- Gap vs component: thumbs up/down with a feedback form and share. Feedback has no destination in a local app (no server to send it to); a local-only "rate" would be a dead control unless wired to the trajectory log (`docs/SWIFT_TRAJECTORY.md`, off by default, plaintext). Share exists already.
- Verdict: IMPROVE only in a small way. Value L, effort S: the row is hover-only; make the last assistant message's bar always visible (keyboard and VoiceOver users, and trackpad-less use) and ensure every icon has a localized name. Skip thumbs. If a rating is ever wanted, store it with the chat locally and use it for nothing but export. Maps to: IconButton.

## message-error
- Existing: `Components/ErrorBanner.swift` (a capsule above the composer bound to `model.error`, with a dismiss button), `Generation/StallRecoveryBannerView.swift` (Keep Waiting / Stop Turn / Restart for a silent engine), `StreamRecoveryEventNoticeView.swift`, `LoadAnywayConfirmation.swift`. `AppChat.swift` stores a `stopReason` on messages but nothing in `MessageRowView`/`ChatTranscriptView` renders it (checked by search).
- Gap: a failed, cut-off, or stopped reply has no in-transcript presence. The banner is global, transient, dismissible and says nothing about the partial text, the context-window cause, or what to do. The registry component states: failed (retry), limit hit, stopped partway (continue).
- Verdict: IMPROVE. Value H, effort M. For a local engine the real cases are: (1) generation stopped by user with partial text, (2) context window full mid-reply (offer Compact and retry, ties to the context ring), (3) engine error or OOM (offer Retry, link to diagnostics, `LoadGate`), (4) tool-loop cap hit (`AppModel+AgentLoop.swift`). Render a footer strip under the assistant message keyed off `stopReason` (extend the enum if free-form string today) with a one-line cause and quiet buttons: Retry (`model.regenerateResponse()`, exists), Continue (new: append an assistant-prefill continuation), Compact. Keep `ErrorBanner` for non-message errors (download, install). No "usage limit" or billing copy: that is cloud-only. Maps to: StatusLabel (`AppAttention` color + glyph, not color alone), Button `.quiet`, optionally Card. Test the cause-to-copy mapping as a pure function like `ChatActivityLine` does.

---
## Suggested order
1. message-error (H), 2. context-meter actions (S), 3. attachments token weight (S), 4. mention grouping (S), 5. empty-state resume view, 6. inline-citation phase 1.
Everything else is already covered or inapplicable. The app is ahead of the web components in most of this category.
