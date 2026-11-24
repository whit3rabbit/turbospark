# Agent activity components vs TurboSpark chat UI

Sources read: designer descriptions plus variant source for all 20 slugs; Swift in `swift/TurboSparkApp/Sources/TurboSparkApp/Generation/` (ToolCallCardView, ToolGroupView, ToolApprovalDropdown, ToolApprovalPreviewSummary, InteractiveQuestionCardView, TodoChecklistViews, SubagentLiveCardView, BackgroundAgentsStripView, BackgroundShellsStripView, ChatInboxModel, ChatSidebarInboxRows, StallRecoveryBannerView, StreamRecoveryEventNoticeView, ChatActivityLine, TrajectoryPaneView, BrowserToolCardView, ToolCodeCellView, ToolCallDiffFormatter), `Components/CronJobsSettingsPaneView.swift`, `Tools/Planning`, `Theme/`. The app is already mature here; most verdicts are SKIP or small IMPROVE. I did not read DESIGN.md in full (only the control vocabulary table); new views must still go through it (localized strings, theme tokens, `.appCardSurface`, `AppIconButton`, `AppChip`, `AppProgressBar`, reduce-motion handling).

## Verdict table

| Slug | Verdict | Value / Effort | One line |
| :--- | :--- | :--- | :--- |
| agent-inbox | SKIP | - | `ChatSidebarMode.inbox` + `ChatInboxModel` already sort chats by needs-you/working/finished |
| agent-thread | SKIP | - | Multi-user comment thread on a run; single-user local app |
| background-tasks | IMPROVE | M / S | Merge the agent strip and shell strip into one status-bar tray with count and progress |
| browser-session | SKIP | - | `BrowserToolCardView` + `ToolGroupView` already cover it |
| checkpoint | ADOPT | H / L | No file rollback exists; add per-turn file snapshots with Restore divider |
| decision-inbox | SKIP | - | Duplicate of inbox "Needs you" section; a cross-chat approval queue is the only new part |
| edit-before-run | ADOPT | M / M | Approval card shows args read-only; let user fix path/command before Allow |
| file-edit | IMPROVE | M / S | Diff counts exist in the header; add a short inline diff preview with live tail |
| generative-form | SKIP | - | `InteractiveQuestionCardView` already takes multi-question; a form builder is model-quality risky on small local models |
| inbox-views | SKIP | - | Saved filters over runs; no run-volume to justify it |
| plan | IMPROVE | H / M | `ExitPlanMode` renders a collapsible text block; add real Approve / Request changes gate |
| question | IMPROVE | L / S | Add 1-9 number keys, a "Something else" free-text row, Esc to skip |
| research-progress | SKIP | - | No research job in the app; `ChatActivityLine` covers phase status |
| run-recovery | IMPROVE | M / M | Stall banner exists; add resume-from-failed-tool after a crash or tool error |
| schedule-card | IMPROVE | M / S | Cron is created by tool call with no in-chat confirmation card showing next runs |
| step-timeline | SKIP | - | `TrajectoryPaneView` is the run trace; add per-step duration only if missing |
| subagent-tree | IMPROVE | L / M | `SubagentLiveCardView` flat list; add parent/child nesting only if nested subagents are possible |
| terminal | IMPROVE | M / S | Live tail of last 3 output lines while a command runs, one-line result when collapsed |
| todo-list | SKIP | - | `TaskChecklistPanelView` + `TodoChecklistSummary` already match; optional shimmer on active row |
| tool-approval | IMPROVE | M / S | Approval exists with once/session/always; add a resolved one-line record and a reason line |
| tool-call | SKIP | - | `ToolCallCardView` (1270 lines) and `ToolGroupView` already do card, line and grouped stack |

---

## agent-inbox
- Existing: `Generation/ChatInboxModel.swift` (`ChatSidebarMode.inbox`, `ChatInboxSection` needsYou / working / finished, priority order, persisted setting) and `Generation/ChatSidebarInboxRows.swift`.
- Gap: component adds a list-plus-detail split. TurboSpark already has the list and the detail is the chat itself.
- Verdict: SKIP. Reason: reimplementing what the sidebar inbox mode does.

## agent-thread
- Existing: `TrajectoryPaneView` (read-only role-separated timeline of model and tool calls) covers "each step it took".
- Gap: "replies from the people who stepped in" is multi-user collaboration.
- Verdict: SKIP. Reason: single-user local app, no shared runs.

## background-tasks
- Existing: `BackgroundAgentsStripView` (subagent runs, Stop/Dismiss) and `BackgroundShellsStripView` (running shells, Kill), both strips under the transcript; `ChromeTelemetryView` and `StatusBarView` in `Chrome/`; `TaskProgressIndicatorView`; downloads got a top-bar entry recently (commit 921db84e).
- Gap: two separate strips that only show for the selected chat, nothing in the status bar when you are in another pane or chat; no unified count, no progress per job; model downloads, image jobs, meeting transcription, off-peak tasks live in other places.
- Verdict: IMPROVE. Value M, effort S to M.
- Approach: a `BackgroundTasksPopover` anchored to a small `AppChip`-style status-bar item in `StatusBarView` ("2 running" with a spinner, "all finished" when idle). Rows: `StatusDot` or `TaskProgressFlameIcon`, title, `AppProgressBar(value:)` when fraction known (indeterminate otherwise), Open (select chat/pane) and Stop `AppIconButton`. Data: a read-only aggregator over `model.backgroundAgentRuns`, `model.backgroundShellSummaries`, existing download manager state and image jobs; do not move ownership. Keep the strips; the tray is the cross-chat view. Map: Chip, AppProgressBar, StatusDot, AppIconButton. Note the component's FAQ trap (progress 0..1 vs 0..100): use one scale.
- Local-resource angle is the real win: include "model loading" and "KV/context" busy states only if they are already observable; otherwise skip.

## browser-session
- Existing: `Generation/BrowserToolCardView.swift` (`BrowserToolCardPresentation`: action, target, outcome running/completed/denied/failed/pendingApproval, duration), `BrowserPermissionApprovalView`, `BrowserDialogApprovalView`, `ToolGroupView` coalescing sequential calls.
- Gap: none that matters. A per-visit collapsed summary line is already what `ToolGroupView` gives.
- Verdict: SKIP.

## checkpoint
- Existing: nothing for chat-level file rollback. `FileReadWriteTools` has a per-file `undo_edit` command the model can call; `WorktreeTimelineView` / `WorktreeView` are git commit graphs (for repos with git, user-driven). `AppModel+MessageEditing` edits and regenerates messages but does not restore files. `st` for rewind/checkpoint/restore found only unrelated hits.
- Gap: after an agent turn edits files, the user can only ask the model to undo or use git. There is no "files go back to before this turn" action.
- Verdict: ADOPT. Value H (trust in agent mode on a local model that makes more mistakes), effort L.
- Approach:
  1. Data: a `TurnFileSnapshot` store (per chat, per turn): before the first write of a file in a turn, copy original bytes (or record "did not exist") into `~/.turbospark/.../checkpoints/<chat>/<turn>/`. Hook where `FileReadWriteTools`, `ApplyPatchTool`, `MultiEditExecutor` write. Cap size, skip files over a limit, record added/removed counts (the app already computes them in `ToolCallDiffFormatter`). Must go through the same path guard and project-permission checks; restoring is a write, so run it through the approval/permission layer, and never restore outside the project roots recorded at snapshot time. Security-adjacent: ask first before implementing.
  2. UI: a divider row in `ChatTranscriptView` after a turn that changed files: capsule "Checkpoint, 3 files, 4 min ago" (disclosure lists path +N -M) and a Restore button. Use `AppChip` for the pill, `.appAction(.quiet)` for Restore, `confirmationDialog` for the confirm (copy: files only; do not claim later messages are removed unless you also truncate, and truncating conflicts with the app's non-destructive transcript stance). Mark restored state on the divider.
  3. Honest limits: bash-made changes outside tracked tools are not snapshotted; say so in the confirm text.
- Maps to: Chip, Button (quiet), Card for the disclosure list, `confirmationDialog`.
- Skip the History variant (a separate list); the sidebar/trajectory pane can link to the same data later.

## decision-inbox
- Existing: `ChatInboxSection.needsYou` (pending approval or question) in the sidebar, plus per-chat approval cards and `ChatActivityStatus.needsConfirmation`.
- Gap: the component is a cross-agent queue with "why" and "what happens if yes". The app's concurrent chats (`AppModel+ConcurrentChats`) mean approvals can wait in chats you are not viewing, and the sidebar only flags them, you must open the chat. A queue that approves from outside the chat would be new, but approving without seeing the transcript context is riskier on a local agent that can run shell commands.
- Verdict: SKIP. Reason: the inbox section already surfaces it; inline approve-from-list adds risk for little gain. If wanted later, show the command preview in the sidebar row tooltip and keep Allow inside the chat.

## edit-before-run
- Existing: `ToolCallCardView` approval section (lines ~1100-1240) with Allow once / Allow for chat / Always Allow This Tool or Server / Deny; `ToolApprovalPreviewSummary` builds read-only previews (multiedit blocks, http_request method/body/headers); `ToolCodeCellView` shows commands with copy and download.
- Gap: a small local model often gets one argument wrong (wrong path, extra flag, wrong cwd). Today the only options are deny and re-prompt, or allow a wrong call.
- Verdict: ADOPT. Value M, effort M.
- Approach: in the pending-approval card, add an "Edit" toggle for a whitelist of simple tools: Bash/command (single `TextEditor` with `.appEditorSurface`), path-based file tools (path `TextField`), http_request URL. On Allow, pass an `editedArguments: [String: String]` to `model.approvePendingToolCall(id:alwaysAllowSession:editedArguments:)`; the pending-call record must be replaced before dispatch and the edit must be re-run through `ToolRiskClassifier` and `ToolCallDispatchGate` (an edited command must not inherit an "always allow" rule that matched the original; force once-only when edited). Record "edited by you" on the call so the transcript and trajectory log show it, and tell the model the arguments were changed (system reminder) so its next turn is consistent. Do not offer editing for browser or MCP calls initially.
- Security: this changes approval semantics; per your rules, ask first and add tests that an edited command is re-classified and never auto-allowed.
- Maps to: Card (existing approval card), `.appEditorSurface`, Button (primary Allow, quiet Cancel edit).

## file-edit
- Existing: `ToolCallCardView` header shows action + target + green/red +N -M (`ToolCallSummaryInfo` from `ToolCallDiffFormatter`, honors `appearance.diffMarkers` for non-color mode). Expanded card renders the diff via `ToolCodeCellView(language: "diff")`. `WorktreeDiffView` is the full git diff.
- Gap: you must expand to see anything; no short peek of the first 6 lines in the collapsed state, no live tail while the write streams, no "open file" action visible on hover.
- Verdict: IMPROVE. Value M, effort S.
- Approach: in the collapsed `ToolCallCardView` for edit/write tools, show a 6-line monospaced preview under the header (reuse `ToolCodeCellView` body without the header bar, `theme.code(.small)`, add/remove line backgrounds from `theme.success`/`theme.danger` at low opacity), "N more lines" button to expand; show the tail while the call is running and the head when done (the component's stated reading order). Add an `AppIconButton` "Open file" that routes to the Files pane. Check the `appearance.diffMarkers` rule and the `DesignRules.json` baseline (no raw colors).
- Skip the Review variant: multi-file list is `WorktreeFileListView`.

## generative-form
- Existing: `InteractiveQuestionCardView` handles several questions, single/multi-select, Submit Answers footer, Skip; `AskUserQuestion` tool schema in `PlanningInteractiveTools`.
- Gap: free-text, number, date fields, model-defined form schema.
- Verdict: SKIP. Reason: small local models already struggle to emit a valid `AskUserQuestion` schema; a richer form schema multiplies malformed calls. The existing duplicated-header handling shows this is a real failure mode. Revisit only if question text input is wanted (see `question`).

## inbox-views
- Existing: the two-mode sidebar (chats / inbox) is the saved-view mechanism; chat search and `ChatSearch.swift`.
- Gap: custom saved filters with counts.
- Verdict: SKIP. Reason: a desktop chat app does not have the run volume of a fleet inbox.

## plan
- Existing: `Tools/Planning` (`EnterPlanMode`, `ExitPlanMode` with plan text and optional file path), `PlanFileNamer`, `ToolCallCardView.submittedPlan` renders the plan as a `CollapsibleMessageContentView` under a "Plan" header; `AppAgentDefinition`/permission modes include a plan mode. `AppArtifact` notes ExitPlanMode finalizes a plan artifact.
- Gap: I could not find an Approve / Request changes gate on the plan itself; the plan is shown as markdown in a tool card and the user types a reply. A plan is the highest-leverage review point for a weak local model.
- Verdict: IMPROVE. Value H, effort M.
- Approach: when `ExitPlanMode` is pending, render a `PlanApprovalCard` instead of the plain tool card: numbered steps parsed from the plan markdown (fallback to the markdown block if parsing fails; never lose text), each step a disclosure row, footer: primary `Start` (maps to approve and exit plan mode into the chosen permission mode), quiet `Request changes` (opens an inline `TextEditor` with `.appEditorSurface`, sends feedback as the tool result so the model stays in plan mode). Resolved state: single line "Plan approved" / "Changes requested". The executor should block on the user decision the way `AskUserQuestion` does (reuse that waiter pattern from `AppModel+Questions`).
- Maps to: Card, Button (primary/quiet), Composer-style editor surface, `.accessibilityAddTraits(.isHeader)` on the title.
- Skip the Phases variant (same behavior, different look).

## question
- Existing: `InteractiveQuestionCardView` (accent-tinted card, single-select tap answers, multi-select with Submit, Skip, accessibility label).
- Gap vs component: number-key shortcuts (1-9), arrow keys, an "other" free-text row with field, Esc to skip, hint line under each option. Check whether `UserQuestionItem` options carry a description to use as the hint.
- Verdict: IMPROVE. Value L, effort S.
- Approach: add `.onKeyPress` (macOS 14) for digits while the card has focus, keep Return owned by the composer (per the stall banner's note, inline strips must not steal Return), add an "Other" row with a `TextField` that submits its text as the answer, `.keyboardShortcut(.cancelAction)` for Skip only when the card is focused. Document in `docs/KEYBOARD_SHORTCUTS.md` and add shortcut hint text through `AppStrings`.
- Skip the Decision list variant.

## research-progress
- Existing: `ChatActivityLine` / `ChatActivityStatus` (reading prompt n/total, running tool, task, thinking, image stage with fraction), `StreamingStatusFooterView`, `AppProgressBar`.
- Gap: there is no multi-phase research job to report (plan / search / read / write with source counts). The deep-research agent, if used, is a subagent run covered by `SubagentLiveCardView`.
- Verdict: SKIP. Reason: no backing feature. If a research workflow is added later, `Workflows/WorkflowRunEngine` events could feed a phase list built from `AppProgressBar` plus `TodoItemRow`.

## run-recovery
- Existing: `StallRecoveryBannerView` (engine silent: Keep Waiting / Stop Turn / Restart), `StreamRecoveryEventNoticeView`, `ErrorBanner`, `MessageActionBarView` Retry message (regenerate), `Docs: swift/docs/SWIFT_CRASH_RECOVERY.md`, `WorkflowJournal` for workflow runs.
- Gap: after a tool error or a crash mid-agent-run, the options are regenerate the whole reply or type a new message. The component's "what finished, what broke, retry from this step, skip it, edit the input" has no equivalent for an agent turn.
- Verdict: IMPROVE. Value M, effort M.
- Approach: on a failed tool result in an agent turn, add a footer to that `ToolCallCardView` with `Retry this call`, `Skip` (feeds the model a "skipped by user" result), and `Edit and retry` (shares the edit-before-run editor). Do not change the transcript structure: results already pair by call id. For a turn interrupted by crash, `AppJSONStore` quarantine and the persisted partial turn could offer "Continue from here"; only do that after confirming what partial state is persisted (read SWIFT_CRASH_RECOVERY.md first). Keep the stall banner as is, it is correct.
- Maps to: ErrorBanner style, Button quiet, StatusDot.

## schedule-card
- Existing: `CronScheduler` + `AppCronJob` (next fire cached), `CronJobsSettingsPaneView` lists jobs with schedule, prompt, next run, history, pause/delete; jobs are created by the model via `CronCreate` or via "Schedule a Task..." composing a chat message; `OffPeakTasksSectionView`.
- Gap: when the model creates a job, the user sees a normal tool card. There is no plain-language "Every weekday at 9:00, next runs Mon 9:00, Tue 9:00, Wed 9:00" confirmation, and no way to adjust before it is live.
- Verdict: IMPROVE. Value M, effort S.
- Approach: a presentation-only `CronCreate` card variant in `ToolCallCardView` (it already switches on tool name for plans and browser). Render the human description, next 3 run times from `CronScheduler` next-fire logic (pure function over the cron expression and `Date`), and Pause/Delete quiet buttons that call the existing store. Does not need an editor; the pane remains the editor. Time zone must be shown explicitly (local), copy through `AppStrings`.
- Maps to: Card, Chip, StatusDot, existing store.
- Skip the Schedule editor variant: the CRON pane header already decided "no create form" to avoid forking validation.

## step-timeline
- Existing: `TrajectoryPaneView` (per-chat sheet, role-separated timeline of model calls and tool runs, read-only, local log, off by default and plaintext per `SWIFT_TRAJECTORY.md`), `ToolGroupView` coalesced groups, `AppToolResult.durationSeconds` shown on cards.
- Gap: wall-clock start time per step and nested sub-steps in the transcript. Duration is already available.
- Verdict: SKIP. Reason: transcript already has durations; the trajectory pane owns the timeline. A tiny follow-up inside the trajectory pane (hover timestamps) is optional, not worth a component.

## subagent-tree
- Existing: `SubagentLiveCardView` (prompt head, tool rows with running/ok/error icons, turn count, status text, Stop, Dismiss, expand), `BackgroundAgentsStripView`, `SubagentRunner`, `SubagentProgress`, `SubagentMessageHub`.
- Gap: the six component layouts mostly address many parallel agents. Locally, parallel subagents are bounded by one engine and memory (a single model resident), so lists stay short.
- Verdict: IMPROVE (low). Value L, effort M. Only two cheap changes are worth it: (1) a collapsed one-line state per card (`isExpanded` defaults to true today, so a long run crowds the strip; default collapsed for finished runs); (2) show the subagent's result handback as a one-line summary under the header when done. Nesting needs a `parentRunID` in `SubagentRunState`, which only matters if subagents can spawn subagents; check `SubagentRunner+Gate` before building any tree. Skip swimlanes, roster, activity feed.

## terminal
- Existing: `ToolCallCardView` renders bash commands and output through `ToolCodeCellView` (copy, download) and `ToolResultOutputView`, `ANSIColorizer` in Presentation, `ToolOutputFormatter`; exit/duration on result; `BackgroundShellManager` for long runs.
- Gap: while a command runs, the collapsed card shows nothing of the output; the component tails the last 3 non-blank lines and, when finished, a one-line summary (last non-blank line, red on non-zero exit) with the full log on expand. Failure state is derived from exit code.
- Verdict: IMPROVE. Value M, effort S.
- Approach: if running output is streamed into the call (check whether `ProcessExecutor` streams chunks into `AppToolResult` before completion; if not, this needs a small streaming hook and drops to M), render a 3-line tail in `theme.code(.small)`, `.lineLimit(1)` each, with `role=log` equivalent (`.accessibilityAddTraits(.updatesFrequently)`) and no per-line animation when Reduce Motion is on. Collapsed finished state: last non-blank line summary. Reuse `ANSIColorizer`.
- Skip the Command history variant: `ToolGroupView` already groups consecutive calls.

## todo-list
- Existing: `TodoChecklistViews.swift` (`TodoChecklistSummary` "2/4 done (1 in progress)", `TodoItemRow`, `TaskChecklistPanelView`), `TodoWriteExecutor`, `TaskProgressIndicatorView`/`TaskProgressFlameIcon` for the active item.
- Gap: strike-through animation and shimmer on the active line; a segmented progress bar (Progress variant).
- Verdict: SKIP. Reason: the feature is there. A segmented `AppProgressBar` strip under the title would be a 10-line polish, only if wanted, and the shimmer fights DESIGN.md motion rules (Reduce Motion, battery on long local runs).

## tool-approval
- Existing: `ToolApprovalDropdown` (composer mode picker), the per-call pending approval card in `ToolCallCardView` (Allow once, allow for chat, Always Allow This Tool / This Server for MCP, Deny, with a risk classifier `ToolRiskClassifier`, `ToolApprovalPreviewSummary`), `ProjectToolApprovalSheet`, `ProjectMcpApprovalSheet`, `NetworkAccessApprovalView`, `BrowserPermissionApprovalView`. Rich already.
- Gap vs component: (1) after the decision, a one-line record ("Allowed once / Denied" with the command, struck through if denied) that stays in the transcript; check what the card shows after approve/deny, it may already have status `.denied` rendering; (2) a short "why" line from the model (`reason`) above the command, and cwd shown beside the title; (3) the banner-docked-to-composer idea is not needed since the card sits in the transcript.
- Verdict: IMPROVE. Value M, effort S. Only (1) and (2); verify against current resolved rendering before touching it.
- Approach: add `cwd` and optional model-supplied `description` (Bash tool already has a description arg in many agent tool schemas) to the card; make the resolved state a `StatusDot` + verb + command line using `.appSuccess` / `.appDanger`. No new approval semantics, so no security-review needed.
- Do not take "Always allow" as a plain button on destructive commands: keep the existing risk-aware behavior.

## tool-call
- Existing: `ToolCallCardView` (header with icon, verb, target, +N -M, duration, status; expandable request/response via `ToolCodeCellView`/`ToolResultOutputView`; summary cache `ToolCallSummaryCache`), `ToolGroupView` (coalesced stack with running/pending-approval auto-expand), `ToolPresentation` (name to label/icon registry), `ToolsStatusChip`.
- Gap: none. The component's three variants (card, inline line, grouped stack) are all already represented.
- Verdict: SKIP.

---

## Suggested order if you act on this
1. Plan approval gate (H, M): highest value for local models, reuses the question-card waiter.
2. Terminal live tail and file-edit preview (M, S each): cosmetic but frequent.
3. Tool approval resolved record plus edit-before-run (M): do the read-only part first, then editing with security review.
4. Checkpoint (H, L): biggest build and a safety surface (file restore); design the snapshot store before UI.
5. Background tasks tray, schedule card, run-recovery footer, question keys, subagent collapse: opportunistic.

## Cross-cutting notes
- Every added string goes through the string catalog (ten-language parity gate); every new color through `.appSuccess/.appWarning/.appDanger/.appAccentText`, and `make design-check` baselines must not be raised.
- The component library uses fixed amber `#e9bd98` and violet accents; ignore those, use `theme.accent` and `AppAttention`.
- Several component demos animate with springs and shimmer. Use `TurboSparkMotion` and honor Reduce Motion; avoid continuous animation that keeps the GPU awake during local inference.
- Not applicable by nature: any component whose value is multi-user review, billing, or fleet-scale run management.
