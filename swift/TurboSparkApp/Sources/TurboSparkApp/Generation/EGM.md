# Generation

## Overview
The chat and project workspace UI of the macOS app: transcript, streaming rows, tool-call cards, the prompt composer, chat and project sidebars, approval cards, command sheets, the artifact panel, and the git inspector. Most views are SwiftUI bound to `AppModel`; they render state and forward every mutation to model methods. A few files are pure, testable logic (`ChatSearch`, `ComposerAutocompleteEngine`, `GoalEvaluator`, `SystemReminders`, `ToolCallDiffFormatter`, `ToolApprovalPreviewSummary`). Approval views gate tool, network, browser, and project-declared MCP and tool execution on explicit user answers. The directory has no child EGM.md files.

## Interface

Transcript and streaming
- `ChatTranscriptView`, `ConversationPaneView`, `OutputPaneView` (host of command sheets and confirmations), `ConversationLayout` (reading width, `conversationColumn()`), `CompactionDividerView`, `SplitChatPanesView` [ChatTranscriptView.swift:ChatTranscriptView] [OutputPaneView.swift:OutputPaneView] [ConversationLayout.swift:ConversationLayout] [SplitChatPanes.swift:SplitChatPanesView]
- `TranscriptFollowRules`, `TranscriptFollower`, `TranscriptScrollObserver`, `JumpToLatestButton` [TranscriptFollow.swift:TranscriptFollowRules]
- `ActiveStreamingRowView`, `StreamingStatusFooterView`, `ChatActivityStatus.resolve(_:)`, `ChatActivityLine`, `ReasoningTraceView`, `LoadingPhrases.phrase(elapsed:turnStartDistance:)`, `StreamRecoveryEventNoticeView` [ActiveStreamingRowView.swift:ActiveStreamingRowView] [ChatActivityLine.swift:ChatActivityStatus] [LoadingPhrases.swift:phrase]
- `MessageRowView`, `MessageActionBarView`, `MessageVariantSwitcherView`, `MessageEditComposerView`, `BranchEditSheet`, `CollapsibleMessageContentView`, `MidTurnTranscriptRowPresentation` [MessageRowView.swift:MessageRowView] [MessageEditingViews.swift:BranchEditSheet] [CollapsibleMessageContentView.swift:CollapsibleMessageContentView]
- `MarkdownContentWithTablesView`, `InteractiveMarkdownTableView` [InteractiveMarkdownTableView.swift:MarkdownContentWithTablesView]

Tool cards and approvals
- `ToolCallCardView`, `ToolCallSummaryCache`, `ToolGroupView`, `ToolResultOutputView`, `ToolCodeCellView`, `TodoChecklistSummary`, `TaskChecklistPanelView`, `TodoItemRow` [ToolCallCardView.swift:ToolCallCardView] [ToolGroupView.swift:ToolGroupView] [TodoChecklistViews.swift:TaskChecklistPanelView]
- `ToolPresentation` (`resolve`, `status(call:result:)`, `webLink`), `BundledToolArtwork`, `OfflineSiteIcons` [ToolPresentation.swift:ToolPresentation]
- `ToolCallDiffFormatter.summarize(callName:arguments:)`, `ToolApprovalPreviewSummary` (`multiEditSummary`, `httpRequestSummary`) [ToolCallDiffFormatter.swift:summarize] [ToolApprovalPreviewSummary.swift:httpRequestSummary]
- `BrowserToolCardPresentation`, `BrowserToolCardMetadataBuilder`, `BrowserPermissionApprovalView`, `NetworkAccessApprovalView`, `ToolApprovalDropdown`, `InteractiveQuestionCardView` [BrowserToolCardView.swift:BrowserToolCardPresentation] [NetworkAccessApprovalView.swift:NetworkAccessApprovalView] [ToolApprovalDropdown.swift:ToolApprovalDropdown]
- `BackgroundAgentsStripView`, `BackgroundShellsStripView`, `SubagentLiveCardView` [BackgroundShellsStripView.swift:BackgroundShellsStripView] [SubagentLiveCardView.swift:SubagentLiveCardView]

Prompt composer
- `PromptComposerView`, `PromptComposerEditor`, `PromptComposerFooter`, `PromptComposerPlusMenu`, `PromptAttachmentsView`, `PromptExamplesView` [PromptComposerView.swift:PromptComposerView] [PromptComposerEditor.swift:PromptComposerEditor]
- Composer pills and controls: `PromptModelSelectorPill`, `PromptProjectContextPill`, `PromptReasoningPillControl`, `ForgeGuardrailsPillControl`, `SteeringPillControl`, `ToolsStatusChip`, `ContextUsageRingView`, `PromptTipsButton`, `SearchToggleButton`, `PromptAudioInputButton`, `ProjectPickerView`, `GhostChatButton`, `NewChatSuggestionBanner` [PromptContextPills.swift:PromptModelSelectorPill] [PromptSafetyPills.swift:ForgeGuardrailsPillControl] [SteeringComposerMenu.swift:SteeringPillControl] [ToolsStatusChip.swift:ToolsStatusChip]
- Autocomplete: `ComposerAutocompleteController` (state), `ComposerAutocompleteEngine.detectTrigger/slashCandidates/mentionCandidates/invocableSkills`, `ComposerSuggestionPopup`, `SlashCommandMenuGroup.grouped(_:)` [ComposerAutocompleteController.swift:ComposerAutocompleteController] [ComposerAutocompleteEngine.swift:detectTrigger] [SlashCommandMenuGroup.swift:grouped]

Sidebars and search
- `ChatSidebarView`, `ChatSidebarChatRowView`, `ChatSidebarProjectsSectionView`, `ChatSidebarGroupedProjectsView`, `ChatSystemPromptSheet`, `ChatShareButton` [ChatSidebarView.swift:ChatSidebarView] [ChatSidebarChatRowView.swift:ChatSidebarChatRowView]
- `ChatSearch` (`Query.parse`, `buildDocuments`, `hits`, `makeSnippet`, `dateBucket`), `ChatSearchOverlayView`, `ChatSearchResultRowView` [ChatSearch.swift:ChatSearch] [ChatSearchOverlayView.swift:ChatSearchOverlayView]

Sheets and panels
- `SessionStatsSheet`, `HelpSheetView`, `GitInfoSheet`, `ContextUsageSheet`, `TasksStatusSheet`, `ToolsListSheet`, `StatusSheet`, `RewindSheet`, `RecapSheet`, `ProfileMemoryCaptureSheet` [ChatDialogSheets.swift:HelpSheetView] [ChatGitSheets.swift:GitInfoSheet] [ChatParitySheets.swift:RewindSheet]
- `ProjectChatSummary` and `ProjectChatSummaryView` (tasks, outputs, web sources) [ProjectChatSummaryView.swift:ProjectChatSummary]
- `ArtifactPanelView`, `ArtifactPanelSource`, `ArtifactWebView`, `ArtifactWebDocument`, `ArtifactContentRuleList` [ArtifactPanelView.swift:ArtifactPanelView] [ArtifactWebView.swift:ArtifactWebView]

Projects, settings, goals
- `ProjectSettingsSheet` (`mergedPermissions(edited:live:loadedOrigins:)`), `ProjectPermissionsSectionView`, `AppProjectGuardrailsOption`, `ProjectRulesSectionView` [ProjectSettingsSheet.swift:mergedPermissions] [ProjectPermissionsSectionView.swift:AppProjectGuardrailsOption]
- `ProjectMcpSettingsSheet`, `ProjectMcpDetectionSectionView`, `ProjectMcpServerRowView`, `ProjectMcpApprovalSheet`, `ProjectToolApprovalSheet` [ProjectMcpSettingsSheet.swift:ProjectMcpSettingsSheet] [ProjectToolApprovalSheet.swift:ProjectToolApprovalSheet]
- `GoalBannerView`, `GoalEvaluator` (`judgeMessages`, `parseVerdict`, `statusRow`), `SystemReminders.reminder(todos:messages:planModeActive:goal:)` [GoalEvaluator.swift:parseVerdict] [SystemReminders.swift:reminder]

Git inspector
- `WorktreeView`, `WorktreeFileListView`, `WorktreeDiffView` (`DiffBlockCache`, `parseDiff`, `parseHunkHeader`), `WorktreeCommitListView`, `WorktreeTimelineView`, `WorktreeWorktreeListView` [WorktreeView.swift:WorktreeView] [WorktreeDiffView.swift:parseDiff]

## Invariants
- Artifact web content is offline and isolated: `.nonPersistent()` data store, offline content rule list that fails closed to an error page if compilation yields nil, and a network-granted page reloads as inline HTML with no file origin so file access and network never coexist. [ArtifactWebView.swift:isolatedForNetworkAccess] [ArtifactWebView.swift:allowsFileNavigation]
- `file:` navigation is allowed only inside the read folder (trailing-slash prefix check) and never when network is allowed; webview identity includes the grant so a flip rebuilds it. [ArtifactWebView.swift:allowsFileNavigation] [ArtifactPanelView.swift:webViewIdentity]
- Artifact source reads are capped at 4,000,000 bytes, and export over an existing file goes through a sibling temp copy plus `replaceItemAt`. [ArtifactPanelView.swift:loadSourceText] [ArtifactPanelView.swift:exportArtifact]
- Activity status priority lives only in `ChatActivityStatus.resolve`; the footer has its own documented precedence ending in Idle. [ChatActivityLine.swift:resolve] [StreamingStatusFooterView.swift:currentStatusView]
- Visibility of background agents, shells, questions, and live runs is by the item's own `chatID` (nil shows everywhere), never by later selection. [ActiveStreamingRowView.swift:liveRunsForChat] [BackgroundShellsStripView.swift:shellsForChat] [ChatTranscriptView.swift:body]
- Chat search excludes ghost chats by the `isGhost` flag, never by emptiness, and never touches the ghost vault; results are recency-sorted with no scoring. [ChatSearch.swift:buildDocuments] [ChatSearch.swift:hits]
- The autocomplete controller never writes composer text (`accept(in:)` returns the replacement); stale async file scans are dropped via `scanGeneration`; skills shadowed by a built-in name are excluded so the popup, help sheet, and plus menu agree with the submit-time parser. [ComposerAutocompleteController.swift:accept] [ComposerAutocompleteController.swift:reset] [ComposerAutocompleteEngine.swift:invocableSkills]
- Composer Return calls `model.run()` only when `canRunOrQueue`, is ignored with Shift or IME marked text, and the autocomplete popup consumes keys first; programmatic inserts use `writePromptTextDirectly` so they are never treated as pastes. [PromptComposerEditor.swift:body] [PromptComposerView.swift:insertPromptText]
- Project, branch, and model navigation (selection, branch switch, project save) is blocked while running, submitting, or a tool call is pending. [ProjectPickerView.swift:navigationBlocked] [WorktreeView.swift:branchMenu] [ProjectSettingsSheet.swift:save]
- Approval cards never default-grant: no Return/Esc binding on the network card, Reject is the default on project tool approval with no approve-all, and a repo-declared MCP server is never dialed before the user answers. [NetworkAccessApprovalView.swift:NetworkAccessApprovalView] [ProjectToolApprovalSheet.swift:content] [ProjectMcpApprovalSheet.swift:ProjectMcpApprovalSheet]
- Approval scope is per host (URL tools) or per exact origin (browser, max 256 per project), never per tool name; high risk hides session and host grants; browser calls without a resolvable origin can only be denied. [ToolCallCardView.swift:approvalPrompt] [BrowserPermissionApprovalView.swift:body]
- Only the primary call of a parked agent batch shows approval controls; a pending approval force-expands its group and card. [ToolCallCardView.swift:isPendingApproval] [ToolGroupView.swift:isExpanded]
- Approval previews mask header values whose names contain authorization, cookie, token, key, or secret, never print the auth token, and use the executor's URL key list (`httpURLKeys`) so the shown URL is the fetched one. [ToolApprovalPreviewSummary.swift:httpRequestSummary] [ToolApprovalPreviewSummary.swift:httpURLKeys]
- `webLink` accepts only http(s) URLs with a host, and tool and site artwork is resolved only from bundled resources, never fetched or read from model-supplied paths. [ToolPresentation.swift:webLink] [ToolPresentation.swift:OfflineSiteIcons]
- Project settings save layers edits over LIVE project state (MCP servers and rules, REPL roots, browser origins unless changed in the sheet); new projects seed `AppProjectPermissions.newProjectDefault`, never `.auto`. [ProjectSettingsSheet.swift:save] [ProjectSettingsSheet.swift:mergedPermissions] [ProjectSettingsSheet.swift:populateInitialValues]
- Reminders from `SystemReminders` are ephemeral (recomputed per history build, appended to the model-bound last user message, never persisted) and stay out of the O(1) SKILL.state history path. [SystemReminders.swift:SystemReminders]
- Goal verdict parsing takes the first `{` through the LAST `}`; a nil verdict is a transient failure that is never "met" and never clears the goal. [GoalEvaluator.swift:parseVerdict]
- Transcript follow pins within 32 pt of the bottom and unpins only on user-driven moves; `NSEvent.keyCode` is read only for `.keyDown` (it raises an ObjC exception otherwise). [TranscriptFollow.swift:isUserScroll]
- Destructive actions are staged and confirmed: chat and project delete, clear history, ghost discard, process kill (`confirmationDialog` or alert). [ChatSidebarView.swift:body] [OutputPaneView.swift:body] [BackgroundShellsStripView.swift:shellToKill] [GhostChatButton.swift:body]
- Rewind moves the conversation only, never files tools wrote; recap Copy is disabled while empty. [ChatParitySheets.swift:RewindSheet] [ChatParitySheets.swift:RecapSheet]
- Diff parsing skips blank and backslash lines (they shift line numbers), reads hunk counts only between the two `@@` markers, and the parse cache is keyed on raw diff text. [WorktreeDiffView.swift:parseDiff] [WorktreeDiffView.swift:parseHunkHeader] [WorktreeDiffView.swift:DiffBlockCache]
- UI strings are catalog keys unless explicitly verbatim (model output, file names, author skill text, MCP names). [ToolPresentation.swift:localizedLabel] [ToolCodeCellView.swift:headerBar] [ChatDialogSheets.swift:HelpSheetView]

## Side Effects
- Pasteboard writes (message text, code, diffs, commit SHAs, CSV/TSV, recaps, tool output), usually with a 1.5 s Task resetting the "Copied" state. [MessageActionBarView.swift:MessageCopyButton] [ToolResultOutputView.swift:copyOutput] [InteractiveMarkdownTableView.swift:flashCopied]
- `NSSavePanel` writes for artifact export, code download, and table CSV; failures surface an `NSAlert`. [ArtifactPanelView.swift:exportArtifact] [ToolCodeCellView.swift:downloadCode] [InteractiveMarkdownTableView.swift:exportCSV]
- `NSOpenPanel` folder picker and Syntext index build, query, and delete on disk. [ProjectSettingsSheet.swift:selectFolder] [ProjectSettingsSheet.swift:reindexProject]
- Disk reads: artifact source UTF-8, attached images via `NSImage`, bundled tool and site artwork (once, from `Bundle.module`), page file when network is enabled. [ArtifactPanelView.swift:loadSourceText] [MessageRowView.swift:imageAttachments] [ToolPresentation.swift:BundledToolArtwork]
- `WKContentRuleList` compiled asynchronously and cached per process; WKWebView loads file or inline HTML. [ArtifactWebView.swift:withOfflineRules] [ArtifactWebView.swift:loadDocument]
- `NSWorkspace` opens or reveals files, PR URLs, and project folders; `NSDataDetector` scans messages for links. [ChatGitSheets.swift:prsBody] [ChatSidebarGroupedProjectsView.swift:projectHeaderRow] [ProjectChatSummaryView.swift:outputs]
- Model mutations are forwarded rather than done locally: `run`, `openChat`, `rewindTo`, `killBackgroundShell`, `setEffectivePermissionMode`, `addNetworkDomain`, `addMcpPermissionRule`, `approvePendingToolCall`, `approvePendingMcpServer`, `createProject`/`updateProject`/`deleteProject`, and `testMcpServer` (spawns or dials an MCP server). [PromptComposerEditor.swift:body] [ToolCallCardView.swift:approvalPrompt] [ProjectMcpSettingsSheet.swift:testServer] [ProjectSettingsSheet.swift:save]
- Git commands via `worktree.refresh`, `switchBranch`, `initRepository`, `loadDiff`; posts `.toggleInspector` and observes `.focusPrompt`. [WorktreeView.swift:branchMenu] [WorktreeView.swift:header] [PromptComposerView.swift:body]
- Async work: `ProjectFileIndex.shared.entries(for:)` for mentions, 300 ms debounced search rebuild, 1 Hz `Timer` and `TimelineView` while live, summarization via `summarizeConversationForProfileMemory`, speech via `AppSpeechSynthesizer.shared`. [ComposerAutocompleteController.swift:recomputeMention] [ChatSearchOverlayView.swift:scheduleRebuild] [StreamingStatusFooterView.swift:timer] [ProfileMemoryCaptureSheet.swift:body] [MessageActionBarView.swift:MessageSpeechButton]
- VoiceOver `AccessibilityNotification.Announcement` on generation start/finish and parked questions; `NSEvent`-driven scroll probing observes `boundsDidChangeNotification`. [OutputPaneView.swift:body] [ChatTranscriptView.swift:body] [TranscriptFollow.swift:TranscriptScrollProbeView]

## Verification
- Run: `swift test --package-path swift/TurboSparkApp`
