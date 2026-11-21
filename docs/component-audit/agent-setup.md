# Agent setup components vs TurboSpark SwiftUI app

Paths are under swift/TurboSparkApp/Sources/TurboSparkApp. Read: swift/AGENTS.md (head), docs/design-system/README.md (head). DESIGN.md itself was not opened in this pass, so the component mapping names (Card, Chip, StatusDot, EmptyState, Composer, Button) come from the design-system folder layout and the existing code (appAction, AppSearchField, ExtensionPaneIntro, EmptyState), not from a read of each guide. Descriptions and variants were read for all 15; source was skimmed in full only for connector-sync and source-ingest, the rest by description and variant name (large files).

## Verdict table

| Slug | Verdict | Value | Effort | Reason |
|---|---|---|---|---|
| agent-card | IMPROVE | L | S | Agents pane has list+detail; add header chip row (model, tool count, scope) only |
| agent-create | SKIP | - | - | AgentEditorSheet exists; "describe it" generator is a cloud-style wizard, low payoff |
| connector-sync | SKIP | - | - | Cloud app connectors (Drive/Slack) do not exist locally; scope tree maps to nothing here |
| event-trigger | SKIP | - | - | App-event webhooks (PR opened) need a server; local scheduler + cron cover the on-device case |
| instructions-editor | IMPROVE | M | S | Add live token estimate to SystemPromptSettingsSection and project rules, using real tokenizer |
| mcp-servers | IMPROVE | M | M | Settings pane is mature; add composer health popover and per-server last-error |
| memory | IMPROVE | M | S | Ledger manager exists; add "saved to memory" in-chat note with Undo and topic grouping |
| model-settings | IMPROVE | M | S | Sampling sliders + named presets exist; add header popover with tone presets |
| retrieval-inspector | ADOPT | M | M | No view of what recall/context engine injected; real value for a local RAG/KV app |
| secrets-manager | SKIP | - | - | Keychain is opt-in per credential by design; a secrets table conflicts with that rule |
| source-ingest | IMPROVE | M | M | Ingest exists for embeddings/attachments; add staged per-file progress (read, split, index) |
| step-inspector | SKIP | - | - | Workflows are scripts (lexer/interpreter), not a node canvas; no step selection surface |
| step-picker | SKIP | - | - | Same: no visual workflow builder to drop steps into |
| tool-picker | IMPROVE | H | M | Per-chat tool toggles in composer plus-menu, grouped by server, with permission state |
| trigger-list | IMPROVE | L | S | CronJobsSettingsPane lacks last-fired and recent firings; WorkflowScheduler has the data |

## agent-card

- Component: one agent as card, list row, directory listing, chat header.
- Existing: Components/AgentsSettingsPaneView.swift (337 lines, scope filter built-in/user/project, shadowed and constrained names), Components/AgentDetailView.swift, Components/AgentEditorSheet.swift, model State/AppAgentDefinition.swift (name, description, tools, disallowedTools, model, maxTurns, scope, isEnabled).
- Gap: only the "chat header" variant is genuinely absent. When a chat runs as a named agent there is no header strip showing which agent, model and tool count are active. Directory variant is cloud-only.
- Verdict: IMPROVE (chat header only), value L, effort S.
- Approach: small HStack strip above ChatTranscriptView in Generation/, shown only when the active chat has an agent: name, a model Chip, "N tools" Chip, a Menu to switch/clear. Data is already on AppAgentDefinition. Maps to Chip + Button (plain style). Skip the grid card.

## agent-create

- Component: new-agent form with preview, or a one-sentence "describe it" flow.
- Existing: AgentEditorSheet.swift (275 lines), AgentContentImportSheet.swift, AgentParser.swift (markdown frontmatter agents), AgentManager+Files.swift.
- Gap: no generation from a sentence, no live "try it" preview. Agents are files, so power users edit markdown anyway.
- Verdict: SKIP. A describe-it flow needs the on-device model to author a valid agent file; with small local models that output is unreliable and needs a validation loop (AgentParser) to be safe. Revisit only if users ask. Cheaper win: a "Test in new chat" button in AgentEditorSheet (effort S) which needs no new component.

## connector-sync

- Component: connected cloud app (Drive) with sync status, item counts, history, failures, folder tree scope picker.
- Existing: nothing equivalent. Nearest are Files/ProjectFileIndex.swift (local index) and Import/ChromeBookmarkImporter.swift.
- Verdict: SKIP. Cloud OAuth connectors are out of scope for an on-device app, and the project rules keep credentials opt-in. The tri-state folder tree (leaf-id selection, mixed parent) is a good pattern if the app ever lets users pick which local folders feed the index: Components/ExcludedScanPathsSectionView.swift and ScanFolderRowView.swift are the place to apply it, but that is a separate, unrequested change.

## event-trigger

- Component: "when this happens in another app, run the agent" with conditions.
- Existing: Workflows/WorkflowScheduler.swift (653 lines), Components/CronJobsSettingsPaneView.swift, Tools/Automation/MonitoringNotificationTools.swift, Components/Hooks/ (tool-event hooks).
- Verdict: SKIP. Third-party app events require a public webhook endpoint or cloud relay. The local equivalents (cron, file watch, hooks on tool events) already exist. Do not build a SaaS-style event picker.

## instructions-editor

- Component: project instructions with token meter, system prompt with variable test preview, rules list with scope and file-backed read-only rows.
- Existing: Components/SystemPromptSettingsSection.swift (157 lines), Generation/ProjectSettingsSheet.swift (rules, agent profile), State/AppChatInstructionPin.swift and AppModel+InstructionPinning.swift, Components/SteeringPresetEditorSheet.swift, SoulSettingsSection.swift.
- Gap: no token count next to editable instruction text. For a local model with a fixed window, instruction tokens are directly KV cost, so this matters more here than on cloud. File-backed rules being read-only with an explanatory note is also a good rule.
- Verdict: IMPROVE, value M, effort S.
- Approach: add a caption row under the TextEditor in SystemPromptSettingsSection and the rules editor in ProjectSettingsSheet: "~N tokens, X% of window" using the loaded model's tokenizer (ffi) and State/AppContextUsage.swift for the window, not the component's chars/4 heuristic (the FAQ itself says it is not a tokenizer). Debounce, tabular digits via themedFont. Maps to existing Form row, caption text style. Show a warning tone (.appWarning) past a threshold. Same-file change, no new model.

## mcp-servers

- Component: settings list with status and tools, add-server form with Test, composer status popover.
- Existing: Components/McpSettingsPaneView.swift, McpServerCardView.swift (test button, enable, tools disclosure, "Tools: N" badge), McpServerEditorSheet.swift (403 lines), McpTransportFieldsView.swift (env, passthrough), McpImportSheet.swift, summary pills (Servers/Active/Tools), Tools/MCP/McpClientEngine.swift, McpToolCatalogCache.swift.
- Gap: list and add/test are already covered. Missing: (1) a compact health popover reachable from the composer or status bar, (2) persistent last error or last connect time per server, (3) env values are plain text in a monospaced editor with no mask.
- Verdict: IMPROVE, value M, effort M.
- Approach: (1) Popover from Chrome/StatusBarView.swift or the composer plus-menu: rows with StatusDot (connected, failed, disabled) + name + tool count + Retry, fed from AppModel+Mcp.swift state. Needs a per-server runtime status published property if not already present (not verified). (2) store lastError/lastConnectedAt next to discoveredTools. (3) Mask env values whose key matches TOKEN|KEY|SECRET in McpTransportFieldsView, but this touches credential handling: ask before doing it, per user rules. Maps to StatusDot, Chip, Button, popover.

## memory

- Component: manage page, "saved in chat" note, profile grouped by topic.
- Existing: App/MemorySettingsPaneView.swift (379 lines: scope Profile/Project, status filter, search, approve/edit/forget, evidence, source-message link, pending count), Components/MemoryCharacterView.swift, Tools/Memory/MemoryTool.swift, AppModel+MemoryScheduler.swift.
- Gap: the manage variant is already stronger than the component (approval workflow, evidence, forget semantics). Missing: the in-transcript note when the model saves or proposes a memory, with Undo/Edit, and topic grouping.
- Verdict: IMPROVE, value M, effort S.
- Approach: when MemoryTool writes a claim, append a transcript-level inline row (not a toast) in Generation/ChatTranscriptView using the same row style as tool-call rows: "Saved to memory: <claim>" with Undo and Open buttons (Open deep-links to the pane via the claim id; Undo calls the existing forget path). Pending-approval claims show Approve inline. Must keep the transcript non-destructive rule; this adds a UI row only. Needs localization keys through the xcstrings parity gates. Skip the topic profile (claims have no topic field).

## model-settings

- Component: run-settings sidebar, quick header popover, presets with tone names hiding the numbers.
- Existing: Diagnostics/GenerationSamplingSection.swift (364 lines: max tokens, temperature, top-k, top-p, repetition penalty, scopes, saved named presets via AppSamplingPreset), Diagnostics/InspectorView.swift (inspector sidebar), Chrome/ModelLoaderControl.swift, Components/PersonalitySettingsSection.swift.
- Gap: sidebar and presets exist. Missing: a one-click quick control near the chat (header or composer footer) and tone-style presets (Precise / Balanced / Creative) as the default entry, with numbers collapsed.
- Verdict: IMPROVE, value M, effort S.
- Approach: add a popover button in Generation/PromptComposerFooter.swift (or top bar) listing the existing model.samplingPresets plus 3 built-in tone presets and a "Customize..." link to the inspector section. Reuse applyPreset(). Built-in tone values must be validated per family with real runs, since a wrong temperature on a small model produces degenerate output; ship them as data, not code, and say they are unmeasured. Maps to Chip/segmented control + Button.

## retrieval-inspector

- Component: for a test query, shows ranked passages, scores, and which entered context; pipeline trace.
- Existing: nothing direct. Syntext context engine (docs/SYNTEXT_CONTEXT.md, conversation recall) and Components/ProjectFileIndex exist, and Generation/SourcesPopoverView.swift shows web sources only (State/AppChatTurnSources.swift). Generation/ContextUsageRingView.swift shows window composition by component and auto-compact point.
- Gap: users cannot see why recalled snippets or file context were injected, what scored, or what was dropped to fit the window. This is a real debugging surface for an on-device agent and maps to local resources (context, KV).
- Verdict: ADOPT (variant "ranked passages", skip the pipeline trace), value M, effort M.
- Approach: extend TurnSources with a retrieval group (kind: recall, file index, memory) carrying score, source chat/file, and an "included / trimmed" flag; render in an extended SourcesPopoverView (or a "Context" tab next to the usage ring popover) as rows: score bar, snippet (2 lines), source, state Chip. Requires the recall path to report scores and the trim decision, which needs a Rust/FFI or Swift change in the context engine: check what Syntext already exposes before committing to M. Maps to Card, Chip, StatusDot. Keep a Diagnostics-style, off-by-default feel (trajectory log is plaintext and local).

## secrets-manager

- Component: API keys table, masked, per environment scope, plus request-a-secret-mid-run.
- Existing: Server/ServerKeychain.swift, TypeSafeKeychain.swift, Components/HfAuthTokenCardView.swift, RemoteProviderSettingsPaneView.swift, Components/ImageSearchSettingsSectionView.swift, profile vault (ProfileUnlockView, AppModel+ProfileVault).
- Verdict: SKIP. Per CLAUDE.md, credentials are not stored by default and Keychain use is opt-in per credential and never at launch for an unused feature. A central secrets table that lists and re-reveals values cuts against that and would be an auth/crypto change (ask first). The "requested in a run" variant (agent needs a token, user pastes it, agent never sees it) is interesting but needs a tool-layer redesign: not a UI port.

## source-ingest

- Component: file drop with per-file stages (reading, splitting, indexing, ready, failed with retry label), and website crawl with include/exclude rules.
- Existing: Files/AttachmentImporter.swift, DocumentTextExtractor (Presentation), Files/ProjectFileIndex.swift, Components/EmbeddingModelPicker.swift, Server/EmbeddingsAPIPaneView.swift, Browser/ (browser pane), FetchedSourceSaveBar.swift.
- Gap: the app has attachment and index machinery but, from what was searched, no staged per-file progress list with a failure reason and fix-it action (for example "scanned PDF, retry with OCR"). Website crawl: web fetch exists as a tool, a rules-based crawler is out of scope.
- Verdict: IMPROVE (file stages only), value M, effort M. SKIP the crawl variant.
- Approach: an Ingest list view in Files/ with a 3-segment progress bar per row (ProgressView per stage or a custom Capsule row using theme tokens), driven by an @Published array on the importer/indexer: stage enum, progress, error, retryLabel. Add Cancel and Retry. Drag-drop via .onDrop/dropDestination. Maps to Card list rows, StatusDot, Button. Needs the indexer to report stages; verify that hook exists before sizing.

## step-inspector

- Component: settings for one workflow step beside a canvas, plus last-run input/output/timing.
- Existing: Workflows/ is a script language (WorkflowLexer, Interpreter, ScriptChecker, RunEngine, Journal). No canvas or node UI found.
- Verdict: SKIP. No step selection model to hang an inspector on. The Last-run variant (input, output, duration per step) could map to WorkflowJournal if a run viewer is ever built, but that is a new feature, not a component mimic.

## step-picker

- Component: palette to add triggers/actions/logic/agents to a workflow canvas.
- Existing: none (workflows are written as scripts).
- Verdict: SKIP, same reason as step-inspector.

## tool-picker

- Component: tools grouped by app with permission per tool; composer quick menu; permissions table.
- Existing: Components/PermissionsSettingsPaneView.swift and PermissionsInfoSectionsView.swift (settings side), Tools/Registry/AppToolCatalog.swift, AppToolCatalog+Mcp.swift, AppToolCatalog+Profiles.swift, AppToolRegistry+ToolSearch.swift, Generation/PromptComposerPlusMenu.swift (files, folders, slash commands, "Add files, folders, and tools"), AppAgentDefinition.tools/disallowedTools.
- Gap: strongest candidate. The composer menu covers attachments and slash commands. I did not find (not exhaustively verified) a per-chat toggle list of tools grouped by source (built-in, each MCP server, plugins) with allow/ask/deny state. For a small local model, every enabled tool costs prompt tokens, so a quick "disable tools for this chat" control has a measurable benefit, not just a UX one.
- Verdict: IMPROVE, value H, effort M.
- Approach: new popover from PromptComposerPlusMenu (or a Tools chip in PromptComposerFooter): DisclosureGroup per source from AppToolCatalog, toggle per tool or per group, a permission glyph (allow, ask, deny) from the existing permission gate, and a footer "N tools, ~T tokens in prompt" reading the catalog's schema size. Store per-chat overrides on AppChat and feed them into the registry filter used by native tool calling (AppModel+NativeToolCalling, AppModel+Gating). Maps to Composer + Chip + Toggle. Respect the permission model: the picker may only narrow, never grant beyond settings. Localization parity gates apply.

## trigger-list

- Component: everything that starts an agent with last-fired and next-run, plus recent firings log.
- Existing: Components/CronJobsSettingsPaneView.swift (234 lines: jobs with cron string, "next run", Paused state, pause and delete), Workflows/WorkflowScheduler.swift, Tools/Automation.
- Gap: has next run and pause; lacks last fired, last result (ok / failed / skipped because app was closed), and a short firing history. The copy says tasks only fire while the app is running, so a missed-run indicator is useful.
- Verdict: IMPROVE, value L, effort S.
- Approach: add a secondary line per row "Last ran 09:00, ok" plus an expandable list of the last 5 firings with StatusDot; persist via the job record in the existing store (check AppJSONStore quarantine rules for new fields; make them optional for backward decoding). Keep the existing EmptyState. Skip the multi-source (webhook/app event) merge since those source types do not exist.

## Cross-cutting notes

- Cheapest wins in order: tool-picker (H), sampling quick popover, instructions token meter, memory in-chat note.
- Do not port: anything web-only (cloud connectors, event webhooks, secrets vault, workflow canvas). The visual style of these components is Tailwind/indigo; none of it transfers, only the interaction patterns do. Use theme accessors (.appSuccess, .appWarning, .appDanger, .appAccentText, .appBorder), not literals, and run make design-check after any UI change.
- Every new string needs the xcstrings catalog and the ten parity gates; use Text(..., bundle: .module) or AppStrings.localized.
- Not verified: whether MCP runtime status, indexer stage progress, and recall scores are already exposed as observable state. Three of the IMPROVE/ADOPT items (mcp-servers popover, source-ingest, retrieval-inspector) depend on that and could grow from S/M to L.
