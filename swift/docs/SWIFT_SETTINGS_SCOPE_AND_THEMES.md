# Settings scope and theme audit, 2026-09-10

Companion to [SWIFT_SETTINGS_AUDIT.md](SWIFT_SETTINGS_AUDIT.md). Source
inspection and automated tests are distinct from visual evidence below.

## Scope and lifecycle contracts

| Surface | Binding and persistence | Consumer and application | Validation / reset |
|---|---|---|---|
| MCP User | `globalMcpServers`, MCP archive under active profile | `AppToolCatalogMcp.resolvedServers`, catalog, permission engine, execution and resources | Existing name collision checks and approval remain; removal deletes this configuration |
| MCP Project | `AppProject.mcpServers`, `enabledMcpServers[server UUID]`, project archive | Same resolver; inherited server override precedes enabled filtering | Nil override restores user default; disabled global names cannot fall through to a duplicate project name |
| Skills User | `DisabledItemStore`, TurboSpark-owned skill directories | `SkillManager.computeEffectiveSkills`, prompts and skill execution | External files cannot be saved or deleted; Copy creates an owned skill; delete removes only that installation path from the ledger |
| Skills Project | Canonical root plus skill name for legacy store keys; `AppProject.enabledSkills` for inherited overrides | Project-over-user name precedence, then project enablement | Nil restores inherited state; old disabled names remain fallback defaults |
| Plugins User | Profile plugin ledger and `pluginEnableState` | `PluginManager` resolves only installations in scope; tools, skills, agents and hooks invalidate together | Enablement and installation are separate; uninstall removes only selected ledger scope |
| Plugins Project | Ledger project path, `AppProject.enabledPlugins` and `localPluginPaths` | Captured project ID selects the editor's read and write target | Nil restores user default; local unregister leaves source directory intact |
| Marketplace sources | Existing user stores plus `AppProject.marketplaces.sources/hidden` by contribution kind | Source browser merges inherited user subscriptions and project additions | Removing an inherited source hides it locally; Restore removes the hide; installed items remain |
| Marketplace install | Explicit captured User/Project target, independent of source scope | Installation methods receive that scope before async work | Missing captured project errors; changing chats cannot redirect installation; failed ledger writes roll back caches and replacements |
| Plugin removal | `uninstallPluginID` invokes throwing manager action | Only matching records removed; shared cache survives other records | Final removal quarantines all cached versions, commits ledger, then removes quarantine; errors surfaced |

An in-flight tool call may finish. Later MCP calls resolve the live project
archive, so a tool disabled after prompt construction cannot start another
call through the old visible list. Plugin hooks retain their existing trust
requirements. External skills offer Reveal, Disable and Copy; plugin
contributions are managed through their owner.

## Persisted-control inventory

Controls sharing a binding are grouped explicitly. This table supplements
the historical field audit rather than treating informational text as a
setting. P = active profile, J = project archive, C = chat archive.

| Controls | Binding / storage | Runtime consumer | Timing / validation / reset |
|---|---|---|---|
| Appearance mode | `AppearanceManager.appearance`, P appearance.json | Scene preferred scheme and theme resolver | Immediate; System/Light/Dark enum; appearance reset |
| Theme gallery, saved themes, save/duplicate/rename/delete/import/export | `selectedThemeID`, `savedThemes`, paired configs, P appearance.json | `ResolvedAppTheme`, `AppThemeInjector`, semantic leaves | Immediate; version 1, nonempty name <=80, finite contrast 0...100, six-digit colors; invalid import is atomic; reset keeps library |
| Accent, background, foreground, contrast, translucent sidebar | `lightConfig` / `darkConfig`, P appearance.json | Resolved palette and sidebar/composer surfaces | Immediate per-mode edit; unmatched palette is Custom |
| UI/code font, weight and size; Text size | Appearance typography, P appearance.json | `AppFontDescriptor`, themed view modifiers and MarkdownUI | Immediate; paired theme choice preserves all typography |
| Pointer cursors, benchmarks, dock icon, motion, diff markers | Appearance preferences, P appearance.json | Cursor modifier, status bar, dock renderer, transitions, diff rendering | Immediate; appearance reset restores defaults |
| Language | Shared locale preference | Bundle localization / relaunch flow | Shared across profiles intentionally; catalog parity tests |
| Ghost startup, interaction mode, menu bar | `MacAppSettings` via General pane, P | Launch/chat creation/menu bar paths | Startup settings affect launch/new chats; ghost data remains transient |
| Auto-compact, retained turns | `autoCompactEnabled`, `compactionKeepRecentTurns`, P | `AppChatCompaction` | Next turn boundary; bounded retention; fixed summarizer sampling intentional |
| Memory enabled | `memoryEnabled`, P; memory content per project | `MemoryPromptBuilder` and memory tools | Next prompt; existing index and injection budgets remain |
| Profile create/rename/delete/switch | Profile registry | `UserProfileStore`, save-and-relaunch switch | Explicit lifecycle actions; model weights/language/server key remain shared |
| Keyboard shortcuts | `KeyboardShortcutCatalog` | Commands and accessibility labels | Read-only reference; tested against command definitions |
| Permission mode, rules, advisory veto | Project permissions or P defaults; `commandAdvisoryVeto` | `AppToolPermissionEngine`, CommandGate | Next call; hard gates preserved; removing rule affects only its owning scope |
| Agent mode hints | `agentModeHints`, P | AgentModeGate classifier routing | Next permission decision; existing fallback behavior retained |
| LM Studio discovery/path, custom model folders | `enableLMStudioDetection`, `lmStudioDirectory`, `customModelDirectories`, P | Model discovery and storage scanner | Rescan actions; no configurable catalog installation destination promised |
| HF endpoint / token | `hfEndpointInput`, P; token store | Endpoint resolver, model catalog and authenticated fetch | Explicit apply/save; shared endpoint resolver refreshes catalog |
| System prompt | `defaultSystemPrompt`, P; chat system override C | Prompt assembly | Next turn; chat override wins; empty means none |
| Temperature, top-K/P toggles and values, repetition penalty, seed, stops, token budget | Sampling fields P; `samplingOverride` C; `samplingPresets` P | `effectiveSamplingSettings` / GenerateOptions | Next turn; clamped sampling; Inspector scope picker selects App defaults or This chat |
| Reasoning | `setReasoning`, `modelReasoningDefaults`, P | Checkpoint capability/template resolution | Next turn; unsupported levels clamped by checkpoint; shared Engine/Server binding |
| Context, expert slots, power, rate cap | `maxContextTokens`, runtime options, P | `buildOpenOptions` | Model reload; automatic context and bounded memory resolution remain |
| Memory guard/custom bytes/automatic context floor | `loadGuard`, `loadGuardCustomBytes`, `minAutoContextTokens`, P | Model load admission | Model reload; Custom tier retained in Server picker |
| Speculation mode/drafter, KV width | `speculation`, `speculativeDrafter`, `kvBits`, P | Model open options and runtime capability resolver | Model reload; unsupported combinations reported by existing resolver |
| Steering enable/preset/path/mode/scale/layers/target/gate | Steering fields/presets P, raw knobs in Inspector | `resolvedSteeringPreset` and engine steering update | Existing apply/reload flow; preset selection and raw values share one resolver |
| Guardrails | `guardrailsMode` P plus project preference | Tool-call guardrail routing | Subsequent turn/call; existing checkpoint capability behavior |
| Fan persistence | `keepFansPinnedOnQuit`, P | Shutdown fan restoration | On quit; requires existing ThermalForge service |
| Server start/stop, port | `serverPinnedPort`, P; server runtime | In-process server | Start action; invalid port rejected; changing model stops server |
| Server API key, generate/copy | Keychain, shared | Server authentication | Explicit action; copy affects clipboard; no keychain mutation in tests |
| Server auto-start/background | `serverAutoStartOnLaunch`, `keepServerRunningInBackground`, P | Launch/termination service flow | Relaunch or window close; runtime server state shown separately |
| Embedding model | `serverEmbeddingModelInput`, P | Embedding endpoint session | Server/model setup action; route is implemented |
| Agents/subagents | Agent files and enable store | Agent discovery and dispatch | Subsequent invocation; unsupported model override control remains removed |
| Hooks | Hook config and plugin trust | HookManager event dispatch | Subsequent hook event; plugin enablement invalidates cache |
| Scheduled tasks | Cron job archive | Scheduler | Save/enable/remove actions; timestamps show current scheduler state |
| Project name/root/rules/tool settings | `AppProject`, J | Project selection, prompt assembly, execution containment | Save action; editor merges current archive to retain extension changes made in nested sheets |
| Inspector counters and storage paths | Runtime telemetry / computed URLs | Display only | Read-only; no persistence promised |

## Theme and rendering boundaries

Nine immutable built-ins have paired previews. A saved theme has a fresh,
stable identity; import never overwrites a built-in or another saved record.
Deleting the active saved theme leaves its colors as Custom. Old archives
retain their colors and fonts. Root background, text, elevated surfaces,
borders and selection resolve from the current environment. App-owned
Markdown links, quotes, tables and code containers use that same palette.

AppKit review: color pickers and file dialogs remain native. PDF and
QuickLook document content retains document styling. `ArtifactWebView`
loads user-authored HTML in its existing sandbox, so its document CSS is
independent. `ResponseMarkdownRenderer` is used by the exported transcript
document controller, not by live chat; its document colors remain separate.
Brand imagery and generated charts are not recolored.

## Search

`settingsControl` declarations live beside their targets. Run
`python3 scripts/generate-settings-catalog.py` after changing declarations;
`--check` detects drift. Results include pane context, navigate, scroll and
highlight the selected target. Search matches source labels and localized
labels. Color search opens Customize. The legacy pane keyword filter remains
as a supplementary category shortcut. Controls inside item editors require
selecting an item first; the source manifest is not a claim that every
conditional editor can be opened from a bare search result.

## Verification evidence

- `SettingsScopeTests`: actual AppModel project enablement, scoped uninstall,
  inherited MCP overrides, legacy skill keys, external-file protection,
  project-only plugin discovery, profile-root cache isolation and source errors.
- `ThemeLibraryTests`: paired selection, typography preservation, custom CRUD,
  malformed import atomicity, legacy archive, reset/library preservation and
  added-palette foreground contrast >=4.5:1.
- Seventeen applied mutations failed their focused tests: MCP override, skill ownership,
  canonical skill key, captured plugin preference, scoped uninstall, complete
  cache-family deletion, theme validation, reset preservation, Custom identity
  detection, inherited source hiding, failed-removal reporting, unreadable source
  archives, plugin discovery scope, installation ledger failure, skill path
  validation, scoped skill ledger cleanup and external-source symlink protection.
  Source restored after each mutation.
- Live isolated QA bundle: paired Sandstone selection, Light/Dark controls,
  theme gallery selection accessibility, custom-theme save and restart persistence,
  Temperature search navigation and highlight, MCP and Skills layout at Extra
  Large text, and the common User/Project selector listing two QA projects. These are observed UI checks,
  not evidence that every nested sheet has been visually inspected.
- Full Swift suite: 1,535 tests, one skipped, zero failures. The final targeted
  checks also cover the external-source symlink guard and localized labels.
- Localization catalog compiled for 21 languages; generated search index and
  whitespace checks passed.
- Required Rust build, formatting and clippy passed in this session. Workspace
  tests encountered an unrelated family-count expectation in
  `crates/runtime/src/real_forward_init.rs` while another session added a model
  family. No Rust inference or ABI implementation changed for this work.
