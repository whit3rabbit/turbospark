# Component audit: designer-mcp AI components vs TurboSpark chat/agent

## START HERE (instructions for a fresh session with no context)

You are picking up a finished audit. Do not redo the audit. The goal of the new session is
to IMPLEMENT every ADOPT and IMPROVE item below, in the tier order given, in the native
SwiftUI macOS app under `swift/TurboSparkApp/`. The owner approved doing all suggestions.

### 0. What this is

- The "designer-mcp" project publishes a library of 109 AI UI components (chat, agent
  activity, artifacts, usage, voice, and so on) as React/shadcn TSX. TurboSpark's app is
  native SwiftUI, so nothing is copied. You port the behavior and information design.
- The library is browsable at
  https://whit3rabbit.github.io/designer-mcp-docker/components/index.html (open it with a
  browser tool or the `web-fetch` agent; it is a static page with a preview per component).
- The same library is served locally by an MCP server named `designer`. Its tools appear
  as `mcp__designer__*`. They may be deferred: load their schemas first with ToolSearch,
  for example `select:mcp__designer__design_component_search,mcp__designer__design_component_get,mcp__designer__design_component_code`.
  If no `mcp__designer__*` tool exists, the server is not running; say so and fall back to
  the web preview above. Do not invent component source.

### 1. How to read a component (MCP)

| Tool | Use | Warning |
|---|---|---|
| `mcp__designer__design_component_search` | List or filter components. Args: `query`, `category`, `limit` (max 100). Categories: `agent activity`, `agent setup`, `chat`, `media generation`, `output & artifacts`, `usage & limits`, `voice`, `writing`. | Output for all 109 is about 73 KB; use `category` or `query` |
| `mcp__designer__design_component_get` | One component: description, install command, FAQ, every variant's source | Can return 20k to 90k tokens. Call it for ONE slug at a time, ideally inside a subagent |
| `mcp__designer__design_component_code` | Source of ONE variant. Args: `slug`, `variant` (id or name; defaults to first) | Prefer this to `design_component_get` when you only need the layout and states |

Slugs are kebab-case, for example `message-error`, `tool-picker`, `plan`. Variant ids look
like `agent-card-header`. Read the source for structure, states, copy and edge cases, then
rebuild in SwiftUI. Ignore Tailwind classes, Radix and Framer Motion specifics.

### 2. The skill (optional, design methodology)

- Skill name: `designer-mcp-skill:designer-mcp`. Invoke it with the Skill tool
  (`/designer-mcp-skill:designer-mcp`). It documents the whole `design_*` tool family
  (styles, design systems, presets, templates) and a research-first workflow.
- You do NOT need it for this task: the component list and verdicts are already decided
  here. Use it only if you need extra tools (`design_preset_search`, `design_docs_search`)
  or the exact tool details (its `references/mcp-tools.md`).
- In THIS repo the design authority is `DESIGN.md` and `docs/design-system/`, not the
  skill's styles. If they conflict, the repo wins.

### 3. Read these repo files before writing any Swift (in order)

1. `CLAUDE.md` / `AGENTS.md` (root) and `swift/AGENTS.md`: rules, verification, safety.
   Also `CLAUDE.local.md` for machine notes.
2. `DESIGN.md` then `docs/design-system/README.md` and the per-control guide under
   `docs/design-system/components/<Control>/README.md` (Button, Card, Chip, Composer,
   EmptyState, IconButton, PillPicker, ProgressBar, SearchField, SectionHeader, StatusDot,
   StatusLabel, and others). Mandatory before any macOS UI change.
3. The per-component report for the item you are building, in
   `docs/component-audit/` (see "Source reports" at the bottom). It names the existing
   Swift file, the gap, the approach and the design-system mapping.
4. `docs/TESTING.md` before adding tests.

### 4. Tools and rules you must follow

- Search with `st` (ripgrep syntax), not `grep`/`rg`. Run `st index` if stale.
- Use the `Edit` tool for changes; keep each change small and reversible.
- Do not touch files already modified in `git status` that you did not change (at audit
  time: Rust GGUF files in `crates/catalog` and `crates/repack`, and `docs/GGUF.md`;
  another session owns them). Run `git status` before editing and stage only your own paths. Read `git diff` of
  any shared file (`CHANGELOG.md`, `*.xcstrings`) before staging. Use
  `git diff --patience` for `.xcstrings`.
- ASCII only in source, comments and docs. No emojis, no em dashes.
- Every user-visible string uses bundled `Text` or `AppStrings.localized`.
- Colors come from theme styles (`.appSuccess`, `.appWarning`, `.appDanger`,
  `.appAccentText`, `.appBorder`), never system colors or literals.
- Never trigger macOS permission prompts from a picker; credentials stay opt-in.
- Ask the owner first (per CLAUDE.md autonomy rules) before: edit-before-run, checkpoint,
  per-hunk accept/reject in approval flows, mcp-servers env masking, anything touching
  auth, crypto, Keychain, access control, dependencies, CI or data migrations. Each
  is marked ASK below. The "do all suggestions" approval does not replace this check.
- Never commit unless the owner asks.

### 5. Per-item workflow (repeat for each component, in tier order)

1. Read the item's report in `docs/component-audit/`. Verify its "gap" claims with `st`
   and by reading the cited Swift files (the audit did not build or run anything).
2. Read the component source only if the report is thin: `design_component_code` for one
   variant, preferably in a subagent so the large output stays out of your context.
3. Write a failing unit test for the logic (parsers, state, formatting), then implement.
   Keep parsing and decisions in pure functions that return nil when unsure.
4. Build the view from design-system controls and tokens. Add VoiceOver labels, keyboard
   defaults, Reduce Motion handling.
5. Verify: `make swift-lib` then `make swift-test` for the touched area, and
   `make design-check` (about 1 s) after every UI change; `make design-check-full` before
   finishing. For visual confirmation use `make swift-app` and launch with an isolated
   state dir (see the memory note on isolated app launch if present). If a build cannot run
   (disk, toolchain), say so; never claim UI verified from a compile alone.
6. Report: files changed, checks run, what you could not verify. One item per change.

### 6. Scope guard

"All suggestions" means all ADOPT and IMPROVE rows in the tier tables and the quick-win
list. SKIP rows stay skipped; each has a reason in "Skipped, and why". Work tier 1 first.
Conditional items (before-after, image-prompt-composer reference images) wait for their
stated condition (before-after is still waiting as of 2026-11-20). Do not widen scope to new engine features (inpainting, guardrail
signals) that the audit found missing.

---

Scope: all 109 components in the designer-mcp AI component library (React, shadcn-style
registry exports; 8 categories) were compared with the existing SwiftUI chat and agent UI.
Six audits ran in parallel, one per category group. Per-component detail (existing file,
gap, approach) is in the scratchpad files listed at the end.

Lens: TurboSpark is a local on-device inference app. Cloud billing, SaaS integrations,
multi-user features and anything that fetches third-party pages at render time were
skipped unless they map to a local resource (memory, context, KV, thermal).

## Result

Verdicts are ADOPT (new), IMPROVE (extend existing) or SKIP. Roughly 9 are ADOPT, 40 or so
IMPROVE, and the remainder SKIP (counts not formally tallied).

The app already covers most of the library. The wins are in failure handling, approvals,
and information the engine already has but the transcript does not show.

## Caveats on evidence

- Several audits read the full component source only for a few slugs. The rest rest on
  the registry description plus a search of the Swift code. The chat-A, chat-B and misc
  audits say so explicitly.
- Unverified gap claims (check before building): plan approval flow, resolved tool-approval
  record, whether terminal output streams before a command finishes, whether
  `ResponseMarkdownRenderer` already holds back partial markers, whether a composer-level
  reasoning-effort control exists, whether MCP runtime status, indexer stage progress and
  recall scores are already observable state, and whether artifact snapshots exist.
- Nothing was built or run. Design-system mapping was by control name from
  `docs/design-system/` folder layout; read the per-control guide before each build.

## Prioritized plan

Tiers order by value over effort, then by dependency. Items marked ASK touch security,
credentials or approval semantics and need a go-ahead per CLAUDE.md before work starts.

### Tier 1: do first (high value, small or medium effort)

| # | Component | Verdict | Value/Effort | Why now |
|---|---|---|---|---|
| 1 | message-error | IMPROVE | H/M | Failures only reach the global `ErrorBanner`; `stopReason` is stored but never rendered |
| 2 | tool-picker | IMPROVE | H/M | Per-chat tool toggles with permission state and prompt-token cost |
| 3 | plan | IMPROVE | H/M | `ExitPlanMode` is a text block; needs Start / Request changes gate |
| 4 | tool-approval | IMPROVE | M/S | Resolved one-line record, cwd, model reason; no semantics change |
| 5 | context-meter | IMPROVE | M/S | Add Compact / Microcompact / New chat actions to existing ring popover |
| 6 | terminal | IMPROVE | M/S | 3-line live tail and one-line result summary |
| 7 | file-edit | IMPROVE | M/S | Collapsed 6-line diff peek; live tail while streaming |
| 8 | mic-picker | IMPROVE | M/S | Pre-record level meter, Voice Call device picker, persist device UID |
| 9 | model-unavailable | IMPROVE | M/S | Typed load-failure strip modeled on `StallRecoveryBannerView` |
| 10 | quota-banner (as resource strip) | IMPROVE | M/S | Memory, thermal, context strip above composer; today only in status bar |

### Tier 2: next (medium value or medium effort)

| # | Component | Verdict | Value/Effort | Notes |
|---|---|---|---|---|
| 11 | test-results | ADOPT | M/M | Parse cargo and swift test output; parsers return nil when unsure |
| 12 | file-tree | ADOPT | M/S | Collapsed Card of files a turn touched, from tool calls |
| 13 | scroll-rail | ADOPT | M/M | Tick rail with hover preview and jump-to-turn |
| 14 | inline-citation | ADOPT | M/M | `[n]` markers validated against source count |
| 15 | edit-before-run | ADOPT | M/M | ASK. Edit command then Allow; re-run risk classification; once-only |
| 16 | run-recovery | IMPROVE | M/M | Retry / Skip / Edit-and-retry on failed tool call |
| 17 | retrieval-inspector | ADOPT/IMPROVE | M/M | Extend `TurnSources` and `SourcesPopoverView`; needs score and trim data |
| 18 | attachments-tray | IMPROVE | M/S | Per-chip token estimate and context-budget warning |
| 19 | reasoning | IMPROVE | M/S | Stored "Thought for N s" and 72pt fade-folded rail preview |
| 20 | schedule-card | IMPROVE | M/S | Plain-language schedule plus next 3 runs for `CronCreate` |
| 21 | model-picker | IMPROVE | M/M | Searchable popover, local/remote grouping, size and fit facts |
| 22 | background-tasks | IMPROVE | M/S | Merge agent and shell strips into one status-bar tray |
| 23 | memory | IMPROVE | M/S | Inline "saved to memory" row with Undo and Open |
| 24 | model-settings | IMPROVE | M/S | Composer-footer popover with Precise / Balanced / Creative presets |
| 25 | instructions-editor | IMPROVE | M/S | Real-tokenizer token and context caption in system prompt editors |
| 26 | artifact-card | IMPROVE | M/S | In-flight "writing" state and revision badge |
| 27 | empty-state | IMPROVE | M/M | Greeting, resume unfinished work, project home |
| 28 | chart | IMPROVE | M/M | Native Swift Charts for bar, line, donut; offline, theme colors |

### Tier 3: later or conditional

| Component | Verdict | Value/Effort | Condition |
|---|---|---|---|
| checkpoint | ADOPT | H/L | ASK. Needs per-turn file snapshot store plus Restore divider |
| diff (per-hunk accept/reject) | IMPROVE | M/M-L | ASK for approval flow; start with worktree side |
| artifact-panel (history) | IMPROVE | M/L | Needs snapshot storage |
| image-generation | IMPROVE | M/M | Blur-to-sharp progress, in-place Regenerate |
| image-results | IMPROVE | M/M | Vary, batch pick row, hover actions |
| before-after | ADOPT | M/S | Only once an image pair (img2img, upscale) exists. Status 2026-11-20: deferred, condition not met (the app has no img2img or upscale path, and adding one is engine scope) |
| prose-diff | IMPROVE | M/M | Per-change accept/reject in `HumanizeDiff`, display-only |
| selection-toolbar | ADOPT | M/S as quote/context actions | SwiftUI `Text` hides selection range; full floating toolbar is L, skip |
| mentions-menu | IMPROVE | M/S then L | Kind grouping and tint first; inline pills deferred |
| point-and-ask | IMPROVE | M/M | Ask-pin tool; gate on vision models |
| image-prompt-composer | IMPROVE | M/L | Local-LLM prompt enhance first; reference images blocked on engine |
| knowledge-sources | IMPROVE | M/M | Per-turn scope popover with token costs |
| mcp-servers | IMPROVE | M/M | ASK for env masking; verify MCP runtime status is observable |
| source-ingest | IMPROVE | M/M | File ingest staging and retry only; no website crawl |

### Low value quick wins (batch with nearby work)

agent-card (chat header strip), trigger-list (last fired and result), question (1-9 keys,
Other row, Esc), code-block (file title, marked lines), message-actions (keep last bar
visible), chain-of-thought (summary header), subagent-tree (collapse finished by default),
slash-command-menu (mode chips), streaming-text (hold back dangling markdown markers only),
text-shimmer (extract `.appShimmer`, honor Reduce Motion), thread-header (title collapse),
source-viewer (open file at returned line range), suggestion-chips (opt-in templates only).

## Integration into the design system

DESIGN.md decides rules; Swift theme code decides values. New views must use existing
controls and tokens (no system colors or literals; `make design-check` ratchets this).

| Component | Design-system control(s) | Tokens and rules |
|---|---|---|
| message-error, model-unavailable, resource strip | `StatusLabel`, `StatusDot`, `Button`, `IconButton` | `.appDanger` / `.appWarning` text styles; never color alone, pair with icon and text |
| plan, edit-before-run, tool-approval, run-recovery | `Card`, `Button` (primary + secondary), `SectionHeader` | Keyboard default action; Esc cancels; VoiceOver labels and focus order |
| context-meter, attachments-tray, retrieval-inspector, tool-picker | `Chip`, `ProgressBar`, `Card` | Token counts as secondary text; popover uses existing `ContextUsagePopoverView` pattern |
| terminal, file-edit, test-results, file-tree | `Card`, `ProgressBar`, `StatusDot` | Mono type from the type scale; collapsed by default; expandable |
| scroll-rail, reasoning, text-shimmer | none new; small custom view | Reduce Motion honored; pause animation when window not key |
| model-picker, mic-picker, model-settings | `PillPicker`, `SearchField`, `Composer` footer | Searchable popover; persist selection |
| empty-state | `EmptyState`, `Chip` | Replace bare `PromptExamplesView` grid |
| checkpoint, schedule-card, memory | `Card`, `Button`, `SectionHeader` | Restore divider is a transcript row, not a modal |

Rules that apply to every item:

- Read `DESIGN.md` and the matching `docs/design-system/components/<Control>/` guide before
  writing a view.
- Every string uses bundled `Text` or `AppStrings.localized`; ten localization parity gates
  apply. Use `--patience` when diffing `.xcstrings`.
- Run `make design-check` (about a second) per change and `make design-check-full` before
  merging; a count below its baseline in `DesignRules.json` must be lowered in the same change.
- macOS privacy prompts appear only after the user chooses Allow (mic-picker must not
  trigger the prompt). Credentials stay opt-in (secrets-manager skipped for this reason).

## Implementing React/shadcn patterns in SwiftUI

The registry components are TSX with Tailwind and Radix primitives. Port the behavior and
information design, not the code.

| React/shadcn idiom | SwiftUI equivalent in this app |
|---|---|
| `Card`, `Badge`, `Alert` | Design-system `Card`, `Chip`, `StatusLabel` |
| `Popover`, `HoverCard` | `.popover(isPresented:)`, as in `ContextUsagePopoverView` |
| `Collapsible`, `Accordion` | `DisclosureGroup` or a custom disclosure row; persist expansion in view state |
| `Tooltip` | `.help(...)` plus accessibility label (tooltip is not enough for VoiceOver) |
| `Command` (cmdk) | Existing `/` and `@` menus in the Prompt composer files; add grouping |
| `Progress`, skeleton | `ProgressBar`; determinate prefill progress already in `ChatActivityLine` |
| `useState` / `useReducer` | `@State` for view-local; `@Observable` models for chat state |
| React context or store | `AppModel` and its `AppModel+*.swift` extensions |
| Streaming text via state updates | Existing streaming path in `ResponseMarkdownRenderer`; keep markdown parsing out of the hot path |
| Framer Motion | `withAnimation` / `.transition`; gate on `accessibilityReduceMotion` |
| Tailwind classes | Theme tokens (`.appSuccess`, `.appBorder`, type scale); no literal colors |
| ECharts / Chart.js | Swift Charts (`Chart`, `BarMark`, `LineMark`, `SectorMark`) |
| Monaco / diff viewers | Existing `WorktreeDiffView`; add hunk model, not a new editor |
| Pure presentational props | A `View` plus a small value-type view model; keep parsing (test output, citations) in unit-testable pure functions that return nil when unsure |

Testing: each parser or state change gets a unit test that exercises the behavior it claims;
mutation-check subtle guards (see CLAUDE.md). CI does not run the app `swift test` suite, so
run `make swift-test` locally.

## Skipped, and why (grouped)

- Cloud and billing: cost-meter, credit-top-up, usage-meter, quota-banner (as quota).
- Integrations not present: calendar-event, email-draft, connector-sync, event-trigger,
  itinerary, product-card, link-preview (fetches third-party pages; conflicts with opt-in network).
- Already covered by a stronger native view: audio-player, branch-picker, conversation,
  edit-message, message, loader, data-table, context-breakdown, live-transcript,
  version-history, session-tabs, sources-list, thread-list, panel-dock, prompt-input,
  agent-inbox, browser-session, todo-list, tool-call, step-timeline, decision-inbox.
- No matching backend: edit-region (no inpaint), guardrail-notice (no withheld signal),
  research-progress, ghost-completion (contends with chat decode), inline-prompt.
- Wrong product shape: agent-thread (multi-user), assistant-launcher, step-inspector,
  step-picker (workflows are scripts), generative-form (small models fail richer schemas),
  inbox-views, agent-create (describe-it generator unreliable on small local models).
- Policy: secrets-manager (conflicts with opt-in Keychain rule).

## Source reports

Per-component detail (existing Swift file, gap, verdict, approach, design-system mapping),
one section per slug with a verdict table at the top of each file:

- agent setup: `docs/component-audit/agent-setup.md`
- agent activity: `docs/component-audit/agent-activity.md`
- output and artifacts: `docs/component-audit/output-artifacts.md`
- chat, first half: `docs/component-audit/chat-a.md`
- chat, second half: `docs/component-audit/chat-b.md`
- media, usage, writing, voice: `docs/component-audit/misc.md`

These were machine-generated by subagents and not independently reviewed; treat file
paths and "gap" claims as leads to verify.
