# Stage 5: Zoom Out and Inventive Catalyst report (OC-09)

Target: docs/designs/opencode2-parity/OC-09-sources-panel.md (full file, 156 lines)
Context: INDEX.md, OPENCODE2_COMPARISON.md, WebSearchExecutor.swift, WebTools.swift, ToolPresentation.swift.
Reviewers: egm-zoom-out (completed), egm-inventive-catalyst (completed). Both read persona and standard.

## Zoom Out findings (egm-zoom-out)

### Major

ZO-1. Lines 38-39, 47-48, 92-108, 126-131, 142-143. The stored collector and new message metadata are redundant. `AppChatMessage` (State/AppChat.swift) already archives `toolCalls` and `toolResults`; a pure function can derive title, URL and domain at render time, with dedupe and cap. That removes TurnSourceCollector, the turn-completion hook, the new Codable field, the write-cost paragraph and the CUJ 3 round-trip test. The doc never says tool calls and results are already archived. Caveat: `promptProjection`, `observation` and `archivalOutput` may move oversized output out of the archived `output`; check truncation cannot drop URLs.

ZO-2. Lines 96, 135. "Structured result payload (already exists)" is unverified and contradicts item 5 ("if it is not already typed"). `WebSearchOutput` and `WebFetchOutput` are typed, but the stored `AppToolResult.output` is a markdown string. `ToolPresentation.swift` only registers names, icons and aliases, the wrong layer. Name the extraction seam (parse the stored string, or add a typed field at the executor boundary in WebTools.swift) and drop item 5 unless reveal needs it.

ZO-3. Lines 40-41, 96-98. "The reply row that follows the turn" is underspecified: tool calls live on assistant messages across an agent loop. Which row owns the chip (final message or union across the turn), how a turn is delimited, and whether variants (`alternates`) and recovery-anchor rows count are not stated.

ZO-4. Lines 44-46, 56-58, CUJ 2. Row actions are not tied to the real browser entry point. `AppModel+Browser.swift` has `setBrowserPaneOpen` and `authorizeBrowserNavigation`; the latter may be agent-oriented and prompt for approval. Name the function or state that a user-initiated route is needed, otherwise the "no ladder change" non-goal cannot be checked.

ZO-5. Lines 44-46, 60-66. Scope creep in v1: three row actions plus domain grouping with counts exceed "see which sources a reply used". "Reveal in tool card" needs scroll and expand plumbing that is not designed. Decide popover now, default to one plain list, cut or specify reveal.

### Nit

ZO-6. Lines 126-143. Once the collector is dropped, `AppChatTurnSources.swift` is a small helper; check Components/ for reusable popover or list rows.
ZO-7. Lines 114-115. "Nothing extra leaves the machine" omits that opening a source makes a network request. Ghost chats (GhostChatVault) are not addressed.
ZO-8. Lines 60-66. Both open questions already carry a lean; read as decisions. Resolve or state what evidence flips them.
ZO-9. Lines 3-8. No one-line summary of what OpenCode's renderer does (a side tab per turn); INDEX says absence claims were not re-verified.

## Inventive Catalyst findings (egm-inventive-catalyst)

IC-1 (Major). Lines 92-104, 129-137. False compromise: the payload "already exists" but `AppToolResult` carries only `output: String`, `promptProjection`, an observation reference and media references; `formatMarkdown()` flattens search output and fetch output has no title. Item 5 targets the wrong file. Fix: add an optional typed `sources` field on `AppToolResult`, emitted by the web executors (tolerant decoding pattern already exists), derive the chip from the turn's results at render time, which removes separate reply metadata and resolves CUJ 3. Liabilities: memoize derived data; state whether old archives get no chip or a backfill.

IC-2 (Major). Lines 38-39, 64-66. "Source" is undefined: failed, blocked or redirected fetches (`WebFetchOutput.url` is post-redirect), pre- versus post-sanitize items (`WebSearchExecutor.sanitize` lines 633-690 drops credentials, private hosts, non-http), and the synthetic Tavily direct-answer item (lines 743-749, "Tavily Direct Answer", https://tavily.com) would appear as a real source. Fix: a source is a post-sanitize record from a successful result tagged search or fetch; flag and skip the Tavily item; record the final URL.

IC-3 (Nit). Lines 96-98, 129-130. Capture at turn completion and storage on reply metadata leaves interrupted, cancelled, regenerated turns and parallel tool-call ordering undefined. Derive on demand, or state the behavior.

IC-4 (Nit). Lines 45-46. "Reveal in tool card" needs a stored first-seen call id; dedupe makes the target ambiguous. Store it or cut the action.

IC-5 (Nit). Lines 106-108. The write-cost claim is unmeasured; acceptable, do not add a global per-chat cap before needed.

No false compromise found in: popover over side panel, privacy and offline stance, reuse of browser-pane policy, Non-Goals.
Note: the reviewer could not read the one-line Section 4 entry of OPENCODE2_COMPARISON.md (tool truncated it) and did not verify against it.
