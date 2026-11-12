# Goldfish Annotated Review Report

Status: resolved

## Applied

* Applied (25): B1, B2, B3, M1, M2, M4, M5, M6, M8, M9, M10, M11, M12, M13, M14, M15, M16, N1, N2, N3, N5, N6, N7, N9, N11.
* Unapplied: M3 and M7 (ESCALATE, author scope decision), N4, N8, N10 (DECLINE).
* Note: the INDEX.md line 89 edit from M6 is outside the target file and was left to the author (recorded in the design doc Dependencies paragraph).

**Target:** `docs/designs/zcode-parity/ZP-22-remote-providers.md` (full file, 266 lines)
**Findings source:** `consolidated_review_report.md` in this folder (30 findings: 3 Blocker, 16 Major, 11 Nit; the consolidated header says 15 Major but lists M1 to M16).

## Summary

| Category | Count | Items |
| --- | --- | --- |
| CONTEST | 0 | none |
| ESCALATE | 2 | M3, M7 |
| PROPOSE FIX | 25 | B1, B2, B3, M1, M2, M4, M5, M6, M8, M9, M10, M11, M12, M13, M14, M15, M16, N1, N2, N3, N5, N6, N7, N9, N11 |
| DECLINE | 3 | N4, N8, N10 |

Security-relevant items (conservative edits, deny/ask by default): B2, M2, M9, M12, N11.

### Verified-false reviewer claims

None. Every code claim checked was true:

* `GenerationToolCall` is used in the local loop (`State/AppModel+Generation.swift:302`) and in `Tools/Core/ToolCallDispatchGate.swift`, `Tools/Core/ToolCallParser.swift`, `State/AppModel+NativeToolCalling.swift`. B1 holds.
* `SettingsPanes.swift` does not exist. Settings panes are split across `App/` (`GeneralSettingsPaneView.swift`, `EngineSettingsPaneView.swift`) and `Components/` (`ModelsSettingsPaneView.swift`, `McpSettingsPaneView.swift`). M11 holds.
* `Networking/` and `Generation/ModelStep.swift` do not exist. `Generation/` holds views (for example `PromptContextPills.swift`); the generation logic is in `State/AppModel+Generation.swift`. M4 and M11 hold.
* `State/ModelProviderDiscovery.swift` and `Components/ModelProviderDiscoverySheet.swift` exist, so the "Provider" name collision in M15 is real.
* `Tools/Web/HttpRequestExecutor.swift` exists and uses `HttpRequestRedirectDelegate`. M2 and M4 reuse suggestions are valid.
* Compaction measures tokens with `session.countTokens` (`swift/docs/SWIFT_COMPACTION.md` lines 33 to 39), a local-engine call. "Compaction works unchanged" is therefore unproven for remote chats. M5 holds.
* `Server/ServerKeychain.swift` (opt-in off for new installs, read failure degrades to no key), `Server/KeychainRead.swift`, `Tools/Hooks/HookOptionKeychain.swift` exist, so the "name one existing keychain wrapper" ask in B2 is satisfiable. `ZCODE_COMPARISON.md` 7.14, 7.15, 7.16 exist (lines 685 to 697).
* `swift/docs/SWIFT_LOCALIZATION.md` says 21 languages. N2 holds.

### ESCALATE items (need the author, listed first)

* **M3 (phasing)** and **M7 (alternatives)**: both turn on whether the author wants to narrow the adopted scope (single OpenAI-compatible family and chat-only first). That is a scope decision, not a defect I can settle from files. `INDEX.md` line 105 already says "verify the turn-loop seam first", which supports a phase split, but the author decided the two-family scope on 2026-11-11. No `.kiro/` spec is readable here either (`.kiro` is absent from this checkout).

## Prioritized Findings & AI Suggestions

### [Issue 1] [`ZO-1, IC-1, CO-1, DC-1, ST-7, S8-14, S8-21`] - Severity: Blocker (B1)
*   **Affected Scope:** [Lines 67-75, 138-149, 214-219](../../../designs/zcode-parity/ZP-22-remote-providers.md#L67-L75) (Open Questions, Technical Plan, Target Files item 4)
*   **Discrepancy Description:** The ModelStep seam is unverified, yet the doc commits to extracting it and to "downstream untouched"; the loop uses local `GenerationToolCall`.
*   **Rationale:** The design depends on an assumption the author deferred to the first implementation task.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Verified. `AppModel+Generation.swift:302` declares `var nativeToolCalls: [GenerationToolCall]` and feeds the result of `AppStreamRecovery.generateUntilComplete` (engine session, `session.info.maxContext`, `session.countTokens`) into the loop. `AppModel+AgentLoop.swift` consumes `GenerationResult` (`handleExtractedToolCall(... result: GenerationResult ...)`). The loop is therefore coupled to engine types, not to a provider-agnostic step. The Open Question's own fallback ("if FFI-coupled, land chat-only") is the likely outcome. I did not read the whole seam, so the exact event type list below is partly unconfirmed.
*   ***Proposed Edit:***
    *   Location: "Open Questions", first bullet (lines 69-75), and Technical Plan paragraph after the diagram (line 151).
    *   Change: Replace the first Open Question with a Decision paragraph in Technical Plan:
        "**Turn loop seam (verified 2026-11-11)**: the loop in `State/AppModel+Generation.swift` calls the engine through `AppStreamRecovery.generateUntilComplete` and carries engine-typed results (`GenerationResult`, `GenerationToolCall`). It is not provider-agnostic today. This design introduces `ModelStep` as a protocol returning engine-neutral step events (content delta, tool call, usage, finish, interrupt); `LocalModelStep` adapts the existing path and maps `GenerationToolCall` into the neutral type. Downstream behavior (steering, queues, ladder, compaction) is unchanged in behavior, but the call sites listed in `AppModel+Generation.swift` and `AppModel+AgentLoop.swift` change type. [Author to confirm the exact call-site list after a read of both files.]"
        Delete the Open Question bullet.

### [Issue 2] [`CO-2, ZO-4, DC-7, ST-4, SY-3, IC-7, S8-11, S8-24`] - Severity: Blocker (B2)
*   **Affected Scope:** [Lines 52-53, 93-96](../../../designs/zcode-parity/ZP-22-remote-providers.md#L93-L96) (Goals, CUJ 1)
*   **Discrepancy Description:** The "store in Keychain" toggle ("defaults on", "off means provider stays disabled") contradicts per-credential opt-in and leaves the key's fate undefined when off. No keychain helper is named.
*   **Rationale:** Credentials handling must be unambiguous.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Security item. Verified: the repo rule (`ServerKeychain.swift` header and `ZCODE_COMPARISON.md` 7.14) is opt-in per credential, never at launch, read failure degrades to "no key". A default-on toggle is the opposite of that rule. The conservative edit: saving the key is the explicit opt-in act, no toggle, key read only at first remote send, denied or failed read yields a named error and no send.
*   ***Proposed Edit:***
    *   Location: CUJ 1 "Then" bullet (lines 93-96) and Goals last bullet (lines 52-53).
    *   Change: Replace the parenthetical with: "the key is stored as a generic-password Keychain item when the user saves it; saving is the opt-in for that credential, and there is no separate toggle. Until saved, the provider is disabled. The key is read from the Keychain only at the first send to that provider in a session, never at launch. A denied or failed read sends nothing and shows a named error (for example 'Keychain access denied')." Add to Goals bullet: "Storage follows the `ServerKeychain` pattern (`Server/ServerKeychain.swift`, `Server/KeychainRead.swift`); see ZCODE_COMPARISON 7.14."

### [Issue 3] [`ST-1, DC-6, SY-9, S8-18`] - Severity: Blocker (B3)
*   **Affected Scope:** [Lines 21, 58, 162, 170, 179](../../../designs/zcode-parity/ZP-22-remote-providers.md#L158-L179) (Overview, Non-Goals, Alternatives)
*   **Discrepancy Description:** Every rejection reads "Rejected by author" with no link or technical reason.
*   **Rationale:** A reader cannot audit the decision.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** The decision is recorded in a readable file: `ZP-90-excluded-and-decision-findings.md` line 35 and line 162 (2026-11-11, 7.1 adopted as ZP-22, 7.2 excluded). The cited roadmap is under `.kiro/steering/`, which is absent here, so link ZP-90 rather than the roadmap. The Cons lists already carry technical reasons; the Reason lines should point at them.
*   ***Proposed Edit:***
    *   Location: Overview line 21, Non-Goals line 58, and the three "Reason for Rejection" lines (162, 170-172, 179).
    *   Change: State the decision once in Overview: "Adopted and scoped by the 2026-11-11 author decision recorded in [ZP-90](ZP-90-excluded-and-decision-findings.md) (7.1 adopted, 7.2 excluded)." Delete the repeated "Author decision 2026-11-11" text at line 58. Replace the reasons with: A, "Rejected: hosted identity contradicts local-first and ZP-90 excludes cloud OAuth permanently." B, "Rejected: adds a loopback hop and new FFI surface for a client-side concern; revisit if CLI parity needs it." C, "Rejected: silent selection breaks trust and reproducibility; see Non-Goals."

### [Issue 4] [`DC-2, ST-3, S8-20`] - Severity: Major (M1)
*   **Affected Scope:** [Whole file](../../../designs/zcode-parity/ZP-22-remote-providers.md#L181-L197) (missing section, insert after Non-Functional Requirements, line 195)
*   **Discrepancy Description:** No Risks and Mitigations section.
*   **Rationale:** Remote egress and an agent loop over untrusted API output carry real risks.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Valid; the doc has no such section. Drafted from the doc's own content and the findings. Text marked [confirm] is my invention and needs author confirmation.
*   ***Proposed Edit:***
    *   Location: new `## Risks and Mitigations` after line 195.
    *   Change:
        * **Provider API drift** (field or event changes): pin tested request shapes in SSE fixtures; unknown event types are ignored, not fatal. [confirm]
        * **Tool-result content leaves the machine**: tool output returned to a remote model is sent to the provider; the first-send notice says so, and tool approvals are unchanged.
        * **Stream truncated mid tool call**: partial tool JSON is discarded, the turn ends with the named "bad stream" error, no tool runs.
        * **Base URL misconfiguration or leakage**: https required (loopback excepted), Authorization stripped on cross-host redirect (see M2).
        * **Remote loop cost**: a runaway agent loop costs money; the usage ledger shows per-provider totals, and the existing step cap applies. [confirm that a step cap exists for remote chats]

### [Issue 5] [`IC-2, DC-3, CO-7, SY-11, S8-13`] - Severity: Major (M2)
*   **Affected Scope:** [Lines 52-53, 119-124, 181-187](../../../designs/zcode-parity/ZP-22-remote-providers.md#L181-L187) (Goals, CUJ 4, Non-Functional)
*   **Discrepancy Description:** Security asserted, not designed: base URL scheme, cross-host redirects, "key-shaped" undefined.
*   **Rationale:** A key can leak to an attacker-chosen or plaintext endpoint.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Security item. Verified `HttpRequestExecutor` uses `HttpRequestRedirectDelegate` and sets `Authorization` headers itself, so a reusable redirect and header policy exists. Conservative default: deny.
*   ***Proposed Edit:***
    *   Location: Non-Functional "Security and privacy" bullet (lines 183-187).
    *   Change: Append: "Base URLs must be https; plain http is accepted only for loopback hosts (localhost, 127.0.0.1, [::1]). Any other http URL is refused at save time. Redirects to a different host drop the Authorization and x-api-key headers (reuse `HttpRequestRedirectDelegate`) or are refused. Redaction patterns: header values for Authorization and x-api-key, and strings matching `sk-` and `sk-ant-` prefixes followed by 16 or more key characters. [confirm patterns and the minimum length]"

### [Issue 6] [`ZO-2, S8-22`] - Severity: Major (M3)
*   **Affected Scope:** [Lines 17-53, 197-250](../../../designs/zcode-parity/ZP-22-remote-providers.md#L17-L53) (Overview, Goals, Detailed Implementation Plan)
*   **Discrepancy Description:** No phasing; two clients, loop refactor, accounting and rollout in one landing.
*   **Rationale:** Large single landing, high risk.
*   ***AI Suggestion:*** ESCALATE
*   ***Justification:*** The reviewer is not wrong that the plan is large (25 files touched across 10 items), but narrowing scope reverses a decision the author made (two families, tools in scope, Open Question 1 unresolved). The evidence for the right cut lives with the author's intent, not in a readable file. INDEX line 105 supports "verify the seam first". Suggested option if the author agrees: Phase 1 OpenAI-compatible, chat-only, flat model list; Phase 2 Anthropic; Phase 3 tools, depending on B1. Needs the author's call.
*   ***Proposed Edit:***
    *   Location: new "Phasing" subsection after Goals, if accepted.
    *   Change: Not applied. Author decides.

### [Issue 7] [`ZO-3, DC-3, DC-10, S8-23`] - Severity: Major (M4)
*   **Affected Scope:** [Lines 206-213](../../../designs/zcode-parity/ZP-22-remote-providers.md#L206-L213) (Target Files items 2-3)
*   **Discrepancy Description:** Two bespoke SSE clients, new `Networking/` directory, no shared parser, no reuse of Tools/Web code.
*   **Rationale:** Duplicated transport and an undeclared new directory.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Verified: `Networking/` does not exist; `Tools/Web/HttpRequestExecutor.swift` does. One caveat: `HttpRequestExecutor` carries a private-address refusal (`refuseConnectedPrivateAddress`) meant for the web-fetch tool, which would block loopback mock servers in the manual test plan; the author must decide whether the remote client reuses only the redirect delegate or the whole executor. That part is unconfirmed.
*   ***Proposed Edit:***
    *   Location: items 2 and 3 (lines 206-213).
    *   Change: Add item "2a. `Networking/SSEParser.swift` (Create; `Networking/` is a new directory): shared line-oriented SSE parser used by both clients." Reword items 2 and 3 to "family client over the shared SSEParser; transport reuses `HttpRequestRedirectDelegate` from `Tools/Web/HttpRequestExecutor.swift`; the private-address refusal is not applied to user-configured provider hosts [confirm]."

### [Issue 8] [`ZO-5, DC-5, CO-9, SY-2, ST-12, IC-4, S8-3, S8-16, S8-25`] - Severity: Major (M5)
*   **Affected Scope:** [Lines 49-51, 76-80, 151, 231-234](../../../designs/zcode-parity/ZP-22-remote-providers.md#L76-L80) (Goals, Open Questions, Technical Plan, item 8)
*   **Discrepancy Description:** Context accounting is open while Goals assume it; "compaction works unchanged" is likely false; no overflow path.
*   **Rationale:** Local token counts drive compaction and window fitting.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Verified: `SWIFT_COMPACTION.md` says `measured` is `session.countTokens` over assembled history and `updateTokenEstimate` feeds the prompt builders. A remote chat has no local session to count. S8 advises against the calibration idea; resolve minimally with option (a).
*   ***Proposed Edit:***
    *   Location: Open Question 2 (lines 76-80) and line 151.
    *   Change: Close the question: "Decision: option (a). The ring shows the last API-reported usage; the pre-send estimate is characters divided by a fixed constant (4, labeled 'estimate' in the ring popover) [confirm constant]." Replace line 151 with: "Compaction for remote chats triggers on the API-reported watermark or the estimate against the model entry's context window, not on `session.countTokens`. A provider context-overflow error is a named error that offers compaction. The summarizing call runs on the chat's selected model. [confirm trigger threshold reuse from SWIFT_COMPACTION.md]"

### [Issue 9] [`ZO-6, DC-4, ST-8, IC-9, SY-14, CO-12, S8-17`] - Severity: Major (M6)
*   **Affected Scope:** [Lines 105, 148, 188](../../../designs/zcode-parity/ZP-22-remote-providers.md#L105-L107) plus `INDEX.md` lines 89-90
*   **Discrepancy Description:** ZP-03 and ZP-23 dependencies unstated; INDEX says "stands alone".
*   **Rationale:** Sequencing and ownership unclear.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Verified: ZP-03 describes `model-io-*.jsonl` rollouts, with Rust-side logging per `INDEX.md` line 31, so CUJ 2's rollout record depends on it. `INDEX.md` line 105 sequences ZP-22 last. The INDEX edit is outside this file; report it to the author.
*   ***Proposed Edit:***
    *   Location: new "Dependencies" paragraph after Non-Goals (line 65).
    *   Change: "**Dependencies**: the rollout-log entry in CUJ 2 needs ZP-03; without ZP-03 remote turns skip that step (soft dependency). ZP-22 does not block ZP-23 (concurrent chats); one in-flight request per chat holds either way." Separately, in `INDEX.md` line 89, replace "stands alone but touches" with "has a soft dependency on ZP-03 and touches". [confirm ZP-23 interaction]

### [Issue 10] [`DC-6, ST-2, ZO-9, SY-5, S8-6, S8-27`] - Severity: Major (M7)
*   **Affected Scope:** [Lines 155-179](../../../designs/zcode-parity/ZP-22-remote-providers.md#L155-L179) (Alternatives Considered)
*   **Discrepancy Description:** Alternatives are thin; real options missing (single OpenAI-compatible family first; size comparison for Alternative B).
*   **Rationale:** The author-rejected options are strawmen; the real choice is scope.
*   ***AI Suggestion:*** ESCALATE
*   ***Justification:*** The missing alternative ("single family first") is the same scope decision as M3. I cannot add it as a rejected option or as a chosen one without the author's call. The Alternative B size comparison needs numbers for Rust-side client cost that are not in any readable file.
*   ***Proposed Edit:***
    *   Location: Alternatives Considered.
    *   Change: Not applied. If M3 is accepted, add "Alternative D: OpenAI-compatible only first" as the chosen path.

### [Issue 11] [`ZO-8, ST-7, SY-4, CO-16, IC-10, S8-15`] - Severity: Major (M8)
*   **Affected Scope:** [Lines 67-84](../../../designs/zcode-parity/ZP-22-remote-providers.md#L81-L84) (Open Questions)
*   **Discrepancy Description:** Decided items sit under Open Questions (Test connection says "Include").
*   **Rationale:** Open Questions should be open.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Valid by reading the text: the bullet says "Include".
*   ***Proposed Edit:***
    *   Location: Goals (after line 53) and Open Questions bullet 3 (lines 81-84).
    *   Change: Add Goal: "A Test connection button per provider reports success or a named failure (auth, network, endpoint)." Reduce the Open Question to: "**Connection probe call**: models-list (free, but not served by every compatible endpoint) versus a one-token completion (costs tokens, universal). Lean models-list with a completion fallback. [confirm]"

### [Issue 12] [`IC-3, SY-6, SY-12, S8-4`] - Severity: Major (M9)
*   **Affected Scope:** [Lines 64-65, 151-153, 183-185](../../../designs/zcode-parity/ZP-22-remote-providers.md#L183-L187) (Non-Goals, Technical Plan, Security NFR)
*   **Discrepancy Description:** Background callers (compaction, titles, memory, cron) and mid-chat model switches can send history remotely; privacy notice content unspecified.
*   **Rationale:** Silent data egress, which the doc claims to prevent.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Security and privacy item. Verified background callers exist (`AppModel+TitleGeneration.swift`, `AppModel+MemoryScheduler.swift`, `AppModel+Microcompact.swift` in `State/`). Their model choice when a chat is remote is not stated. Conservative rule: deny by default, background callers use only the chat's selected model, and a notice precedes the first remote send.
*   ***Proposed Edit:***
    *   Location: Non-Goals (after line 65) and Security NFR (line 183).
    *   Change: Add Non-Goal: "No background feature (titles, memory, cron, microcompact) selects a remote model on its own; each uses the model the chat selected, and a feature with no chat context stays local." Replace the privacy sentence with: "The first remote send per provider per chat shows a notice naming the provider host and stating 'prompts and tool results leave this machine'; the send waits for confirmation. Switching a chat from local to remote mid-chat triggers the same notice." [confirm notice wording and whether confirmation blocks]

### [Issue 13] [`IC-5, CO-8, CO-14, SY-7, DC-8, S8-1, S8-10`] - Severity: Major (M10)
*   **Affected Scope:** [Lines 188-193](../../../designs/zcode-parity/ZP-22-remote-providers.md#L188-L193) (Performance, Reliability)
*   **Discrepancy Description:** Reliability has no values or mapping to local error handling.
*   **Rationale:** Untestable as written.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Valid. S8 advises against a retry subsystem; resolve with two timeouts and an explicit no-retry statement. Numeric values are set at implementation, per S8.
*   ***Proposed Edit:***
    *   Location: Reliability bullet (lines 191-193).
    *   Change: Append: "One first-byte timeout and one stall timeout (values fixed at implementation, stated in `SWIFT_REMOTE_PROVIDERS.md`). No automatic retry: every error ends the turn with its named error and the user resends. Each named error (auth, rate limit, overloaded, network, bad stream, context overflow) maps onto the local engine's error row type. [author to confirm the mapping target after reading the local error row code]"

### [Issue 14] [`ZO-7, CO-6, ST-10, SY-8, S8-19`] - Severity: Major (M11)
*   **Affected Scope:** [Lines 214, 220](../../../designs/zcode-parity/ZP-22-remote-providers.md#L214-L220) (Target Files items 4-5)
*   **Discrepancy Description:** Target files named by "basename"; `SettingsPanes.swift` does not exist.
*   **Rationale:** An implementer cannot find the files.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Verified (see summary). `State/AppModel+Generation.swift` and `State/AppModel+AgentLoop.swift` exist. Settings panes live in `App/` and `Components/`; the pane list is driven by `App/SettingsTab.swift` and `App/AppSettingsView.swift`.
*   ***Proposed Edit:***
    *   Location: items 4 and 5.
    *   Change: Item 4: "`Generation/ModelStep.swift` (Create) plus `State/AppModel+Generation.swift` and `State/AppModel+AgentLoop.swift` (Modify)". Item 5: "`App/SettingsTab.swift` and `App/AppSettingsView.swift` (Modify: add the Providers tab) plus `Components/ProviderSettingsPaneView.swift` (Create)". Consider placing `ModelStep.swift` under `State/` instead of `Generation/` (which holds views); [confirm].

### [Issue 15] [`ST-5, CO-11, SY-13, S8-7, S8-12`] - Severity: Major (M12)
*   **Affected Scope:** [Lines 117-124, 204-250](../../../designs/zcode-parity/ZP-22-remote-providers.md#L117-L124) (CUJ 4, Target Files)
*   **Discrepancy Description:** No owner for log redaction, settings export, Keychain code, ATS handling.
*   **Rationale:** Security duties without a file.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Security item. `ProfileExport` and `ProfileBackup` exist (`AppModel+ProfileExport.swift`, `AppModel+ProfileBackup.swift`), so an export path is real and must not carry the key; CUJ 4's "exports settings" is ambiguous. Conservative: key excluded from every export and backup by construction (it is never in the persisted shape).
*   ***Proposed Edit:***
    *   Location: item 1 (line 201-205) and CUJ 4 (lines 120-124).
    *   Change: Item 1 "Proposed Changes" append: "owns the Keychain read and write for provider keys and the redaction helper that the clients call before any log write; the redaction helper is the single egress point for provider logging." CUJ 4 "When": "the user inspects settings.json, a profile export or backup, or logs". CUJ 4 "Then": "none of them contains the key". [confirm ATS: https-only needs no exception; loopback http needs an ATS local-networking allowance, check `Info.plist`]

### [Issue 16] [`ST-6, DC-13, S8-5`] - Severity: Major (M13)
*   **Affected Scope:** [Lines 3-7, 33-53, 55-65](../../../designs/zcode-parity/ZP-22-remote-providers.md#L55-L65) (Non-Goals)
*   **Discrepancy Description:** Presets, provider JSON import, and opencode import are neither Goals nor Non-Goals.
*   **Rationale:** The cited ZCode parity targets include them.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Valid; the doc cites the ZCode provider JSON schema doc (lines 6-7) but never says whether the schema is adopted.
*   ***Proposed Edit:***
    *   Location: Non-Goals, add a bullet after line 61.
    *   Change: "No provider presets, no import of ZCode's provider JSON or another tool's provider config, and no settings sync; providers are entered by hand in this change. A later spec may add import. [confirm]"

### [Issue 17] [`IC-6, S8-2`] - Severity: Major (M14)
*   **Affected Scope:** [Lines 35-44, 204-213](../../../designs/zcode-parity/ZP-22-remote-providers.md#L35-L44) (Goals, Target Files)
*   **Discrepancy Description:** "OpenAI-compatible" is treated as one protocol; vision, reasoning, and usage field differences unaddressed.
*   **Rationale:** Compatible endpoints differ in practice.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Valid. S8 rejects a quirk-profile layer, so state it as a known limit.
*   ***Proposed Edit:***
    *   Location: Non-Goals, add a bullet.
    *   Change: "Known limit: 'OpenAI-compatible' endpoints differ in vision input, reasoning fields, and usage reporting. The client implements the common `/chat/completions` subset; a field a provider omits (for example usage) degrades to the estimate and is labeled. No per-vendor quirk profiles."

### [Issue 18] [`S8-29, S8-30`] - Severity: Major (M15)
*   **Affected Scope:** [Lines 3, 206-219, 224-227](../../../designs/zcode-parity/ZP-22-remote-providers.md#L3-L7) (header, names)
*   **Discrepancy Description:** Cites only 7.1/7.2, ignoring 7.14, 7.15 (picker contract), 7.16; "Provider" collides with `ModelProviderDiscovery.swift`.
*   **Rationale:** Naming collision and missing source links.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Verified: `State/ModelProviderDiscovery.swift` and `Components/ModelProviderDiscoverySheet.swift` exist, and `ZCODE_COMPARISON.md` lines 685, 691, 697 are 7.14, 7.15, 7.16. 7.16 says Test connection mirrors ZCode's `ModelConnectivityResult`; 7.15 describes the picker the Remote section must extend.
*   ***Proposed Edit:***
    *   Location: line 3 and names throughout.
    *   Change: Line 3: "Findings 7.1 and 7.2 (API-key half), with 7.14 (credential rules), 7.15 (composer picker), and 7.16 (connectivity checks) in [ZCODE_COMPARISON.md]." Rename `ProviderStore` to `RemoteProviderStore` (file `State/RemoteProviderStore.swift`) and `ProviderSettingsPaneView` to `RemoteProviderSettingsPaneView`; keep "Providers" as the user-visible tab name. [confirm names]

### [Issue 19] [`CO-3, CO-4, CO-5, CO-15, DC-9, SY-1, ST-11, S8-31`] - Severity: Major (M16)
*   **Affected Scope:** [Lines 3, 26-32, 52, 114, 140, 232-234](../../../designs/zcode-parity/ZP-22-remote-providers.md#L26-L32) (Problem, Goals, CUJ 3, diagram)
*   **Discrepancy Description:** Uncited claims and undefined terms (server "proves" both families; "repo rules"; "permission ladder"; 7.3/7.4).
*   **Rationale:** A fresh reader cannot check them.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Verified: `ZCODE_COMPARISON.md` section above 7.14 (line ~680) documents the server serving Anthropic and OpenAI-compatible shapes; "proves" overstates a serving-direction fact. The permission ladder is a repo term (`Tools/Guardrails/`, `PermissionsSettingsPaneView.swift`).
*   ***Proposed Edit:***
    *   Location: The Problem paragraph (lines 26-32), line 52, line 114.
    *   Change: Replace "the app's own server code proves the codebase already speaks both API families, but only in the serving direction" with "the app's server already serves both endpoint shapes (ZCODE_COMPARISON 7.13), but only in the serving direction." Line 52: "repo rules (root AGENTS.md and ZCODE_COMPARISON 7.14)". Line 114: add gloss "(the permission ladder: the app's tiered tool approval, deny or ask by default; see swift/docs/SWIFT_TOOLS.md [confirm anchor])". Line 59-61: "findings 7.3/7.4 (hosted billing and quota reset, excluded in ZP-90)". Note: the reviewer's 7.1/7.13 numbering for the server claim is from the consolidated report; I did not confirm that the server finding is numbered 7.13.

### [Issue 20] [`ST-9`] - Severity: Nit (N1)
*   **Affected Scope:** [Lines 7-86](../../../designs/zcode-parity/ZP-22-remote-providers.md#L24-L86) (headings)
*   **Discrepancy Description:** Headings "The Problem", "Technical Plan (Architecture)"; Open Questions precedes CUJs.
*   **Rationale:** Template conformance.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Cheap and only counted against `feature-spec-standard`, which lives in the plugin and not in the repo; the sibling ZP docs would show the house order. I did not compare them.
*   ***Proposed Edit:***
    *   Location: headings at lines 24, 126, and the Open Questions block (lines 67-84).
    *   Change: Rename "The Problem" to "Problem Statement" and "Technical Plan (Architecture)" to "Proposed Architecture"; move Open Questions after Critical User Journeys. [confirm against sibling ZP docs]

### [Issue 21] [`ZO-10, CO-13, DC-11, S8-28`] - Severity: Nit (N2)
*   **Affected Scope:** [Lines 194, 253-258](../../../designs/zcode-parity/ZP-22-remote-providers.md#L253-L258) (Localization, Verification)
*   **Discrepancy Description:** "Ten gates", "21 languages" unlinked; missing test cases (401, mid-stream disconnect, partial tool JSON).
*   **Rationale:** Unverifiable references.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Verified: `swift/docs/SWIFT_LOCALIZATION.md` states 21 languages.
*   ***Proposed Edit:***
    *   Location: lines 194 and 258; RemoteProviderTests bullet (lines 236-242).
    *   Change: Link "[21 languages and the parity gates](../../../swift/docs/SWIFT_LOCALIZATION.md)". Add to the test list: "401 and 429 responses, disconnect mid-stream, truncated tool-call JSON".

### [Issue 22] [`CO-10, SY-9`] - Severity: Nit (N3)
*   **Affected Scope:** [Lines 21-22, 57-58](../../../designs/zcode-parity/ZP-22-remote-providers.md#L57-L58) (Overview, Non-Goals)
*   **Discrepancy Description:** "No provider family switching" is ambiguous; the author decision is repeated five times.
*   **Rationale:** Redundancy and an unclear phrase.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Valid by reading. Folded with B3 (state the decision once).
*   ***Proposed Edit:***
    *   Location: Non-Goals first bullet (lines 57-58).
    *   Change: "**No OAuth.** No account login and no hosted identity; API keys are the only credential type (decision in ZP-90). A provider's family is fixed when it is created; changing family means adding a new provider." [confirm that is the intended meaning of "family switching"]

### [Issue 23] [`SY-10`] - Severity: Nit (N4)
*   **Affected Scope:** [Lines 19-20, 28, 40, 226](../../../designs/zcode-parity/ZP-22-remote-providers.md#L19-L20) (subjective wording)
*   **Discrepancy Description:** "Clearly labeled", "visually distinct", "frontier model" are subjective.
*   **Rationale:** Unverifiable acceptance wording.
*   ***AI Suggestion:*** DECLINE
*   ***Justification:*** Nit. The concrete treatment belongs in the Design System step that `DESIGN.md` requires before any UI change; specifying it here would pre-empt it. Recorded, not fixed.
*   ***Proposed Edit:***
    *   Location: none.
    *   Change: none.

### [Issue 24] [`SY-15`] - Severity: Nit (N5)
*   **Affected Scope:** [Lines 216-219](../../../designs/zcode-parity/ZP-22-remote-providers.md#L216-L219) (Target Files item 4)
*   **Discrepancy Description:** "Untouched" while modifying turn-loop files.
*   **Rationale:** Contradiction.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Valid; line 219 says "downstream ... untouched" while the item modifies the loop.
*   ***Proposed Edit:***
    *   Location: line 219.
    *   Change: Replace "untouched" with "unchanged in behavior".

### [Issue 25] [`SY-16`] - Severity: Nit (N6)
*   **Affected Scope:** [Lines 243-245](../../../designs/zcode-parity/ZP-22-remote-providers.md#L243-L245) (item 9)
*   **Discrepancy Description:** Mutation checks nested under Proposed Changes.
*   **Rationale:** They are verification steps.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Valid. Root AGENTS.md asks for mutation checks on subtle guards, so they should be in the Verification Plan.
*   ***Proposed Edit:***
    *   Location: lines 243-245 and Verification Plan.
    *   Change: Move the "Mutation checks" bullet to Verification Plan as its own bullet, and add "assert the mutation applied before treating a survivor as evidence".

### [Issue 26] [`SY-17`] - Severity: Nit (N7)
*   **Affected Scope:** [Lines 35-37, 144-145, 210](../../../designs/zcode-parity/ZP-22-remote-providers.md#L35-L37) (terminology)
*   **Discrepancy Description:** `openai-compatible`, `OpenAICompat`, `OpenAI-compatible` all used; "family client" vs "API family".
*   **Rationale:** Inconsistent terms.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Valid by reading.
*   ***Proposed Edit:***
    *   Location: Goals bullet 1 (lines 35-38).
    *   Change: Add: "An API family is the wire protocol, either `anthropic` or `openai-compatible` (persisted strings); class names `AnthropicClient` and `OpenAICompatClient` are the family clients." Use "OpenAI-compatible" in prose throughout.

### [Issue 27] [`ST-14, DC-12`] - Severity: Nit (N8)
*   **Affected Scope:** [Lines 126-149](../../../designs/zcode-parity/ZP-22-remote-providers.md#L126-L149) (diagram)
*   **Discrepancy Description:** Diagram omits key read, redaction, and cancel points.
*   **Rationale:** Optional completeness.
*   ***AI Suggestion:*** DECLINE
*   ***Justification:*** Nit and marked optional by the reviewer. The key-read and redaction points are covered in prose by the B2 and M12 edits. Recorded, not fixed.
*   ***Proposed Edit:***
    *   Location: none.
    *   Change: none.

### [Issue 28] [`IC-8`] - Severity: Nit (N9)
*   **Affected Scope:** [Lines 55-65](../../../designs/zcode-parity/ZP-22-remote-providers.md#L55-L65) (Non-Goals)
*   **Discrepancy Description:** Loopback server routing to remote models unstated.
*   **Rationale:** The server serves local models; a user could expect it to proxy remote ones.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** One line; a proxy would also forward provider keys through the server, which is a security expansion, so the conservative default is "no".
*   ***Proposed Edit:***
    *   Location: Non-Goals, after line 65.
    *   Change: "The TurboSpark API server never proxies to a remote provider, and provider keys are never visible to it."

### [Issue 29] [`ST-13`] - Severity: Nit (N10)
*   **Affected Scope:** `INDEX.md` lines 50, 63-72 (other file)
*   **Discrepancy Description:** 7.1 is listed in both the drafted table and the ZP-90 list.
*   **Rationale:** Duplicate listing.
*   ***AI Suggestion:*** DECLINE
*   ***Justification:*** Nit, and not in the target file. Checked: `INDEX.md` line 72 and `ZP-90` line 35 deliberately say "7.1 is adopted as ZP-22", so the second listing is a pointer, not an error. Recorded, not fixed.
*   ***Proposed Edit:***
    *   Location: none.
    *   Change: none.

### [Issue 30] [`SY-11`] - Severity: Nit (N11)
*   **Affected Scope:** [Lines 186-187](../../../designs/zcode-parity/ZP-22-remote-providers.md#L186-L187) (Security NFR)
*   **Discrepancy Description:** The "future cloud surface" sentence is speculative.
*   **Rationale:** Describes a feature the doc excludes.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Valid; the doc excludes hosted service in Non-Goals. Deleting it removes a claim, not a protection.
*   ***Proposed Edit:***
    *   Location: lines 186-187.
    *   Change: Delete "Remote chats are excluded from any future cloud surface by the same rules."
