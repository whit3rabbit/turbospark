# Critic report: ZP-22 remote providers (Stage 7)

Target: docs/designs/zcode-parity/ZP-22-remote-providers.md, full file. One round only; convergence not assessed. Standard feature-spec-standard for design and structural; style had none.

## Design critic (egm-design, persona agents/design_critic.md, standard feature-spec-standard.md)

Verdict: not ready to approve. Caveat: ZP-03 and ZP-23 not read; findings 4 and 5 rest on INDEX.md.

DC-1 Blocker, lines 69-75, 214-219. Core premise (provider-agnostic model-step seam) unverified; no fallback costed. Fix: audit the seam now, name the loop functions and FFI types, or add a Risks section sizing both outcomes.

DC-2 Major, whole file. No Risks and Limits section. Missing: provider schema drift, API version pinning (Anthropic version header), cost runaway from agent loops, prompt injection via tool results sent to a third party, half-emitted tool call on partial stream. Fix: add Risks and Mitigations with failure mode, impact, mitigation.

DC-3 Major, lines 181-190. Security asserted, not designed: user-controlled base URL, no https rule, no cross-host redirect rule, no loopback/link-local rule; reuse of HttpRequestExecutor (has a redirect delegate) not stated; "key-shaped strings" undefined; nothing on whether attachments, memory or tool results are sent. Fix: add a threat model: https except loopback, redirect policy, defined redaction patterns, explicit "what leaves the machine".

DC-4 Major, lines 99-107, 143-149, 188. ZP-03 rollout log is Rust-side while the remote path is Swift-only; INDEX says ZP-22 "stands alone"; "one in-flight request per chat" ignores ZP-23 concurrency (rate limits, per-provider caps, concurrent key reads). Fix: state ZP-03 and ZP-23 dependencies, where remote rollout entries are written, per-provider concurrency and 429 handling.

DC-5 Major, lines 49-51, 76-80, 151, 228-234. "Compaction works unchanged" unchecked; no compaction trigger for remote chats, no handling of a wrong or missing context window, no "context too long" path, no token-cost note. Fix: specify trigger, over-limit path and cost; close the open question.

DC-6 Major, lines 155-179. Alternatives A and C rejected only as "Rejected by author"; B thin and unquantified; missing alternatives: single OpenAI-compatible family first, Rust HTTP stack reuse, third-party Swift SDK, importing existing provider config (State/ModelProviderDiscovery.swift is the cited Swift side of 7.1 and is never mentioned). Fix: technical rationales or a decision-record link; add those alternatives.

DC-7 Major, lines 88-97. CUJ 1 Keychain toggle contradicts AGENTS.md (credentials not stored by default); no session-only path; read timing, mid-turn Keychain prompt and denied access undefined. Fix: default off with session-memory key, or justify default-on and define denied-access error.

DC-8 Nit, lines 33-53, 189. No measurable acceptance criteria or timeout/retry values. Fix: state defaults.

DC-9 Nit, lines 25-31. "Server code proves..." has no file reference. Fix: cite server files.

DC-10 Nit, lines 128-149, 204-219. Networking/ directory does not exist and is not called out as new; reuse of HttpRequestExecutor unstated; generation files named only by "basenames". Fix: name actual files.

DC-11 Nit, lines 253-258. "Ten parity gates", "21 languages" unlinked; no tests for rate limit, 401 key rotation, mid-stream disconnect, partial tool-call JSON. Fix: add cases, link swift/docs/SWIFT_LOCALIZATION.md.

DC-12 Nit, lines 126-149. AppJSONStore found only in State/AppStorageRoot.swift; confirm name and location. Fix: link the file.

DC-13 Nit, lines 3-7. 7.1 and 7.2 listed separately in sources; schema parity (presets, opencode import, per-model overrides) in/out of scope unstated. Fix: add a short schema-parity table.

## Structural reviewer (egm-structural, persona agents/structural.md, standard feature-spec-standard.md)

Verdict: FAIL (1 Blocker, 7 Major, 6 Nit). Heading map is clean (no skipped levels). Not verified: external GitHub links on lines 4-7; 7.1/7.2 body text only grepped.

ST-1 Blocker, lines 162, 170, 179 (also 21, 58-59). Each rejection says "Rejected by author" with no link; the standard forbids attributing rejection to a maintainer without a link. Fix: use the concrete trade-offs already in the Cons and link the decision record (ZP-90, roadmap or issue).

ST-2 Major, lines 155-179. Alternatives set thin: A is a posture; missing SDK vs hand-rolled SSE, OpenAI-compatible only, Responses API vs chat/completions, do-nothing; open-question options (a)/(b) not evaluated. Fix: add SDK vs hand-rolled and single-family options with technical reasons.

ST-3 Major, no Risks and Mitigations section. Unaddressed: http/SSRF/redirect key leak, cost runaway, schema drift, Keychain prompt friction, truncated SSE mid tool call, workspace content leaving via tool results. Fix: add the section, each risk with an owner section or test.

ST-4 Major, lines 93-96. Keychain opt-in sentence garbled; conflicts with line 94 and INDEX lines 119-120. Fix: two testable sentences: toggle default, off meaning, first Keychain read.

ST-5 Major, lines 204-250. Target Files lack owners for log redaction, settings export, ZP-03 remote marker, Keychain item code, Test connection call, ATS handling for arbitrary base URLs. Fix: add entries for each.

ST-6 Major, lines 3-7, 33-53. Scope gap against cited parity targets: presets, provider JSON import and opencode-config import appear in neither Goals nor Non-Goals. Fix: add an explicit Goal or Non-Goal.

ST-7 Major, lines 67-84 vs 36-53, 99-107, 214-219. Decided content under Open Questions while Goals and CUJs assume the unresolved answers; no chat-only branch in the plan; "Connection checks ... Include" is a decision. Fix: move Include into Goals; add a contingent chat-only phase keyed to the seam task.

ST-8 Major, INDEX 89-90 vs lines 105, 146-148, 188. INDEX says "stands alone" but ZP-22 depends on ZP-03 and interacts with ZP-23; "one in-flight request per chat" needs reconciling. Fix: add a Dependencies note and update the INDEX bullet.

ST-9 Nit, lines 7-86. Headings: "The Problem" to "Problem Statement", "Technical Plan (Architecture)" to "Proposed Architecture"; Open Questions precedes CUJs. Fix: rename; move Open Questions later.

ST-10 Nit, lines 214, 220. Target Files 4 and 5 are not concrete paths and bundle files. Fix: one entry per real file.

ST-11 Nit, lines 26-28. Problem claim unsourced, need not quantified. Fix: cite server files and one concrete trigger case.

ST-12 Nit, lines 76-80. Context accounting question does not cite finding 2.12 or say the exact-count guarantee does not carry over. Fix: cite and state.

ST-13 Nit, INDEX 63-72 vs 27-52. 7.1 listed both as drafted (ZP-22) and in ZP-90. Fix: remove from the ZP-90 list or state ZP-90 holds a pointer.

ST-14 Nit, lines 128-149. Diagram omits key read point, redaction point, interrupt cancelling the URLSession task; undefined terms. Fix: add a short sequence and define terms at first use.

## Style reviewer (egm-style, persona agents/style.md, no standard)

No Blockers. ZCODE_COMPARISON.md not read by this reviewer.

SY-1 Major, lines 27-29. "Server code proves..." uncited and overstated; "primary model source" uncited. Fix: name the server files or reword to "exposes both endpoint shapes (path)".
SY-2 Major, lines 48-51 vs 76-80. Goal commits to "a documented local estimate" while the question is open. Fix: reword the Goal to defer to the open question, or close it.
SY-3 Major, lines 93-96. CUJ 1 sentence unparseable; two opt-in models at once; fate of the entered key when off unstated. Fix: two sentences: default, and what off does.
SY-4 Major, lines 81-84. Connection check labeled open but says "Include". Fix: move to Goals; keep only the probe call open.
SY-5 Major, lines 167-169. "Small against the FFI/wire/binding cost" unquantified; "FFI symmetry" undefined. Fix: list concrete items avoided or drop the term.
SY-6 Major, lines 151-153. "Compaction works unchanged" uncited; omits that compaction sends history to the provider. Fix: cite the code path; add the remote-send note to the privacy NFR.
SY-7 Major, lines 188-193. Timeouts and stall detection have no values or owner; "matching local engine error handling" unverified. Fix: state first-byte and max SSE gap thresholds and configurability.
SY-8 Major, lines 214, 220. "basenames" entries name no files. Fix: list actual paths.
SY-9 Nit, lines 21-22, 57-58, 162, 170, 179. Author decision repeated five times; "Rejected by author" restates Cons. Fix: state once; give substantive reasons.
SY-10 Nit, lines 19-20, 28, 40, 226. "Clearly labeled", "visually distinct", "frontier model" are subjective or jargon. Fix: specify label text or token treatment; say "a larger hosted model".
SY-11 Nit, lines 186-187. "Future cloud surface" sentence speculative; "key-shaped" undefined. Fix: drop it; define (sk-, sk-ant- prefixes, Authorization headers).
SY-12 Nit, lines 183-185. "State this plainly" gives no location or content. Fix: put the notice text in plan item 5 (provider host, "prompts leave this machine").
SY-13 Nit, lines 117-121. "Exports settings" is an uncited feature. Fix: cite or remove.
SY-14 Nit, line 105. ZP-03 unlinked; ZP-23 not mentioned; INDEX says "stands alone". Fix: link and state dependencies.
SY-15 Nit, lines 216-219. "Untouched" while the same item modifies turn-loop files. Fix: "unchanged in behavior".
SY-16 Nit, lines 243-245. Mutation checks nested under Proposed Changes. Fix: move to Verification Plan.
SY-17 Nit, terminology. openai-compatible / OpenAICompat / OpenAI-compatible; family client / family-native / API family. Fix: pick one, define "family" once.
