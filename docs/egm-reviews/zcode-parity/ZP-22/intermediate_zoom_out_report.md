# Intermediate zoom out report: ZP-22 remote providers (Stage 8)

Reviewer: egm-zoom-out (persona agents/zoom_out.md, standard zoom-out-standard.md). Target unchanged since Stage 5. Verdict: context sufficient, root problem clear, design not ready. Stage 6 and 7 reports mostly restate Stage 5; about one third of their proposed mitigations are scope creep.

Spot checks: ModelProviderDiscovery.swift, AppModel+Generation.swift, AppModel+AgentLoop.swift, PromptContextPills.swift, HttpRequestExecutor.swift exist. No SettingsPanes.swift; panes live in Components/*SettingsPaneView.swift.

## A. Reject as scope creep

S8-1 Major. IC-5 retry, backoff, Continue row, "Retry on local model" (188-193). Keep the taxonomy and one sentence: no automatic retry; errors surface as named rows.
S8-2 Major. IC-6 quirk profile and Capability mapping (35-44, 204-213). Ship one OpenAI-compatible path; record quirks as a known limit.
S8-3 Major. IC-4 calibrated ratio plus count endpoint (76-80). Close as option (a), fixed chars-per-token constant, labeled estimate.
S8-4 Major. IC-3 egress matrix and consent gate (64-65, 151-153, 183-185). Keep one rule: background callers use the chat's selected model; first-send notice is per provider per chat.
S8-5 Major. ST-6/DC-13 schema-parity table (3-7). One Non-Goals bullet: no presets, no provider JSON or opencode import.
S8-6 Major. DC-6/ST-2 extra alternatives. Add only "single OpenAI-compatible family first".
S8-7 Nit. ST-5 extra target-file entries. Fold redaction and Keychain into ProviderStore and the egress type; drop settings export.
S8-8 Nit. ST-9, ST-13, ST-14, SY-10, SY-16, SY-17 (keep terminology only), DC-11. Cosmetic or implementation-spec material.
S8-9 Nit. CO-4/CO-5 inline definitions. Keep one gloss for "permission ladder" and the finding numbers.
S8-10 Nit. DC-8/CO-14/SY-7 numeric timeouts. Say one first-byte and one stall timeout, values set at implementation.

## B. Keep (removes ambiguity)

S8-11 CO-2/ST-4/SY-3/DC-7/IC-7 (93-96). Drop the toggle: saving the key is the opt-in; Keychain read on first remote send; denied access gives a named error. Also conflicts with ZCODE_COMPARISON 7.14.
S8-12 CO-11/SY-13 (122). Reword CUJ 4 to "settings.json never contains the key."
S8-13 IC-2/DC-3 narrowed (52-53, 183-187). https except loopback; strip Authorization on cross-host redirect; define key-shaped as sk-/sk-ant- prefixes plus Authorization headers; reuse the HttpRequestExecutor redirect delegate. Reject the non-Codable wrapper detail.
S8-14 IC-1/DC-1/CO-1/ST-7 (69-75). Run the seam read now, name functions near AppModel+Generation.swift:302 and the AgentLoop call site, delete the Open Question.
S8-15 ST-7/SY-4/ZO-8/CO-16 (81-84). Test connection into Goals; prefer a free models-list probe.
S8-16 SY-2/CO-9 (48-51, 76-80). Close option (a) in one place.
S8-17 DC-4/ST-8/IC-9/ZO-6/SY-14 (105, 148, 188, INDEX 89-90). One Dependencies note: ZP-03 soft dependency; ZP-22 does not block ZP-23; per-chat in-flight limit unchanged; fix INDEX "stands alone".
S8-18 ST-1/SY-9 (21, 58, 162, 170, 179). State the decision once with a link; keep technical Cons per alternative.
S8-19 CO-6/ST-10/ZO-7/DC-10 (214, 220). Real paths: State/AppModel+Generation.swift, State/AppModel+AgentLoop.swift, Components/ProviderSettingsPaneView.swift; mark Networking/ as new.
S8-20 DC-2/ST-3 reduced. Five-line Risks list: schema or API-version drift, agent-loop cost, tool-result content sent to a third party, truncated stream mid tool call, base URL leakage.

## C. Stage 5 findings still open

S8-21 Blocker (ZO-1). Lines 67-75, 138-149, 214-219. Seam unverified.
S8-22 Major (ZO-2). Lines 17-53, 197-250. No phasing. Fix: Phase 1 OpenAI-compatible chat-only, flat model list; Anthropic and tools later; update INDEX line 105 and Goals.
S8-23 Major (ZO-3). Lines 206-213. Shared SSE parser and transport on HttpRequestExecutor/URLSession.
S8-24 Major (ZO-4). Lines 53, 92-96. Name one keychain wrapper (ServerKeychain, TypeSafeKeychain or HookOptionKeychain) and cite ZCODE_COMPARISON 7.14.
S8-25 Major (ZO-5). Lines 151-153, 231-234. Specify remote compaction trigger (reported usage watermark) and the context-overflow error path.
S8-26 Major (ZO-6). Covered by S8-17.
S8-27 Nit (ZO-9). Lines 164-172. One sentence on why serving-side Rust code is not a reusable client.
S8-28 Nit (ZO-10). Lines 194, 258. Link swift/docs/SWIFT_LOCALIZATION.md for gate and language counts.

## D. New findings

S8-29 Major, lines 3, 224-227. Cites only 7.1/7.2; ignores 7.15 (picker is "Aligned" and filters to live-loadable installs, so a Remote section changes that contract), 7.16 (connectivity checks "Different") and 7.14 (credential rules). Fix: add to header; note the picker contract change in the PromptContextPills item.
S8-30 Major, lines 206-219. ModelProviderDiscovery.swift already uses "provider" for on-disk GGUF locations; ProviderStore collides. Fix: rename to RemoteProvider/RemoteProviderStore.
S8-31 Nit, lines 26-29. "Proves" claim is supported by ZCODE_COMPARISON 7.1 ("serves those endpoints, never calls them"). Fix: cite 7.1/7.13; reword to "serves both endpoint shapes".
