# Consolidated review report: ZP-22 remote providers

Target: docs/designs/zcode-parity/ZP-22-remote-providers.md (full file, 266 lines)
Preset: Feature spec only (stages 5, 6, 7, 8). No remediation, target not edited. Stage 7 ran one round only, so convergence was not assessed. All reviewers completed; none failed (two spawns were retried after "Concurrent subagent limit reached" and then succeeded).
Source IDs: ZO = zoom_out_report (zoom-out), IC = zoom_out_report (inventive catalyst), CO = comprehension_report, DC/ST/SY = critic_report (design, structural, style), S8 = intermediate_zoom_out_report.
Persona files read (per each reviewer's adapter and report): egm-zoom-out (Stages 5 and 8) agents/zoom_out.md with zoom-out-standard; egm-inventive-catalyst agents/inventive_catalyst.md; egm-comprehension agents/comprehension.md; egm-design agents/design_critic.md with feature-spec-standard; egm-structural agents/structural.md with feature-spec-standard; egm-style agents/style.md. Caveat: reviewers did not read ZCODE_COMPARISON 7.1/7.2 in full (the Stage 8 reviewer did).

Counts after deduplication: Blocker 3, Major 15, Nit 11.

## Blocker

B1. Lines 67-75, 138-149, 214-219. The ModelStep seam is unverified, yet the doc commits to extracting it and to "downstream untouched"; the loop uses local `GenerationToolCall` (AppModel+Generation.swift:302). Fix: do the seam read now, name the real call site and event types, delete the Open Question, or cut tools from scope. [ZO-1, IC-1, CO-1, DC-1, ST-7, S8-14, S8-21]
B2. Lines 52-53, 93-96. CUJ 1 Keychain toggle ("defaults on", "off means disabled") contradicts per-credential opt-in and says nothing of the key when off; no keychain helper named. Fix: drop the toggle; saving the key is the opt-in; read on first remote send; named error on denied access; name one existing keychain wrapper and cite ZCODE_COMPARISON 7.14. [CO-2, ZO-4, DC-7, ST-4, SY-3, IC-7, S8-11, S8-24]
B3. Lines 21, 58, 162, 170, 179. Every rejection reads "Rejected by author" with no link or technical reason; the standard forbids this. Fix: state the decision once with a link (ZP-90 or roadmap) and keep the technical Cons per alternative. [ST-1, DC-6, SY-9, S8-18]

## Major

M1. Whole file. No Risks and Mitigations section. Fix: five-line list (API drift, agent-loop cost, tool-result content sent to a provider, truncated stream mid tool call, base URL leakage), each with one mitigation. [DC-2, ST-3, S8-20]
M2. Lines 52-53, 119-124, 181-187. Security asserted, not designed: base URL scheme, cross-host redirects, "key-shaped" undefined. Fix: https except loopback; strip Authorization on cross-host redirect; define patterns (sk-, sk-ant-, Authorization); reuse HttpRequestExecutor redirect delegate. [IC-2, DC-3, CO-7, SY-11, S8-13]
M3. Lines 17-53, 197-250. No phasing; two clients, loop refactor, accounting and rollout in one landing. Fix: Phase 1 OpenAI-compatible chat-only with a flat model list; Anthropic and tools later; update INDEX. [ZO-2, S8-22]
M4. Lines 206-213. Two bespoke SSE clients, new Networking/ directory, no shared parser or reuse of Tools/Web URLSession code. Fix: shared SSE parser plus transport on HttpRequestExecutor; state what is reused; mark the directory new. [ZO-3, DC-3, DC-10, S8-23]
M5. Lines 49-51, 76-80, 151, 231-234. Context accounting open while Goals assume it; "compaction works unchanged" is likely false (local token counts drive it); no overflow path. Fix: close as option (a) with a fixed chars-per-token constant; specify remote compaction trigger and provider overflow error. [ZO-5, DC-5, CO-9, SY-2, ST-12, IC-4 (S8 advises rejecting the calibration idea), S8-3, S8-16, S8-25]
M6. Lines 105, 148, 188 and INDEX 89-90. ZP-03 (Rust-side rollout log) and ZP-23 dependencies unstated; INDEX says "stands alone". Fix: one Dependencies note: ZP-03 soft, ZP-22 does not block ZP-23; fix INDEX. [ZO-6, DC-4, ST-8, IC-9, SY-14, CO-12, S8-17]
M7. Lines 155-179. Alternatives thin; real options missing. Fix: add "single OpenAI-compatible family first" (becomes Phase 1); give Alternative B a size comparison. [DC-6, ST-2, ZO-9, SY-5, S8-6, S8-27]
M8. Lines 67-84. Decided items sit under Open Questions (Test connection "Include"). Fix: move Test connection to Goals; keep only the probe call open, prefer a free models-list probe. [ZO-8, ST-7, SY-4, CO-16, IC-10, S8-15]
M9. Lines 64-65, 151-153, 183-185. Background callers (compaction, titles, memory, cron) and mid-chat switch can send history remotely; privacy notice content unspecified. Fix: one rule: background callers use the chat's selected model; first-send notice per provider per chat, with provider host and "prompts leave this machine". [IC-3, SY-6, SY-12, S8-4]
M10. Lines 188-193. Reliability has no values or mapping to local error handling. Fix: one first-byte and one stall timeout, values set at implementation; map taxonomy to the local model; state no automatic retry (S8 rejects a retry subsystem). [IC-5, CO-8, CO-14, SY-7, DC-8, S8-1, S8-10]
M11. Lines 214, 220. Target files named "basenames", `SettingsPanes.swift` does not exist. Fix: State/AppModel+Generation.swift, State/AppModel+AgentLoop.swift, Components/ProviderSettingsPaneView.swift. [ZO-7, CO-6, ST-10, SY-8, S8-19]
M12. Lines 204-250. No owner for log redaction, settings export, Keychain code, ATS handling. Fix: fold redaction and Keychain into ProviderStore and the egress path; drop "settings export" from CUJ 4 (reword to "settings.json never contains the key"). [ST-5, CO-11, SY-13, S8-7, S8-12]
M13. Lines 3-7, 33-53. Presets, provider JSON and opencode import neither in Goals nor Non-Goals. Fix: one Non-Goals bullet. [ST-6, DC-13, S8-5]
M14. Lines 35-44, 204-213. "OpenAI-compatible" treated as one protocol; vision, reasoning and usage field differences unaddressed. Fix: state as a known limit for the single compat path (S8 rejects a quirk-profile layer). [IC-6, S8-2]
M15. Lines 3, 206-219, 224-227. Cites only 7.1/7.2, ignoring 7.14, 7.15 (picker contract) and 7.16; "Provider" collides with ModelProviderDiscovery.swift. Fix: cite them; rename to RemoteProvider/RemoteProviderStore. [S8-29, S8-30]
M16. Lines 3, 26-32, 52, 114, 140, 232-234. Uncited claims and undefined terms (server "proves" both families; repo rules; permission ladder; 7.3/7.4). Fix: cite ZCODE_COMPARISON 7.1/7.13, reword to "serves both endpoint shapes", one-line gloss of permission ladder and finding numbers. [CO-3, CO-4, CO-5, CO-15, DC-9, SY-1, ST-11, S8-31]

## Nit

N1. Lines 7-86. Headings: "Problem Statement", "Proposed Architecture"; Open Questions precedes CUJs. Fix: rename. [ST-9]
N2. Lines 194, 253-258. "Ten gates", "21 languages" unlinked; missing test cases (401, mid-stream disconnect, partial tool JSON). Fix: link swift/docs/SWIFT_LOCALIZATION.md. [ZO-10, CO-13, DC-11, S8-28]
N3. Lines 21-22, 57-58. "No provider family switching" ambiguous; author decision repeated five times. Fix: clarify or delete; state once. [CO-10, SY-9]
N4. Lines 19-20, 28, 40, 226. "Clearly labeled", "visually distinct", "frontier model" subjective. Fix: specify label or token treatment. [SY-10]
N5. Lines 216-219. "Untouched" while modifying turn-loop files. Fix: "unchanged in behavior". [SY-15]
N6. Lines 243-245. Mutation checks nested under Proposed Changes. Fix: move to Verification Plan. [SY-16]
N7. Terminology: openai-compatible / OpenAICompat / OpenAI-compatible; family client vs API family. Fix: one spelling, define "family". [SY-17]
N8. Lines 126-149. Diagram omits key read, redaction and cancel points. Fix: short sequence (optional). [ST-14, DC-12]
N9. Non-Goals. Loopback server routing to remote models unstated. Fix: one line. [IC-8]
N10. INDEX 63-72. 7.1 listed in both drafted table and ZP-90. Fix: remove from the ZP-90 list or note it is a pointer. [ST-13]
N11. Lines 186-187. "Future cloud surface" sentence speculative. Fix: drop. [SY-11]

## Traceability

- zoom_out_report.md: ZO-1 to ZO-10 (zoom-out), IC-1 to IC-10 (inventive catalyst). Feeds B1-B2, M2-M10, M14, N9.
- comprehension_report.md: CO-1 to CO-16. Feeds B1-B2, M2, M5, M6, M8, M10-M12, M16, N2, N3.
- critic_report.md: DC-1 to DC-13, ST-1 to ST-14, SY-1 to SY-17. Feeds all Blockers and most Majors and Nits.
- intermediate_zoom_out_report.md: S8-1 to S8-31. Confirms all Stage 5 findings open. Recommends rejecting scope-adding asks (IC-3 consent matrix, IC-4 calibration, IC-5 retry subsystem, IC-6 quirk profiles, ST-6 parity table, extra alternatives, numeric timeouts) and keeping ambiguity-removal fixes (B1-B3, M1, M2, M6, M8, M11, M12).
- Merge notes: IC-5, IC-6, IC-4 are kept as findings but carry the Stage 8 advice to resolve them minimally.
