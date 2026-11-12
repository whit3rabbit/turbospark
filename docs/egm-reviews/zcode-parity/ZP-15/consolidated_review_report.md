# Consolidated review report: ZP-15 browser video recording

Target: docs/designs/zcode-parity/ZP-15-browser-webm-recording.md (full file, 186 lines)
Preset: Feature spec only (stages 5, 6, 7, 8). No remediation, target not edited. Stage 7 ran one round only, so convergence was not assessed. All reviewers completed; none failed. One spawn (inventive catalyst) hit the concurrent subagent limit once and succeeded on retry.
Source IDs: ZO = zoom_out_report (zoom-out), IC = zoom_out_report (inventive catalyst), CO = comprehension_report, DC/ST/SY = critic_report (design, structural, style), S8 = intermediate_zoom_out_report.

Counts after deduplication: Blocker 2, Major 11, Nit 8.

## Blocker

B1. Lines 1-3, 27, 62-65, 148-150, 161-163. Deliverable format undecided: title, filename, INDEX row (line 42) say WebM, plan writes MP4, a "format" setting exists. Fix: decide MP4, retitle to browser session video recording, drop the format field, update INDEX. [ZO-3, IC-7, CO-1, DC-2, ST-4, SY-1, S8-B1]

B2. Lines 51-61, 98-101, 146-151. Capture mechanism left open while plan and tests assume SCStream plus a snapshot fallback; a window crop still captures overlays and cannot show an occluded or hidden view, so "page-only" is not guaranteed; "solves it" is unverified. Fix: hardware spike as step 0, state the invariant that the captured surface holds only the web view, pick one mechanism, move the fallback to a follow-up. [ZO-1, IC-1, CO-2, DC-1, ST-2, SY-2, SY-4, S8-B2]

## Major

M1. Lines 77-83, 133-135, 156-160. Session-scoped approval contradicts "ask-gated always"; `AppToolPermissionEngine` ladder position not named (INDEX requires it); "stops at session end or tool call" and "debuggable flow" vague; two-tool path adds scope. Fix: per-call ask, name the ladder position, sharpen CUJ 2, ship the user button first and agent tools as a second task. [ZO-6, IC-3, CO-4, DC-4, ST-6, SY-6, SY-8, S8-M1, S8-M7]

M2. Lines 51-61, 146-151, 180-182. Owning process and Screen Recording permission unexamined (helper vs main app, denied or revoked path); reuse claim imprecise (SCStream in `TurboSparkComputerUse/WindowCapturer.swift` and `TurboSparkApp/Audio/AudioCapture.swift`, no AVAssetWriter in the app). Fix: name the process and files to reuse, add a permission-denied CUJ and one test. [ZO-2, IC-2, CO-3, DC-3, S8-M2]

M3. Lines 37, 99-105, 132-133. "Per-chat artifact storage" and the transcript-attach API are unverified; three phrasings of the folder. Fix: cite the real type and API once, or add a sized task to build it. [ZO-4, CO-7, IC-9, S8-M3]

M4. Lines 69-76, 85-90, 108-110. Failure handling covers only the size cap; quit, crash, disk full, SCStream error, tab close or navigation, resize, lease loss and hidden pane undefined, so "nothing is truncated corrupt" is unsupported. Fix: one compact stop-and-finalize table (fragmented MP4 only if one line). [ZO-10, IC-4, IC-5, CO-6, DC-6, S8-M5]

M5. Missing section. No Risks and Mitigations section required by the standard. Fix: add a short section pointing to M2, M4, M7 and the privacy notice, without new content. [ST-3]

M6. Lines 39-40, 108-110, 162-163. Caps premature and underspecified: duration cap has no default, "named notice" and disk threshold undefined, settings fields speculative. Fix: one fixed size-cap constant plus disk precheck now, settings later; if kept, give the values (reviewers conflict: ZO/S8 say trim, SY/DC say specify). [ZO-5, IC-9, CO-5, DC-5, SY-5, ST-7, S8-M4]

M7. Lines 112-128, 49-65. Alternatives give "Rejected by author" with no trade-off; Alternative A is both rejected and the shipped fallback; container and capture options sit in Open Questions, not Alternatives. Fix: restate reasons as technical trade-offs and move the options into Alternatives. [ST-1, ST-2, SY-2, DC-7, S8-N3]

M8. Lines 14-16, 27-29. No rationale for who consumes the video or why it beats screenshots given disk cost and the permission prompt. Fix: one paragraph. [ZO-7, S8-M6]

M9. Lines 132-135. Page-only recording still captures secrets; no start-time notice; export and model exposure unstated. Fix: one-line sensitive-content notice at start and one sentence that files stay local and are not sent to the model; defer retention and export rules. [IC-8, DC-4, S8 recommendation]

M10. Lines 31-32, 98-100, 109-110, 150-155. "Agent-leased tab" undefined; the lease and tab owner (`BrowserAutomation/WebKitBrowserEngine.swift`) is not in the plan though it is the privacy boundary. Fix: define "leased" and name where tab identity resolves to a view rect. [CO-8, ZO-11]

M11. Lines 30-34, 163. "Opt-in per session" has no control or setting. Fix: drop the phrase since the user starts recording explicitly, or add the setting. [CO-5, SY-7, S8 recommendation]

## Nit

N1. Lines 136-141. "21 languages" conflicts with INDEX "ten localization parity gates". Fix: use one verified wording. [ZO-12, CO-13, ST-9, S8-N4]

N2. Lines 92-110, 164-172. Implementation-level detail (CMSampleBuffer wiring, mutation checks) in a feature spec. Fix: trim to the start/stop contract and single-recording invariant. [ZO-9, S8-N2]

N3. Lines 63-64, 114-120. Alternative A's no-permission advantage not weighed against the new TCC prompt; fallback fps unstated. Fix: add a sentence and the fps. [ZO-8, CO-11, S8-N1]

N4. Lines 155-160. Text-only hiding rule and "media capability pattern" unjustified and unlinked. Fix: drop the rule or justify it. [IC-6, CO-12, SY-15, S8-N5]

N5. Lines 12, 22-25, 89-90, 118-119. Wording: "nothing is truncated corrupt", "choppy", filler Overview. Fix: "the finalized file plays to the last written frame", state the frame-rate and timing-drift fact. [SY-10, SY-11, SY-13, S8-N7]

N6. Lines 3-6, 18-20, 24-27, 176-182. Unlinked references: ZCode recording spec (docs/browser-use/2026-08-12-browser-webview-video-recording-spec.md), kiro browser-automation exclusion, ZP-14, hardware-gate rule, ZCode gloss, `make swift-lib`. Fix: add links and one-line glosses. [ST-5, SY-3, SY-12, CO-9, CO-10, CO-14, CO-15, DC-8, S8-N6]

N7. Lines 92-110, 94-106. Diagram omits the process boundary and agent tool path; heading is "Technical Plan". Fix: label the diagram primary path only; rename optional. [ST-8, SY-14]

N8. Lines 136-138, 164-172. Hardware-encode and no-render-cost claims unmeasured; no CPU or frame-drop budget. Fix: say expected, not verified; measure later. [SY-9, DC-8, SY-16, ST-10]

## Traceability

- zoom_out_report.md: ZO-1 to ZO-12 (zoom-out), IC-1 to IC-9 (inventive catalyst). Feeds B1, B2, M1-M10, N1-N4.
- comprehension_report.md: CO-1 to CO-15. Feeds B1, B2, M1-M4, M6, M10, M11, N1, N3, N4, N6.
- critic_report.md: DC-1 to DC-8 (design), ST-1 to ST-10 (structural), SY-1 to SY-16 (style). Feeds all Blockers and Majors, N1-N8.
- intermediate_zoom_out_report.md: S8-B1, S8-B2, S8-M1 to S8-M7, S8-N1 to S8-N7. Confirms all Stage 5 findings open; recommends rejecting scope-adding asks (retention and export analysis, MB-per-minute and CPU budgets, extra alternatives, duration cap default, diagram redraws) and keeping ambiguity-removal fixes.
- Merge notes: the S8 promotion of the container and capture decisions to Blocker is adopted; no earlier reviewer rated them Blocker (all rated Major). ST-10 requires no change.
