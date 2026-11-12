# Intermediate zoom out report (Stage 8): ZP-15 browser video recording

Target: docs/designs/zcode-parity/ZP-15-browser-webm-recording.md (full file). Reviewer: egm-zoom-out, completed. The target was not edited after Stage 5, so every Stage 5 finding is still open. Stage 6 and 7 findings largely duplicate Stage 5 (container, capture mechanism, permission, approval semantics, failure handling, artifact folder, tab lease).

## Blockers

S8-B1 Lines 1-3, 62-65, 156-163. The deliverable is undecided: title and filename say WebM, the plan writes MP4, a "format" setting exists. Fix: decide MP4, retitle to browser session video recording, drop the format setting, update INDEX line 42.

S8-B2 Lines 51-61, 98-101, 146-151. The capture decision is open while the plan assumes SCStream plus a snapshot fallback; a crop of a window captures overlays and cannot show an occluded or hidden view, so page-only is not guaranteed. Fix: hardware spike as step 0, state the invariant that the captured surface contains only the web view, pick one mechanism, move the fallback to a follow-up.

## Majors

S8-M1 Lines 77-83, 133-135, 156-160. Session approval contradicts ask-gated always; ladder position not named (INDEX lines 121-122). Fix: per-call ask, name the position, align CUJ 2 and tests.

S8-M2 Lines 54-57, 146-151, 180-182. Reuse claim imprecise: SCStream code is in `TurboSparkComputerUse/WindowCapturer.swift` and `TurboSparkApp/Audio/AudioCapture.swift`; no AVAssetWriter exists; TCC owner may change. Fix: name the owning process and files to reuse, add a permission-denied journey.

S8-M3 Lines 37, 99-105. "Existing per-chat artifact storage" unverified. Fix: cite the type and API or add a sized task.

S8-M4 Lines 40, 108-110, 162-163. Size cap, duration cap, settings and disk precheck are premature. Fix: one fixed size cap constant plus disk precheck, no settings fields yet.

S8-M5 Lines 69-76, 85-90. Hidden, collapsed or background tab, quit, crash, tab close and lease loss are undefined. Fix: short stop-and-finalize rule.

S8-M6 Lines 14-16, 27-29. No rationale for who consumes the video versus screenshots. Fix: one paragraph.

S8-M7 Lines 77-83, 156-160. Two-tool agent path adds scope. Fix: ship the user button first, agent tools as a second task, or justify in one sentence.

## Nits

S8-N1 Lines 63-64, 114-118. Alternative A's no-permission advantage not weighed against the new TCC prompt.
S8-N2 Lines 92-110, 164-172. Implementation-level detail in a feature spec.
S8-N3 Lines 112-128. "Rejected by author" without trade-off.
S8-N4 Lines 136-141. "21 languages" vs INDEX "ten localization parity gates".
S8-N5 Lines 156-160. Text-only hiding rule unjustified.
S8-N6 Lines 3-6, 18-20. ZCode spec and kiro exclusion cited by name only.
S8-N7 Lines 89-90. "Nothing is truncated corrupt" ungrammatical.

## Recommendation on Stage 6 and 7 additions

Accept, each in one sentence or short paragraph: MP4 decision and retitle; capture decision, spike and overlay invariant; owning process and permission-denied CUJ with one test; per-call ask and ladder position; compact failure rule (quit, crash, disk full, tab close, lease loss, hidden tab); artifact folder defined once; "leased tab" defined; missing links (ZP-14, ZCode spec, kiro spec, hardware-gate rule); short Risks section that only points to the above; one-line sensitive-content notice; wording nits.

Reject as scope bloat: retention, cleanup, export and ZP-22 exposure analysis (say once that files stay local and are not sent to the model); MB-per-minute, fps and CPU budgets (state one fixed profile, measure later); extra alternatives beyond a one-line mention; a duration cap default and an "opt-in per session" control (drop the phrase); diagram redraws and heading rename; extra manual checks beyond cap, indicator and denied permission.

Net effect: the draft shrinks to a user-button MP4 recorder with one cap, agent tools as an optional second task. S8-B1 and S8-B2 gate the other fixes.
