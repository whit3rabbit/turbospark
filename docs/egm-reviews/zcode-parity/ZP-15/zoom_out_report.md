# Zoom out report (Stage 5): ZP-15 browser video recording

Target: docs/designs/zcode-parity/ZP-15-browser-webm-recording.md (full file, 186 lines).
Reviewers: egm-zoom-out (ZO), egm-inventive-catalyst (IC). Both completed.

## Zoom-out reviewer (ZO)

ZO-1 (Major) Lines 51-61, 98-101, 146-151. The central capture decision is open, yet the plan, tests and file list assume (a) SCStream plus a snapshot-stitch fallback. Two pipelines means two test matrices. Nothing shows a content-rect crop works for an occluded, scrolled-off or other-Space WKWebView. Fix: add a hardware spike as step 0, pick one mechanism, make the fallback a separate follow-up.

ZO-2 (Major) Lines 54-57, 146-151, 180-182. Reuse claim is partly wrong. SCStream/SCContentFilter live in `Sources/TurboSparkComputerUse/WindowCapturer.swift` (desktopIndependentWindow) and `Sources/TurboSparkApp/Audio/AudioCapture.swift` (display filters, SCStreamOutput/Delegate). The draft says "the helper app" but the controller is in the main app target, which changes the TCC owner. No AVAssetWriter exists in the app today. Fix: name the owning process, reuse the AudioCapture lifecycle and WindowCapturer filter code, name the shared seam.

ZO-3 (Major) Lines 1-3, 62-65, 156-163. Container is unresolved: title and Goal say WebM, the plan writes MP4, and a "format" setting is speculative. Fix: decide MP4, retitle to video recording, drop the format setting.

ZO-4 (Major) Lines 37, 99-105. "Existing per-chat artifact storage" is not verified; no obvious store or transcript-attach API found. Fix: cite the exact type and API, or add and size a task to build it.

ZO-5 (Major) Lines 40, 108-110, 162-163. Size cap plus duration cap plus settings plus disk precheck is premature. Fix: ship one size cap (500 MB constant) first.

ZO-6 (Major) Lines 77-83, 156-160. Agent tool path adds scope; "one-time session approval" conflicts with "ask in every mode"; no `AppToolPermissionEngine` ladder position (INDEX requires it); two tools. Fix: ship the user button first, or state ladder position and reconcile approval semantics.

ZO-7 (Major) Lines 5, 14-16, 27-29. No rationale for who consumes the video or why it beats screenshots given disk cost and the screen-recording prompt. Fix: add one paragraph.

ZO-8 (Nit) Lines 63-64, 114-118. Alternative A is rejected while its no-permission advantage is never weighed against the new TCC prompt. Fix: weigh it.

ZO-9 (Nit) Lines 92-110, 164-172. Implementation-level detail (CMSampleBuffer wiring, mutation checks) in a feature spec. Fix: trim to the start/stop contract and the single-recording invariant.

ZO-10 (Nit) Lines 85-90, 108. "Nothing truncated corrupt" ignores crash and app quit. Fix: state finalize-on-quit or discard.

ZO-11 (Nit) Lines 32, 150-155. `BrowserAutomation/WebKitBrowserEngine.swift` (lease and tab owner, already uses takeSnapshot) is not in the plan although the lease is the privacy boundary. Fix: name where tab identity resolves to a view rect.

ZO-12 (Nit) Lines 136-141. "21 languages" versus INDEX "ten localization parity gates". Fix: use one verified wording.

## Inventive catalyst (IC)

IC-1 (Major) Lines 51-61, 98-100. Content-rect crop of a window still includes overlays composited over the web view (sheets, popovers, indicator), so "page-only" is not guaranteed. Fix: make "capture a window containing only the web view, overlays excluded" the invariant (dedicated borderless window or filter excluding app windows) before choosing (a).

IC-2 (Major) Lines 51-61, 146, 180-182. Screen Recording permission and owning process unexamined; may mean a second TCC prompt. Fix: name the owning process and a preflight/denial flow.

IC-3 (Major) Lines 79-83 vs 132-135. Session-scoped approval contradicts "ask-gated always". Fix: pick per-call ask, align CUJ 2 and tests.

IC-4 (Major) Lines 85-90, 108-110. Failure handling covers only the size cap (quit, crash, sleep, disk full, SCStream error, tab close, resize, lease loss missing); size is not bounded by bitrate. Fix: add a failure matrix, fragmented MP4, fixed resolution/fps/bitrate profile.

IC-5 (Major) Lines 69-76, 136-137. No rule for a hidden, collapsed or background tab; CUJ 1 can record a blank surface. Fix: add a visibility requirement and pause or stop rule.

IC-6 (Nit) Lines 156-160. Tool hidden for text-only configs, but the model does not consume the video. Fix: drop the rule or justify it.

IC-7 (Nit) Lines 1, 62-65, 161-163. WebM/MP4 leaks into title, settings, strings. Fix: close as MP4, remove format field.

IC-8 (Nit) Lines 132-135. Page-only capture still records secrets on the page; no start or export notice. Fix: add a sensitive-content notice.

IC-9 (Nit) Lines 37-38, 108-109. Artifact attach/playback contract and duration cap default undefined. Fix: state artifact kind and playback path, give a default.
