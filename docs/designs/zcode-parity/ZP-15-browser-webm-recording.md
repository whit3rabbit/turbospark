# ZP-15: Browser session video recording (WebM)

Finding 6.12 in [ZCODE_COMPARISON.md](../../ZCODE_COMPARISON.md).
Parity target: ZCode's
[browserVideoRecorder](https://github.com/zai-org/ZCode/blob/main/packages/desktop/src/main/browserView/browserVideoRecorder.ts)
and its recording spec.

* Author: whit3rabbit
* Status: Draft
* Date: 2026-11-11

--------------------------------------------------------------------------------

## Overview

Record the embedded browser tab to a video file while the agent or the
user works, for later review of what the agent did on a page. The
earlier `browser-automation` kiro spec excluded "browser-use recording to
video"; that exclusion is reversed, confirmed by author decision
2026-11-11.

## The Problem

After an agent browser session, the only evidence is the screenshot
tool's captures and the tool transcript. Multi-step interactions (form
flows, animations, timing-dependent behavior) cannot be replayed from
still images. ZCode records the tab to WebM with a CI smoke; TurboSpark
has no recording path in Browser/ or BrowserAutomation/.

## Goals

* Record the agent-leased tab (and only that tab) to a video file,
  started and stopped by a user action in the browser pane and by an
  agent tool, opt-in per session.
* Recording captures page content only: the WKWebView surface, not the
  whole window, and never other panes or chats.
* Output lands in the chat's artifact folder and attaches to the
  transcript as a file artifact when the recording stops.
* Long-session safety: size cap with automatic stop and a named notice;
  disk-space precheck before start.

## Non-Goals

* No system-audio or microphone capture in the recording (page-only
  video; the browser's audio policies are governed separately by ZP-14).
* No streaming upload; files stay local (share/export remain the
  existing local flows).

## Open Questions

* **Capture mechanism**: options: (a) `SCStream` (ScreenCaptureKit)
  filtered to the WKWebView's window content with a content rect; (b)
  periodic `takeSnapshot` stitching into a video at fixed fps; (c) a
  `WKWebView`-private API (none is public). The helper app already uses
  ScreenCaptureKit for front-window capture (Computer Use), so (a) has
  in-repo precedent, but it captures windows, not views; a borderless
  hosting window or content-rect crop solves it. Option (b) is
  dependency-free but loses smoothness and costs CPU at capture rate.
  This is the central implementation decision; the draft leans (a) with
  a content-rect crop, falling back to (b) if SCStream framing proves
  unreliable.
* **Container**: WebM is ZCode's choice; AVFoundation natively writes
  MOV/MP4 (H.264). Options: write MP4 natively and label it as such, or
  transcode to WebM. Lean MP4 native and update the finding's framing
  (the parity goal is the capability, not the container), author decides.

## Critical User Journeys

### CUJ 1: record an agent session

* **Given**: the agent is driving a form flow in the browser pane.
* **When**: the user starts recording from the pane toolbar.
* **Then**: page content records to a file in the artifact folder; a
  recording indicator shows in the tab; stopping attaches the file as a
  transcript artifact with duration and size.

### CUJ 2: agent-started recording

* **Given**: a debuggable flow.
* **When**: the agent calls the recording tool (ask-gated).
* **Then**: recording starts with the user's one-time approval for the
  session, stops at session end or tool call, and attaches the same
  artifact.

### CUJ 3: cap safety

* **Given**: a long session hits the size cap.
* **When**: the cap is reached.
* **Then**: recording stops automatically, the file is finalized, and a
  named notice explains the stop; nothing is truncated corrupt.

## Technical Plan (Architecture)

```text
BrowserPaneView toolbar / recording tool
       |
       v
BrowserRecordingController (new)
  start(tabId): disk precheck, SCStream config
    (content rect = WKWebView bounds), AVAssetWriter
  frames: CMSampleBuffer -> writer input
  stop(): finalize, attach artifact, audit log entry
       |
       v
Artifact folder (existing per-chat artifact storage)
```

Recording is bounded by a size cap (default 500 MB, settings-tunable)
and a duration cap; both stop cleanly. The controller enforces one
recording at a time across tabs.

## Alternatives Considered

### Alternative A: snapshot stitching at 5 fps

* **Pros**: no screen-capture permission, pure WKWebView APIs.
* **Cons**: choppy output, CPU cost during agent runs, and captures can
  interleave badly with programmatic scrolling; timing bugs become
  invisible exactly when recording is used to find them.
* **Reason for Rejection**: Rejected by author: fallback option only.

### Alternative B: record the whole window

* **Pros**: simplest SCStream config.
* **Cons**: captures chat content and other panes into a shareable
  file; privacy surface grows for no capability gain.
* **Reason for Rejection**: Rejected by author: page-only capture is
  the privacy boundary.

## Non-Functional Requirements

* **Security/privacy**: capture is page-content-only; files live in the
  profile's artifact storage; the agent-facing tool is ask-gated always
  (never auto-allowed in any mode, matching the computer-use family's
  posture); starting and stopping writes audit entries.
* **Performance**: hardware-encoded H.264 via AVAssetWriter; frame
  delivery off the main thread; recording adds no work to the transcript
  render path.
* **Localization**: indicator, notices, and settings strings in the
  catalog for 21 languages.

## Detailed Implementation Plan

### Target Files to Create/Modify

1. `swift/TurboSparkApp/Sources/TurboSparkApp/Browser/BrowserRecordingController.swift` (Create)
   * **Type / Scope**: recording pipeline.
   * **Proposed Changes**: SCStream content-rect capture of the browser
     webview (or snapshot-stitch fallback), AVAssetWriter MP4 output,
     size and duration caps with clean finalization, single-recording
     enforcement, disk precheck.
2. `swift/TurboSparkApp/Sources/TurboSparkApp/Browser/BrowserPaneView.swift` (Modify)
   * **Type / Scope**: UI.
   * **Proposed Changes**: record button and indicator, stop and attach
     flow, notices.
3. `swift/TurboSparkApp/Sources/TurboSparkApp/Tools/Browser/BrowserToolDefinitions.swift` (Modify)
   * **Type / Scope**: agent tool surface.
   * **Proposed Changes**: `browser_start_recording` and
     `browser_stop_recording` tools, ask-gated in every mode, hidden for
     text-only configurations (matching the media capability pattern).
4. `swift/TurboSparkApp/Sources/TurboSparkApp/State/MacAppSettings.swift` (Modify)
   * **Type / Scope**: settings.
   * **Proposed Changes**: size cap, duration cap, and format fields.
5. `swift/TurboSparkApp/Tests/TurboSparkAppTests/BrowserRecordingTests.swift` (Create)
   * **Type / Scope**: unit tests.
   * **Proposed Changes**: controller state machine with a stubbed
     frame source (start/stop/cap/finalize), single-recording
     enforcement, disk precheck refusal, artifact attachment payload,
     tool gating (ask in every mode).
   * Mutation checks: drop the single-recording guard and assert the
     enforcement test fails; auto-allow the tool in one mode and assert
     the gating test fails.

### Verification Plan

* **Compilation/Build**: `make swift-lib`, then build the app scheme.
* **Unit Tests**: `swift test` for the controller tests.
* **Manual**: record 30 seconds of an agent-driven page, stop, attach,
  and play back the file; exercise the cap with a small configured cap.
* **Note**: ScreenCaptureKit capture of a live window requires a real
  screen and screen-recording permission; this manual gate runs on
  hardware, not in CI, consistent with the repo's hardware-gate rules.

--------------------------------------------------------------------------------

*Drafted with EGM*
