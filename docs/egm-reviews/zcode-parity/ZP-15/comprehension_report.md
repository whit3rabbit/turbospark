# Comprehension report (Stage 6): ZP-15 browser video recording

Target: docs/designs/zcode-parity/ZP-15-browser-webm-recording.md (full file). Reviewer: egm-comprehension, completed.

## What this proposes

Add a feature to the macOS app's embedded browser pane that records the agent-leased tab (page content only, not the window or other panes) to a video file. The user starts and stops it from the pane toolbar, or the agent does so through two ask-gated tools. A new BrowserRecordingController would capture frames using ScreenCaptureKit (SCStream) cropped to the web view, write them with AVAssetWriter, and enforce one recording at a time, a size cap (default 500 MB), a duration cap and a disk-space precheck. The finished file goes in the chat's artifact folder and is attached to the transcript. The document leans toward writing MP4 rather than WebM, and the author has not yet decided.

## How the existing system works, per this document

TurboSpark is a Swift macOS app with an embedded browser (WKWebView) in Browser/ and BrowserAutomation/. Today the only evidence of an agent browser session is screenshot-tool captures and the tool transcript. A separate "helper app" already uses ScreenCaptureKit for front-window capture (Computer Use). A per-chat artifact storage exists, along with an ask-gating permission model, a "computer-use family" of tools, a "media capability pattern" that hides tools for text-only configurations, and an audit log. The document does not describe the details of any of these.

## Findings

CO-1 (Major) Lines 1-6, 27, 62-65, 148-150, 162-163. Title and ZCode reference say WebM, body says MP4; deliverable format never stated. Fix: settle the container or mark it provisional and rename title and file.

CO-2 (Major) Lines 30-34, 49-61, 92-106. "Only that tab" and page-only goals conflict with window-region pixel capture; occlusion, scroll, resize and hidden-pane behavior unstated. Fix: add a paragraph on those cases and whether recording pauses.

CO-3 (Major) Lines 51-61, 98-100, 146-151, 180-182. Helper app uses ScreenCaptureKit but the controller sits in the main app; permission owner, grant flow and denial behavior unstated. Fix: name the owning process and add a permission-denied journey.

CO-4 (Major) Lines 77-90, 133-135. CUJ 2 is unclear: "stops at session end or tool call" is ambiguous; one-time session approval conflicts with "ask-gated always"; "debuggable flow" undefined. Fix: define approval scope, stop trigger and precondition.

CO-5 (Major) Lines 30-34, 108-110, 163. "Opt-in per session" has no control described; no duration cap default; "format fields" contradict a single format. Fix: define the opt-in control, give the default, drop the format field.

CO-6 (Major) Lines 85-90, 108-110. "Nothing truncated corrupt" and "named notice" are unsupported for crash, quit, disk full; "named" is undefined. Fix: state finalize-on-quit and crash behavior and the notice text.

CO-7 (Major) Lines 33-37, 105, 132-133. "Chat's artifact folder", "per-chat artifact storage", "profile's artifact storage" may differ; no naming, retention or deletion rules. Fix: define the folder once with path scheme and cleanup rules.

CO-8 (Major) Lines 31-32, 98-100, 109-110. "Agent-leased tab" and "tabId" undefined; lease end, navigation or close mid-recording unstated. Fix: define "leased" and the mid-recording behavior.

CO-9 (Major) Lines 44-47, 133, 139. ZP-14 referenced without link; audit log location and fields unnamed. Fix: link ZP-14 and name the audit log and entry fields.

CO-10 (Nit) Lines 3-6, 18-20. ZCODE_COMPARISON finding, "recording spec", "kiro spec", "CI smoke" unexplained. Fix: one-line gloss or link each.

CO-11 (Nit) Lines 51-61, 114-120. Alternative A (5 fps) is also the fallback; fps never specified elsewhere and the fallback changes the permission story. Fix: state default fps and the permission change.

CO-12 (Nit) Lines 156-160. "Media capability pattern" and text-only gating undefined and unjustified. Fix: link the pattern or justify in one sentence.

CO-13 (Nit) Lines 139-140. "21 languages" and "the catalog" assumed. Fix: link the localization doc.

CO-14 (Nit) Lines 176-182. "Hardware-gate rules" and `make swift-lib` unexplained; no manual check for caps, indicator or denied permission. Fix: link the rule and add the checks.

CO-15 (Nit) Lines 5, 22-28. ZCode introduced without a description. Fix: one-line description at first mention.
