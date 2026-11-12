# Stage 6: Comprehension report (OC-02)

Reviewer: egm-comprehension (completed). Read the document alone, no GoldfishContext. No Blockers.

## What this proposes

The document proposes out-of-app desktop notifications for TurboSpark (a macOS app) covering four events: turn finished, needs permission, needs answer, and error. Each event has a persisted on/off toggle and an optional named system sound. A notification posts only when the app is not the active app. At most one notification per chat is pending at a time, and a newer event replaces the older one. Clicking a notification activates the app and selects that chat. macOS authorization is requested only when the user enables the first toggle and presses Allow in an explainer. The work is split into a pure policy object, a notification-center wrapper, a settings section and hooks in the app model. The existing automation code is migrated to the wrapper. Dock badges and in-window sidebar states are explicitly out of scope.

## How the existing system works, per this document

The app runs agents locally and unattended and currently gives no out-of-window signal when a background turn finishes or blocks. UserNotifications is used only inside automation executors, directly. There are no sound assets, no per-event settings in the General pane, and no dock badge or attention request on turn completion. In-window cards and toasts already exist for focused sessions, and an "AppModel" with turn, permission-gate and question-flow lifecycle points exists. The document also refers to existing VoiceOver announcements, a settings search catalog fed by `.settingsControl`, a persisted archive, and a rule that the app never prompts at launch. All of these are mentioned but not described.

## Findings

CO-1 Major, lines 133-135, 171-173. Contradiction: AutomationExecutors "keeps its own direct use in v1" and "migrates to the wrapper in the same change"; file item 5 modifies it. Pick one.

CO-2 Major, lines 79-84. "Event source wiring" reads as a decision, not a question. "Auto-resolve countdown" is undefined here and depends on ZP-02. The doc does not say how the dedupe slot handles a needsAnswer event followed by a countdown-expiry event.

CO-3 Major, lines 51-55, 91-95. Dedupe release is unclear: "until the app is focused again", "clear on click", "until handled" are three conditions. Whether a replacing event of a different kind (error over needsPermission) sounds again is unstated. The injected clock (lines 119-122) has no described time-based behavior.

CO-4 Major, lines 109-111 vs 56-58. Toggle "stays on" after declined authorization contradicts "declining leaves all events off".

CO-5 Major, lines 56-59, 104-108. Authorization flow incomplete: no behavior when the user later changes authorization in System Settings, or when automation already obtained authorization.

CO-6 Major, lines 79-82, 164-169. Hook points vague. "Turn finished" vs error vs cancellation not distinguished; AppModel+Cancellation.swift is modified but whether a cancelled turn notifies is not stated.

CO-7 Major, lines 47-50. "No window focused" is equated with NSApp not active. Multi-window support unstated; "focused session" vs the event's chat unrelated. A background chat in an active app gets no notification: intended? CUJ 2 covers only the typing case.

CO-8 Major, lines 139-140, 123-129. "Under 200 ms" has no start, end or measurement and no test. "Sounds never delay the post" is not reflected in the sequential diagram.

CO-9 Nit, lines 3-6, 25-29, 71. External references needed for scope; "shared state source" undefined; "when it lands" gives no order or fallback.

CO-10 Nit, lines 76-77. "the app already has 18" is missing its noun (panes).

CO-11 Nit, lines 16-17, 115-117. `AppAttentionEvents.notify` appears only in the diagram, not the file list. New type, facade, or part of the policy file?

CO-12 Nit, lines 43-46, 60-63. "A small set of system sounds" not enumerated; how NSSound respects alert volume is asserted; default sound and default toggle states missing.

CO-13 Nit, lines 31-41, 144-147. "Existing announcements" for VoiceOver not described. "In-window cards always appear" vs cards for focused sessions only: do cards appear when the app is in the background?

CO-14 Nit, lines 177-178, 195-198. "21 languages", localization gates, `make design-check`, DESIGN.md section 9 are undefined or unlinked; "mutation check" introduced only at lines 182-183.

CO-15 Nit, lines 31-41. "Attention request" (presumably NSApp.requestUserAttention) is undefined and has no owner in Goals or Non-Goals.
