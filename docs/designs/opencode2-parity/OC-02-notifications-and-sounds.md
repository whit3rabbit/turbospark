# OC-02: Per-event desktop notifications and sounds

Findings: Section 3, "Notification and sound settings - Missing in
Swift" and the notification half of Section 9a, "Session unread and
needs-attention state" in
[OPENCODE2_COMPARISON.md](../../OPENCODE2_COMPARISON.md).
Parity targets: OpenCode's
[notifications page](https://github.com/anomalyco/opencode/blob/dev/packages/app/src/settings/notifications/notifications.tsx),
[coordinator](https://github.com/anomalyco/opencode/blob/dev/packages/app/src/shell/notifications/coordinator.ts),
and
[sound](https://github.com/anomalyco/opencode/blob/dev/packages/app/src/shell/notifications/sound.ts).

* Author: whit3rabbit
* Status: Draft
* Date: 2026-11-11

--------------------------------------------------------------------------------

## Overview

A four-event notification matrix (turn finished, needs permission, needs
answer, error) delivered through UserNotifications when no app window is
focused, each event with an on/off toggle and an optional system sound,
plus click-through that focuses the window and selects the chat. The
dock-badge half of the same gap is already in the
[desktop-shell-hardening](../../../.kiro/specs/desktop-shell-hardening/brief.md)
spec; the in-window sidebar row states are
[ZP-02](../zcode-parity/ZP-02-unread-pending-badges.md). This draft owns
the out-of-app delivery channel.

## The Problem

TurboSpark's agents run locally and unattended, and nothing tells the
user when a background turn finishes or blocks. `UserNotifications` is
used only inside automation executors
([AutomationExecutors.swift](swift/TurboSparkApp/Sources/TurboSparkApp/Tools/Automation/AutomationExecutors.swift));
there are no sound assets, no per-event settings in the General pane,
and no dock badge or attention request on turn completion. OpenCode's
answer is a per-event toggle matrix with sounds, delivered only when no
window is focused and deduplicated across windows. The same shape fits
TurboSpark without any cloud component.

## Goals

* Events: `turnFinished`, `needsPermission`, `needsAnswer`, `error`.
  Each has a persisted toggle and a sound picker (none plus a small set
  of system sounds; no bundled audio files).
* Delivery rule: a notification is posted only when `NSApp` is not
  active (no window focused). Focused sessions get the existing in-window
  cards and toasts instead. This mirrors OpenCode's unfocused-only rule.
* Deduplication: at most one outstanding notification per chat until the
  app is focused again; a newer event replaces the pending one for that
  chat instead of stacking.
* Click action: activate the app, select the chat, and clear that chat's
  dedupe slot.
* Authorization is user-initiated: enabling the first event toggle shows
  a short explainer with an Allow button; declining leaves all events
  off and nothing else in the app changes. No prompt at launch, matching
  the app's permission rules.
* Sounds play through `NSSound` named system sounds only when the event
  toggle is on; sounds respect the system alert volume and are not a new
  audio pipeline.

## Non-Goals

* No notification when the app is focused (not a setting in v1; the
  in-window attention surfaces own that case).
* No per-chat or per-project notification overrides.
* No custom sound file import.
* No dock badge work here (owned by the desktop-shell-hardening spec);
  this draft only calls its shared state source when it lands.

## Open Questions

* **Settings placement**: a Notifications section in the General pane
  (OpenCode's shape) or a dedicated pane given the app already has 18.
  Lean: General section; promote to a pane only if it grows a third
  block (sounds plus scheduling).
* **Event source wiring**: fired from `AppModel` turn lifecycle
  (finish, error), from the permission engine gate (needsPermission),
  and from the question flow (needsAnswer). Whether the question event
  also arms when the Auto-resolve countdown in
  [ZP-02](../zcode-parity/ZP-02-unread-pending-badges.md) is about to
  elapse (lean: yes, one "needs you" notification, not two).

## Critical User Journeys

### CUJ 1: unattended turn needs permission

* **Given**: a chat running a tool that requires approval, all toggles
  on, app in background.
* **When**: the approval card opens.
* **Then**: a notification appears ("Chat X needs approval") with a
  sound if configured; clicking it brings the app forward with that chat
  selected; no second notification for the same chat until handled.

### CUJ 2: focused app is silent

* **Given**: the same event while the user is typing in the app.
* **When**: the approval card opens.
* **Then**: no notification and no sound; the in-transcript card and
  pending pill (ZP-02, when built) are the cues.

### CUJ 3: authorization declined

* **Given**: the user enables "Turn finished" for the first time and
  declines the macOS authorization prompt.
* **When**: turns finish afterwards.
* **Then**: nothing is posted, the toggle stays on but shows a
  "notifications are off in System Settings" state with an Open System
  Settings action, and everything else works unchanged.

## Technical Plan (Architecture)

```text
AppModel turn/permission/question/error hooks
        |  (one call: AppAttentionEvents.notify(event, chat))
        v
AttentionEventPolicy
  - suppress if NSApp.isActive
  - replace per-chat pending item (dedupe slot)
        v
AppNotificationCenter (UNUserNotificationCenter wrapper)
  - request authorization only from the settings Allow button
  - post with chat id in userInfo
  - delegate callback: click -> select chat, clear slot
        v
NSSound play (event sound, if configured)
```

The policy object is plain and testable with an injected
`isActiveProvider` and clock, so the delivery matrix is covered by unit
tests without posting real notifications. `AutomationExecutors` keeps
its own direct `UNUserNotificationCenter` use in v1 and migrates to the
wrapper in the same change so there is one authorization owner.

## Non-Functional Requirements

* **Latency**: event to posted notification under 200 ms; sounds never
  delay the post.
* **Privacy**: notification bodies carry the chat title only, never
  transcript content; the permission event names the tool, not its
  arguments.
* **Accessibility**: VoiceOver users get the same cues in-window via the
  existing announcements; notifications are a visual/auditory channel
  and never the only signal (in-window cards always appear).

## Detailed Implementation Plan

### Target Files to Create/Modify

1. `swift/TurboSparkApp/Sources/TurboSparkApp/App/AppNotificationCenter.swift` (Create)
   * **Rationale**: the single `UNUserNotificationCenter` owner:
     authorization state, posting, click routing, settings-mismatch
     surfacing.
2. `swift/TurboSparkApp/Sources/TurboSparkApp/App/AttentionEventPolicy.swift` (Create)
   * **Rationale**: the pure delivery matrix (event toggles, focus rule,
     per-chat dedupe slots). No framework types in the decision path so
     it tests headlessly.
3. `swift/TurboSparkApp/Sources/TurboSparkApp/App/GeneralSettingsPaneView.swift` (Modify)
   * **Rationale**: the Notifications section: four event toggles, sound
     pickers, authorization status row. Canonical location; the settings
     search catalog picks it up via `.settingsControl`.
4. `swift/TurboSparkApp/Sources/TurboSparkApp/State/AppModel+Queue.swift`,
   `swift/TurboSparkApp/Sources/TurboSparkApp/State/AppModel+Cancellation.swift`,
   `swift/TurboSparkApp/Sources/TurboSparkApp/State/AppModel+Questions.swift`,
   `swift/TurboSparkApp/Sources/TurboSparkApp/State/AppModel+Tools.swift` (Modify)
   * **Rationale**: emit the four events at the existing lifecycle
     points (turn completion path, error path, permission gate, question
     park). Each addition is one policy call.
5. `swift/TurboSparkApp/Sources/TurboSparkApp/Tools/Automation/AutomationExecutors.swift` (Modify)
   * **Rationale**: route through the shared wrapper so authorization
     and dedupe stay in one place.
6. `swift/TurboSparkApp/Sources/TurboSparkApp/State/AppModel+Persistence.swift` (Modify)
   * **Rationale**: persist the event toggle and sound-choice records
     with the archive.
7. `swift/TurboSparkApp/Localization/Localizable.xcstrings` (Modify)
   * **Rationale**: all new strings, 21 languages, same change.
8. `swift/TurboSparkApp/Tests/TurboSparkAppTests/AttentionEventPolicyTests.swift` (Create)
   * **Rationale**: matrix coverage: focused suppress, unfocused
     deliver, dedupe replace, toggle off silent, sound gating.
     Mutation check: invert the focus rule and assert the focused test
     fails.
9. `swift/TurboSparkApp/Tests/TurboSparkAppTests/AppNotificationCenterTests.swift` (Create)
   * **Rationale**: wrapper behavior with a mocked center: authorization
     request only from the Allow path, userInfo round trip, declined
     authorization state string.

### Verification Plan

* **Build**: `make swift-lib`, then the app scheme build.
* **Tests**: the new suites plus the localization gates and
  `make design-check`.
* **Manual**: with toggles on, run one turn unfocused: the notification
  arrives and clicking it focuses the right chat. Repeat focused: nothing
  arrives. Decline authorization and confirm the degraded-state row.
  Run the full DESIGN.md section 9 checklist on the new settings
  section.

--------------------------------------------------------------------------------

*Drafted with EGM*
