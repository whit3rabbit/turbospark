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
* Status: Partial (implemented 2026-11-14; deferred items below)
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

## Implementation record (2026-11-14)

Status is Partial: the notification channel, settings and tests are built;
two items are deferred (below) and the manual passes were not run.

Built:

* `AttentionEventPolicy.swift`: the pure matrix (toggles, focus rule,
  per-chat slots with priority) plus the persisted types
  (`AttentionEvent`, `AttentionSound`, `AttentionPreferences`).
* `AppNotificationCenter.swift`: the wrapper (authorization state, post,
  sound, click routing) over an `AttentionNotificationBackend` protocol; the
  real backend wraps `UNUserNotificationCenter` and answers `.unavailable`
  outside a signed .app bundle.
* `AppModel+AttentionEvents.swift`: the four emit points and the Settings
  mutators. One type plus a small backend protocol replaced the draft's
  `AppAttentionEvents.notify` facade (review M8).
* `NotificationsSettingsSection.swift`, added to the General pane: four
  toggles, a sound picker per event, one authorization row (Allow, or Open
  System Settings after a refusal). Four `SettingsControlCatalog` entries
  plus the section and the two actions; `generate-settings-catalog.py
  --check` passes.
* `MacAppSettings.attentionEvents` (tolerant decode; absent or junk means
  every event off). Persisted in the settings file, not with the chat
  archive (review M10; the draft's file 6 `AppModel+Persistence.swift` is
  the load/save of that field).
* 16 catalog keys in 21 languages.
* Tests: `AttentionEventPolicyTests` (14), `AppNotificationCenterTests` (13),
  `AttentionEventHooksTests` (14); the new enums are in
  `LocalizationParityTests.testEnumBackedDisplayStringsAreCatalogKeys`.

Event wiring (the draft's Open Question on source wiring):

* `needsPermission`: `pendingToolCall.didSet` (a card opening, read one hop
  later because the chat id is assigned after it) and `deferToolProposal`
  (a background chat parking a proposal).
* `needsAnswer`: `waitForUserAnswers`, when the question card publishes.
* `error`: `finishCancelled` with reason `error` or `context_overflow`.
* `turnFinished`: `finishTurnTail`, last, only when the chat is idle and not
  generating afterwards. A user Stop, an error, a parked approval and a
  turn that handed over to a queued one do not fire it (review M7).

Open Questions, resolved:

* Settings placement: a Notifications section in General (the draft's lean).
  Not promoted to a pane.
* Auto-resolve countdown arming the question cue: moot. No such countdown
  exists in the code today, so there is nothing to arm; one cue per
  question, raised when the card publishes.

Review findings, how each was settled:

* B1, B2 (Alternatives, Risks), short form. Rejected: a single master
  toggle (loses the "only errors" case the matrix exists for); posting the
  banner through content.sound alone (see M3); always posting regardless of
  focus (duplicates the in-window cards). Risks: flood across chats (one
  slot per chat, replaced in place, no rate cap); stale slot (released on
  app activation, on click, and when a post fails); revoked authorization
  (status re-read on activation and when Settings appears, events dropped
  while not authorized); click on a deleted chat (the app comes forward and
  selection is unchanged); double sound (played once per delivered post);
  chat title on a lock screen (kept, as the draft states; a Ghost chat
  never raises a banner).
* M1: `AutomationExecutors.PushNotification` is NOT migrated. It requests
  authorization at tool-call time, which contradicts the Allow-only rule,
  and routing an agent push through the focus rule would change a shipped
  behavior. Deferred (below). The wrapper never prompts from an event.
* M2: the toggle stays on while authorization is denied and the row says so
  (CUJ 3); "declining leaves all events off" in Goals is superseded.
* M3: kept NSSound named system sounds as the Goals say, with two stated
  limits: macOS Focus and Do Not Disturb do not silence it, and it plays
  only after a successful post. The banner itself carries no sound.
  Default sound is none, so enabling an event is a silent banner until the
  user picks one. Sounds offered: Glass, Ping, Pop, Hero, Basso. The
  alert-volume claim was dropped (unverified).
* M4: priority added. needsPermission and needsAnswer are never replaced by
  turnFinished or error; error is not replaced by turnFinished. One release
  rule with three triggers: app activation, click, post failure.
* M5: declined. The rule stays "app not active" as the Goals say. A
  background chat in an active app gets the sidebar pill and the parked
  card, not a banner. Reopen if that proves quiet in practice.
* M6: no ZP-02 state source is observed; the hooks fire from the lifecycle
  points directly (the draft's fallback). ZP-02 phase 1 and the dock badge
  spec do not expose a transition stream to compose with. Dock badge stays
  out of scope.
* M9: the delegate is set when the backend is created at `AppModel` init
  (setting a delegate does not prompt). A click selects the chat if it
  still exists.
* M11 and N-items: banner bodies are the chat title plus a fixed line; the
  permission line names the tool, never its arguments (tested).
* Latency: nothing blocks the main actor; the post is an async task and the
  sound follows it. The "under 200 ms" figure was not measured.

Deviations:

* No keyboard shortcut was added, so the OC-01 catalog is untouched.
* Tests do not cover `SystemAttentionBackend` itself (it needs a signed
  bundle); the delegate hop and the real `UNUserNotificationCenter` calls
  are unexercised.

Deferred:

* Migrating `AutomationExecutors.PushNotification` to the shared wrapper
  (needs a decision on whether agent pushes bypass the focus rule and
  whether the tool may keep prompting at call time).
* Dock badge and attention request (desktop-shell-hardening owns it).

Not run: manual unfocused/focused/declined passes, the DESIGN.md section 9
checklist (text sizes, themes, German and Arabic, VoiceOver), a real
notification in the installed app bundle, `make swift-app`.

Verification: focused app tests (AttentionEventPolicyTests 14,
AppNotificationCenterTests 13, AttentionEventHooksTests 14,
LocalizationParityTests, KeyboardShortcutCatalogTests,
SettingsWindowLayoutTests, DesignTokenRatchetTests: 66 pass),
`make compile-strings`, `make design-check`,
`generate-settings-catalog.py --check`. Mutation checks: inverting the
focus rule and disabling the rank check each fail the policy tests. The
finish-cue cancel guard was not mutation-checked. `make swift-lib` and
`make swift-app` were not run.

--------------------------------------------------------------------------------

*Drafted with EGM*
