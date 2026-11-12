# Stage 5: Initial Zoom Out report (OC-02)

Target: docs/designs/opencode2-parity/OC-02-notifications-and-sounds.md (full file, 202 lines). Reviewers: egm-zoom-out (ZO), egm-inventive-catalyst (IC). Both completed.

## Zoom Out (ZO)

ZO-1 Major, lines 131-135, 171-173, 56, 124. Contradiction: AutomationExecutors "keeps its own direct use in v1" and "migrates to the wrapper in the same change". PushNotificationExecutor (AutomationExecutors.swift lines 50-51) calls requestAuthorization at tool-call time, which conflicts with "authorization only from the settings Allow button". The doc does not say whether agent-initiated PushNotification is exempt from the unfocused-only rule and dedupe, and it also drives an in-app toast through `onNotificationPushed`. Migrating is a refactor outside the stated scope: decide here or defer.

ZO-2 Major, lines 115-129, files 1-2. Two new types plus an `AppAttentionEvents.notify` facade that is not in the file list. The policy is about three rules; one small type with an injected `isActive` closure and a center protocol gives the same testability. Define or drop `AppAttentionEvents`.

ZO-3 Major, lines 22-29, 70-71, 79-84, INDEX dependency note. Hidden dependency on ZP-02, the desktop-shell-hardening spec (brief did not resolve in this worktree) and implicitly ZP-23. No interface is named. Without ZP-23 only the selected chat generates, so unfocused delivery may fire only when the user switches away mid-turn. State which scenarios trigger in v1.

ZO-4 Major, lines 79-84, file 4. Hook points in four AppModel extension files are asserted, not verified. "Each addition is one policy call" is unconfirmed. Name the existing seam. Defer the Auto-resolve countdown question until ZP-02 decides.

ZO-5 Major, lines 20-23, 45-47, 60-62. Per-event sound picker is a speculative surface. Alternative: per-event sound on/off using `content.sound = .default` (the existing executor does this), dropping NSSound. If UNNotification carries the sound, a separate NSSound step may double-play. Say which channel owns the sound.

ZO-6 Major, line 109, CUJ 3, lines 56-59. Line 109 says the toggle "stays on" after decline; line 58 says declining leaves all events off. Resolve, and specify the authorization-state UI in one place.

ZO-7 Major, lines 48-50, 66-67. "NSApp is not active" differs from "no window focused" (hidden or minimized windows, other app windows). No trigger named to clear slots and remove delivered notifications on app activation.

ZO-8 Nit, lines 139-143. "Under 200 ms" is unmeasured; reword as "never blocks the main actor". Chat title is user content on the lock screen: record redaction as an open question.

ZO-9 Nit, lines 148-198, file 6. Mutation-check phrasing and test inventory are low altitude. Persisting toggles "with the archive" is likely the wrong layer; ZP-02 uses `MacAppSettings`.

ZO-10 Nit, lines 3-6, 35-37. Executor link on line 36 is relative to the doc folder and will not resolve. The only existing UserNotifications use is AutomationExecutors.swift line 50 (verified by search). The dock-badge claim could not be verified because the brief was not found.

## Inventive Catalyst (IC)

IC-1 Major, lines 115-118, 164-169. Imperative `notify` call sites in four AppModel files duplicate state ZP-02's ChatBadgeState already computes; they can drift from sidebar state. INDEX.md line 73 says OC-02 should share ZP-02's state source but the diagram has no connection. Fix: make the policy an observer of badge-state transitions (pending-appeared, unread-raised, running-ended, error-raised); keep a small hook for error. Cost: hard dependency on ZP-02's state shape.

IC-2 Major, lines 51-53, 54-55, 94-95. Dedupe contradicts itself: newer event replaces pending (51-53), no second notification until handled (95), three clear conditions (refocus, click, handled). A finished or error event could replace a blocking permission cue. Fix: priority ordering (permission and answer above error above finished), replace only upward; retract delivered notifications with removeDeliveredNotifications when state resolves or app focuses. Verify re-sound on identifier repost.

IC-3 Major, lines 56-59, 133-135, 171-173. Same AutomationExecutors contradiction as ZO-1. Fix: wrapper owns authorization and posting only; agent-initiated pushes use an explicit channel that bypasses the focus rule, reuses authorization state, and never prompts at call time. Keep the PushNotification result text ("posted" vs "in-app only") truthful.

IC-4 Major, lines 60-62, 127-128. NSSound is a second uncoordinated channel: ignores Focus and Do Not Disturb, plays when authorization is declined, can sound without a banner. Fix: set the sound on the notification content; use NSSound only after confirmed authorization and a successful post. Named system sounds may need a copy to Library/Sounds, which touches "no bundled audio".

IC-5 Major, lines 48-50, 97-102. No cue when the app is focused on a different chat (ZP-23 background chats): the app is active, no notification, and ZP-02 is "when built". Fix: suppress only when the app is active and the event chat is selected, or state the interim behavior.

IC-6 Nit, lines 174-176. Toggles persisted "with the archive" ties preferences to chat export and migration. Use settings as ZP-02 does.

IC-7 Nit, lines 141-143, 93. Privacy statement inconsistent: "chat title only" then "names the tool". Title can be sensitive on lock screen. Fix: generic-text option, default as written.

IC-8 Nit, lines 137-140, 79-84. 200 ms has no measurement method. Auto-resolve "one notification" lean depends on ZP-02's open question; record as blocking or defer.
