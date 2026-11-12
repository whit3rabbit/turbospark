# Stage 8: Intermediate Zoom Out report (OC-02)

Reviewer: egm-zoom-out (completed). Target unchanged since Stage 5, so all Stage 5 findings remain open. Context is sufficient; no Blockers at zoom-out altitude. Reuse verified: AutomationExecutors.swift lines 49-56 keep the bundle guard, tool-call-time requestAuthorization (line 51) and content.sound = .default (line 56); line 42 fires onNotificationPushed. ZP-02 has Alternatives (line 116), ChatBadgeState (line 101), MacAppSettings (line 107).

## Open Stage 5 findings (S8-M)

S8-M1 Major, lines 131-135, 171-173, 56. Keep-vs-migrate contradiction; executor requests authorization at tool-call time. Simplest fix: defer migration, make the change additive, note two-owner overlap. If migrating, agent pushes bypass the focus rule and never prompt. [ZO-1, IC-3, CO-1, ST-4, SY-1]
S8-M2 Major, lines 56-58 vs 109-111. Declined-authorization contradiction. Keep toggle on with a status row; state the UI once. [ZO-6, CO-4, DC-8]
S8-M3 Major, lines 60-62, 127-128. NSSound is a second uncoordinated channel. Simplest: per-event sound on/off via notification content, no NSSound, no picker; or gate on a successful post and name the owner. [ZO-5, IC-4, DC-9]
S8-M4 Major, lines 115-129, files 1-2. `AppAttentionEvents.notify` appears only in the diagram; fold the policy into one type with an injected isActive closure and a small center protocol, or drop the facade. [ZO-2, CO-11, ST-6, SY-13]
S8-M5 Major, lines 79-84, 164-170. Hook points unverified; state a cancelled turn does not fire turnFinished; move "Event source wiring" out of Open Questions; defer Auto-resolve question. [ZO-4, DC-7, CO-6, SY-15, ST-3]
S8-M6 Major, lines 26-29, 70-71, 73-84. Dependency interface gap: one or two sentences that the policy observes ZP-02 ChatBadgeState transitions, fallback minimal hooks if ZP-02 slips, and that without ZP-23 only the selected chat generates. Brief link unresolved: fix or mark unverified. [ZO-3, DC-6, ST-7]
S8-M7 Major, lines 48-50, 97-102. Define the focus rule in one sentence (active app and the event's chat selected means suppress); one trigger to clear slots on activation. [ZO-7, IC-5, DC-4, CO-7]

## Keep vs decline (Stage 6 and 7 asks)

Keep (ambiguity removal): the contradictions above; one-line dedupe priority (permission and answer not replaced by finished or error) with one release rule; name the sound channel; one-sentence ZP-02 dependency; one-line click routing (delegate set at app start so cold-launch clicks are not lost; click on a deleted chat only activates the app); fix broken link and MacAppSettings storage reference.

Alternatives Considered and Risks and Mitigations sections (DC-1, ST-1, SY-16, ST-2, DC-10): zoom-out sees these as process requirements; keep each to 3-5 bullets since the standard requires them. Alternatives worth listing: content.sound vs NSSound, single master toggle vs four events, observe badge state vs imperative hooks, unfocused-only vs always post. Risks: flood across chats, stale slot, revoked authorization, double sound, lock-screen title. Do not add rate caps or coalescing.

Decline (altitude drop or speculation): persisted-store migration and defaults story, Focus and locked-screen manual checks, occlusion and Spaces definitions, identifier scheme, revocation refresh, acceptance criteria and mock seam listing, sound name enumeration, hook-emission and persistence tests, VoiceOver and DESIGN.md linking, alert style None, subagent coverage, 200 ms measurement method, Open Questions reordering, file-item formatting.

## Nits (S8-N)

S8-N1 lines 36, 174-176, 139-143, 76-77, 59: broken executor link and overstated "nothing tells the user"; MacAppSettings vs "archive"; reword 200 ms as "never blocks the main actor"; "18" lacks noun; bare "permission rules" reference.
S8-N2 lines 148-198: plan items 8 (mutation check) and 9 (test inventory) are already low altitude; trim, do not add. Target net change about 20-30 lines.
