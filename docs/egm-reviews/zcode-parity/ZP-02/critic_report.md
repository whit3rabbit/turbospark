# Stage 7: Critic report (ZP-02)

Reviewers: egm-design (DC), egm-structural (ST), egm-style (SY), run as one round via direct dispatch, standard feature-spec-standard for design and structural. All three completed. One round only, so convergence was not assessed. No remediation applied.

## Design critic (DC)

DC-1 (Blocker) Lines 116-130. Alternatives fail the rule: both say "Rejected by author" with a one-line rationale, neither is a real competitor. Missing: last-seen timestamp versus revision CAS (ZCode uses a persisted `unreadAt`), unread from assistant-reply count, the three auto-resolve options, a system notification for pending interactions.

DC-2 (Blocker) Lines 17-19, 41-42, 58-62, 85, 163-172. Countdown and auto-resolve are undecided yet the plan builds on them. No default window or snooze values, no timer owner or quit and deselect behavior. Safety: auto-resolving an approval with an empty payload is undefined; if it means approve it is a privilege escalation through `AppToolPermissionEngine`. Lines 170-172 (dismissal path) conflict with option (b) (empty payload). Scope the countdown to questions, display-only by default, or state approval semantics and ladder position.

DC-3 (Major) Lines 36-38, 69-76, 87-92 versus ZP-23 83-87. Priority pending > unread > running means a streaming background chat shows the dot, never the spinner, contradicting ZP-23 CUJ 1. CUJ 3 defines the spinner for the selected row only. Error state dropped without comment. Define unread at turn completion or put running above unread; define background spinner.

DC-4 (Major) Lines 63-65, 69-76, 110-114, 137-139, 147-151. CAS is unspecified: no evidence the revision exists or its granularity. If it bumps per streamed token, selecting a streaming chat re-raises the dot on the selected row. State what the revision is, when it bumps, and why a simple flag is insufficient.

DC-5 (Major) Lines 63-65, 69-72. Unread definition open, CUJ 1 silently picks the broader one; tool-only churn raises the dot per chunk.

DC-6 (Major) Lines 41-42, 83-85, 160-162. Click behavior ambiguous (navigate versus snooze); no keyboard or VoiceOver path; snooze-once persistence across relaunch or deselection unstated.

DC-7 (Major) Lines 36, 103-104, 154-155. Pending-state source unproven: app model gates on a single `pendingToolCall == nil` (global, not per chat). Add a hard ZP-23 prerequisite and interface for per-chat pending state, and what ships if ZP-23 slips.

DC-8 (Major) Lines 142-186. Persistence and migration underspecified: storage layer unnamed (profile DB, archive, chat record), no schema change, default for existing chats on upgrade, cleanup on archive or delete, branching and cron-fired messages.

DC-9 (Major) Whole document. No Risks and Limits section: timers while backgrounded, a countdown that never resolves, lost CAS write, many rows lighting at once, auto-resolve security, redraw cost, success metrics.

DC-10 (Major) Lines 137-139 versus 160-166. "No polling" beside a live countdown; state timer mechanism, tick rate, owner, Reduce Motion behavior.

DC-11 (Nit) Lines 44-54. Non-Goals thin and one bullet is a dependency; dock badge feed has no interface or file.

DC-12 (Nit) Lines 173-179. Tests omit countdown expiry and auto-resolve, snooze persistence, the selected-and-streaming re-raise, upgrade migration, accessibility labels.

DC-13 (Nit) Lines 134-136, 181-185, 195. "21 languages" and "ten gates" unsourced; row-label wording unspecified.

DC-14 (Nit) Lines 9-11. Status Draft but reads as decided ("adopted") beside open questions; state what ZP-02 does before ZP-23 (code largely dead).

## Structural (ST)

Heading hierarchy is clean; fenced block closed.

ST-1 (Blocker) Lines 116-130. Same as DC-1: "Rejected by author" with no linked issue; A is a strawman, B is declared a complement at line 30. Move the Open Question options and the timestamp-versus-flag choice here.

ST-2 (Major) Whole document. Missing "Risks and Mitigations": existing-chat migration, stale revision token, privacy via badges and accessibility labels, recompute cost, empty-payload auto-resolve and hooks.

ST-3 (Major) Lines 56-65 versus 41-43, 84-85, 110-114, 173-179. Two open decisions treated as settled elsewhere; resolve or mark dependent goals, CUJs and tests.

ST-4 (Major) Lines 47-54. Second Non-Goals bullet is a dependency; missing exclusions (error state, reordering, dock badge work); ZP-90 reference unlinked.

ST-5 (Major) Lines 44-45, 113-114, 24-32. Stale text contradicts ZP-23 adoption.

ST-6 (Major) Lines 87-92, 100-105 versus ZP-23 CUJ 1 (83-87). Spinner scoped to selected row; add a background-running journey.

ST-7 (Major) Lines 41, 44, 84-85, 167-172. Pending pill scope ambiguous: countdown for approvals unspecified; selected chat with open card undefined versus "exactly one slot".

ST-8 (Major) Lines 144-185, 150-151, 181. Architecture incomplete: persistence location unnamed, no pending-state data flow or timer lifecycle, no catalog or settings UI file entries; item 7 "Unchanged" belongs in Verification.

ST-9 (Nit) Lines 30, 44, 127. Dock badge spec unlinked (INDEX links `.kiro/specs/desktop-shell-hardening/brief.md`); name the owner of the shared count.

ST-10 (Nit) Line 94. Heading differs from standard "Proposed Architecture" and ZP-23's pattern.

ST-11 (Nit) Lines 179-180. Mutation checks sit as a sibling bullet outside the Type / Scope / Proposed Changes schema.

ST-12 (Nit) Lines 118-130. Pros are one-liners; section reads as perfunctory.

## Style (SY)

SY-1 (Major) Lines 58-62, 18, 41-42. "Most defensible" gives no criterion; goals assume the countdown the question leaves open.

SY-2 (Major) Lines 63-66, 71-72. "Matches chat conventions" unsupported; CUJ 1 contradicts the open question.

SY-3 (Major) Lines 49-55, 113-114. Dependency in Non-Goals; lines 113-114 describe the pre-adoption state.

SY-4 (Major) Lines 44-45. "Future concurrent-chats world" stale; name ZP-23 and the dock-badge spec.

SY-5 (Major) Lines 25-33. Claims about existing behavior (unread search, progress flame icon) lack file or symbol citations.

SY-6 (Major) Lines 134-136, 183-185, 195. "21 languages" and "ten gates" uncited.

SY-7 (Major) Lines 136-138, 82, 168-172. "No polling" qualify: badge state not polled, countdown uses a timer on pending rows.

SY-8 (Nit) Lines 17-21. "State machine" with no transition table; "sized for" is filler.

SY-9 (Nit) Lines 121-123, 128-130. Slogan rationales; "trivial"; "across generation sessions" unclear.

SY-10 (Nit) Lines 83-85, 40, 165, 88-92. Oblique "navigation no longer stops anything"; "interval" versus "extension" naming; awkward "yields".

SY-11 (Nit) Lines 28, 36-37, 105, 160. Inconsistent state names (default icon, default bubble, progress flame); running definition disagrees across Goals, CUJ 3 and plan.
