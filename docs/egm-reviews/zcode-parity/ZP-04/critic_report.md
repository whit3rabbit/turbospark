# Stage 7: critic report (ZP-04)

Target: docs/designs/zcode-parity/ZP-04-subagent-coordinator-messaging.md (full file, 193 lines)
Reviewers: egm-design (DC, standard feature-spec-standard), egm-structural (ST, standard feature-spec-standard), egm-style (SY). All completed. One round only (no remediation applied, so convergence was not assessed).

## Design critic (DC)

DC-1 Blocker. Lines 61-69, 36-47, 114-117. Persistence is open yet Goals, CUJ 2 and the plan depend on it; Goals say "persisted transcript" while option (a) is in-memory. In-memory retention pins child context in a memory-limited local app. Fix: pick one for v1 or split SendMessage into its own spec; state whether resume reuses KV prefix or re-prefills.

DC-2 Blocker. Lines 38-47, 100-112, 167-170. Delivery to an idle coordinator is unspecified; a message could wait forever. Queue lifetime on session close, switch or app restart is undefined; interaction with ZP-23 not mentioned. Fix: define wake behavior (as task-notifications do) and queue lifetime.

DC-3 Major. Lines 63-69, 143-145. 4 KiB cap is only "suggested" and not in NFRs; oversize behavior undefined; worst case is 8 x 4 KiB per child with no aggregate cap; oldest-drop loses the key message silently. Fix: state values, overflow behavior, aggregate cap, tell the child about drops.

DC-4 Major. Lines 138-147. Prompt-injection through child text is asserted away by attribution; no escaping mechanism. Fix: escape carrier tags in the body, state carriers are untrusted data, add a test.

DC-5 Major. Lines 140-142, 163-166. RespondToCoordinator denied in read-only blocks the likeliest user (an explorer child); SendMessage resume re-enables tool use under an unstated permission state; hook re-firing unstated. Fix: allow in read-only or justify; state resume permission and hook behavior.

DC-6 Major. Lines 50-57, 146, 107. Depth-cap interaction with resume unstated; SendMessage addressing key undefined; SendMessage to a running, nonexistent or already-resumed child undefined. Fix: define the id and each case.

DC-7 Major. Lines 119-136. Alternatives are weak; missing: streaming partial results, scratch channel, spawn fresh child with prior result pasted (much cheaper than resume), and the persistence options. Steering-pill and hook-path claims uncited. Fix: add real alternatives, cite.

DC-8 Major. Lines 36-47, 71-90, 182-189. No failure modes (child cancelled with unread messages, messages after finish), no UI or transcript visibility for carriers, no localization note. Fix: state visibility and persistence of carriers and handle failures.

DC-9 Nit. Lines 50-54. Stray "Non-Go" duplicates Non-Goals. Fix: delete line 50.

DC-10 Nit. Lines 3-7, 17-23. Kiro brief unnamed; finding 3.5 is a negative search over two roots; overlap with finding 3.4 not noted. Fix: link and note.

DC-11 Nit. Lines 41, 106, 146. Carrier naming inconsistent. Fix: define one schema.

DC-12 Nit. Lines 163-167, 182-189. File 3 label malformed; file 4 either/or path; no Rust-side check if KV reused; tests miss oversize, wake-on-idle, escaping. Fix: resolve and add.

## Structural (ST)

ST-1 Blocker. Lines 127-128, 135-136. Both rejections read "Rejected by author" and restate the design; the standard forbids unlinked maintainer attribution. Fix: concrete trade-offs or a linked decision record.

ST-2 Major. Lines 119-136. Alternatives narrow; none for the retain-versus-persist fork or for resume versus fresh child; B is a straw man. Fix: add per major decision.

ST-3 Major. Missing section. No Risks and Mitigations section; risks scattered (prompt injection, resume after restart, depth interaction, message loss, context growth). Fix: add the H2.

ST-4 Major. Line 50. Stray "Non-Go" line. Fix: delete.

ST-5 Major. Lines 61-69, 114-117, 158-162, 43-44. Core decision open while the plan assumes an answer; message size bound is a requirement, not an open question. Fix: resolve or gate; move size to NFRs.

ST-6 Major. Lines 47-48, 149-181. Goal "catalog exposure via frontmatter" has no target file; item 3 label malformed; item 4 path hedged; RespondToCoordinator definition and executor unassigned. Fix: add rows, pin paths.

ST-7 Major. Lines 140-142, 163-166. Ladder position stated inconsistently for the two tools, not in ladder vocabulary. Fix: one table of tool, tier, mode.

ST-8 Major. Lines 38-41, 100-107, 146-147. Carrier terminology drift. Fix: one definition.

ST-9 Major. Lines 9-11, 51-60. No link to INDEX entry or to ZP-23 and cross-session-context specs. Fix: add relative links.

ST-10 Nit. Lines 52-60. Resolved by ST-4 fix.

ST-11 Nit. Lines 94-112. Diagram SendMessage arrow points into prompt assembly; should go coordinator to runner. Fix: redraw.

ST-12 Nit. Lines 1-7. No size estimate (INDEX says medium). Fix: optional.

ST-13 Nit. Lines 71-90. Only happy-path CUJs. Fix: add failure CUJ.

ST-14 Nit. Lines 178-180. Mutation checks nested unlabeled. Fix: label.

## Style (SY)

SY-1 Major. Lines 50-58. Duplicate non-goal. Fix: delete line 50.

SY-2 Major. Lines 61-69, 43-44, 88, 111-112, 161-162, 175. Plan assumes "retained", Goals say "persisted"; tests assume an option. Fix: one term, state which option tests assume.

SY-3 Major. Lines 68-69, 143. Bounds are "suggest" values; no oversize test or behavior. Fix: state defaults and add a test.

SY-4 Major. Lines 143-145. Oldest-drop unreported; "cannot balloon" claim unproven (no global cap). Fix: state per-child bound or add a global cap.

SY-5 Major. Lines 140-142, 165-166. Ladder inconsistent; "limited to the child session" understates a resumed child that can write files. Fix: name the rung, drop or define the claim.

SY-6 Major. Lines 41, 106, 146-147. Carrier attributes differ. Fix: define once.

SY-7 Major. Lines 121-136. Repeated "Rejected by author" adds no rationale; hook-path and steering-pill claims uncited. Fix: cite or drop.

SY-8 Major. Lines 15-23, 114, 107, 170. Claims about existing code (closure-bound port, injection point, depth caps, gate) cite no file or symbol; kiro brief and audit unlinked. Fix: add symbols and links.

SY-9 Nit. Lines 3-7. Swift-side "absent" evidence not stated. Fix: add a line.

SY-10 Nit. Lines 17-35. Overview and Problem repeat. Fix: trim.

SY-11 Nit. Lines 41, 46-47, 102, 106. "step" and "step boundary" interchangeable. Fix: one term.

SY-12 Nit. Lines 18, 43, 86, 88. "context intact" used repeatedly, undefined. Fix: define once.

SY-13 Nit. Line 164. Malformed label. Fix: match others.

SY-14 Nit. Line 167. Target file undecided. Fix: name it.

SY-15 Nit. Lines 178-180, 79-81. Mutation bullet unlabeled; CUJ 1 final-result delivery untested. Fix: label, add test.

SY-16 Nit. Lines 182-189. Manual step lacks pass criteria. Fix: state the observable.

SY-17 Nit. Line 185. `swift test` lacks package path or target. Fix: add.

SY-18 Nit. Lines 47-49. Frontmatter goal has no file. Fix: add or drop.

SY-19 Nit. Line 59. Overlaps depth-cap non-goal. Fix: merge.
