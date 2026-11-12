# Consolidated review report: ZP-04 subagent-to-coordinator messaging

Target: docs/designs/zcode-parity/ZP-04-subagent-coordinator-messaging.md (full file, 193 lines)
Preset: Feature spec only (stages 5, 6, 7, 8). No remediation, target not edited. Stage 7 ran one round only, so convergence was not assessed. All reviewers completed; none failed.
Source IDs: ZO = zoom_out_report (zoom-out), IC = zoom_out_report (inventive catalyst), CO = comprehension_report, DC/ST/SY = critic_report (design, structural, style), S8 = intermediate_zoom_out_report.

Counts after deduplication: Blocker 4, Major 11, Nit 9.

## Blocker

B1. Lines 61-69 (also 36-47, 43-44, 88, 114-117, 158-162). Resume persistence is undecided, yet Goals say "persisted transcript", the plan says "retained", and tests assume one. Fix: decide in-memory retention with the transcript kept on the existing background-run entry (priorHistory already exists), prune with it, defer durable persistence. [CO-1, DC-1, ST-5, SY-2, IC-1, ZO-3, S8-1]

B2. Lines 38-47, 100-112, 167-171. Delivery to an idle coordinator or a coordinator blocked on a foreground child is unspecified; a message can wait indefinitely and CUJ 1 fails. Fix: reuse the existing park-and-drain turn-tail path, state queue lifetime on session switch, and scope RespondToCoordinator to background children or state foreground behavior. [DC-2, IC-2, ZO-4, S8-3, S8-4]

B3. Lines 119-136. Alternatives are "Rejected by author" with circular rationale, and the set omits the real forks (retain vs persist, resume vs spawn a fresh child with the prior result pasted). Fix: rewrite as concrete trade-offs and add those alternatives. [ST-1, ST-2, DC-7, SY-7, S8-7]

B4. Missing section. No Risks and Mitigations section, required by the standard; risks are scattered in NFRs. Fix: add a short list (prompt injection, message loss, resume after restart, depth interaction, context growth, child cancelled with unread messages). [ST-3, DC-8, S8 scope guidance]

## Major

M1. Lines 100-103, 153-157. New `SubagentMessageHub` duplicates the existing `pendingTaskNotifications` queue; "outbound" queue unexplained; ordering risk. Fix: add a message case to the existing queue, reuse drain and escaping. [ZO-1, S8-2, CO-8]

M2. Lines 167-171. Injection seam is hedged and wrong; task-notifications are drained in AppModel, AppModel+Queue and AppModel+Subagents, not SystemReminders. Fix: name the real seam. [ZO-2, CO-5, ST-6, SY-14, S8-4]

M3. Lines 138-148. Prompt injection via child text is handled only by labeling. Fix: escape carrier tags with existing escaping, mark content as untrusted data, add a test. [DC-4, IC-5, S8-6]

M4. Lines 68-69, 143-145. Size and depth bounds are only "suggested", oversize behavior undefined, oldest-drop silent, "cannot balloon" unproven. Fix: fixed values in NFRs, truncate with a marker, return "mailbox full" to the child. [DC-3, IC-3, SY-3, SY-4, CO-10, ZO-7, S8-5]

M5. Lines 140-142, 163-166. Ladder position inconsistent and not in ladder vocabulary; read-only exclusion blocks explorer children; "limited to the child session" understates a resumed child; resume permission and hooks unstated. Fix: one table (tool, tier, mode), allow in read-only, resumed child inherits tools and gate. [DC-5, ST-7, SY-5, IC-7, CO-6, S8-8]

M6. Lines 42-44, 50-57, 86-90, 146. SendMessage target key undefined; running, unknown, pruned, concurrently resumed child cases and depth-cap interaction unstated. Fix: define the id; error for running, not-found for unknown. [DC-6, IC-4, CO-9, S8-9]

M7. Lines 17, 36-47, 163-166. Two features with different cost and risk are bundled. Fix: ship SendMessage first (reuses priorHistory) or split into phases. [ZO-4, S8-3]

M8. Lines 40-41, 106, 146-147. Carrier tag and attributes differ across sections; "launch id" undefined. Fix: one schema. [CO-4, ST-8, SY-6, DC-11, S8-12]

M9. Lines 47-48, 149-181. Frontmatter exposure goal has no target file; RespondToCoordinator definition and executor unassigned; item 3 label malformed. Fix: add rows or drop the goal. [ST-6, SY-18, IC-6, ZO-6]

M10. Lines 15-23, 107, 114, 170. Claims about existing code (closure-bound port, injection point, steering precedent, hook path) cite no symbol; kiro brief and audit unlinked; "resolves the contradiction" overstated. Fix: cite files and symbols, link the brief, record the search. [SY-8, CO-3, CO-7, DC-10, IC-8, S8-13]

M11. Lines 36-47, 71-90, 182-189. No failure modes and no statement of whether carriers are visible in the transcript or persisted, no localization note. Fix: state visibility and failure handling. [DC-8]

## Nit

N1. Line 50 (also 59). Stray "Non-Go" duplicates Non-Goals; reviewers rated it Major but it is a one-line deletion. Fix: delete. [ZO-5, ST-4, SY-1, DC-9, CO-2, IC-6, SY-19, S8-10]

N2. Lines 163-167. Malformed "Type / Spawn-tool schema" label; target file for item 4 undecided. Fix: match other items. [CO-11, SY-13, SY-14, ZO-6, S8-11]

N3. Lines 94-112. SendMessage arrow points into prompt assembly. Fix: draw coordinator to runner. [ST-11, S8-14]

N4. Lines 172-189. Mutation checks nested and low altitude; manual step lacks pass criteria; `swift test` and scheme unspecified. Fix: trim, label, add package path. [ZO-7, ST-14, SY-15, SY-16, SY-17, CO-15, S8-15]

N5. Lines 9-11, 56-60. No links to INDEX, ZP-23 or cross-session-context; ZP-03 overlap not noted. Fix: add relative links. [ST-9, CO-12, S8-13]

N6. Lines 13-35, 127, 135. Undefined terms: ZCode, coordinator, steering pills, kiro, "context intact". Fix: one defining sentence each. [CO-13, CO-14, SY-12]

N7. Lines 17-35, 41-47. Overview and Problem repeat; "step" and "step boundary" used interchangeably. Fix: trim, one term. [SY-10, SY-11]

N8. Lines 71-90, 79-81. CUJs cover only happy paths; final-result delivery, escaping and oversize untested. Fix: add a failure CUJ and tests. [ST-13, SY-15, DC-12, S8-15]

N9. Lines 3-7, 9-11. Swift-side "absent" evidence not stated; no size estimate (INDEX says medium). Fix: add a line. [SY-9, ST-12]

## Traceability

- zoom_out_report.md: ZO-1 to ZO-7, IC-1 to IC-8. Feeds B1, B2, M1, M2, M3, M4, M6, M7, M9, M10, N1-N4.
- comprehension_report.md: CO-1 to CO-15. Feeds B1, M2, M4-M6, M8, M10, N1, N2, N4-N6.
- critic_report.md: DC-1 to DC-12, ST-1 to ST-14, SY-1 to SY-19. Feeds all Blockers and Majors, N1-N9.
- intermediate_zoom_out_report.md: S8-1 to S8-15. Confirms all Stage 5 findings open. Recommends rejecting scope-adding asks (IC-1 durable log plus cache, IC-2 wake feature, IC-3 coalescing, IC-4 hub state machine, DC-3 aggregate cap) and keeping ambiguity-removal fixes. These scope conflicts are the main decisions for the author.
- Merge notes: IC-1 and S8-1 disagree on mechanism (S8 prefers in-memory); both agree the choice must be made. ST-4 and SY-1 downgraded from Major to Nit as a trivial deletion.
