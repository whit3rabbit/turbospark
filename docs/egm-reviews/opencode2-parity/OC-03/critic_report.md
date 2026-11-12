# Stage 7: critic report (OC-03)

One round only (convergence not assessed). Design and structural used feature-spec-standard; style had no standard. All three reviewers completed.

## Design critic (DC)

Verdict: not ready.

DC-1 (Blocker, whole file): no Alternatives Considered section. Candidates: pinned Needs you / Working strip above existing buckets, badge-only via ZP-02, sort toggle, filter chip. Fix: add the section with concrete rejection reasons.
DC-2 (Blocker, whole file): no Risks and Mitigations section (ghost chats, token-rate invalidation, rows that jump, duplicated priority logic). Fix: add it.
DC-3 (Blocker, lines 44-46 vs ZP-02 36-37): priority order contradicts ZP-02 (pending > unread > running); the swap mutation test encodes the OC-03 order. Fix: pick one order, justify any divergence, test a running chat with unread content.
DC-4 (Major, lines 100-102, 129-131): ChatInboxModel is a second derivation beside ZP-02 ChatBadgeState; INDEX 73-74 wants one source. Fix: group over ChatBadgeState; pre-ZP-02 fallback is a thin adapter.
DC-5 (Major, lines 20-25, 47-48): "works today" unproven; before ZP-23 only the selected chat can be parked or running; Finished impossible for unselected chats. Fix: gate on ZP-23 or state the mode is near-empty before it.
DC-6 (Major, lines 50-51, 72-74, 61, 137-138): Finished defined two ways; last-ended timestamp is in memory and resets on relaunch. Fix: one definition, session-scoped, state relaunch behavior.
DC-7 (Major, lines 81-93): idle read non-archived chats undefined. Fix: say whether shown and under which heading.
DC-8 (Major, lines 88-93): row movement not handled; opening a Finished row marks it read and it leaves under the cursor. Fix: keep the selected row pinned until deselect, or define the transition and VoiceOver announcement.
DC-9 (Major, lines 117-118): performance claim ignores per-token body re-runs (ChatSidebarView 304-307) and per-chat vault decrypt check (line 436). Fix: transitions-only invalidation, avoid per-chat decryption.
DC-10 (Major, lines 53-55, 434-436): ghost chats may leak titles into a glanceable inbox; filter and project scope undefined. Fix: exclude ghost chats or redact; define filter and scope behavior.
DC-11 (Major, lines 53-55): archive or delete on a busy chat unaddressed; existing archived rows use Restore and Delete. Fix: list actions per section; delegate busy-chat rules to ZP-23 guards.
DC-12 (Major, lines 42-43, 98, 112, 133-135): toggle placement and composition with Chat/Projects unspecified; persistence contradicts non-goal; no settings file in list. Fix: specify position, add MacAppSettings.swift.
DC-13 (Major, INDEX 76-77 vs 91-95): sequencing puts OC-03 before ZP-23. Fix: move after ZP-23 or acknowledge near-empty value.
DC-14 (Nit, line 32): broken relative link. Fix: ../../../swift/...
DC-15 (Nit, lines 56-58, 121-123): bucket header Text at ChatSidebarView 343-348 has no .isHeader trait, so "existing" patterns do not exist. Fix: say the trait is new.
DC-16 (Nit, lines 129-147): file list omits ChatSidebarChatRowView, ChatSidebarLayout, ChatSidebarOrganizer. Fix: list them.
DC-17 (Nit, whole file): no success metric. Fix: add a manual time-to-find measure.
DC-18 (Nit, lines 6-9): OpenCode parity targets not marked unverified. Fix: repeat the INDEX note.

## Structural (ST)

Heading hierarchy, tables and footer are fine.

ST-1 (Blocker, whole file): no Alternatives Considered heading. Fix: add H2 with at least two rejected options.
ST-2 (Blocker, whole file): no Risks and Mitigations heading. Fix: add H2 (state drift, sequencing, list cost, delete from a new surface).
ST-3 (Major, lines 43, 61): persistence contradiction. Fix: reword non-goal and name storage.
ST-4 (Major, line 32): broken link. Fix: ../../../swift/...
ST-5 (Major, lines 88-93): CUJ 2 malformed (two When). Fix: one Given, When, Then.
ST-6 (Major, lines 67-75): Open Questions precede CUJs and Technical Plan; standard order differs. Fix: move after Risks.
ST-7 (Nit, line 42): "vocabulary pill picker" undefined. Fix: link DESIGN.md.
ST-8 (Nit, line 107): "no fifth chrome band" undefined. Fix: link or define.
ST-9 (Nit, lines 85-86): "normal sections or absent" vague. Fix: state which chats appear.
ST-10 (Nit, line 155): "DESIGN.md section 9" unlinked. Fix: link ../../../DESIGN.md.
ST-11 (Nit, lines 125-146): no sequencing note. Fix: add a "Depends on" line.

## Style (SY)

No Blockers.

SY-1 (Major, lines 49-51, 69-75): Goals settle Finished as "since you last opened" while the lean is "ended while unfocused". Fix: align or mark pending Open Question 2.
SY-2 (Major, lines 42-43, 69-71): default mode asked but Goals state none. Fix: state Sessions default and drop the question.
SY-3 (Major, lines 22-28, 47-48): "works today" claim unevidenced. Fix: name which sections work today.
SY-4 (Major, lines 79-93): CUJ 1 "normal sections or absent" untestable; CUJ 2 no defined outcome. Fix: rewrite as single When/Then with Working vs Finished condition.
SY-5 (Major, lines 117-118): "identical to existing grouping cost" unverified. Fix: say same asymptotic order or drop.
SY-6 (Nit, line 38): "glanceable triage pass" subjective. Fix: "a single grouped view of chats waiting on the user".
SY-7 (Nit, lines 36-38): "hunt across rows" colloquial. Fix: "scan rows".
SY-8 (Nit, lines 111-114): "fifth chrome band" uncited; sentence repeats lines 19-21. Fix: cite or drop.
SY-9 (Nit, lines 53-57): "existing patterns", "existing confirmation dialogs" unnamed. Fix: name the view or file.
SY-10 (Nit, lines 61-63): persistence conflict. Fix: "beyond the mode preference".
SY-11 (Nit, lines 3-9): OpenCode links use mutable `dev` branch. Fix: pin to commit 7ce35879b8.
SY-12 (Nit, lines 22-28, 132, 143-145): protocol/"before and after ZP-02" stated three times. Fix: state once.
SY-13 (Nit, lines 145-146): mutation check phrased as instruction. Fix: "the ordering test must fail when priorities are swapped".
SY-14 (Nit, lines 154-155): unlinked DESIGN.md. Fix: link.
SY-15 (Nit, line 42): "Sessions" vs "chats" terminology. Fix: define mapping once or use "Chats".
