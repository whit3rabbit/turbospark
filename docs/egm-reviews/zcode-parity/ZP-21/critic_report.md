# Stage 7: Critic report (ZP-21 workspace tabs)

Reviewers: egm-design (DC, standard feature-spec-standard), egm-structural (ST, feature-spec-standard), egm-style (SY). One round only, convergence not assessed. All three completed (style and structural were first refused with "Concurrent subagent limit reached", then ran on retry; spawned one at a time).

## Design (DC)

DC-1 Blocker, lines 115-133. No Risks & Mitigations section (state leakage, memory growth, restore failure, ZP-23 dependence, cap interaction). Fix: add the section.
DC-2 Major, lines 115-133. Rejections attributed to "author" without link; no "ZP-23 plus project switcher, no tabs" alternative. Fix: concrete reasons plus that alternative.
DC-3 Major, lines 33-41, 54-57, 99-102 (ZP-23 line 49). ZP-23 cap of 2 and model-switch refusal unreconciled with three tabs each running a chat. Fix: state tab behavior at cap and whether model is global.
DC-4 Major, lines 33-35 vs 58-60. Settings-tab goal contradicts the open question; reserved id is dead schema. Fix: decide, move to Non-Goal or drop from schema.
DC-5 Major, lines 54-57 vs 69-71, 185-187. Stale conditionals after "resolved" prerequisite. Fix: remove.
DC-6 Major, lines 98-103, 36-37, 160. Tab identity undefined: duplicates, deleted workspace, close with running chat or unsent draft. Fix: specify policies.
DC-7 Major, lines 36-38, 105-108, 138-139. Tabs archive is plain AppJSONStore while chats live in per-profile SQLCipher; profile scoping, vault unlock ordering (finding 1.12), sensitive paths and drafts unaddressed; draft and scroll persistence vs Goals unclear. Fix: state scoping, ordering and persisted fields.
DC-8 Major, lines 73-78, 138-139. Restore failure modes (missing folder, revoked permission, deleted chat), write timing and crash behavior missing. Fix: define.
DC-9 Major, lines 134-137. "Lightweight state" unsupported; no tab cap or latency target. Fix: set cap and target or say unmeasured.
DC-10 Major, lines 39-41. "Without schema changes" unjustified against ZP-23 coordinator state. Fix: show mapping or drop claim.
DC-11 Major, lines 70-71, 84-86, 160-161. Tab badge aggregation undefined (ZP-02 defines row badges). Fix: aggregate over all chats with ZP-02 priority.
DC-12 Major, lines 88-103, 118-120, 152-157. AppModel refactor hand-waved; partial migration leaks global state (git pane, file surfaces, sidebar filter). Fix: list properties that move vs stay, add leak tests.
DC-13 Nit, lines 4-5. ZCode link path differs from comparison doc. Fix: use `store/tabStore.ts`.
DC-14 Nit, lines 15-16, 46-47. "Availability" undefined; "creation order" claim uncited. Fix: define and cite.
DC-15 Nit, lines 158, 163-164. Strip placement and `Generation/` folder odd. Fix: clarify.
DC-16 Nit, lines 140-142. No VoiceOver or shortcut collision check (Cmd+W). Fix: add to NFRs and tests.
DC-17 Nit, lines 169-187. No tests for missing-workspace restore or cap interaction; no first-launch migration for users without an archive. Fix: add.

## Structural (ST)

ST-1 Blocker, lines 123-124, 132. "Rejected by author" with no linked record. Fix: technical rationale or link.
ST-2 Blocker, between lines 134 and 144. No Risks & Mitigations section (close while generating, missing workspace, duplicate tab, memory across tabs, persisted state security, refactor failure). Fix: add H2.
ST-3 Major, lines 115-132. Both alternatives are layout variants; no low-cost or status quo option. Fix: add per-project state cache without strip.
ST-4 Major, line 5. ZCode path likely wrong. Fix: `packages/ui/src/store/tabStore.ts`.
ST-5 Major, lines 54-57. Resolved item under Open Questions; sequencing mixed in. Fix: move to Dependencies note.
ST-6 Major, lines 70, 186-187. Stale conditionals. Fix: unconditional wording.
ST-7 Major, lines 33-35 vs 58-60, 94. Goals and Open Questions disagree on settings tab. Fix: goal becomes "reserve id only".
ST-8 Major, line 50. ZP-90 cross-reference for remote-session rekeying may be dangling (1.6 not in INDEX list). Fix: confirm or re-point.
ST-9 Major, lines 70-86, 136-137. CUJ 3 titled "switch guard" but has none; badge not a Goal; lightweight-state NFR vs ZP-23 resource cost. Fix: retitle, add goal, reconcile.
ST-10 Nit, line 37. "Per-tab availability" undefined. Fix: define or drop.
ST-11 Nit, lines 148-158. Strip view under `Generation/`, mount location ambiguous. Fix: relocate, state position.
ST-12 Nit, lines 174-176. "Mutation checks" bullet unlabeled and mis-nested. Fix: label.
ST-13 Nit, lines 15-17, 26-27. "Largest navigation change" is rhetoric not echoed in INDEX. Fix: remove.

## Style (SY)

SY-1 Major, lines 26-29, 54-57, 70, 185-186. Stale gating language; "before this draft" should be "before implementation". Fix: state once, delete conditionals.
SY-2 Major, lines 16-17, 33-35, 58-60. Settings tab both goal and undecided question; no decision owner. Fix: goal "reserve id; Settings stays a window in v1".
SY-3 Major, lines 15, 37, 47. "Availability" undefined. Fix: define or list persisted fields.
SY-4 Major, lines 33, 40, 70-71, 84-86, 99-101, 160-161. Badge semantics inconsistent; "one active chat per workspace" conflicts with ZP-23. Fix: one aggregation rule, one term.
SY-5 Major, line 5 (and line 15). Parity link inconsistent with comparison doc; "session restore per tab" vs "pane session restore". Fix: verified URL, state scope.
SY-6 Major, lines 40, 110-113, 136-139. Uncalibrated claims (lightweight, views unchanged, like every AppJSONStore file, single-tab fallback, no schema changes). Fix: cite code or say "expected" with a check.
SY-7 Major, lines 38-39, 64-71, 85. "Restore exactly" vs enumerated Goals; sidebar state missing from Goals. Fix: enumerate.
SY-8 Major, lines 15-23, 90-100. Workspace, project and tab used interchangeably. Fix: define once.
SY-9 Nit, lines 22-26. Subjective phrasing. Fix: concrete statement.
SY-10 Major, lines 56, 80-86, 173. CUJ 3 mislabelled; "guard" never defined. Fix: rename and state guard defers to ZP-23.
SY-11 Nit, lines 120-132. Alternatives padded or unsupported. Fix: concrete cost, one-sentence reason.
SY-12 Nit, lines 108, 141, 148-163. "Token ratchet" undefined; strip placement and folder. Fix: link DESIGN.md, state position.
SY-13 Nit, lines 17, 45-46. "Deliberately single-window" and "matching ZCode" contradictory. Fix: reword.
SY-14 Nit, lines 147-176. No tests listed for strip view or shortcuts. Fix: add or state manual only.

Persona files read: design read agents/design_critic.md; structural read agents/structural.md; style read agents/style.md (all under /Users/whit3rabbit/.claude/plugins/cache/egm/egm/0.1.1/).
