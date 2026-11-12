# Stage 5: Initial zoom out report (ZP-02)

Target: docs/designs/zcode-parity/ZP-02-unread-pending-badges.md (full file). Reviewers: egm-zoom-out (ZO), egm-inventive-catalyst (IC). Both completed.

## Zoom-out (ZO)

Verdict: objective and motivation are clear. No single Blocker, but compare-and-set and countdown should not be implemented as written. Code-search evidence: AppChat (State/AppChat.swift:411) has `updatedAt` but no revision field; no "revision" in ChatSidebarOrganizer.swift; no "unread" in chat, sidebar or organizer files; no dock or badge file under `.kiro/specs` in this worktree.

ZO-1 (Major) Lines 33-45, 110-114, 146-151. The CAS revision token rests on an unverified claim ("chat archive revision the organizer already reads"); no revision concept exists, only `updatedAt`. The doc admits only the selected chat generates today, ZP-23 is a draft, and the app is single-process with paired devices excluded. ZCode's CAS arbitrates desktop versus web clients, which TurboSpark lacks. Alternative: persist `lastReadAt` or last-read message count and compute unread from it. If CAS stays, name the real revision source and add it to the file list.

ZO-2 (Major) Lines 56-62, 78-85, 107-108, 163-172. Countdown and snooze serve an auto-resolve semantic the app lacks; the toggle is default off, so the countdown does nothing. Bundled settings fields and a question-lifecycle change are gated on an unresolved Open Question; file item 5 cannot be reviewed as written. A static pending pill solves "no cue that a chat needs me". Cut countdown, snooze and settings, or spin auto-resolve into its own draft.

ZO-3 (Major) Lines 17-21, 94-108, 146-151. A new `ChatBadgeState.swift` mixes a derived view value with a persisted flag. The one persisted field belongs on AppChat (already Codable); derived priority can live in ChatSidebarOrganizer. State the decode default for old archives.

ZO-4 (Major) Lines 29-32, 44-45, 50-54. ZP-23 dependency is hidden in Non-Goals. Under the single-active model, a non-selected chat never receives content, so the dot and pill cannot fire; ZP-02 delivers nothing visible until ZP-23. State the hard ordering or merge into the ZP-23 rollout. The `desktop-shell-hardening` spec was not found under `.kiro/specs` in this worktree (may exist elsewhere); verify, or the dock-badge goal is dangling. "any future concurrent-chats world" is stale since ZP-23 is adopted.

ZO-5 (Major) Lines 33-45, 17-19, 90, 100-105. ZCode's priority is error > unread > spinner. This draft drops error without recording it as a deliberate divergence. CUJ 3 says the spinner yields to unread, so a generating chat with an old unread dot looks idle.

ZO-6 (Nit) Lines 63-66. Unread definition left open but CUJ 1 assumes "generation appends content" (broad). Pick one and make CUJ 1 consistent.

ZO-7 (Nit) Lines 100-108, 168-172, 187-198. Technical Plan reads like a type sketch; "ten gates" and "21 languages" repeated; `make design-check-full` use unexplained.

ZO-8 (Nit) Lines 118-130. Alternatives are weak: A rejected because "persistence is the point", B is not a sidebar design. Missing real alternatives: persisted `lastReadAt` instead of CAS, and pill without countdown.

Limits: ZCODE_COMPARISON.md read only to line 237; ZCode source not verified.

## Inventive catalyst (IC)

IC-1 (Major) Lines 56-61, 94-114, 167-172. False compromise: countdown is framed as display-only versus auto-resolve, and machinery is planned before auto-resolve is wanted; a countdown on a pill that cannot expire is a misleading cue. Split: phase 1 pill with no countdown or snooze; phase 2 countdown, snooze and auto-resolve as one unit after the question is decided.

IC-2 (Major) Lines 38-40, 110-114, 152-156. CAS copied from ZCode's multi-writer model; here one process and one UI actor serialize writers. Use a persisted watermark (`lastSeenRevision` or message count); unread = revision > watermark. Needs a revision bumped on every content change, coalesced for streaming deltas, plus a one-time migration default.

IC-3 (Major) Lines 49-54, 83-85, 112-114 (INDEX 78-79, ZP-23 44-45). Coupling to ZP-23 forces the sidebar to wait. Define badge state as a pure function over a small read-only provider (running, openApproval, openQuestion, revision); test against a fake and the single-active adapter now, ZP-23 implements it later.

IC-4 (Major) Lines 36-37, 88-92 (ZCODE_COMPARISON.md line 172). Error state dropped without a decision; a failed background chat would show as unread or default. Record the exclusion or add it. Consider spinner base layer with unread corner overlay, pending exclusive on top.

IC-5 (Nit) Lines 63-65, 33-39. Unread Open Question versus assumed append semantics; under a watermark, any revision change is natural, but tool-only chatter may be noisy.

IC-6 (Nit) Lines 112-114. "Semantics exist for the concurrent future" is speculative; name a concrete interleaving that mismatches today, or drop CAS.

IC-7 (Nit) Lines 137-141. "No polling" is untrue for the countdown; state a single shared tick scoped to visible pill rows that stops when the window is not key.

IC-8 (Nit) Lines 17-21, 144-185. Auto-resolve with an empty answers payload is not equivalent to a decline for the model; say what the model sees (a named timed-out result).
