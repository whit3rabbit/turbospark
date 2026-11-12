# Stage 6: Comprehension report (OC-04 session tabs)

Reviewer: egm-comprehension, read the full file alone. Completed.

**What this proposes**

Add a browser-style tab strip to the macOS app so many chat sessions stay open at once. There are two kinds of tab: session tabs (one per open chat) and draft tabs (an unsent composer holding typed text). The open tabs, selected tab, and draft text persist across launches. Submitting a draft turns that tab into a session tab in place. Closing a session tab pushes it onto a bounded "Recently Closed" stack, and reopening restores it at its old position. Cmd+T, Cmd+W and option+arrow handle new, close and cycle. The strip is a sub-row of the existing working-panes band, backed by a new SessionTabStore, a strip view, and edits to the root view and model.

**How the existing system works, per this document**

The app restores one chat selection at launch. A separate split-panes view can show several chats side by side, but those are window-level panes and not tabs. There is no persisted open-tab set, draft tab, closed-tab stack, or per-tab scroll and composer memory. The window has four fixed chrome bands, and a fifth is forbidden by DESIGN.md. There is a single window, a per-domain JSON archive store with corruption quarantine, an existing chat selection path and submit path, and a keyboard shortcut catalog. The document does not say whether the "existing submit path" works without an already-selected chat.

## Findings

CO-1 Major. Lines 46, 105-109, 134, 140-145. Persistence goals conflict: line 46 persists per-tab draft text, line 134 persists only ids for session tabs, CUJ 3 promises session-tab drafts survive. Fix: state where session-tab composer text lives and make Goals, Plan and CUJ 3 agree.
CO-2 Major. Lines 49, 91-94. "No composer flash" is undefined; the in-place kind swap during asynchronous chat creation, and failure, are not described. Fix: describe the promotion sequence and failure case.
CO-3 Major. Lines 50-52, 98-102. Reopen is underspecified: position out of range, chat already open in another tab, chat deleted, and where "draft intact" is kept. Fix: define each case.
CO-4 Major. Lines 20-21, 84, 116. Recently Closed is called session-scoped but stored in the persisted store. Fix: decide and say it once.
CO-5 Major. Lines 24-27, 54-55, 66. ZP-21 and ZP-23 are load-bearing but defined only by link; relation of session tabs to workspace tabs unclear. Fix: add two or three summarizing sentences.
CO-6 Major. Lines 56-58, 167-172. Cmd+W edge cases (last tab, draft-only), option+arrow direction, and catalog timing ("once both land" vs files 6 and 7) unclear. Fix: add a chord table and one timing decision.
CO-7 Major. Lines 75-80. Strip-hiding rule is ambiguous, conflicts with Cmd+T, and placement is called open while the plan assumes (a). Fix: rewrite as a table and mark placement decided or open consistently.
CO-8 Major. Lines 68-69, 81-83. Split-pane interaction is both a lean and a non-goal; behavior when panes are open and tabs change is not stated. Fix: say what tab selection does to panes.
CO-9 Nit. Lines 1-10, 33-39. Project terms undefined (AppJSONStore, AppAdaptiveHeader, PromptComposer, working-panes band, "park chat selection"). Fix: one-line definitions.
CO-10 Nit. Lines 38-39. "Its report" has no antecedent. Fix: name the report and ranking basis.
CO-11 Nit. Lines 120-121. "Never truncate" vs overflow; draft tab title rule, overflow order, and selected-tab visibility unstated. Fix: state them.
CO-12 Nit. Lines 138-139, 163-166. "Resident" chat undefined. Fix: define or cite the mechanism.
CO-13 Nit. Lines 183-188. Localization gate and language list not named. Fix: name them.
