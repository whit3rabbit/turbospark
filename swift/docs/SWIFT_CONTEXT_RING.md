# Swift context ring (the composer's context-usage indicator)

What the circular meter in the prompt composer's footer shows, where its
numbers come from, and the two rules that keep it honest. The model layer is
`State/AppContextUsage.swift`; the ring and its popover are
`Generation/ContextUsageRingView.swift`; the shared assembly is
`State/AppModel+History.swift`'s `buildEstimateParts`.

## The ring

A 28 pt button in `PromptComposerFooter`'s trailing cluster, first position
(so it cannot move when the conditional clear button appears). It is a
`Circle().trim` over a track: empty at zero tokens, filling clockwise from
12 o'clock, tinted by `ContextUsageSummary.tint(forFraction:)` -- green
while comfortable, yellow from 70 percent, red from 90. Claude Code's
thresholds. The fill animates 150 ms linear so a jump reads as motion rather
than a flicker.

It renders NOTHING until an engine session is loaded: without one there is
no resolved window to be full. The fraction is the same reading the status
bar's context meter shows (`estimatedContextTokens` over
`resolvedContextTokens`), so the two surfaces cannot disagree.

## The popover

Clicking opens a 320 pt popover (the `ToolApprovalDropdown` /
`FanReadoutView` popover pattern, arrow up), refreshed via
`updateTokenEstimate()` on appear:

- The headline: `used / window (NN%)` in compact number format.
- A stacked capsule bar, one segment per component, widths as fractions of
  the WINDOW (so free space stays visibly track).
- One row per component: colored dot, label, tokens and percent of window.
  A final "Free space" row.
- The footer: "Auto-compact at ~NNk tokens" (or "Auto-compact off"), plus
  the "Compacting conversation..." state while the summarizer runs.

The rows: System prompt (the user-authored prompt, agent instructions,
workspace and project rules), Tools & skills (the tool addendum, which
carries the skill listing inside it), MCP servers, Memory, Conversation
(transcript rows after the compaction boundary), Compaction summary,
Attachments, Draft. There are NO plan-limit rows: those are Anthropic-side.
The compaction trigger is the one real limit this app has, and it is
computed through `AppChatCompaction.triggerNumerator/Denominator` with
ceiling division so a prompt AT the trigger satisfies
`shouldAutoCompact` and one token under does not.

## Where the numbers come from

`updateTokenEstimate`'s existing debounced task (250 ms, skipped while
generating, cancelled rather than queued -- the reasons are in its own
comment) is the ONLY writer. It builds `buildEstimateParts()` ONCE and
takes two readings over that one snapshot:

1. the EXACT count: `session.countTokens(history)`, the template render the
   turn itself would price;
2. the per-piece counts: `session.countTokens(in:)` on each labeled slice,
   raw string counts with no template render (about 8 cheap dispatches).

`estimatedPromptTokens` gets the first; `contextUsageSummary` gets both.
Because the two readings ride one task over one assembly, the ring, the
status bar and the popover can never be looking at different conversations.
Opening the popover re-kicks the same function, so stale numbers refresh on
open; mid-turn nothing updates, and the popover shows the last snapshot.

**The rows APPORTION, they do not sum.** The exact total includes
chat-template role framing, which lands on the conversation as a whole and
not on any one slice; the rows are raw-string counts. The headline always
reads the exact number when one exists (`usedTokens` falls back to the rows'
sum only before the first estimate lands), and the popover says so in its
footnote.

**The system prompt IS the join of its sections.** `buildSystemPrompt`
folds `buildSystemPromptSections`, and `buildEstimateParts` groups those
same tagged sections into pieces. One builder, three consumers; a section
that changes text or order changes everywhere at once, and a section
dropped from a piece group reddens
`testProjectSectionsAreGroupedIntoLabeledPieces` rather than silently
vanishing from the popover.

## The estimate the ring reads (state#116)

`estimatedContextTokens` was `transcriptChars/4 + attachmentChars/4 +
estimatedPromptTokens` from before the estimate was exact. Once
`estimatedPromptTokens` became the exact count of the assembled prompt (which
already includes every transcript row), the chars/4 term double counted the
transcript, overstating the fill by roughly a quarter of the conversation --
invisible while the meter was merely informational, wrong the moment the
ring's tiers keyed on it. It is `attachmentChars/4 + estimatedPromptTokens`
now: attachments are the one thing the exact count cannot see (a render
emits one marker per image), so they are the one thing still approximated.
Images are priced at that approximation everywhere here; the page-token
expansion happens in the engine's splice and no client-side count sees it.

## Tests

`Tests/TurboSparkAppTests/ContextUsageTests.swift`, all session-free:
tint tiers at both boundaries, exact-vs-rows headline preference, the clamp
of free space and fraction at the window, the trigger's agreement with
`shouldAutoCompact` at its boundary, the section-join equality, the piece
assembly against the history (boundary skip, raw summary, draft), the
project section grouping, and the double-count regression. Each has been
mutation-checked to redden only its own case; the raw-string counting seam
itself needs a live tokenizer and is the manual GUI smoke's.
