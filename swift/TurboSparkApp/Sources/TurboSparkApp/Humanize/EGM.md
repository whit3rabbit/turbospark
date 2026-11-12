# Humanize

## Overview
The Humanize section: paste a draft, run it through the dedicated `humanizer-12b` raw-completion rewriter, and review the rewrite with a word-level diff, audit checks and user rules. The directory holds the state machine and engine seam (`HumanizeController`), the long-draft planner that splits text into parts, the user rules store, the diff, and the SwiftUI views (two aligned cards, a run rail, a header model bar and an Advanced rules panel). It owns its own `TurboSparkSession` rather than sharing the chat session, because the humanizer is not a chat-templated model and needs an 8,192-token window.

## Interface
- Controller and engine seam [HumanizeController.swift:HumanizeController]
  - `HumanizeEngine` protocol (`humanize(_:options:)`, `cancel()`); `TurboSparkSession` conforms [HumanizeController.swift:HumanizeEngine]
  - `HumanizeController`: `Phase` (`idle`, `loadingModel`, `running`), published `draft`, `rules`, `rewrite`, `sourceDraft`, `segments`, `plan`, `checks`, `ruleFindings`, `stats`, `stopReason`, `prefillProgress`, `partProgress`, `errorMessage`, `engineLoaded`; actions `load()`, `run()`, `cancel()`, `clearRewrite()`, `clearAll()`, `unload()`, `refreshContext()`, `usesModel(at:)`; derived `canRun`, `currentPlan`, `isBusy`, `isInstalled`, `wasTruncated` [HumanizeController.swift:HumanizeController]
  - Owner-injected closures: `installedPath`, `notConnectedStatement`, `modelMoving`, `installHandler`, `baseOptions`, `openEngine`, `countDraftTokens`; constants `modelAlias` ("humanizer-12b") and `contextTokens` [HumanizeController.swift:HumanizeController]
  - `HumanizeError`, `HumanizeChecks`, `HumanizeStats` [HumanizeController.swift:HumanizeError]
- Planning [HumanizeParts.swift:HumanizePlanner]: `HumanizePart`, `HumanizeSegment`, `HumanizePlan`, `HumanizePlanner` (`window` 8192, `promptReserve` 256, `partTokens` 800, `outputTokens(for:)`, `fits`, `plan(_:count:)`), actor `HumanizeTokenCounter`.
- Rules [HumanizeRules.swift:HumanizeRules]: `HumanizeRules` (protected terms, avoid terms, replacements, fixed seed), `HumanizeRuleFindings`, `HumanizeRulesStore` (key `TurboSpark.humanizeRules`), `HumanizeInstruction` (read-only display copy and fingerprint of core's instruction).
- Diff [HumanizeDiff.swift:HumanizeDiff]: `HumanizeDiff.compare(source:rewrite:segments:)`, `changedShare`.
- Views
  - `HumanizeView` (root), `HumanizeHeader`, `HumanizeAdvancedPanel` [HumanizeView.swift:HumanizeView, HumanizeHeader.swift:HumanizeHeader, HumanizeAdvancedPanel.swift:HumanizeAdvancedPanel]
  - `HumanizeRewriteRail`, `HumanizeProgressStatus`, `HumanizeFeedback`, `HumanizeAuditLines` [HumanizeStatus.swift:HumanizeRewriteRail]
  - `HumanizeMetrics`, `HumanizeCard`, `HumanizeModelBar`, `HumanizeRunButton`, `HumanizeSkeleton` [HumanizeChrome.swift:HumanizeCard]
- Subdirectories: none.

## Invariants
- The controller never imports `AppModel`; install path, move state, drive state and download are injected closures set by the owner. Engine open options are read at open time (`baseOptions()`), then `maxContext` is forced to `.fixed(contextTokens)`. Only quality-neutral resource settings belong in `baseOptions`, since the checkpoint's quality gates were frozen at defaults (no chat steering, KV quantization, speculation or rate cap) [HumanizeController.swift:baseOptions, openEngine].
- `run()` sets `phase` and `operationPath` synchronously before scheduling the task. `canRun` reads `phase`, so a second click in the gap would otherwise start a second rewrite on one engine [HumanizeController.swift:run].
- Never open the engine from a path that is being moved: `modelMoving` is checked in `load()`, in `perform` (before the planner reads the tokenizer) and in `ensureEngine`, the single place the engine opens. `ensureEngine` also requires `installedPath() == operationPath` before and after the async open, and checks cancellation after, because native opening can finish after Stop or Unload and must not be republished [HumanizeController.swift:ensureEngine].
- A passage that cannot fit is refused (`passageTooLong`) before mapping weights and never truncated. Fit is per part: `part.tokens + promptReserve + outputTokens <= window`, where `outputTokens = min(2048, max(256, 5/2 tokens))` mirrors core's `default_max_new`. `partTokens = 800` keeps every part inside that budget so no per-call limit is sent [HumanizeParts.swift:HumanizePlanner].
- Planning splits a draft at paragraph breaks, falls back to sentence ends for an over-long paragraph, joins neighbors greedily up to `partTokens`, and re-inserts the draft's own text between parts. A draft at or under 800 tokens goes through as one part exactly as typed [HumanizeParts.swift:plan, HumanizeController.swift:rewrite].
- The tokenizer count runs in the `HumanizeTokenCounter` actor, off the main actor, trimming whitespace before counting [HumanizeParts.swift:HumanizeTokenCounter]. `refreshContext` discards its result if the draft or installed path changed while counting; `currentPlan` returns nil until `countedDraft` and `countedPath` match [HumanizeController.swift:refreshContext, currentPlan].
- Rules work outside the model: the engine refuses every option except output budget and seed, and the checkpoint was trained on one exact instruction, so no rule reaches the prompt. Replacements edit each part's finished text; protected and avoid terms are checked once against the whole rewrite. `HumanizeInstruction.text` is display only and `HumanizeRulesTests` hashes it against `fingerprint` so it cannot drift from core's `PROMPT_FINGERPRINT` [HumanizeRules.swift:HumanizeInstruction].
- Replacement matching is exact-case and whole-word; avoid matching is case-insensitive; Han, Hiragana and Katakana terms skip word boundaries. Protected terms count as missing only if the draft contained them exactly [HumanizeRules.swift:check, regex(for:)].
- A rule run uses the rules captured when it started (`run()` copies `rules`). One seed applies to every part when `fixedSeed` is on, else no seed is sent so each run is a new sample [HumanizeController.swift:run, rewrite].
- Result content is authoritative; the stream is a progressive view. Replacements apply before the `HumanizeSegment` is recorded so highlights match the text the user copies. A cancelled run leaves finished parts visible, sets `stopReason = .cancelled`, and sets `checks = nil` so partial text is not reported "missing" numbers. A stream that ends without `.finished` and no cancel clears `audited` and `ruleFindings` [HumanizeController.swift:perform, rewrite].
- `clearRewrite()` and `clearAll()` refuse while busy, because the running task writes those fields [HumanizeController.swift:clearRewrite].
- Diff is display only: it never mutates text, and copy always sends the original `controller.rewrite` string. Work is bounded: over 4000 tokens either side, or a product over 4,000,000, falls back to paragraph alignment. Unchanged runs shorter than half of each neighboring change are folded into the change, up to 6 passes. One or two punctuation marks are not marked. A part with no segment (a stopped tail) is not compared [HumanizeDiff.swift:compare, cleanup].
- The root view recomputes the diff off the main actor with `Task.detached` keyed by `DiffKey`, and debounces the token count by 180 ms [HumanizeView.swift:body].
- Rules persistence fails soft: a value that no longer decodes yields empty rules instead of failing the section [HumanizeRules.swift:HumanizeRulesStore.load].
- UI: audit sentences go through `AppStrings.localized` so they follow the in-app language picker; findings are pointers for the proofreader, not claims the rewrite is wrong. The run arrow uses `arrow.forward` so it flips for right-to-left layouts. Card frames scale with `AppMetrics.textScale`. Dividers do not soften under Increase Contrast. Clearing a rewrite asks to confirm (it cannot be regenerated identically); a lone draft clears without a dialog [HumanizeStatus.swift, HumanizeView.swift:clearAll].

## Side Effects
- Persists rules as JSON in `UserDefaults` under `TurboSpark.humanizeRules` on every change [HumanizeRules.swift:HumanizeRulesStore.save, HumanizeController.swift:rules]. `@AppStorage("TurboSpark.humanizeAdvancedVisible")` stores panel visibility [HumanizeView.swift].
- FFI: opens a `TurboSparkSession(modelPath:options:)` (maps the weights, compiles Metal pipelines) and streams `GenerationEvent`s; `TurboSparkTokenizer(modelPath:)` reads the tokenizer from the model path [HumanizeController.swift:openEngine, HumanizeParts.swift:HumanizeTokenCounter].
- Tasks: the run/load task stored in `task`, `Task.detached` for diffs, a debounced `.task(id:)` for token counting [HumanizeController.swift, HumanizeView.swift].
- Clipboard: reads `NSPasteboard.general` for Paste and writes the rewrite for Copy [HumanizeView.swift:draftPanel, resultPanel].
- Deletes the model: `model.deleteModel(_:)` after confirmation; navigation via `model.activeSection`, `model.isDownloadManagerExpanded` [HumanizeView.swift, HumanizeChrome.swift:HumanizeModelBar].
- Timers and animations are state-driven (`TimelineView`, `phaseAnimator`) and stop under reduce motion [HumanizeChrome.swift:HumanizeRunButton].
- No network or Keychain access in this directory (the download is delegated through `installHandler`).

## Verification
- Run: `swift test --package-path swift/TurboSparkApp`
