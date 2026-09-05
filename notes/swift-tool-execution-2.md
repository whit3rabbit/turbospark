---
uuid: "73c2c625-6312-4bc9-931d-fc1f1391506c"
title: "TurboSparkApp: the permission gate's learned classifier"
summary: "The bundled hazard model's veto ships OFF: it adds zero true positives against this app's own contract lists and false-positives on ordinary commands like python3 -m pytest. Only its advisory sentence ships on"
tags: ["swift", "app", "security", "permission-gate"]
source: "docs/PERMISSION_GATE.md, swift/CLAUDE.md Gotcha 29"
created: "2026-09-05"
updated: "2026-09-05"
depends_on: ["8e1f5553-7073-4050-8396-6c6a02917eea"]
---

## What does the permission gate's local classifier actually gate, and can I trust it?

`TerminalCommandClassifier.isAutoApprovable` is a POSITIVE allowlist: it
runs a command unwatched only when it's a single simple invocation (no
shell metacharacters, no quote/backslash/`=` in the head word) AND the
program is on a read/build allowlist. This replaced an ~20-regex denylist
a non-adversarial rewrite defeated on 18 of 23 corpus strings
(`r""m -rf ~/Documents`, `eval $(printf ...)`, `` `echo rm` -rf ~ ``),
since a denylist over a string bound for `/bin/zsh -c` names specific
patterns instead of the shape of the problem. `CommandGate` is a bundled two-head logistic classifier (hazard,
obfuscation) running AFTER the allowlist, so it only sees a command
already admitted and can move a verdict to `.ask`, never approve one the
allowlist refused (`HazardVeto` has no `approve` case).

**The veto ships off.** Measured against this app's own must-run and
must-ask test lists: the allowlist already refuses every evasion string
(zero true positives left for the classifier to add), while on ordinary
commands it false-positives on 1 to 3 of 18 depending on threshold, and
`python3 -m pytest tests/ -q` scores 1.000 so no threshold removes it. The
corpus grades general hazard, not allowlist evasion, the question that
actually matters here. What ships ON is `advisoryReason`: a sentence
decorating a verdict already heading to the approval sheet, never a
changed outcome.

## Don't

- Don't enable `MacAppSettings.commandAdvisoryVeto` expecting real
  protection. It measurably adds no true positives here and costs real
  false positives on ordinary developer commands.
- Don't trust the classifier's in-distribution numbers. Hazard reads ROC
  AUC 0.9972 in-distribution but 0.7010 held out by source, largely the
  model recognizing one generator's phrasing (181 of 246 training rows
  came from one source).
- Don't port the feature vectorizer without the `\b` word-boundary anchors,
  or expect Swift's `split(separator: " ")` to match Python's `str.split()`
  (any-whitespace-run). Both mismatches produce a WORKING model that
  scores DIFFERENTLY (0.9320 against a true 0.9931), not an obviously
  broken one. Each is pinned by its own test.
- Don't trust a ported scorer just because it returns plausible numbers on
  a small fixture. A model returning a CONSTANT for every input passes a
  narrow fixture. This is exactly the defect that got the originally
  published `command_hazard_model.onnx` withdrawn (a feature-scaling bug
  saturated the sigmoid). Check for spread, not just presence of output.
- Don't wire in the `prompt_injection_watch` model. Its 0.98 pooled ROC AUC
  is source identification, not injection detection: the same features
  predict WHICH source dataset a row came from just as well (0.9998,
  0.9841, 0.9878), teaching non-transferable notions of injection.

See the paired page for the tool execution pipeline this gate sits inside.
