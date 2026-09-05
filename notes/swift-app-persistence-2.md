---
uuid: "46b9d0c4-642c-4f54-a02a-c3787756f7fd"
title: "TurboSparkApp: testing traps"
summary: "A Swift trap (out-of-range slice, Int(1e300)) aborts xctest with signal 5 and prints no 'Test Case failed' line, so a harness keying on that string calls it SURVIVING when it reproduced the bug exactly"
tags: ["swift", "app", "testing"]
source: "swift/CLAUDE.md"
depends_on: ["b8b6dcbc-87fe-49b1-a76c-7047aabf5047"]
created: "2026-09-05"
updated: "2026-09-05"
---

## What's different about mutation-checking and testing this app specifically?

`swift test --filter SuiteName/testCaseName` runs one case in ~0.3s against
~15s for the whole suite, which is what makes a mutate/run/restore loop
cheap enough to do per assertion. Two harnesses run and print two
summaries: swift-testing's `Test run with 0 tests in 0 suites passed` is
NOT the result, and a `| tail` lands on exactly that line. Read `Executed N
tests, with M failures` instead.

## Don't

- Don't trust a harness that keys on the string `Test Case ... failed (` to
  detect a mutation. A genuine Swift trap (`Int(1e300)`, an out-of-range
  slice) aborts the whole xctest process with `exited with unexpected
  signal code 5` and prints `Fatal error:`, never that string. A harness
  that only looks for the failure line reports this as SURVIVING or as a
  build error when the mutation actually reproduced the bug. Read the raw
  output before believing either verdict.
- Don't run `swift build` from a subdirectory of the package to confirm a
  fix. It fails with `linker command failed` and no error line above it,
  since `-L../TurboSpark/Sources/CTurboSpark` resolves against the CWD, not
  the package root. Build and test from `swift/TurboSparkApp` itself.
- Don't assume a red pre-existing test after a state fix is a regression
  you caused. It's at least as likely to be the defect reproducing
  correctly. The tell is one question about the ASSERTION, not the diff:
  does it state a requirement, or restate the implementation? "Home should
  read `.granted`" is the second wearing the first's clothes, when that row
  could never have reported anything else. Update the assertion with the
  reason, don't revert the fix or delete the case.
- Don't write a fixture for a USER-scope skill or agent. Unlike the
  project-scoped and global-tools directories (both safely redirected),
  `CustomToolManager.userHomeToolsDirectory`,
  `SkillManager.defaultUserSkillsDirectory`, and
  `AgentManager.defaultUserAgentsDirectory` are literal `~/.turbospark/...`
  paths with no test seam. This is the archive-overwrite failure mode
  through a fourth API, on the developer's own home directory.
- Don't index into `AgentManager.shared.builtInAgents` by position in a new
  test. `[0]` is `explore`, which disallows every write, and multiple cases
  have failed on that before switching to `findAgent(name:
  "general-purpose")` by name instead.
- Don't take a tidy confirmation `swift build` at face value. An incremental
  build that compiled nothing prints the same `Build complete!` as one that
  built your change (`[0/3] Write swift-version...` vs `[N/M] Compiling
  ...`). Read the output of the run that actually compiled, or `touch` the
  files first.
