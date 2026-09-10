# Handoff

Session state for the next agent. Written 2026-09-09. This file is
SHORT-LIVED by design: once its "start here" items are done, delete the
section rather than letting it rot beside `ROADMAP.md`, which is the real
tracker.

## Start here

**Nothing in the last change was ever compiled.** It was authored in a
Linux container with no Swift toolchain (`swift: command not found`), so
`swift build` and `swift test` did not run and neither did any mutation
check. Verify before building anything on top of it:

```sh
make swift-lib                       # both swift test targets fail without this
cd swift/TurboSparkApp && swift build && swift test
```

Read `Executed N tests, with M failures`. Do not read swift-testing's
`Test run with 0 tests in 0 suites passed`, which is the other harness and
is what a `| tail` lands on (`swift/CLAUDE.md` Gotcha 44).

`ROADMAP.md` Priority 0 items 6 to 9 are this work, in the tracker's own
format and in the order to do them. Item 6 is the verification above and is
blocking; item 7 is the root cause that was deliberately left open.

## What landed, and what it actually was

`QwenParityFeaturesTests.testCronOneShotFiresAndRemovesItself` was
intermittently red with `["wake word", "wake word"]` against
`["wake word"]`, and passed 5 of 5 under `--filter`.

The reported diagnosis was test-to-test store leakage. That was wrong, and
the tell was in the report: **`--filter` passing 5 of 5 is the mechanism,
not a coincidence.** There was only ever ONE job. It was delivered twice.

`AppModel.init` calls `startCronScheduler()`, which puts a 20-second
repeating `Timer` on the main run loop calling
`CronScheduler.shared.fireDueJobs()`. Nothing invalidates it, and its block
does not reference the model, so a deallocated `AppModel` leaves it
running. About twenty test files build an `AppModel`; a `--filter` run
builds none. `fireDueJobs` then AWAITS its delivery while `completeFire`
rewrites `nextFireAt` only after that await returns, so a tick landing in
the window took the same job again.

That is a live production bug, not a test artifact: any cron delivery that
outlives one tick double-submits into the user's chat.

Two changes went in:

- `takeDueJobs` marks a taken id `inFlight` and skips one already there;
  `completeFire` clears it. Pinned by
  `testAnOverlappingPollDoesNotDeliverTheSameJobTwice`.
- `CronScheduler` takes `init(directory:)` and the four `execute*` statics
  take `scheduler: CronScheduler = .shared`, so a case gets its own
  instance and its own file. `AppStorageRoot` isolates the FILE; it never
  isolated the OBJECT.

The leaked timer itself is untouched. That is `ROADMAP.md` P0 item 7 and is
the actual root cause.

## Two things worth carrying past this task

**A store seam is not a concurrency seam.** `AppStorageRoot` makes a test's
writes land somewhere private and says nothing about a `.shared` whose
behaviour other code is still driving. Full write-up in
`swift/docs/SWIFT_STORAGE.md`, closing section.

**"Passes under `--filter`, fails in a full run" names the mechanism.** It
is not ordinary order-sensitivity. It says something else in the process is
acting on the same object, and the thing to look for is a timer, a task or
an observer installed by another test's fixture -- not a value left behind
in a file. The first theory here was a stale file reachable through a
reused pid, which was plausible, cost an hour, and was refuted by that one
sentence in the bug report.
