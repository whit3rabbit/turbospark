# Crash recovery and containment limits

What the app does when a render or model failure kills the process, and what
it cannot do. Added with ZP-17
([draft](../../docs/designs/zcode-parity/ZP-17-error-boundaries.md)).

## What cannot be contained

SwiftUI has no error boundary. A trap raised while a view body runs (force
unwrap, array bounds, integer overflow, `precondition`, `try!`, out of memory)
ends the process, and no sibling view can catch it. Swift `throw` is not the
failure class: it is typed and already handled at the call site. Do not add a
wrapper view that claims to isolate a pane, and do not describe the app as
crash proof.

## What the app does instead

1. **Typed errors at the data layer.** A parse or decode API that reads
   untrusted or truncated input returns a `Result` or throws a typed error, and
   the view renders a failure state. This is a per-surface rule that each
   surface's own change owns (see "Deferred" below).
2. **Detect an unclean exit.** `App/CrashRecoveryStore.swift` writes
   `session-marker.json` (format version and start time only) under the
   profile store root once the main window appears, and deletes it in
   `applicationWillTerminate`, after every other quit step. A marker found at
   the next start produces one warning toast. The marker is replaced at that
   moment, so the notice shows once per crash.
3. **The in-flight turn.** A turn that was streaming is not recovered by this
   feature. On an orderly quit `AppModel.shutdown()` saves the partial reply as
   an interrupted row. After SIGKILL or a crash nothing runs, the partial reply
   existed only in memory, and the notice says it "may be incomplete". The
   selected chat and window state already persist through the normal stores, so
   there is no separate restore action or snapshot.

## Accepted false positives

Force Quit, a reboot or power loss mid-session, and a second instance on the
same store root all leave a marker. The toast therefore says the app "did not
quit cleanly", never "crashed". A marker that cannot be written costs only the
next run its notice. The vault lock keeps the process alive, so it never
clears the marker, and a second `beginSession` in one process returns clean.

## Stalled engine and guarded relaunch (OC-14)

A turn whose engine goes silent for 90 seconds inside one generate call
shows a strip above the composer: Keep Waiting, Stop Turn, Restart TurboSpark
(`Diagnostics/GenerationStallWatchdog.swift`,
`Generation/StallRecoveryBannerView.swift`). Stop cannot free a wedged
engine, only end the turn. Restart (`Diagnostics/GuardedRelaunch.swift`)
flushes chats first (`AppModel.flushForRelaunch`), spawns a detached `open`
helper, and quits through the normal path, with a 10 second hard exit
behind it. The hard exit clears `session-marker.json` only when the flush
finished. Relaunches are capped at 2 per 10 minutes in
`relaunch-guard.json`. Nothing is logged, reported or sent. A wedged main
thread cannot show this strip; there is no heartbeat for it yet. Record:
`docs/designs/opencode2-parity/OC-14-hang-watchdog.md`.

## Coding rule

No force unwrap (`!`), `try!` or unchecked subscript on data that came from
disk, a document, the network or a model in code a view body reaches. Use
`guard let`, `try?` with a visible failure state, or a bounds check. This is a
review rule, not a lint gate: there is no tool that decides "reachable from a
view body" mechanically, and a repository-wide `!` scan would be noise. Do not
add a baseline file for it.

## Per-surface failure states

Each surface owns its typed error and failure state, as the rule above says.
All three are built: the trajectory log reader (ZP-03 `RolloutStore`) skips a
truncated final line and shows a count, office rendering (ZP-08
`OfficeRenderModel`, `OfficeRenderFailureView`) shows a bounded error card on
malformed XML with an external-opener fallback, and plugin page loads (ZP-06
`PluginPageFailureCard`) show a typed failure with Try again. What stays
unbuilt is the generic typed-error contract the ZP-17 draft first proposed.
