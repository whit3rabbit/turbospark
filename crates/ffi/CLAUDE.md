# turbospark-ffi

The C ABI a native GUI drives the engine through, and the SwiftPM package
over it. Produces a `staticlib` plus a hand-written
`include/turbospark.h`; `swift/TurboSpark` wraps it and
`swift/TurboSparkDemo` is a SwiftUI chat app that verifies the stack end to
end.

## Safety

This crate carries `unsafe` and cannot forbid it: it IS the ABI layer, so
every entry point takes raw pointers. It joins `model-io` and `streaming` as
the third such crate (AGENTS.md Gotcha 9).

## Directory & File Structure

```
crates/ffi/
+-- Cargo.toml
+-- include/
|   \-- turbospark.h        # THE CONTRACT. Hand-written; see Gotcha 2.
+-- src/
|   +-- lib.rs              # every `extern "C"` entry point
|   +-- abi.rs              # status codes, the per-thread error slot, `guard`
|   +-- strings.rs          # borrowing C strings in, handing owned ones out
|   +-- wire.rs             # the JSON shapes (camelCase)
|   +-- session.rs          # the opaque handle; where the cancel flag lives
|   +-- open.rs             # opening an install (macOS)
|   +-- generate.rs         # one turn: render, decode, stream, report
|   +-- models.rs           # catalog, probe, install (portable)
|   \-- telemetry.rs        # phase counters and peak footprint
\-- tests/
    \-- c_surface.rs        # the C entry points, through the `rlib` face
```

## Development & Test Commands

```sh
cargo test -p turbospark-ffi

# Build the staticlib and stage it for SwiftPM.
make swift-lib

# The Swift side. `swift-test-real` needs a real install and takes minutes.
make swift-test
make swift-test-real MODEL=~/models/gemma4.gturbo
make swift-demo
```

## Crate Gotchas

1. **THE CANCEL FLAG LIVES OUTSIDE THE SESSION MUTEX, AND THAT IS THE WHOLE
   DESIGN.** A GUI generates on a background thread, which holds the engine
   lock for the entire turn, and presses Stop on the main thread. Move the
   `AtomicBool` inside the `Mutex` and `ts_session_cancel` blocks until the
   generation it is trying to stop has finished -- a Stop button that works
   only once the model is done, which a user experiences as a frozen window
   rather than as a bug report. Note the failure mode: it does not error, it
   HANGS, so the test that covers it
   (`cancelling_from_another_thread_stops_a_running_generation`) would time
   out rather than fail loudly.

   `Session::arm` clears the flag at the start of every generation, so a
   Stop pressed between turns is discarded rather than cancelling the next
   one. `a_cancel_between_turns_does_not_cancel_the_next_one` pins that.

2. **THE HEADER IS HAND-WRITTEN AND ONLY THE SWIFT TEST TARGET CAN CHECK
   IT.** `tests/c_surface.rs` reaches the same function bodies through the
   `rlib`, so it passes against a header that declares the wrong signature
   entirely. Only linking the `staticlib` and calling through
   `turbospark.h` can catch a mismatch, which is why
   `swift/TurboSpark/Tests` exists and why it is not optional. It has
   already earned its place once: the first draft assumed `crates/catalog`'s
   rows were camelCase and they are snake_case, and nothing on the Rust side
   noticed.

   A generator (cbindgen) was declined rather than overlooked. It adds a
   build-dependency and a codegen step to a workspace that has declined even
   LTO, and it would only ever restate the Rust side to itself -- the Swift
   target is the stronger check.

3. **OPTIONS AND RESULTS ARE JSON; THE PER-TOKEN PATH IS NOT.** That single
   decision removes about fifteen struct definitions from the header and
   every layout, alignment and versioning question with them, and it makes
   adding a knob a field in `wire.rs` rather than an ABI break. The cost is a
   serde round trip on calls that happen once per session or once per second.
   The streaming callback carries a pointer and a length and no JSON, so the
   hot path pays nothing.

   Field names are camelCase so a Swift `Codable` needs no `CodingKeys`.
   **The two exceptions are `CatalogEntry` and `InstalledModel`**, which are
   `crates/catalog`'s own on-disk shapes (`models.json` and
   `~/.turbospark/installed.json`) passed through unchanged so a GUI's rows
   and the CLI's rows are provably the same data. Those two DO need
   `CodingKeys` on the Swift side, and that asymmetry is deliberate.

4. **EVERY `extern "C"` BODY IS A CALL TO `abi::guard` AND NOTHING ELSE.**
   Unwinding across the FFI boundary is undefined behaviour, and this
   workspace cannot opt out of unwinding: the root `Cargo.toml` records that
   `panic = "abort"` must stay off because two `Drop` impls are load-bearing
   on the unwind path. So a panic escaping into C is a real hazard rather
   than a theoretical one. The two deliberate exceptions are
   `ts_last_error` (it returns a length, and must not be able to clear the
   slot it is reading) and `ts_peak_footprint_bytes` (nothing in it can
   fail, and there is no code to return through).

5. **`open.rs` MIRRORS `crates/cli/src/generate.rs::open_session` AND
   SHOULD BE MAINTAINED BY READING IT.** Both of that function's documented
   traps are live here: the resolved context window is carried on the
   session and never re-read from the request (under `auto` the request
   carries no number and the KV cache is already allocated), and the
   RESOLVED slot count is reported rather than the requested one. Do not
   "simplify" either by reading the caller's options back out.

   **SPECULATION IS THE ONE PLACE IT FOLLOWS THE SERVER INSTEAD.** The three
   decisions are the shared `runtime::speculation_policy`'s, in the same
   order, and the drafter is resolved BEFORE the open because the policies
   name exactly one. But `resolve_speculation` is called with
   `deterministic: true` -- the install half is what is fixed at open, and
   the per-turn half is `generate::turn_block`, because a GUI's temperature
   belongs to the request the way a server's does and not to the process the
   way the CLI's does. A sampled turn falls back SILENTLY: this binding's
   sampling default is 0.2, so sampled is the normal case and a per-turn
   warning would fire on it. The session-level answer is in
   `sessionInfo.speculation` instead, said once.

   Two smaller decisions inside it. Every option is MAPPED before anything
   is read from disk, so a misspelled key outranks a bad path in the error
   -- which is also what lets the SwiftPM target reach these spellings with
   no install on the machine, the only kind of check Gotcha 2 admits. And
   `Session` carries a plain `Option<usize>` block rather than a
   `runtime::SpeculationPlan`, because that type is macOS-only and this
   struct is not; the human-readable half already lives in `SessionInfo`.

6. **`Engine::Scripted` IS ABSENT FROM THE HEADER ON PURPOSE.** It is
   reachable only through `session_for_testing` on the `rlib` face, and it
   exists so the whole generate path -- the channel split, the cancel
   plumbing, the event callback, the result JSON -- can be tested on any
   platform with no multi-gigabyte install. Without it the threading
   contract in Gotcha 1 would be covered by nothing but a manual click.

7. **THE INSTALL BYTE CALLBACK IS CALLED CONCURRENTLY FROM WORKER
   THREADS**, because `HttpRangeSource` splits a large range across
   connections. The STAGE lines arrive on the calling thread. That is why
   `models::install` takes them as two separate bounds (`FnMut` and
   `Arc<dyn Fn + Send + Sync>`) rather than as two arms of one callback, and
   why the header states the obligation on the caller. A progress bar driven
   from those events must take the MAXIMUM rather than the latest, or it
   jumps backwards.

8. **A SwiftPM `-L` FLAG IS RESOLVED AGAINST THE PACKAGE BEING BUILT, NOT
   THE ONE THAT DECLARED IT.** So `swift/TurboSpark`'s own
   `-LSources/CTurboSpark` is correct when its tests link and wrong for
   every consumer, and `swift/TurboSparkDemo` has to repeat the flag with
   its own view of the same directory. That is a SwiftPM limitation rather
   than a mistake; the fix for a published package is an `.xcframework`
   binary target, which resolves paths for its consumers properly. A
   two-package repository does not need the packaging step.

9. **SwiftPM DOES NOT TREAT THE STATICLIB AS A BUILD INPUT, so `swift test`
   will happily link the PREVIOUS one.** The `-L` path arrives as an unsafe
   linker flag, which SwiftPM passes through without adding a dependency
   edge, so an archive that changed under an unchanged set of `.swift` files
   triggers no relink at all. That is not cosmetic here: this target is the
   ONLY thing that can check the hand-written header (Gotcha 2), and a stale
   link makes it check the build before the one being tested.

   Measured 2026-08-21 while mutation-checking the FFI's speculation
   options: two mutations of `open.rs` in a row both read as the FIRST one's
   failure, and the RESTORED tree still read red until the Swift sources
   were touched by hand. A mutation check on this crate was unreliable for
   as long as the seam existed. `scripts/swift-lib.sh` now `touch`es every
   `.swift` file in both packages after staging the archive; deleting
   `.build` would also work and costs a full package rebuild each time.

10. **`scripts/swift-lib.sh` SETS `MACOSX_DEPLOYMENT_TARGET`, AND IT MUST
   MATCH BOTH `Package.swift` FILES.** Without it cargo builds for the host
   SDK's default (macOS 26.5 on this machine) while SwiftPM links for 13.0,
   which draws an `ld` warning per object file. The warnings are the visible
   half; the real problem is an app claiming to support macOS 13 while
   containing objects built against a much newer SDK.
