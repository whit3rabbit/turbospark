# turbospark-ffi

The C ABI a native GUI drives the engine through, and the SwiftPM package
over it. Produces a `staticlib` plus a hand-written
`include/turbospark.h`; `swift/TurboSpark` wraps it and
`swift/TurboSparkApp` is a SwiftUI chat app that verifies the stack end to
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
|   +-- lib.rs              # crate root, type aliases, re-exports
|   +-- abi.rs              # status codes, the per-thread error slot, `guard`, ABI helpers
|   +-- strings.rs          # borrowing C strings in, handing owned ones out
|   +-- wire.rs             # the JSON shapes (camelCase)
|   +-- session.rs          # the opaque handle (Session over shared SessionCore); the cancel flag
|   +-- open.rs             # opening an install (macOS)
|   +-- generate/           # one turn: render, decode, stream, report
|   |   +-- mod.rs          # turn execution driver, budget clamp, speculation block
|   |   +-- channel.rs      # channel split: separating reasoning and content
|   |   +-- prompt.rs       # message decoding, prompt template rendering, tokenization
|   |   +-- vision.rs       # image attachment and injection lifecycle
|   |   \-- tests.rs        # unit tests: speculation, wire compatibility, image parts
|   +-- models.rs           # catalog, probe, install (portable)
|   +-- server.rs           # the in-process HTTP server: background thread, tokio runtime, lifecycle
|   +-- server_model.rs     # ChatModel adapter over SessionCore, for server.rs
|   +-- server_registry.rs  # the models a RUNNING server serves, and its event ring
|   +-- telemetry.rs        # phase counters and peak footprint
|   +-- testing.rs          # session_for_testing (scripted testing harness)
|   +-- vision.rs           # Image data URL decoding & vision token prep
|   \-- api/                # C ABI entry points (extern "C")
|       +-- core.rs         # errors, strings, system telemetry
|       +-- session.rs      # session lifecycle and introspection
|       +-- generate.rs     # generation, prompt rendering, tokenization, window fit
|       +-- models.rs       # catalog, recommendations, probe, install
|       \-- server.rs       # start / attach / detach / stop / info / poll_events
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

# BLOCKED is a SECOND install and covers what MODEL structurally cannot:
# every speculation assertion is a property of the install being opened, so
# one variable gates one shape. Point it at a MoE or sub-4-bit install and
# the refusal cases run too (see Gotcha 11).
make swift-test-real MODEL=~/models/qwen38-27b-mtp.gturbo \
                     BLOCKED=~/models/ornith35b.gturbo
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

   **THE LOAD-GUARD WORK IS THE SECOND TIME THIS EARNED ITS PLACE, and this
   one was a SIGNATURE change rather than a field.** `ts_recommend_json` grew
   an `options_json` argument (`uint32_t, const char *, char **`), which the
   `rlib` face cannot notice at all -- `tests/c_surface.rs` calls the Rust
   function directly and would pass against a two-argument declaration in the
   header forever. Only `swift/TurboSpark/Tests` links the `staticlib` and
   goes through `turbospark.h`. Anything that changes an ARITY here is
   unverified until `make swift-test` has run, and `make swift-lib` must run
   first or that suite links the previous archive (Gotcha 9).

   Note what did NOT need a signature: `loadGuard` and `minAutoContext` are
   `OpenOptions` fields, so they cost a line in `wire.rs` and a paragraph in
   the header's comment. That asymmetry is Gotcha 3's whole argument in one
   change -- the options bag absorbed two knobs for free and the one call
   taking a bare integer had to break.

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
   every consumer, and `swift/TurboSparkApp` has to repeat the flag with
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

11. **ONE INSTALL VARIABLE CAN ONLY EVER GATE ONE SHAPE, and speculation is
   a property of the install.** `TURBOSPARK_TEST_MODEL` opens one artifact
   per run, so every speculation assertion in `RealModelTests` is conditional
   on which one -- `testADeclinedDflashDrafterCanBeAskedForByName` SKIPS
   unless it happens to be the DFlash2 install, and until 2026-08-21 the
   REFUSAL path was covered by nothing at all: it was swept by hand and
   written into `DEVIATIONS.md`, which no run re-checks.
   `TURBOSPARK_TEST_MODEL_NO_SPECULATION` (`BLOCKED=` on the make target) is
   a second install, so one `swift test` covers two shapes.

   **It must point at a MoE or sub-4-bit install and the tests CHECK that**,
   rather than trusting the caller. A dense INT4 install with no MTP head
   also reports speculation off with a reason and would satisfy every other
   line, so a variable aimed at the wrong artifact would read green while
   re-testing a case already covered -- the fixture-must-discriminate rule
   (AGENTS.md Gotchas 48, 50, 51) applied to an env var. Verified by aiming
   it at `qwen38-27b` on purpose: both cases fail, and both say what to point
   it at instead.

   The refusal case is also this crate's regression guard for the wrong-cause
   bug fixed the same day (`crates/runtime/CLAUDE.md` Gotcha 0): a named block
   on a MoE install used to be refused for its missing HEAD rather than its
   architecture, which on a GUI surface is a message telling someone to
   download 4.4 GB that cannot help.

12. **THE LOAD GUARD MUST BE THE SAME ON BOTH CALLS A GUI MAKES.**
    `ts_recommend_json` ranks under the tier it is given and `ts_session_open`
    refuses under the tier IT is given, and the two share one memory budget by
    construction -- which is what makes a hub verdict worth showing. A host
    ranking under `relaxed` while opening under `strict` promises a fit the
    loader then refuses, in the one place a user cannot see the two disagree.
    Neither call can detect the mismatch, so nothing here will ever raise it;
    the Swift side keeps one `AppModel.activeLoadGuard` accessor for exactly
    that reason (`swift/CLAUDE.md` Gotcha 25).

    Absent, `null` and `{}` all mean `relaxed` on both calls, which is what
    this ABI did before the option existed and what every frozen footprint row
    describes. An unrecognized STRING is refused rather than defaulted, for
    `sized`'s reason and one of its own: quietly ranking under the default
    when the caller asked for `strict` is the exact failure the option exists
    to prevent.

13. **`ts_server_start` SHARES THE OPEN MODEL RATHER THAN OPENING A SECOND
    ONE, AND THAT IS WHY `Session` SPLIT INTO A THIN HANDLE OVER
    `SessionCore`.** Added 2026-08-30. The alternative -- have this crate open
    a full `turbospark_server::RealChatModel` of its own -- would reuse 100%
    of that crate's tested speculation/chunked-prefill/vision/guardrail logic
    with zero duplication, and was declined: a GUI already holding one
    `RealForwardRunner` open (Gotcha 1's whole design assumes this) would pay
    a SECOND multi-gigabyte mapping and Metal pipeline compile just to expose
    the same model over HTTP, which is not survivable on the machine the GUI
    itself is running on for a double-digit-gigabyte install.

    `SessionCore` carries the engine, the tokenizer and everything resolved
    at open; `Session` is `Arc<SessionCore>` behind a newtype, reachable via
    `Deref` so every pre-existing `session.tokenizer` / `session.engine` /
    ... call site needed no change (Rust's field-access autoderef, same
    mechanism method-call autoderef uses). `Session::core()` hands a clone of
    that `Arc` to `server::Server::start`, which builds an `FfiChatModel`
    (`server_model.rs`) around it and passes that to
    `turbospark_server::build_router_with_options` -- the SAME router
    function the standalone binary uses. `ts_session_close` on the handle
    that started the server drops only the caller's own reference; the
    engine stays resident until `ts_server_stop` (or process exit) releases
    the server's clone. `tests/c_surface.rs`'s
    `closing_the_session_does_not_stop_a_server_still_serving_it` is the
    end-to-end proof: it drops the `Session` and then makes a real HTTP
    request against the server that outlived it.

    **`FfiChatModel` IS A SECOND COPY OF `RealChatModel::run_completion`'S
    DISPATCH, NOT A REUSE OF IT.** It replicates speculation and chunked
    prefill (both already live on this crate's own `generate/` for the SAME
    session, so skipping either here would make the in-process server slower
    than a direct `ts_generate` call for no reason a caller could see), and
    since 2026-08-30 it replicates `RealChatModel`'s image-under-one-lock
    handling too.

    **THIS ENTRY USED TO SAY VISION COULD NOT BE SERVED HERE, AND THE
    SENTENCE WAS TRUE WHEN WRITTEN.** Images were a server-only feature, the
    FFI surface took none, and an in-process server built on a session that
    could not encode one would have been claiming a capability nothing behind
    it could reach -- so the refusal was correct rather than lazy. Nothing
    about that reasoning had to change for it to become wrong: only the
    arrival of `ts_generate`'s image content parts, which is AGENTS.md Gotcha
    62's tell at the level of a capability. The refusals that remain are the
    honest ones, a SCRIPTED session and off-macOS.

    **THE BIND CANNOT HAPPEN ON THE CALLING THREAD, and the first version of
    this got that wrong.** `Server::start` originally called
    `rt.block_on(TcpListener::bind(..))` on the thread `ts_server_start`
    itself runs on, to read the resolved port back before spawning the
    long-lived server thread. That is exactly
    `tokio::runtime::Runtime::block_on`'s documented panic -- "Cannot start a
    runtime from within a runtime" -- and it fired immediately, not on some
    exotic caller: `tests/c_surface.rs`'s own `#[tokio::test]` server tests
    hit it on the first run, because the test body itself runs inside a
    Tokio runtime and `ts_server_start` tried to start a SECOND one on top of
    it. A C caller is never inside a Tokio runtime, so this specific trap
    could not have reached a real Swift host, but the fix is not a test
    workaround: any async Rust host embedding this library through the
    `rlib` face would have hit the identical panic. The bind now happens
    ON the background thread, inside its OWN runtime, and the resolved
    port (or a bind failure) is sent back to `start`'s caller over a plain
    `std::sync::mpsc::channel` -- a blocking OS-level wait with no
    restriction on which thread or runtime calls `recv()` on it, unlike
    `tokio::sync`'s channels or `block_on`.

    Guardrails are NOT overridden and take whatever `ChatModel::guardrails()`
    trait default is (on), matching `ScriptedChatModel`'s own choice to leave
    it alone rather than reason about a feature this session's `generate/`
    never exercises either.

    **`ServerInfo` REPORTS THE BOUND HOST AS WELL AS THE BOUND PORT, AND THE
    SECOND FIELD EXISTS BECAUSE THE FIRST ONE'S DISCIPLINE WAS ONLY HALF
    APPLIED.** Added 2026-08-30. `Server::start` binds `format!("127.0.0.1:
    {port}")` and then reads `local_addr()` back -- and for the life of the
    feature it took the PORT off that read and threw the IP away, leaving
    every consumer to restate the literal. `AppModel.startServer` toasted
    `"Server listening on 127.0.0.1:\(info.port)"`, half observed and half
    asserted, and `AppSettingsView`'s Address row did the same. Each was TRUE
    and neither was a reading; nothing about either would have had to change
    for it to become wrong, which is Gotcha 62's tell at the root. The channel
    carries `SocketAddr` now and `ServerInfo` carries `host`.

    **THE TEST THAT COVERS IT HAD TO BE A SEPARATE ONE, because the three
    round-trip cases structurally cannot see a wrong host.** `server_base_url`
    interpolated the same `127.0.0.1` literal beside the read-back port, so
    every server case here would have passed unchanged under any bind change
    -- in the one file positioned to catch it. It builds from `info["host"]`
    now, and `the_reported_host_is_the_address_actually_bound` asserts the
    value. Mutation-checked by rebinding to `0.0.0.0`: exactly one case
    reddened and the three HTTP round trips stayed GREEN, because this
    platform happily answers a request addressed to `0.0.0.0`. A URL that
    connects is not evidence that the address reported is one a caller should
    copy.

14. **A SERVER SERVES A SET OF MODELS THAT CHANGES WHILE IT RUNS, AND
    DETACHING IS WHAT RELEASES ONE.** Added 2026-08-30, extending Gotcha 13
    (which describes the same design with one model, and whose every word
    about sharing rather than re-opening still holds -- per entry).

    `ts_server_start` takes a NULLABLE session: null starts a server with
    nothing attached, which is the state a GUI wants (the socket binds and
    the address can be shown before the user has chosen what to load), and a
    non-null session is exactly an immediate `ts_server_attach_session`
    rather than a second code path. `ts_server_detach_model` removes one by
    id; `ts_server_stop` still releases the whole set.

    **`server_registry.rs`'s `LiveRegistry` IS THE STORAGE AND NOT THE
    POLICY.** The routing rule -- exact id, else the single attached model
    whatever the name, else a 404 -- is
    `turbospark_server::registry::resolve_among`'s and is deferred to rather
    than copied. A second copy here would be a second place for the
    Claude-Code fallback to drift.

    **A DUPLICATE MODEL ID IS REFUSED BY NAME RATHER THAN SUFFIXED.** The id
    is the install directory's own file name, which is what a request's
    `model` field carries and what `detach` keys on. A silently-renamed
    second copy would be addressable under a name the caller never learned,
    and a detach under the name they DO know would then remove the wrong
    one. Two sessions on one install directory is a caller mistake and this
    is the cheapest place to say so;
    `attaching_a_second_model_under_the_same_id_is_refused` also asserts the
    refused attach left nothing half-added.

    **THE OBLIGATION THIS CREATES ON A HOST IS THE SAME ONE GOTCHA 13
    STATED, NOW PER MODEL.** Each entry holds an `Arc<SessionCore>`, so
    `ts_session_close` on the session it came from frees only the caller's
    handle. A host that drops its own reference without detaching keeps the
    weights, the KV cache and the compiled Metal pipelines resident with
    nothing in its UI still showing the model as loaded. `swift/CLAUDE.md`
    Gotcha 26 is the app-side half.

    **DETACHING DOWN TO ONE MODEL RE-ENABLES THE FALLBACK, which is worth
    knowing before it surprises somebody.** After detaching `alpha` from a
    two-model server, a request naming `alpha` is still ANSWERED -- by the
    survivor, under the single-model rule. Detaching stops that ENGINE from
    answering, not that name from being accepted.
    `detaching_removes_a_model_from_routing` asserts it, and has to read
    `requestRouted.served` off the event stream to do so: an OpenAI response
    echoes the request's own `model` field, so a 200 carrying
    `"model":"alpha"` is equally consistent with either model having served
    it, which is exactly the pair the test must tell apart.

15. **EVERY SERVER THIS CRATE STARTS RECORDS, AND THE RING REPORTS ITS OWN
    OVERFLOW.** `Server::start` always passes an `EventRing` observer, where
    the standalone `turbospark-server` binary passes `None` -- a host
    embedding this has a console to show the rows in, and its own stdout is
    not where they would go.

    `ts_server_poll_events_json` DRAINS: an event is returned exactly once,
    so a host polling on a timer appends what it gets rather than
    deduplicating. `max` bounds ONE call rather than the buffer, and the
    remainder stays queued, so a burst arrives late and never silently
    short. `max == 0` means unbounded, which is what a final drain before
    shutdown wants.

    **`dropped` IS IN THE PAYLOAD RATHER THAN BEING A SEPARATE QUERY, and
    that is the point of the field.** A ring that quietly discarded its
    oldest rows would make a console read "nothing happened in that window",
    which is the same thing it reads when the server genuinely was idle --
    and those are the two states somebody watching it is trying to tell
    apart. It counts per DRAIN rather than per lifetime, so a gap already
    shown is not reported again on the next poll. Three unit tests in
    `server_registry.rs` cover the bound, the count and the reset; making
    `drain` a peek reddens all three plus the three C-surface polling cases,
    which is the invariant doing the work rather than an over-broad test.

16. **`open.rs`'S `open()` OPTS INTO PREFIX KV REUSE, THE ONE PLACE IT
    DIVERGES FROM `open_session`, AND `generate/` OPTS INTO CHUNKED
    PREFILL TOO -- NEITHER TAKES A FLAG.** Added 2026-09-01. Gotcha 5 already
    said `open.rs` mirrors `crates/cli/src/generate.rs::open_session` step
    for step; that function backs the CLI's single-shot `--prompt` path, not
    `--chat`, so it never calls `runner.set_prefix_reuse(true)`.
    `TurboSparkApp` is multi-turn by construction -- one long-lived session
    per loaded model, `generate` called once per chat message for the
    session's whole lifetime -- exactly the shape `--chat`'s own header names
    as its reason for opting in ("the ONE caller that opts in by default,
    because it is the one that is multi-turn by construction and is named by
    no frozen row"), so this file now does too. Safe unconditionally: a
    family this cannot help (recurrent state, a sliding-window ring past its
    slack) silently returns 0 reused tokens rather than erroring
    (`crates/runtime/CLAUDE.md` Gotcha 30), so the floor is "no worse than
    before", never a new failure mode.

    `generate/` separately routes `(Engine::Real(runner), None)` through
    `run_raw_completion_chunked_cancellable` at `foundation::DEFAULT_CHUNK_SIZE`
    whenever `runner.supports_chunked_prefill()` -- the SAME predicate
    `crates/server/CLAUDE.md` Gotcha 19 documents for `RealChatModel`, added
    here because this crate had simply never been wired to it (unlike prefix
    reuse, there was no historical reason: chunked-vs-sequential is a
    STANDING byte-identity guarantee this crate did not need to re-prove, only
    to reach). **Gated on `image_parts.is_empty()` as well**, matching
    `crates/server/CLAUDE.md` Gotcha 21's own discipline: the vision-capable
    family's chunked driver refuses an open image prompt BY NAME
    (`crates/runtime/CLAUDE.md` Gotcha 14), so composing the two on a call
    this crate's own `attach_images` already handles separately would turn an
    image turn that used to succeed into one that fails, on a family whose
    general chunked-prefill support has nothing to do with whether THIS call
    happens to carry a picture.

    `GenerateResult.reusedPrefixTokens` reports the count (0 when nothing was
    reused), for the reason `crates/cli/src/chat.rs`'s `[prefix-reuse]`
    footer line exists: an integration that silently no-ops reads exactly
    like one that works, and that file's own header records the feature
    reading 0/33 through two rounds of an implementation that looked correct
    everywhere else. **That exact failure mode is what caught the first
    version of this wiring, in Swift rather than in Rust.** A
    `testASecondTurnReusesThePreviousTurnsKV` case
    (`swift/TurboSpark/Tests/TurboSparkTests/RealModelTests.swift`) sends two
    turns on one real session and asserts the second turn's
    `reusedPrefixTokens` is a MAJORITY of the first turn's `promptTokens`, not
    merely nonzero -- measured 14 of 18 (78%) on the real Gemma 4 install,
    consistent with `--chat`'s own recorded 13/33 and 29/49 rather than full
    recovery, which the header there also explains. Only `swift test` can run
    it (Gotcha 6: `tests/c_surface.rs` is deliberately model-free), which is
    why the assertion belongs there and not in this crate's own suite.

17. **THE HEADER STATED THE ALLOWED SLOT SET AND `open()` ENFORCED NOTHING,
    AND THIS IS THE ONE FRONT END WHERE THAT IS FATAL.** `turbospark.h` has
    said `expertCacheSlots number | "auto" | null (default auto; 8/16/24/32)`
    for as long as the option has existed, and `docs/SWIFT_BINDINGS.md` says
    the same -- so the CONTRACT was documented and correct. What was missing
    was the check: `sized` proved only "a non-negative integer", and
    `ExpertCacheSlots::Fixed` was built from whatever came through.

    Every other front end validates against
    `foundation::runtime_config::ALLOWED_CACHE_SLOTS` --
    `crates/invocation`'s parser, `crates/bench`'s and `crates/server`'s
    argument loops, `catalog::entry`. This binding was the one that did not,
    and it is the one whose caller is a PICKER rather than a typed flag: a GUI
    offers a menu, a user chooses from it, and this engine is linked into that
    GUI's process, so a value the engine refuses is not an error anyone can
    show -- `crates/streaming`'s expert cache panics and the abort takes the
    whole app with it (root Gotcha 64 is the reachable case, at
    `slots == top_k` on the first multi-token prompt). `swift/TurboSparkApp`
    duly offered `[0, 4, 8, 16, 32, 64, 128]`, four of them fatal.

    The check sits with the other option mapping, BEFORE anything is read from
    disk, for Gotcha 5's reason: the error then names the option rather than
    the path, which is also what lets
    `an_out_of_set_expert_cache_slot_count_is_refused_before_the_model_is_read`
    exercise it with no install on the machine. That test asserts the LEGAL
    values still get past the option check as well, or the guard would pass by
    refusing everything -- a gate that cannot fail (`swift/CLAUDE.md`
    Gotcha 22).

    **A DOCUMENTED CONTRACT IS NOT AN ENFORCED ONE.** Reach for the header
    when asking what a caller may send, and for the code when asking what
    happens if they send something else.
