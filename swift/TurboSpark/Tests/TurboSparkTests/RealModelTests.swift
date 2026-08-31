import Foundation
import XCTest

@testable import TurboSpark

/// End-to-end against a real install, gated on `TURBOSPARK_TEST_MODEL`.
///
/// The Rust and Swift tests beside this one prove the ABI and the plumbing
/// with no model. **This is the only thing that proves the whole stack**:
/// SwiftUI-shaped async code, through the C boundary, into a real Metal
/// forward pass and back. It is gated rather than `#if`'d out for the same
/// reason `crates/bench`'s gates are -- an unset variable SKIPS with a note,
/// and a variable pointing at a missing install FAILS.
///
///     TURBOSPARK_TEST_MODEL=~/models/gemma4.gturbo swift test
///
/// Minutes, not seconds: opening maps gigabytes and compiles pipelines.
final class RealModelTests: XCTestCase {

    /// Resolves the test model path from the environment or skips if unset.
    private func modelPath() throws -> String {
        guard let raw = ProcessInfo.processInfo.environment["TURBOSPARK_TEST_MODEL"],
            !raw.isEmpty
        else {
            throw XCTSkip("set TURBOSPARK_TEST_MODEL to a .gturbo install to run this")
        }
        // Passed through with its tilde intact ON PURPOSE: the wrapper
        // expands it, and expanding here too would test this test rather
        // than the wrapper.
        return raw
    }

    /// An install this engine CANNOT speculate on, for the refusal cases.
    ///
    /// **A SECOND VARIABLE RATHER THAN A SECOND RUN, because one process
    /// holds one session per open and the shapes it has to compare are
    /// properties of DIFFERENT installs.** `TURBOSPARK_TEST_MODEL` covers
    /// whatever it points at, and every speculation assertion in this file is
    /// therefore conditional on which artifact that is -- which is how the
    /// blocked path came to be swept by hand and asserted by nothing.
    ///
    ///     TURBOSPARK_TEST_MODEL=~/models/qwen38-27b-mtp.gturbo \
    ///     TURBOSPARK_TEST_MODEL_NO_SPECULATION=~/models/ornith35b.gturbo \
    ///       swift test
    ///
    /// Any install whose ARCHITECTURE blocks the batched verify will do: a
    /// MoE one (no batched routed pair) or a sub-4-bit one (the GEMM is
    /// INT4-only). It must NOT be a merely headless dense INT4 install --
    /// that one is refused for a different reason, and the case below is
    /// written to tell the two apart.
    private func blockedModelPath() throws -> String {
        guard let raw = ProcessInfo.processInfo.environment["TURBOSPARK_TEST_MODEL_NO_SPECULATION"],
            !raw.isEmpty
        else {
            throw XCTSkip(
                "set TURBOSPARK_TEST_MODEL_NO_SPECULATION to a MoE or sub-4-bit install "
                    + "to run this")
        }
        return raw
    }

    /// Tests opening a real model and inspecting its resolved session parameters.
    func testOpensAndDescribesItself() async throws {
        let session = try await TurboSparkSession(modelPath: try modelPath())
        // The RESOLVED values. Under automatic sizing nothing was asked for,
        // so these are the only numbers that exist.
        XCTAssertGreaterThan(session.info.maxContext, 0)
        XCTAssertGreaterThan(session.info.vocabSize, 0)
        XCTAssertFalse(session.info.family.isEmpty)
        // `drafter` is non-nil exactly when `block` is, which is the one
        // invariant the two fields can break independently -- a drafter
        // named beside a null block would read as "on" to a status panel
        // that checked the wrong field.
        let speculation = session.info.speculation
        XCTAssertEqual(
            speculation.block != nil, speculation.drafter != nil,
            "block and drafter must agree, got \(String(describing: speculation))")
        print(
            "open: family=\(session.info.family) context=\(session.info.maxContext) "
                + "slots=\(session.info.expertCacheSlots) vocab=\(session.info.vocabSize) "
                + "speculation=\(speculation.block.map(String.init) ?? "off")"
                + "\(speculation.drafter.map { " via \($0.rawValue)" } ?? "")"
                + "\(speculation.reason.map { " (\($0))" } ?? "")")
    }

    /// Tests streaming generation of coherent text against a real model.
    func testGeneratesCoherentTextAndStreamsIt() async throws {
        let session = try await TurboSparkSession(modelPath: try modelPath())
        var options = GenerateOptions()
        options.maxNewTokens = 120
        options.seed = 20260721

        var streamed = ""
        var sawPrefill = false
        var result: GenerationResult?

        for try await event in session.generate(
            [ChatMessage(role: .user, content: "Explain how coastal wetlands reduce flood damage.")],
            options: options
        ) {
            switch event {
            case .prefill: sawPrefill = true
            case .content(let c): streamed += c
            case .reasoning: break
            case .finished(let r): result = r
            }
        }

        let r = try XCTUnwrap(result)
        XCTAssertTrue(sawPrefill, "prefill progress should reach the caller")
        XCTAssertGreaterThan(r.newTokens, 20)
        // The streamed text and the accumulated result must agree, or a
        // caller trusting one of them is wrong about the other.
        XCTAssertEqual(streamed, r.content)
        // Coherence, checked as weakly as an automated test honestly can:
        // real words, not markup or a single repeated token.
        XCTAssertTrue(
            r.content.lowercased().contains("wetland") || r.content.lowercased().contains("flood"),
            "expected an on-topic answer, got: \(r.content.prefix(200))")
        print("generate: \(r.newTokens) tokens at \(r.tokensPerSecond ?? 0) tok/s, \(r.stopReason)")
    }

    /// A page to send. A THIRD variable, for Gotcha 11's reason: a vision
    /// install is a shape `TURBOSPARK_TEST_MODEL` alone cannot guarantee, and
    /// one variable can only ever gate one shape.
    ///
    ///     TURBOSPARK_TEST_MODEL=~/models/qwen38-27b-vision.gturbo \
    ///     TURBOSPARK_TEST_IMAGE=~/models/vision-probe-qwen38/imgs/oracle/medium.png \
    ///       swift test
    private func testImagePath() throws -> String {
        guard let raw = ProcessInfo.processInfo.environment["TURBOSPARK_TEST_IMAGE"],
            !raw.isEmpty
        else {
            throw XCTSkip(
                "set TURBOSPARK_TEST_IMAGE to a page (and TURBOSPARK_TEST_MODEL to an "
                    + "install with a vision tower) to run this")
        }
        let expanded =
            raw.hasPrefix("~/")
            ? FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent(String(raw.dropFirst(2))).path
            : raw
        // FAILS rather than skips when the file is missing: a path that was
        // named and is not there is a broken invocation, not an absent one.
        XCTAssertTrue(
            FileManager.default.fileExists(atPath: expanded),
            "TURBOSPARK_TEST_IMAGE names \(expanded), which does not exist")
        return expanded
    }

    /// **THE END-TO-END VISION ARM, AND ITS LOAD-BEARING ASSERTION IS THE
    /// TOKEN COUNT RATHER THAN THE TEXT.**
    ///
    /// A dropped image is the failure mode this whole path has, and it does
    /// not error: the prompt length agrees, nothing warns, and the model
    /// answers fluently about a picture it was never shown. That is measured
    /// history rather than caution -- `crates/cli` Gotcha 13's first
    /// end-to-end `--image` run transcribed a page it had not seen, with
    /// every shape and length in agreement.
    ///
    /// So this compares the SAME prompt with and without the picture. The
    /// template renders one `<|image_pad|>` marker either way, and the
    /// engine's splice expands that marker to the page's merged-token count
    /// -- hundreds of positions. If the image were silently dropped, the two
    /// prompts would be within a token or two of each other. Nothing about
    /// the page's content has to be known for that to discriminate, which is
    /// what makes it immune to where the transcription happens to stop (the
    /// trap that cost `vision_memory_oracle` two marker designs).
    func testAnImageReachesTheModelAndLengthensThePrompt() async throws {
        let image = try testImagePath()
        let session = try await TurboSparkSession(modelPath: try modelPath())

        // CHECKED, not assumed: a text-only install passes every other line
        // of this test while proving nothing, which is the fixture-must-
        // discriminate rule applied to an env var (Gotcha 11).
        guard session.info.vision.active else {
            return XCTFail(
                "TURBOSPARK_TEST_MODEL points at an install that cannot serve images"
                    + (session.info.vision.reason.map { " (\($0))" } ?? "")
                    + "; point it at one with a vision tower, e.g. qwen38-27b-vision.gturbo")
        }
        XCTAssertNotNil(
            session.info.vision.imageTokenId,
            "an active tower reports the marker id it splices at")

        var options = GenerateOptions()
        options.maxNewTokens = 60
        options.seed = 20260721
        let question = "Transcribe the first line of this page."

        func run(_ message: ChatMessage) async throws -> GenerationResult {
            var result: GenerationResult?
            for try await event in session.generate([message], options: options) {
                if case .finished(let r) = event { result = r }
            }
            return try XCTUnwrap(result)
        }

        let withImage = try await run(
            ChatMessage(role: .user, content: question, images: [.path(image)]))
        let textOnly = try await run(ChatMessage(role: .user, content: question))

        // The whole assertion: a page is worth hundreds of positions, so a
        // dropped image cannot pass this however plausible its answer reads.
        XCTAssertGreaterThan(
            withImage.promptTokens, textOnly.promptTokens + 100,
            "the image prompt is \(withImage.promptTokens) tokens against "
                + "\(textOnly.promptTokens) text-only; the splice did not expand the marker, "
                + "so the picture never reached the model")
        XCTAssertFalse(withImage.content.isEmpty)

        // Printed so a run can be READ. A vision feature needs an arm whose
        // output a person can check against the page: shapes and lengths all
        // agreed in the bug this test exists for.
        print("vision: \(withImage.promptTokens) prompt tokens (text-only \(textOnly.promptTokens))")
        print("vision transcription: \(withImage.content.prefix(300))")
    }

    /// An image sent to an install that cannot serve one is refused BY NAME
    /// rather than dropped, which is the difference between a user seeing a
    /// message and a user seeing a confident answer about nothing.
    func testAnImageIsRefusedByNameWhenTheInstallHasNoTower() async throws {
        let session = try await TurboSparkSession(modelPath: try blockedModelPath())
        guard !session.info.vision.active else {
            throw XCTSkip(
                "TURBOSPARK_TEST_MODEL_NO_SPECULATION happens to carry a vision tower; "
                    + "this case needs an install without one")
        }
        var options = GenerateOptions()
        options.maxNewTokens = 8
        do {
            for try await _ in session.generate(
                [ChatMessage(role: .user, content: "What is this?", images: [.path("/nope.png")])],
                options: options)
            {}
            XCTFail("an image on a tower-less install should be refused")
        } catch {
            let message = "\(error)".lowercased()
            XCTAssertTrue(
                message.contains("vision") || message.contains("image"),
                "the refusal should name what was wrong, got: \(error)")
        }
    }

    /// **The one that matters for a GUI.**
    ///
    /// Cancels from the MAIN actor while the model decodes on the session's
    /// own queue. If the cancel flag were behind the engine lock this would
    /// not fail, it would hang until the full budget was generated -- which
    /// is exactly what a user experiences as a frozen window.
    func testStopReachesTheEngineWhileItIsDecoding() async throws {
        let session = try await TurboSparkSession(modelPath: try modelPath())
        var options = GenerateOptions()
        // Big enough that finishing on its own would take far longer than
        // the cancel point, so an early stop can only be the cancel.
        options.maxNewTokens = 4000

        var tokens = 0
        var result: GenerationResult?
        let started = Date()

        for try await event in session.generate(
            [ChatMessage(role: .user, content: "Write a long detailed essay about tide tables.")],
            options: options
        ) {
            if case .content = event {
                tokens += 1
                if tokens == 20 {
                    // Not inline on the engine's thread: this is the caller's
                    // context, which is the situation being tested.
                    session.cancel()
                }
            }
            if case .finished(let r) = event { result = r }
        }

        let r = try XCTUnwrap(result)
        XCTAssertEqual(r.stopReason, .cancelled)
        XCTAssertLessThan(
            r.newTokens, 500, "should have stopped near the cancel point, not run the budget")
        // The partial turn survives, so a GUI can leave it on screen.
        XCTAssertFalse(r.content.isEmpty)
        print(
            "cancel: stopped after \(r.newTokens) tokens in "
                + String(format: "%.1fs", Date().timeIntervalSince(started)))
    }

    /// **The only thing that drives a batched verify through this binding.**
    ///
    /// A greedy turn is what the speculative loop needs -- acceptance is
    /// `argmax(target) == proposal`, exact only at temperature 0 -- and
    /// every other case in this file samples, so without this the branch
    /// added for speculation is reached by nothing at all. On an install
    /// carrying no drafter it still tests the greedy path, which is also
    /// covered nowhere else here.
    ///
    /// It asserts COHERENCE and not speed. The speedup is measured on the
    /// engine's own probes (`docs/DFLASH2.md`, `docs/MTP.md`) against a
    /// quiet machine; a tok/s figure taken here would be a number from
    /// whatever else the machine was doing.
    func testAGreedyTurnRunsWhateverSpeculationResolvedTo() async throws {
        let session = try await TurboSparkSession(modelPath: try modelPath())
        var options = GenerateOptions()
        options.maxNewTokens = 120
        // EXACTLY zero. `is_deterministic` tests `== 0.0`, so the standing
        // smoke's 0.0001 is a SAMPLED run as far as this gate is concerned
        // and would silently take the sequential loop.
        options.temperature = 0.0

        var streamed = ""
        var result: GenerationResult?
        for try await event in session.generate(
            [ChatMessage(role: .user, content: "Explain how coastal wetlands reduce flood damage.")],
            options: options
        ) {
            if case .content(let c) = event { streamed += c }
            if case .finished(let r) = event { result = r }
        }

        let r = try XCTUnwrap(result)
        XCTAssertGreaterThan(r.newTokens, 20)
        XCTAssertEqual(streamed, r.content)
        XCTAssertTrue(
            r.content.lowercased().contains("wetland") || r.content.lowercased().contains("flood"),
            "expected an on-topic answer, got: \(r.content.prefix(200))")
        let block = session.info.speculation.block
        print(
            "greedy: \(r.newTokens) tokens, \(r.stopReason), "
                + (block.map { "speculative block \($0)" } ?? "sequential (no drafter)"))
    }

    /// Tests that a DFlash2 drafter `auto` declined can be asked for by name.
    ///
    /// Conditional on the install, and deliberately so: this is the only
    /// path through the binding that reaches the block drafter, and it
    /// exists on exactly one artifact here. On any other install the first
    /// open reports no dflash note and the test has nothing to say, which
    /// is reported rather than passed silently.
    ///
    /// It costs a SECOND model open, because the drafter's state is
    /// allocated at open and cannot be switched on a live session -- which
    /// is the same reason both other front ends make it a process-level
    /// setting.
    func testADeclinedDflashDrafterCanBeAskedForByName() async throws {
        let auto = try await TurboSparkSession(modelPath: try modelPath())
        guard auto.info.speculation.reason?.contains("dflash") == true else {
            throw XCTSkip(
                "this install carries no DFlash2 drafter; auto said "
                    + (auto.info.speculation.reason ?? "nothing"))
        }
        // Under `auto` it is OFF, which is the measured decision rather
        // than an accident: 0.96x throughput on prose.
        XCTAssertNil(auto.info.speculation.block)

        var options = OpenOptions()
        options.speculativeDrafter = .dflash
        let named = try await TurboSparkSession(modelPath: try modelPath(), options: options)
        XCTAssertEqual(named.info.speculation.drafter, .dflash)
        let block = try XCTUnwrap(
            named.info.speculation.block, "asking for it by name must turn it on")
        XCTAssertNil(named.info.speculation.reason, "an enabled drafter has nothing to explain")
        print("dflash: block \(block) after auto declined it")
    }

    /// Tests reading phase metrics after completing a generation request.
    func testPhasesAreReadableAfterAGeneration() async throws {
        let session = try await TurboSparkSession(modelPath: try modelPath())
        var options = GenerateOptions()
        options.maxNewTokens = 30
        for try await _ in session.generate(
            [ChatMessage(role: .user, content: "Hi")], options: options)
        {}

        let phases = try await session.phases()
        XCTAssertGreaterThan(phases.calls, 0)
        XCTAssertGreaterThan(phases.totalMsPerCall, 0)
        let peak = try XCTUnwrap(TurboSparkSession.peakFootprintBytes)
        print(
            "phases: \(phases.calls) calls at \(phases.totalMsPerCall) ms, "
                + "hit rate \(phases.expertHitRate.map { String($0) } ?? "n/a"), "
                + "peak \(peak / 1_048_576) MiB")
    }

    /// End to end for the in-process server: start it from a REAL open
    /// session, make an actual HTTP request against the port it bound, and
    /// confirm the reply carries real generated text. No other gate proves
    /// this whole path together -- `crates/ffi/tests/c_surface.rs` covers
    /// the same shape against a SCRIPTED session with no model, and
    /// `testServerCABISymbolsLinkAndValidateNullArgs` only proves the header
    /// links, never that a real forward pass answers through it.
    func testInProcessServerServesARealGeneration() async throws {
        let session = try await TurboSparkSession(modelPath: try modelPath())
        let server = try await session.startServer()
        defer { server.stop() }

        let info = try server.info()
        XCTAssertNotEqual(info.port, 0, "port 0 must resolve to the actually bound port")
        XCTAssertFalse(info.authEnabled)
        XCTAssertEqual(
            info.host, "127.0.0.1",
            "the engine binds loopback and info must REPORT it, not leave a caller to assume it")

        // The URL comes from `info`, not from a literal beside it. Spelling
        // the host by hand here would keep this test green against any bind
        // the reported host no longer matched, which is exactly the gap on
        // the app side that this field closes.
        let base = try XCTUnwrap(info.baseURL)
        var request = URLRequest(
            url: base.appendingPathComponent("v1/chat/completions"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: [
            "model": "m",
            "max_tokens": 8,
            "temperature": 0.0,
            "messages": [["role": "user", "content": "Say hello in one short sentence."]],
        ])

        let (data, response) = try await URLSession.shared.data(for: request)
        let http = try XCTUnwrap(response as? HTTPURLResponse)
        XCTAssertEqual(http.statusCode, 200)

        let body = try XCTUnwrap(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        let choices = try XCTUnwrap(body["choices"] as? [[String: Any]])
        let message = try XCTUnwrap(choices.first?["message"] as? [String: Any])
        let content = try XCTUnwrap(message["content"] as? String)
        XCTAssertFalse(content.isEmpty)
    }

    /// **`info()` AND `stop()` RACING MUST NOT TOUCH A FREED HANDLE.**
    ///
    /// `ts_server_stop` frees its C handle, so any window between reading the
    /// `stopped` flag and dereferencing the pointer is a use-after-free, and
    /// `TurboSparkServer` exists to have no such window. The bug this covers
    /// read the flag under the lock, UNLOCKED, and then made the C call --
    /// which looks careful and is exactly the race.
    ///
    /// The assertion is that this terminates without crashing and that every
    /// call after the stop throws rather than answering. A crash here is the
    /// failure; there is no softer signal a use-after-free gives.
    func testServerInfoRacingStopNeverTouchesAFreedHandle() async throws {
        let session = try await TurboSparkSession(modelPath: try modelPath())
        let server = try await session.startServer()

        // Prove it answers before the race, or a passing race proves nothing.
        XCTAssertNotEqual(try server.info().port, 0)

        await withTaskGroup(of: Void.self) { group in
            for _ in 0..<64 {
                group.addTask {
                    // Either outcome is legal: a read that beat the stop, or
                    // the already-stopped error. Reading a freed pointer is
                    // not, and would crash rather than land here.
                    _ = try? server.info()
                }
            }
            group.addTask { server.stop() }
            group.addTask { server.stop() }
        }

        XCTAssertThrowsError(try server.info()) { error in
            let message = String(describing: error)
            XCTAssertTrue(
                message.contains("already been stopped"),
                "a stopped server must refuse by name, got: \(message)")
        }
    }

    /// **A SESSION DROPPED MID-TURN MUST NOT CLOSE UNDER THE GENERATION.**
    ///
    /// `turbospark.h` forbids `ts_session_close` while `ts_generate` is in
    /// flight, and what upholds that on the Swift side is `generate`'s worker
    /// capturing `self` strongly. Before the fix the capture list named
    /// `handle` alone, so nothing retained the session for the turn: a caller
    /// releasing its last reference while decoding ran `deinit`, and
    /// `ts_session_close` raced `ts_generate` on the queue.
    ///
    /// The session is created and released INSIDE this function with no local
    /// binding surviving, which is the only way to reproduce it -- every
    /// other test here (and `AppModel.executeGenerationTurn`) holds a strong
    /// reference across the turn and therefore cannot see the defect.
    func testAReleasedSessionSurvivesTheTurnItLeftRunning() async throws {
        let path = try modelPath()
        var messages = [ChatMessage(role: .user, content: "Count slowly from one to twenty.")]
        var options = GenerateOptions()
        options.maxNewTokens = 64
        options.temperature = 0.0

        // The stream outlives every strong reference this scope holds: the
        // session is not bound after the call, so its refcount rests entirely
        // on `generate`'s own capture.
        let stream: AsyncThrowingStream<GenerationEvent, Error> = try await {
            let session = try await TurboSparkSession(modelPath: path)
            return session.generate(messages, options: options)
        }()

        var sawContent = false
        var finished = false
        for try await event in stream {
            switch event {
            case .content(let chunk): sawContent = sawContent || !chunk.isEmpty
            case .finished: finished = true
            default: break
            }
        }

        XCTAssertTrue(finished, "the turn must reach .finished with no live caller reference")
        XCTAssertTrue(sawContent, "the turn must produce text, not just survive")
        messages.removeAll()
    }

    /// **`auto` ON AN INSTALL THAT CANNOT SPECULATE OPENS ANYWAY, and says
    /// why.** That is the warn-half of the hard-fail/warn split reaching the
    /// C ABI, and it is the half a GUI depends on: most installs carry no
    /// usable drafter, so a refusal here would make the common case a
    /// failure to open a model that runs perfectly well.
    ///
    /// The three `speculation` fields are checked together because they are
    /// only meaningful as a set -- a null block with a null reason is "the
    /// caller asked for off", and that is a different state from this one.
    func testAutoOpensAnInstallItCannotSpeculateOnAndReportsWhy() async throws {
        let session = try await TurboSparkSession(modelPath: try blockedModelPath())
        XCTAssertNil(session.info.speculation.block, "this install cannot be verified against")
        XCTAssertNil(session.info.speculation.drafter, "no block means no drafter to name")
        let reason = try XCTUnwrap(
            session.info.speculation.reason,
            "auto declining without saying why is the silence this feature exists to end")

        // ASSERT THE FIXTURE DISCRIMINATES. A dense INT4 install with no head
        // also reports off with a reason, and would pass every line above --
        // so a variable pointed at the wrong artifact would make this file
        // read green while testing the case it already covers. The refusal
        // case below is written against the ARCHITECTURAL reason specifically.
        //
        // "no MoE drafter" is `MOE_SPECULATION_BLOCKER_MARKER`
        // (`crates/runtime/src/families/qwen/mod.rs`), the one place both Rust
        // blockers take that phrase from. Swift cannot import it, so this is
        // the single literal copy; keep it in step with that constant. Its
        // predecessor here was "dense-only", which went stale when the routed
        // pair got a batched kernel and no test could see it.
        XCTAssertTrue(
            reason.contains("no MoE drafter") || reason.contains("INT4-only"),
            "point TURBOSPARK_TEST_MODEL_NO_SPECULATION at a MoE or sub-4-bit "
                + "install; this one says: \(reason)")

        // The blocker's own claim, checked rather than quoted: "the model
        // decodes normally, only speculation is unavailable".
        var options = GenerateOptions()
        options.maxNewTokens = 40
        options.temperature = 0.0
        var result: GenerationResult?
        for try await event in session.generate(
            [ChatMessage(role: .user, content: "Name three coastal plants.")], options: options
        ) {
            if case .finished(let r) = event { result = r }
        }
        let r = try XCTUnwrap(result)
        XCTAssertGreaterThan(r.newTokens, 5, "an install that cannot speculate must still decode")
        print("blocked auto: off (\(reason)), still decoded \(r.newTokens) tokens")
    }

    /// **A NAMED BLOCK IT CANNOT SERVE FAILS THE OPEN, AND NAMES THE
    /// ARCHITECTURE RATHER THAN THE MISSING HEAD.**
    ///
    /// Two claims, and the second is a regression guard with a date on it.
    /// Until 2026-08-21 this reported "carries no multi-token-prediction head
    /// ... stream an install that adds the official checkpoint's last shard",
    /// which sent a caller after a 4.4 GB shard that cannot help: no published
    /// MoE conversion of this architecture ships a drafter this port can
    /// ingest, so no shard of any checkpoint changes the answer. `auto`
    /// got the same install right, which is why only a NAMED block can see
    /// it, and why the case above is not enough on its own.
    ///
    /// The failure being at OPEN is the point of the first claim: a caller
    /// who named a block is measuring, and a session that quietly did not
    /// speculate is the number that ends up in a table.
    func testANamedBlockIsRefusedForTheArchitectureRatherThanTheHead() async throws {
        let path = try blockedModelPath()
        var options = OpenOptions()
        options.speculation = .block(2)

        do {
            _ = try await TurboSparkSession(modelPath: path, options: options)
            XCTFail("a named block on an install that cannot serve it must not open quietly")
        } catch let error as TurboSparkError {
            XCTAssertEqual(error.code, .open)
            // Same single literal copy of `MOE_SPECULATION_BLOCKER_MARKER` as
            // the case above; see the note there.
            XCTAssertTrue(
                error.message.contains("no MoE drafter") || error.message.contains("INT4-only"),
                "the refusal must name the architectural obstacle, got: \(error.message)")
            XCTAssertFalse(
                error.message.contains("multi-token-prediction head"),
                "no checkpoint helps here, so the head must not be blamed: \(error.message)")
            print("blocked named: refused with \(error.message)")
        }
    }

    /// **THE SAME CLAIM WITH THE DRAFTER NAMED, which reached the bug by a
    /// second route until 2026-08-22.**
    ///
    /// The case above leaves the drafter at `auto`, which resolves to `mtp`,
    /// so it exercises the MTP arm alone. Naming `dflash` takes the other arm
    /// of the same `match`, and that one had no headless guard for a day: it
    /// reported "asks for the DFlash2 drafter, but this install carries none
    /// ... stream it beside the trunk", sending a caller after a checkpoint
    /// that does not exist for this half of the architecture, since the
    /// published DFlash2 drafter targets the DENSE half.
    ///
    /// Not a duplicate of the case above, for the reason the Rust siblings are
    /// not duplicates either: `drafter` selects which arm runs, so one arm's
    /// guard says nothing about the other's. What is shared below it is the
    /// `resolve_speculation` plumbing, and that is not what was wrong.
    func testANamedBlockNamingDflashIsAlsoRefusedForTheArchitecture() async throws {
        let path = try blockedModelPath()
        var options = OpenOptions()
        options.speculation = .block(2)
        options.speculativeDrafter = .dflash

        do {
            _ = try await TurboSparkSession(modelPath: path, options: options)
            XCTFail("a named block on an install that cannot serve it must not open quietly")
        } catch let error as TurboSparkError {
            XCTAssertEqual(error.code, .open)
            XCTAssertTrue(
                error.message.contains("no MoE drafter") || error.message.contains("INT4-only"),
                "the refusal must name the architectural obstacle, got: \(error.message)")
            // THE DISCRIMINATING HALF, and the one the marker check cannot
            // make: "stream it beside the trunk" is `DflashState::build`'s
            // open-time advice, correct on a dense install and unsatisfiable
            // on this one. Its absence is what says the headless arm ran.
            XCTAssertFalse(
                error.message.contains("stream it beside the trunk"),
                "no DFlash2 checkpoint of this architecture exists, so the "
                    + "refusal must not send a caller after one: \(error.message)")
            print("blocked named dflash: refused with \(error.message)")
        }
    }
}
