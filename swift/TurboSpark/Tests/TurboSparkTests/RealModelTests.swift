import Foundation
import XCTest

@testable import TurboSpark

/// Shared base test case for end-to-end real model tests.
///
/// Resolves test model, blocked speculation model, and vision image paths
/// from environment variables.
class RealModelTestCase: XCTestCase {

    /// Resolves the test model path from the environment or skips if unset.
    func modelPath() throws -> String {
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
    func blockedModelPath() throws -> String {
        guard let raw = ProcessInfo.processInfo.environment["TURBOSPARK_TEST_MODEL_NO_SPECULATION"],
            !raw.isEmpty
        else {
            throw XCTSkip(
                "set TURBOSPARK_TEST_MODEL_NO_SPECULATION to a MoE or sub-4-bit install "
                    + "to run this")
        }
        return raw
    }

    /// A page to send. A THIRD variable, for Gotcha 11's reason: a vision
    /// install is a shape `TURBOSPARK_TEST_MODEL` alone cannot guarantee, and
    /// one variable can only ever gate one shape.
    ///
    ///     TURBOSPARK_TEST_MODEL=~/models/qwen38-27b-vision.gturbo \
    ///     TURBOSPARK_TEST_IMAGE=~/models/vision-probe-qwen38/imgs/oracle/medium.png \
    ///       swift test
    func testImagePath() throws -> String {
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
}

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
final class RealModelTests: RealModelTestCase {

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
        var stopEvent: String?

        for try await event in session.generate(
            [ChatMessage(role: .user, content: "Explain how coastal wetlands reduce flood damage.")],
            options: options
        ) {
            switch event {
            case .prefill: sawPrefill = true
            case .content(let c): streamed += c
            case .reasoning: break
            case .stopped(let reason, _, _):
                stopEvent = reason
            case .toolCall:
                break
            case .finished(let r): result = r
            }
        }

        let r = try XCTUnwrap(result)
        XCTAssertTrue(sawPrefill, "prefill progress should reach the caller")
        // The terminal stream event must agree with the result it precedes.
        XCTAssertEqual(stopEvent, r.stopReason.rawValue)
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

    /// A second turn on the same session continues from the first turn's KV
    /// instead of re-prefilling the whole transcript from scratch.
    ///
    /// This is what `open.rs`'s `runner.set_prefix_reuse(true)` is FOR --
    /// `crates/cli/src/chat.rs`'s `--chat` REPL is the only other caller
    /// that opts in, and that file's own header records the failure mode
    /// this test exists to catch: the feature read 0/33 through two rounds
    /// of an implementation that looked correct everywhere else. Asserting
    /// only that `reusedPrefixTokens >= 0` (its type's own floor) would
    /// never fail on that bug.
    ///
    /// **The threshold is a MAJORITY of turn 1's prompt, not all of it.**
    /// Measured here: 14 of 18 reused (78%) on a two-token question; the
    /// real `--chat` REPL's own recorded numbers are 13/33 and 29/49 (39%
    /// and 59%) -- so a shortfall below the full prompt is the normal case
    /// on this engine rather than a partial-failure signal, and asserting
    /// full recovery would make this test flaky on the real thing it is
    /// meant to verify. What would still fail it: reuse silently degrading
    /// to a handful of coincidentally shared BOS/system tokens.
    func testASecondTurnReusesThePreviousTurnsKV() async throws {
        let session = try await TurboSparkSession(modelPath: try modelPath())
        var options = GenerateOptions()
        options.maxNewTokens = 40
        options.seed = 20260721

        let turn1Messages = [ChatMessage(role: .user, content: "Name one primary color.")]
        var turn1Result: GenerationResult?
        for try await event in session.generate(turn1Messages, options: options) {
            if case .finished(let r) = event { turn1Result = r }
        }
        let r1 = try XCTUnwrap(turn1Result)
        XCTAssertEqual(
            r1.reusedPrefixTokens, 0,
            "a session's first turn has nothing to reuse yet")

        // The transcript a real chat client sends: turn 1's exchange
        // followed by a new question. Re-rendering this is what gives the
        // longest-common-prefix match something to find.
        let turn2Messages =
            turn1Messages + [
                ChatMessage(role: .assistant, content: r1.content),
                ChatMessage(role: .user, content: "Now name a different one."),
            ]
        var turn2Result: GenerationResult?
        for try await event in session.generate(turn2Messages, options: options) {
            if case .finished(let r) = event { turn2Result = r }
        }
        let r2 = try XCTUnwrap(turn2Result)

        // Not just "> 0": a session whose reuse silently degrades to a
        // handful of coincidentally shared BOS/system tokens would still
        // clear that bar. A majority of turn 1's own prompt is the bound;
        // see the doc comment above for why not all of it.
        XCTAssertGreaterThanOrEqual(
            r2.reusedPrefixTokens, (r1.promptTokens + 1) / 2,
            "expected turn 2 to continue from a majority of turn 1's prompt, "
                + "got \(r2.reusedPrefixTokens) reused of \(r2.promptTokens) "
                + "(turn 1 prompt was \(r1.promptTokens))")
        print(
            "prefix-reuse: turn2 reused \(r2.reusedPrefixTokens)/\(r2.promptTokens) prompt tokens "
                + "(turn1 prompt was \(r1.promptTokens))")
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
}
