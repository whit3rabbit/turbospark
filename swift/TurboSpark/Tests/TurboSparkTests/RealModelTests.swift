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

    func testOpensAndDescribesItself() async throws {
        let session = try await TurboSparkSession(modelPath: try modelPath())
        // The RESOLVED values. Under automatic sizing nothing was asked for,
        // so these are the only numbers that exist.
        XCTAssertGreaterThan(session.info.maxContext, 0)
        XCTAssertGreaterThan(session.info.vocabSize, 0)
        XCTAssertFalse(session.info.family.isEmpty)
        print(
            "open: family=\(session.info.family) context=\(session.info.maxContext) "
                + "slots=\(session.info.expertCacheSlots) vocab=\(session.info.vocabSize)")
    }

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
}
