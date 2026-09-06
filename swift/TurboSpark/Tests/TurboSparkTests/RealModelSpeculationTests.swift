import Foundation
import XCTest

@testable import TurboSpark

/// End-to-end speculation tests against a real install.
final class RealModelSpeculationTests: RealModelTestCase {

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
