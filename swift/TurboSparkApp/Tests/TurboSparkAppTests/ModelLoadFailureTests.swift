import TurboSpark
import XCTest
@testable import TurboSparkApp

/// The load-failure strip's classifier (`ModelLoadFailure.Kind`). The engine
/// message is free text, so each specific kind is pinned to the exact sentence
/// the Rust side writes (`crates/model-io/src/error.rs`,
/// `crates/model-io/src/context_policy.rs`). Anything else, including vague
/// text that merely mentions memory or files, must stay `.other`: a wrong
/// specific cause sends the user to the wrong fix.
final class ModelLoadFailureTests: XCTestCase {
    private func kind(_ message: String) -> ModelLoadFailure.Kind {
        ModelLoadFailure.Kind.classify(engineMessage: message)
    }

    func testMemoryRefusalsFromTheContextPolicy() {
        // ContextTooLarge, ContextOverCap and ContextFloorUnmet.
        XCTAssertEqual(kind(
            "--max-context 131072 needs 16.0 GiB of KV cache; 4.0 GiB available "
                + "(36.0 GiB physical - 28.0 GiB weights and expert cache - 4.0 GiB reserve)."),
            .notEnoughMemory)
        XCTAssertEqual(kind(
            "--load-guard custom would allocate 20.0 GiB at --max-context 8192 "
                + "(2.0 GiB slot cache + 1.0 GiB KV), over the 16.0 GiB ceiling"),
            .notEnoughMemory)
        XCTAssertEqual(kind(
            "--min-auto-context 32768 was not met: `auto` resolved 8192 "
                + "(largest this machine affords: 8192). Try a smaller model."),
            .notEnoughMemory)
    }

    func testMissingFiles() {
        XCTAssertEqual(kind("model.gturbo directory at /x is missing manifest.json"), .missingFiles)
        XCTAssertEqual(kind("model.gturbo is missing required file model_weights.bin"), .missingFiles)
    }

    func testDamagedInstall() {
        XCTAssertEqual(kind(
            "SHA-256 of model_weights.bin does not match manifest.files[model_weights.bin].sha256"),
            .damaged)
        XCTAssertEqual(kind("resident index is corrupt: manifest.json: expected value at line 1"), .damaged)
        XCTAssertEqual(kind("no IndexEntry named blk.0.attn_q in model_weights.bin"), .damaged)
    }

    func testUnsupportedFormat() {
        // IndexCorrupt wraps the unknown family, so the family check must win
        // over the generic "corrupt" one.
        XCTAssertEqual(kind("resident index is corrupt: unknown arch.family \"mamba9\""), .unsupported)
        XCTAssertEqual(kind("model capability \"vision\" is not supported by the text-model loader"), .unsupported)
        XCTAssertEqual(kind("manifest version 9.0 is not supported (this build reads 1.0 through 1.4)"), .unsupported)
        XCTAssertEqual(kind(
            "model_weights.bin index format 3 is not supported (this build reads format 2); "
                + "reinstall the model or upgrade turbospark"),
            .unsupported)
        XCTAssertEqual(kind("manifest.json magic does not equal \"GTURBO\""), .unsupported)
    }

    func testVagueMessagesStayOther() {
        let vague = [
            "",
            "out of memory",
            "not enough memory",
            "file not found",
            "No such file or directory (os error 2)",
            "corrupt",
            "unsupported",
            "manifest.arch.hidden_size = 4096; expected 2048",
            "open failed: Metal device lost",
            "a panic was caught inside the library",
        ]
        for message in vague {
            XCTAssertEqual(kind(message), .other, message)
        }
    }

    func testGateRefusalsMapWithoutGuessing() {
        XCTAssertEqual(ModelLoadFailure.Kind(gate: .notConnected(libraryName: "Archive")),
                       .notConnected(libraryName: "Archive"))
        XCTAssertEqual(ModelLoadFailure.Kind(gate: .missing), .missingFiles)
        XCTAssertNil(ModelLoadFailure.Kind(gate: .open))
        XCTAssertNil(ModelLoadFailure.Kind(gate: .warn(.streamsFromDrive)))
    }

    func testActionsPutTheUsefulFixFirst() {
        // Retry only where a retry can change the answer.
        XCTAssertEqual(ModelLoadFailure.Kind.notConnected(libraryName: nil).actions, [.retry, .chooseModel])
        XCTAssertEqual(ModelLoadFailure.Kind.notEnoughMemory.actions, [.chooseModel, .retry, .engineSettings])
        XCTAssertEqual(ModelLoadFailure.Kind.other.actions, [.retry, .chooseModel])
        for kind: ModelLoadFailure.Kind in [.missingFiles, .damaged, .unsupported] {
            XCTAssertEqual(kind.actions, [.chooseModel], "\(kind)")
        }
    }

    func testOtherShowsTheEngineTextAndSpecificKindsDoNot() {
        let model = InstalledModel(alias: "m", repo: "", path: "/m", family: "")
        let other = ModelLoadFailure(model: model, kind: .other, detail: "open failed: Metal device lost")
        XCTAssertEqual(other.cause(language: .english), "open failed: Metal device lost")
        let memory = ModelLoadFailure(model: model, kind: .notEnoughMemory, detail: "--max-context 1 needs")
        XCTAssertNotEqual(memory.cause(language: .english), memory.detail)
        XCTAssertFalse(memory.cause(language: .english).isEmpty)
    }

    /// The strip and `model.error` carry the same failure; the banner must
    /// step aside for exactly that text and no other.
    func testTheStripCoversOnlyTheErrorItWasRaisedWith() {
        let model = InstalledModel(alias: "m", repo: "", path: "/m", family: "")
        let failure = ModelLoadFailure(
            model: model, kind: .other, detail: "boom", errorText: "Failed to load model: boom")
        XCTAssertTrue(failure.covers(error: "Failed to load model: boom"))
        XCTAssertFalse(failure.covers(error: "Some later, unrelated error"))
        XCTAssertFalse(failure.covers(error: nil))
        let refusal = ModelLoadFailure(model: model, kind: .missingFiles, detail: "")
        XCTAssertFalse(refusal.covers(error: ""), "A gate refusal never set `error`.")
    }
}
