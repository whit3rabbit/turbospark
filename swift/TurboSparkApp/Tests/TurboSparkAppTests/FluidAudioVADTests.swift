import XCTest
@testable import TurboSparkApp

/// The speech-detector entry point without a model: routing and refusal.
final class FluidAudioVADTests: XCTestCase {
    private func emptyDirectory() throws -> URL {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("vad-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    /// Only the Silero family builds a detector. The recognizers and
    /// synthesizers are refused before any file is read.
    func testOnlyTheVadFamilyBuildsADetector() async throws {
        let dir = try emptyDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        for family in FluidAudioFamily.allCases where family != .sileroVAD {
            do {
                _ = try await FluidAudioEngine.shared.speechDetector(for: family, installRoot: dir)
                XCTFail("\(family.rawValue) built a speech detector")
            } catch {
                XCTAssertEqual(error as? FluidAudioModelError, .notDetector(family))
            }
        }
        // The reverse: the VAD family is not a recognizer.
        do {
            _ = try await FluidAudioEngine.shared.transcriber(for: .sileroVAD, installRoot: dir)
            XCTFail("the VAD family built a transcriber")
        } catch {
            XCTAssertEqual(error as? FluidAudioModelError, .notSpeech(.sileroVAD))
        }
    }

    /// A folder without the model is refused, and nothing is fetched or
    /// created on the way: FluidAudio's own cache folders stay absent.
    func testAMissingModelIsRefusedWithoutAFetch() async throws {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let caches = [home.appendingPathComponent(".cache/fluidaudio"),
                      home.appendingPathComponent("Library/Application Support/FluidAudio")]
        let absent = caches.filter { !FileManager.default.fileExists(atPath: $0.path) }
        let dir = try emptyDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let started = Date()
        do {
            _ = try await FluidAudioEngine.shared.speechDetector(for: .sileroVAD, installRoot: dir)
            XCTFail("an empty folder loaded")
        } catch {}
        XCTAssertLessThan(Date().timeIntervalSince(started), 5)
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: dir.path), [])
        for cache in absent { XCTAssertFalse(FileManager.default.fileExists(atPath: cache.path), cache.path) }
    }

    /// The signature names the file the catalog row ships, so a pin bump that
    /// renames the export fails here and not on a user's first load.
    func testTheSignatureNamesAFileInTheRecordedExport() {
        let signature = FluidAudioSignatures.SileroVad.model
        XCTAssertEqual(signature.file, "silero-vad-unified-256ms-v6.2.1.mlmodelc")
        XCTAssertEqual(signature.inputs.map(\.name), ["audio_input", "hidden_state", "cell_state"])
        XCTAssertEqual(signature.outputs.map(\.name), ["vad_output", "new_hidden_state", "new_cell_state"])
    }
}
