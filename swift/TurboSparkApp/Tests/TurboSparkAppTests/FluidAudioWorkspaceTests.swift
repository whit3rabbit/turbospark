import TurboSpark
import XCTest
@testable import TurboSparkApp

/// The workspace side of the FluidAudio speech rows: how a window's result is
/// saved, which installs residency accepts, and which pages and jobs reach the
/// rows. No model runs here (FluidAudioASRRealModelTests runs them).
@MainActor
final class FluidAudioWorkspaceTests: XCTestCase {
    private func token(_ text: String, _ start: Double, _ end: Double) -> FluidAudioTranscript.Token {
        FluidAudioTranscript.Token(text: text, start: start, end: end)
    }

    /// Two overlapping windows hear the same words. Each keeps only the words
    /// that start in its own part of the recording, so the saved text has
    /// every word once.
    func testOverlappingWindowsKeepEachWordOnce() {
        let windows = AudioTranscriptionPlan.windows(duration: 50, length: 30, overlap: 2, skipsCoveredTail: true)
        XCTAssertEqual(windows.count, 2)
        // Recording times 28.5 and 29.2 lie in the overlap [28, 30]; the cut is 29.
        let first = FluidAudioTranscript(text: "", tokens: [token("A", 27.0, 27.4), token("B", 28.5, 28.9), token("C", 29.2, 29.6)])
        let second = FluidAudioTranscript(text: "", tokens: [token("B", 0.5, 0.9), token("C", 1.2, 1.6), token("D", 3.0, 3.4)])
        let saved = [first.transcript(in: windows[0], language: "ja"), second.transcript(in: windows[1], language: "ja")]
        XCTAssertEqual(saved.map(\.text).joined(), "ABCD")
        XCTAssertEqual(saved[0].text, "AB")
        XCTAssertEqual(saved[1].segments.first?.startSeconds, 1.2, "times stay relative to the window")
    }

    /// Token results become one segment per sentence with measured timing.
    func testTokensBecomeSentencesWithTokenTiming() {
        let window = AudioTranscriptionWindow(start: 0, duration: 10)
        let tokens = [token("Kyo", 0.1, 0.3), token("wa", 0.3, 0.5), token("\u{3002}", 0.5, 0.6),
                      token(" Ashita", 1.0, 1.4), token("?", 1.4, 1.5), token("mo", 2.0, 2.2)]
        let transcript = FluidAudioTranscript(text: "", tokens: tokens).transcript(in: window, language: "ja")
        XCTAssertEqual(transcript.segments.map(\.text), ["Kyowa\u{3002}", "Ashita?", "mo"])
        XCTAssertEqual(transcript.segments.map(\.timing), ["token", "token", "token"])
        XCTAssertEqual(transcript.segments[1].startSeconds, 1.0)
        XCTAssertEqual(transcript.segments[1].endSeconds, 1.5)
        XCTAssertEqual(transcript.language, "ja")
    }

    /// Without token times the text spans the window, marked as an extent the
    /// editor labels approximate. Silence saves nothing.
    func testTextOnlyResultsSpanTheirWindow() {
        let window = AudioTranscriptionWindow(start: 15, duration: 14.99)
        let transcript = FluidAudioTranscript(text: " ni hao ", tokens: []).transcript(in: window, language: "zh")
        XCTAssertEqual(transcript.segments.count, 1)
        XCTAssertEqual(transcript.segments[0].text, "ni hao")
        XCTAssertEqual(transcript.segments[0].timing, "window")
        XCTAssertEqual(transcript.segments[0].startSeconds, 0)
        XCTAssertEqual(transcript.segments[0].endSeconds, 14.99)
        XCTAssertEqual(FluidAudioTranscript(text: "  ", tokens: []).transcript(in: window, language: "zh").segments, [])
    }

    private func makeController(_ root: URL) throws -> AudioWorkspaceController {
        let vault = ProfileVaultStore(rootProvider: { root }, profileIDProvider: { "audio-fluid" }, migrateLegacyData: false)
        _ = try vault.prepareForLaunch()
        return AudioWorkspaceController(
            library: AudioLibraryStore(repository: ProfileRepository(store: vault)), assets: ManagedAssetStore(vault: vault))
    }

    private func catalogRow(_ alias: String) throws -> AudioProfile {
        try XCTUnwrap(AudioCatalog.profiles().first { $0.identity.alias == alias }, alias)
    }

    func testTranscribePageOffersBothSpeechRows() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let controller = try makeController(root)
        defer { controller.shutdown() }
        controller.profiles = try AudioCatalog.profiles()
        controller.selectPage(.transcribe)
        let offered = Set(controller.availableProfiles.map(\.identity.alias))
        XCTAssertTrue(offered.isSuperset(of: ["parakeet-ja-coreml", "paraformer-zh-coreml"]), "\(offered)")
    }

    /// A folder the user picked never reaches FluidAudio: the picker offers
    /// none and residency refuses one, so only verified downloads load.
    func testResidencyLoadsOnlyAVerifiedDownload() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let controller = try makeController(root)
        defer { controller.shutdown() }
        let profile = try catalogRow("parakeet-ja-coreml")
        XCTAssertFalse(AudioRuntimeProvider(profile).acceptsLocalFolder)
        XCTAssertTrue(AudioRuntimeProvider(try catalogRow("whisper-base")).acceptsLocalFolder)
        controller.preferences.localModels[profile.identity.alias] = Data([1, 2, 3])
        do {
            _ = try await controller.residentFluidTranscriber(for: profile, family: .parakeetJa)
            XCTFail("a model with no verified download loaded")
        } catch {
            XCTAssertEqual(error.localizedDescription, AppStrings.localized("Download this model first."))
        }
        XCTAssertNil(controller.loadedModelAlias)
    }

    /// The load is admitted against the audio memory budget before any byte
    /// is verified.
    func testResidencyRefusesAModelOverTheMemoryBudget() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let controller = try makeController(root)
        defer { controller.shutdown() }
        let row = try catalogRow("paraformer-zh-coreml")
        let oversized = AudioProfile(identity: row.identity, family: row.family, displayName: row.displayName,
                                     capabilities: row.capabilities, pcmFormat: row.pcmFormat,
                                     residentMemoryBytes: .max, readiness: row.readiness)
        controller.installedPaths[row.identity.alias] = root.path
        do {
            _ = try await controller.residentFluidTranscriber(for: oversized, family: .paraformerZh)
            XCTFail("an oversized model was admitted")
        } catch {
            XCTAssertEqual(error.localizedDescription, AppStrings.localized("This checkpoint exceeds the available audio model memory budget."))
        }
        XCTAssertNil(controller.opening, "nothing was verified")
    }

    /// A finished recording is refined with a FluidAudio row even though the
    /// row's Rust `canRun` is false by design.
    func testRefinementStartsForAFluidAudioRow() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let controller = try makeController(root)
        defer { controller.shutdown() }
        let profile = try catalogRow("paraformer-zh-coreml")
        XCTAssertFalse(profile.capabilities.canRun, "the Rust verdict alone would skip this row")
        controller.profiles = [profile]
        controller.installedPaths[profile.identity.alias] = root.appendingPathComponent("not-installed").path
        var item = AudioLibraryItem(title: "Take", kind: .recording)
        item.recipe.task = AudioTask.speechToText.rawValue
        item.recipe.modelID = profile.identity.alias
        let asset = try controller.assets.store(data: Data(repeating: 0, count: 64), fileName: "take.wav", mimeType: "audio/wav")
        item.clips = [AudioLibraryClip(assetReference: asset.storedReference, sampleRate: 16_000, channels: 1, frameCount: 16_000)]
        try controller.save(item)
        controller.refineRecording(item.id)
        XCTAssertTrue(controller.isBusy, "refinement did not start")
        controller.cancel()
    }
}
