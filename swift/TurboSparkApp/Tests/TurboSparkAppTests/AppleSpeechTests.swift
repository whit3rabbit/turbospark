import AVFoundation
import Foundation
import TurboSpark
import XCTest

@testable import TurboSparkApp

/// The macOS built-in recognizer as a zero-download Speech-to-Text engine:
/// it is offered only where `SpeechAnalyzer` exists, routes to its own
/// provider, counts as installed, and loses to any installed catalog model.
@MainActor
final class AppleSpeechTests: XCTestCase {
    private var vault: ProfileVaultStore?
    private var root: URL?

    override func tearDown() {
        if let root { try? FileManager.default.removeItem(at: root) }
        super.tearDown()
    }

    private func controller() throws -> AudioWorkspaceController {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        self.root = root
        let vault = ProfileVaultStore(rootProvider: { root }, profileIDProvider: { "apple-speech" }, migrateLegacyData: false)
        _ = try vault.prepareForLaunch()
        self.vault = vault
        let controller = AudioWorkspaceController(library: AudioLibraryStore(repository: ProfileRepository(store: vault)),
                                                  assets: ManagedAssetStore(vault: vault))
        addTeardownBlock { controller.shutdown() }
        return controller
    }

    private var apple: AudioProfile { AppleSpeech.profile(displayName: "Apple Speech") }

    private func catalogProfile(_ alias: String) -> AudioProfile {
        AudioProfile(
            identity: AudioProfileIdentity(task: .speechToText, alias: alias, repository: "o/\(alias)", revision: "abcdef0123", assetFingerprint: "f"),
            family: alias.hasPrefix("whisper") ? "whisper" : "parakeet_tdt", displayName: alias,
            capabilities: AudioCapabilities(operations: ["transcribe"], backend: "coreml", cancellation: "window", canRun: true),
            pcmFormat: AudioPCMFormat(sampleRate: 16_000, channels: 1), readiness: "implemented_unqualified")
    }

    // MARK: Availability

    /// SFSpeechRecognizer is never a fallback (it can send audio to Apple),
    /// so before macOS 26 the engine is off even if the framework answers.
    func testSupportNeedsMacOS26AndTheFramework() {
        XCTAssertFalse(AppleSpeech.isSupported(osMajor: 25) { true })
        XCTAssertFalse(AppleSpeech.isSupported(osMajor: 26) { false })
        XCTAssertTrue(AppleSpeech.isSupported(osMajor: 26) { true })
        XCTAssertTrue(AppleSpeech.isSupported(osMajor: 27) { true })
    }

    /// Older Macs get no row at all, so their pickers stay exactly the catalog.
    func testBuiltInRowsFollowSupport() {
        let rows = AudioWorkspaceController.builtInAudioProfiles
        XCTAssertEqual(rows.map(\.identity.alias), AppleSpeech.isSupportedOnThisMac ? [AppleSpeech.alias] : [])
    }

    // MARK: Routing

    func testProfileRoutesToItsOwnProviderAndTakesNoFolder() {
        let provider = AudioRuntimeProvider(apple)
        XCTAssertEqual(provider, .appleSpeech)
        XCTAssertFalse(provider.acceptsLocalFolder)
        XCTAssertEqual(provider.canRun(apple), AppleSpeech.isSupportedOnThisMac)
        // The Rust worker cannot run it, so Rust-only paths (live drafts) skip it.
        XCTAssertFalse(apple.capabilities.canRun)
        XCTAssertTrue(apple.identity.repository.isEmpty, "nothing to download")
    }

    func testAppleBackendWithAnotherFamilyOrTaskIsUnsupported() {
        let wrongFamily = AudioProfile(
            identity: apple.identity, family: "whisper", displayName: "x", capabilities: apple.capabilities,
            pcmFormat: apple.pcmFormat, readiness: apple.readiness)
        XCTAssertEqual(AudioRuntimeProvider(wrongFamily), .unsupported)
        let wrongTask = AudioProfile(
            identity: AudioProfileIdentity(task: .textToSpeech, alias: AppleSpeech.alias, repository: "", revision: "", assetFingerprint: ""),
            family: AppleSpeech.family, displayName: "x", capabilities: apple.capabilities,
            pcmFormat: apple.pcmFormat, readiness: apple.readiness)
        XCTAssertEqual(AudioRuntimeProvider(wrongTask), .unsupported)
    }

    /// It transcribes a whole take but has no streaming adapter yet, so it
    /// is a final-pass choice only.
    func testOfferedForFinalTranscriptOnly() {
        XCTAssertTrue(SpeechToTextJob.final.accepts(apple))
        XCTAssertFalse(SpeechToTextJob.live.accepts(apple))
        XCTAssertFalse(SpeechToTextJob.speakers.accepts(apple))
    }

    // MARK: Installed and ranking

    func testCountsAsInstalledWithNoDownload() throws {
        let controller = try controller()
        XCTAssertTrue(controller.isInstalled(apple))
        XCTAssertEqual(controller.installStatus(for: apple), .installed)
        XCTAssertFalse(controller.canLoad(apple), "a system process: nothing to load")
        XCTAssertFalse(controller.isInstalled(catalogProfile("parakeet-ultra-coreml")))
    }

    /// The zero-download default: with nothing installed the final pass uses
    /// the built-in engine, and an installed catalog model still wins.
    func testDefaultsToBuiltInUntilACatalogModelIsInstalled() throws {
        try XCTSkipUnless(AppleSpeech.isSupportedOnThisMac, "needs SpeechAnalyzer")
        let controller = try controller()
        let profiles = ["whisper-base", "parakeet-redux-coreml", "parakeet-ultra-coreml"].map(catalogProfile) + [apple]
        func final() -> String? {
            SpeechToTextModels.resolve(.notes, .final, profiles: profiles, selected: [:],
                                       isInstalled: { controller.isInstalled($0) })?.identity.alias
        }
        XCTAssertEqual(final(), AppleSpeech.alias)
        controller.installedPaths = ["whisper-base": "/models/whisper-base"]
        XCTAssertEqual(final(), "whisper-base")
        controller.installedPaths["parakeet-ultra-coreml"] = "/models/ultra"
        XCTAssertEqual(final(), "parakeet-ultra-coreml")
    }

    /// A user's explicit choice of the built-in engine survives installing
    /// a catalog model.
    func testExplicitChoiceOfBuiltInIsKept() throws {
        try XCTSkipUnless(AppleSpeech.isSupportedOnThisMac, "needs SpeechAnalyzer")
        let controller = try controller()
        controller.installedPaths = ["parakeet-ultra-coreml": "/models/ultra"]
        let chosen = SpeechToTextModels.resolve(
            .meeting, .final, profiles: [catalogProfile("parakeet-ultra-coreml"), apple],
            selected: [SpeechToTextModels.key(.meeting, .final): AppleSpeech.alias], isInstalled: { controller.isInstalled($0) })
        XCTAssertEqual(chosen?.identity.alias, AppleSpeech.alias)
    }

    // MARK: Result mapping

    func testWindowKeepsOnlyThePiecesItOwns() {
        var window = AudioTranscriptionWindow(start: 58, duration: 60)
        window.ownedFrom = 59; window.ownedUntil = 117
        let pieces = [
            AppleSpeech.Piece(text: "Heard by the previous window.", start: 0.2, end: 0.9, words: []),
            AppleSpeech.Piece(text: " Hello there. ", start: 1.5, end: 2.5, words: [
                AudioWord(text: "Hello", startSeconds: 1.5, endSeconds: 1.9),
                AudioWord(text: "there.", startSeconds: 2.0, endSeconds: 2.5),
            ]),
            AppleSpeech.Piece(text: "   ", start: 3, end: 4, words: []),
            AppleSpeech.Piece(text: "No word times.", start: 10, end: 12, words: []),
            AppleSpeech.Piece(text: "Belongs to the next window.", start: 59.5, end: 60, words: []),
        ]
        let transcript = AppleSpeech.transcript(pieces, in: window, language: "en")
        XCTAssertEqual(transcript.segments.map(\.text), ["Hello there.", "No word times."])
        XCTAssertEqual(transcript.segments.map(\.timing), ["word", "segment"])
        XCTAssertEqual(transcript.segments[0].words?.count, 2)
        XCTAssertNil(transcript.segments[1].words)
        XCTAssertEqual(transcript.segments[0].startSeconds, 1.5, "times stay window-relative")
        XCTAssertEqual(transcript.text, "Hello there. No word times.")
        XCTAssertEqual(transcript.language, "en")
    }

    // MARK: Real recognizer

    /// Runs the real recognizer on a `say` clip. Opt in, because a first run
    /// makes macOS download its language model.
    func testRealRecognizerTranscribesSpeech() async throws {
        guard ProcessInfo.processInfo.environment["TURBOSPARK_APPLE_SPEECH_REAL"] == "1" else {
            throw XCTSkip("set TURBOSPARK_APPLE_SPEECH_REAL=1 to run the macOS recognizer")
        }
        guard #available(macOS 26, *), AppleSpeech.isSupportedOnThisMac else { throw XCTSkip("needs SpeechAnalyzer") }
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let clip = dir.appendingPathComponent("clip.wav")
        let say = Process()
        say.executableURL = URL(fileURLWithPath: "/usr/bin/say")
        say.arguments = ["-o", clip.path, "--file-format=WAVE", "--data-format=LEF32@16000",
                         "The quick brown fox jumps over the lazy dog."]
        try say.run(); say.waitUntilExit()
        let file = try AVAudioFile(forReading: clip)
        let buffer = try XCTUnwrap(AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length)))
        try file.read(into: buffer)
        let pcm = Array(UnsafeBufferPointer(start: buffer.floatChannelData![0], count: Int(buffer.frameLength)))

        let locale = try await AppleSpeechTranscriber.locale(preferred: Locale(identifier: "en-US"))
        if await AppleSpeechTranscriber.needsAssets(locale) { try await AppleSpeechTranscriber.installAssets(locale) { _ in } }
        let pieces = try await AppleSpeechTranscriber.transcribe(pcm, locale: locale)
        let text = pieces.map(\.text).joined(separator: " ").lowercased()
        for word in ["quick", "brown", "fox", "lazy", "dog"] { XCTAssertTrue(text.contains(word), "missing \(word) in \(text)") }
        let words = pieces.flatMap(\.words)
        XCTAssertFalse(words.isEmpty, "audioTimeRange gives word times")
        XCTAssertTrue(words.allSatisfy { $0.startSeconds >= 0 && $0.endSeconds >= $0.startSeconds && $0.endSeconds <= Double(pcm.count) / 16_000 + 0.5 })
    }
}
