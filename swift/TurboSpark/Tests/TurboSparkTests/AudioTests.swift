import AVFoundation
import XCTest
@testable import TurboSpark

final class AudioTests: XCTestCase {
    func testNativeErrorsRemainActionableInUserFacingDescriptions() {
        let message = "audio device is busy; retry opening the model after the current job finishes"
        XCTAssertEqual(TurboSparkError(code: .open, message: message).localizedDescription, message)
    }
    func testCatalogSeparatesReadinessAndCallableCapabilities() throws {
        let profiles = try AudioCatalog.profiles()
        let whisper = try XCTUnwrap(profiles.first { $0.family == "whisper" })
        XCTAssertTrue(whisper.capabilities.canRun)
        XCTAssertEqual(whisper.capabilities.timing, "window")
        let kokoro = try XCTUnwrap(profiles.first { $0.family == "kokoro" })
        XCTAssertFalse(kokoro.capabilities.canRun)
        XCTAssertNotNil(kokoro.capabilities.unavailableReason)
        let families = try AudioCatalog.capabilities()
        XCTAssertEqual(Set(families.map(\.id)).count, families.count)
        XCTAssertTrue(families.contains { $0.family == "sts/mel_roformer" && !$0.canRun })
        XCTAssertTrue(families.filter(\.component).allSatisfy { !$0.canRun })
    }
    func testCloneRequestFieldsCrossTheWireInSnakeCase() throws {
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        func wire(_ request: AudioRequest) throws -> [String: Any] {
            try XCTUnwrap(JSONSerialization.jsonObject(with: encoder.encode(request)) as? [String: Any])
        }
        var request = AudioRequest(task: .textToSpeech)
        request.pW = 1.5
        request.tW = 3
        request.referenceSampleRate = 24_000
        request.consent = true
        let json = try wire(request)
        XCTAssertEqual((json["p_w"] as? NSNumber)?.floatValue, 1.5)
        XCTAssertEqual((json["t_w"] as? NSNumber)?.floatValue, 3)
        XCTAssertEqual(json["reference_sample_rate"] as? Int, 24_000)
        XCTAssertEqual(json["consent"] as? Bool, true)
        // Unset options are omitted so the runtime applies its own defaults.
        let plain = try wire(AudioRequest(task: .textToSpeech))
        XCTAssertNil(plain["p_w"])
        XCTAssertNil(plain["t_w"])
        XCTAssertNil(plain["reference_sample_rate"])
        XCTAssertEqual(plain["consent"] as? Bool, false)
    }
    func testVoiceCloneCapabilityAndAcknowledgmentsDecode() throws {
        let clone = AudioCapabilities(operations: ["synthesize", "voice_clone"], backend: "portable_cpu",
                                      cancellation: "diffusion_step", canRun: true)
        XCTAssertTrue(clone.supportsVoiceClone)
        XCTAssertFalse(AudioCapabilities(operations: ["synthesize"], backend: "metal",
                                         cancellation: "sentence", canRun: true).supportsVoiceClone)
        let family = try XCTUnwrap(AudioCatalog.capabilities().first { $0.family == "tts/megatts3" })
        XCTAssertTrue(family.canRun)
        XCTAssertEqual(family.operations, ["synthesize", "voice_clone"])
        // The native catalog always sends the list; an older payload without it still decodes.
        XCTAssertTrue(try AudioCatalog.profiles().allSatisfy { $0.acknowledgments != nil })
        let profile = AudioProfile(
            identity: AudioProfileIdentity(task: .textToSpeech, alias: "x", repository: "", revision: "", assetFingerprint: ""),
            family: "megatts3", displayName: "X", capabilities: clone,
            pcmFormat: AudioPCMFormat(sampleRate: 24_000, channels: 1), readiness: "implemented_unqualified",
            acknowledgments: [AudioAcknowledgment(id: "terms", summary: "Accept the terms.")])
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        var object = try XCTUnwrap(JSONSerialization.jsonObject(with: encoder.encode(profile)) as? [String: Any])
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let round = try decoder.decode(AudioProfile.self, from: JSONSerialization.data(withJSONObject: object))
        XCTAssertEqual(round.requiredAcknowledgments.map(\.id), ["terms"])
        object.removeValue(forKey: "acknowledgments")
        let old = try decoder.decode(AudioProfile.self, from: JSONSerialization.data(withJSONObject: object))
        XCTAssertEqual(old.requiredAcknowledgments, [])
    }
    /// The catalog sends placement, the app-only flag and a fixed layout under
    /// these snake_case keys. Payloads from before them decode to nil.
    func testPlacementAppOnlyAndLayoutFieldsDecodeFromTheCatalogKeys() throws {
        let capabilities = AudioCapabilities(operations: ["synthesize"], backend: "swift_coreml",
                                             cancellation: "chunk", canRun: false,
                                             compute: "gpu", httpApi: false)
        let profile = AudioProfile(
            identity: AudioProfileIdentity(task: .textToSpeech, alias: "x", repository: "", revision: "", assetFingerprint: ""),
            family: "fluidaudio_pocket_tts", displayName: "X", capabilities: capabilities,
            pcmFormat: AudioPCMFormat(sampleRate: 24_000, channels: 1), readiness: "runtime_pending",
            installSubdir: "Models/pocket-tts")
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        var object = try XCTUnwrap(JSONSerialization.jsonObject(with: encoder.encode(profile)) as? [String: Any])
        var wireCapabilities = try XCTUnwrap(object["capabilities"] as? [String: Any])
        XCTAssertEqual(wireCapabilities["compute"] as? String, "gpu")
        XCTAssertEqual(wireCapabilities["http_api"] as? Bool, false)
        XCTAssertEqual(object["install_subdir"] as? String, "Models/pocket-tts")
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let round = try decoder.decode(AudioProfile.self, from: JSONSerialization.data(withJSONObject: object))
        XCTAssertEqual(round, profile)
        wireCapabilities.removeValue(forKey: "compute")
        wireCapabilities.removeValue(forKey: "http_api")
        object["capabilities"] = wireCapabilities
        object.removeValue(forKey: "install_subdir")
        let old = try decoder.decode(AudioProfile.self, from: JSONSerialization.data(withJSONObject: object))
        XCTAssertNil(old.capabilities.compute)
        XCTAssertNil(old.capabilities.httpApi)
        XCTAssertNil(old.installSubdir)
    }
    func testInstallWithOptionsRefusesAStaleIdentityBeforeDownloading() {
        let stale = AudioProfileIdentity(task: .textToSpeech, alias: "no-such-profile", repository: "x/y",
                                         revision: String(repeating: "0", count: 40), assetFingerprint: "none")
        XCTAssertThrowsError(try AudioCatalog.install(stale, acknowledged: ["anything"]))
    }
    func testTranscriptPersistencePreservesAbsentMetadataAndTiming() throws {
        let original = AudioTranscript(text: "Example", language: "English", segments: [AudioSegment(startSeconds: 1, endSeconds: 2, text: "Example", timing: "clip")], tokenLogprobs: [-0.5])
        let copy = try JSONDecoder().decode(AudioTranscript.self, from: JSONEncoder().encode(original))
        XCTAssertEqual(copy, original)
        XCTAssertNil(copy.segments[0].speaker)
        XCTAssertNil(copy.segments[0].words)
        XCTAssertEqual(copy.segments[0].timing, "clip")
    }
    func testPCMFormatUsesInterleavedFramesForDuration() {
        let pcm = AudioPCM(format: AudioPCMFormat(sampleRate: 48_000, channels: 2), samples: Array(repeating: 0, count: 96_000))
        XCTAssertEqual(pcm.durationSeconds, 1)
    }
    func testUnsupportedTaskFailsBeforeOpeningAModel() {
        XCTAssertThrowsError(try AudioSession(modelPath: "/missing/audio", task: .codec))
    }
    func testNativeFixturePcmOwnershipAndClose() async throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let model = root.appendingPathComponent("crates/audio/testdata/minimax_music3/converted_plain").path
        let session = try AudioSession(modelPath: model, task: .music)
        var request = AudioRequest(task: .music)
        request.caption = "piano"; request.lyrics = "[instrumental]"
        request.durationSeconds = 0.12; request.steps = 1; request.seed = 7
        let result = try await session.execute(request)
        let pcm = try XCTUnwrap(result.pcm)
        XCTAssertEqual(pcm.format.channels, 2)
        XCTAssertEqual(pcm.format.sampleRate, 44_100)
        XCTAssertFalse(pcm.samples.isEmpty)
        XCTAssertTrue(pcm.samples.allSatisfy(\.isFinite))
        session.close()
        XCTAssertFalse(pcm.samples.isEmpty, "Swift copied callback-owned memory before native close")
        do { _ = try await session.execute(request); XCTFail("closed session ran") } catch { }
    }
    func testCancellationDoesNotWaitForOperationLock() async throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let model = root.appendingPathComponent("crates/audio/testdata/minimax_music3/converted_plain").path
        XCTAssertThrowsError(try AudioSession(modelPath: model, task: .music, expectedFamily: "whisper"))
        let session = try AudioSession(modelPath: model, task: .music, expectedFamily: "minimax_music3")
        defer { session.close() }
        var request = AudioRequest(task: .music)
        request.caption = "piano"; request.lyrics = "[instrumental]"
        request.durationSeconds = 0.12; request.steps = 1; request.seed = 7
        do {
            _ = try await session.execute(request) { _ in session.cancel() }
            XCTFail("cancelled job succeeded")
        } catch {
            XCTAssertTrue(String(describing: error).contains("cancel"))
            XCTAssertEqual((error as? TurboSparkError)?.code, .cancelled,
                           "a caller-cancelled job is cancelled, not a generation failure")
        }
        let recovery = try await session.execute(request)
        XCTAssertFalse(try XCTUnwrap(recovery.pcm).samples.isEmpty)
    }
    func testExternalNativePermitIsExclusiveAndReleaseIsIdempotent() throws {
        let first = try XCTUnwrap(NativeHeavyWorkPermit.tryAcquire())
        XCTAssertNil(try NativeHeavyWorkPermit.tryAcquire())
        first.release(); first.release()
        let next = try XCTUnwrap(NativeHeavyWorkPermit.tryAcquire())
        next.release()
    }

    /// A real MegaTTS3 clone through this binding, the C ABI and the runtime.
    /// Set TURBOSPARK_TEST_MEGATTS3 (a converted or installed model folder) and
    /// TURBOSPARK_TEST_VOICE_REFERENCE (a mono WAV). With
    /// TURBOSPARK_TEST_MEGATTS3_EXPECTED, the WAV `turbospark-clone speak
    /// --steps 4 --seed 7 --wav-format float32` wrote for the same reference and
    /// text on the same device (Metal, the default for both on macOS), the
    /// binding must reproduce the CLI sample for sample: the reference goes in
    /// at its own rate, so both resample it the same way.
    func testRealMegaTTS3CloneThroughTheBinding() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let model = environment["TURBOSPARK_TEST_MEGATTS3"],
              let referencePath = environment["TURBOSPARK_TEST_VOICE_REFERENCE"] else {
            throw XCTSkip("Set TURBOSPARK_TEST_MEGATTS3 and TURBOSPARK_TEST_VOICE_REFERENCE for the real MegaTTS3 gate")
        }
        let (reference, rate) = try Self.monoWAV(URL(fileURLWithPath: referencePath))
        let session = try AudioSession(modelPath: model, task: .textToSpeech, allowPortable: true, expectedFamily: "megatts3")
        defer { session.close() }
        var request = AudioRequest(task: .textToSpeech)
        request.text = "Hello world, this is a short test."
        request.steps = 4
        request.seed = 7
        request.referenceSampleRate = UInt32(rate)
        do {
            _ = try await session.execute(request, pcm: reference)
            XCTFail("cloning without consent must be refused")
        } catch let error as TurboSparkError {
            XCTAssertTrue(error.localizedDescription.contains("consent"), error.localizedDescription)
        }
        request.consent = true
        let steps = StepCounter()
        let result = try await session.execute(request, pcm: reference) { if $0.stage == "diffusion" { steps.add() } }
        let pcm = try XCTUnwrap(result.pcm)
        XCTAssertEqual(pcm.format.sampleRate, 24_000)
        XCTAssertEqual(pcm.format.channels, 1)
        XCTAssertTrue(pcm.samples.allSatisfy(\.isFinite))
        XCTAssertGreaterThan(pcm.samples.map(abs).max() ?? 0, 0.01, "the clone is silent")
        XCTAssertGreaterThan(steps.count, 0, "no diffusion progress reached Swift")
        if let expected = environment["TURBOSPARK_TEST_MEGATTS3_EXPECTED"] {
            let (cli, cliRate) = try Self.monoWAV(URL(fileURLWithPath: expected))
            XCTAssertEqual(cliRate, 24_000)
            XCTAssertEqual(pcm.samples.count, cli.count)
            XCTAssertTrue(pcm.samples == cli, "the binding and the CLI disagree")
        }
    }

    private static func monoWAV(_ url: URL) throws -> ([Float], Int) {
        let file = try AVAudioFile(forReading: url)
        XCTAssertEqual(file.processingFormat.channelCount, 1, "use a mono WAV")
        let buffer = try XCTUnwrap(AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: 16_384))
        var samples: [Float] = []
        // One read can stop short of the buffer (44,032 of 44,715 frames here).
        while file.framePosition < file.length {
            try file.read(into: buffer)
            guard buffer.frameLength > 0 else { break }
            let channel = try XCTUnwrap(buffer.floatChannelData)[0]
            samples += UnsafeBufferPointer(start: channel, count: Int(buffer.frameLength))
        }
        XCTAssertEqual(Int64(samples.count), file.length)
        return (samples, Int(file.processingFormat.sampleRate))
    }
}

private final class StepCounter: @unchecked Sendable {
    private let lock = NSLock()
    private var value = 0
    var count: Int { lock.lock(); defer { lock.unlock() }; return value }
    func add() { lock.lock(); value += 1; lock.unlock() }
}
