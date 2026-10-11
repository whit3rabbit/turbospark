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

}
