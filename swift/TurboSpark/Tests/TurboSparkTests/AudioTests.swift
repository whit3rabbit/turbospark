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
        } catch { XCTAssertTrue(String(describing: error).contains("cancel")) }
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
