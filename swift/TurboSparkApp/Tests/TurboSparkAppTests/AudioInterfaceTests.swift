import AVFoundation
import Foundation
import Speech
import XCTest
@testable import TurboSpark
@testable import TurboSparkApp

/// The pure halves of the audio interface (docs/AUDIO_UI.md): the
/// capability decision table, attachment classification against the
/// ENGINE's decoder list, the send-time refusal, archive compatibility, and
/// view arithmetic. No microphone, no recognizer, no model.
final class AudioInterfaceTests: XCTestCase {
    private func inputs(
        enabled: Bool = true,
        mic: AVAuthorizationStatus = .authorized,
        speech: SFSpeechRecognizerAuthorizationStatus = .authorized,
        onDevice: Bool = true,
        choice: SpeechEngineChoice = .automatic,
        engineSTT: Bool = false,
        engineTTS: Bool = false,
        modelsConfigured: Bool = false,
        taps: Bool = true
    ) -> AudioCapabilityInputs {
        AudioCapabilityInputs(
            experimentalEnabled: enabled,
            microphone: mic,
            speech: speech,
            onDeviceSpeechSupported: onDevice,
            choice: choice,
            engineSpeechToText: AudioTaskStatus(active: engineSTT, reason: engineSTT ? nil : "engine says no"),
            engineTextToSpeech: AudioTaskStatus(active: engineTTS, reason: engineTTS ? nil : "engine says no"),
            engineSpeechModelConfigured: modelsConfigured,
            engineVoiceModelConfigured: modelsConfigured,
            supportsProcessTaps: taps)
    }

    // MARK: - Capability table

    func testFlagOffDisablesEverythingButReadAloud() {
        let snapshot = AudioCapabilities.resolve(inputs(enabled: false))
        XCTAssertEqual(snapshot.micCapture, .disabled)
        XCTAssertEqual(snapshot.transcription, .disabled)
        XCTAssertEqual(snapshot.systemCapture, .disabled)
        XCTAssertEqual(snapshot.attachments, .disabled)
        // Read aloud predates the flag and must keep working without it.
        XCTAssertEqual(snapshot.readAloud, .available(.apple))
    }

    func testTheEngineIsPreferredWheneverItHasAModel() {
        let snapshot = AudioCapabilities.resolve(
            inputs(engineSTT: true, engineTTS: true, modelsConfigured: true))
        XCTAssertEqual(snapshot.transcription, .available(.engine))
        XCTAssertEqual(snapshot.readAloud, .available(.engine))
    }

    func testAnActiveEngineWithNoConfiguredModelFallsBack() {
        let snapshot = AudioCapabilities.resolve(inputs(engineSTT: true, modelsConfigured: false))
        XCTAssertEqual(snapshot.transcription, .available(.apple))
    }

    func testAutomaticFallsBackToAppleOnDevice() {
        XCTAssertEqual(
            AudioCapabilities.resolve(inputs()).transcription, .available(.apple))
        XCTAssertEqual(
            AudioCapabilities.resolve(inputs(speech: .notDetermined)).transcription,
            .needsPermission(.speechRecognition))
        XCTAssertEqual(
            AudioCapabilities.resolve(inputs(speech: .denied)).transcription,
            .denied(.speechRecognition))
        // No on-device model: refused, never silently sent to a server.
        if case .unavailable = AudioCapabilities.resolve(inputs(onDevice: false)).transcription {
        } else {
            XCTFail("an off-device-only locale must be unavailable")
        }
    }

    func testEngineOnlyCarriesTheEnginesOwnReason() {
        let snapshot = AudioCapabilities.resolve(inputs(choice: .engineOnly))
        XCTAssertEqual(snapshot.transcription, .needsModel("engine says no"))
        XCTAssertEqual(snapshot.transcription.reason, "engine says no")
        XCTAssertEqual(snapshot.readAloud, .needsModel("engine says no"))
        XCTAssertFalse(snapshot.transcription.isUsable)
    }

    func testMicrophonePermissionStates() {
        XCTAssertEqual(AudioCapabilities.resolve(inputs(mic: .notDetermined)).micCapture,
                       .needsPermission(.microphone))
        XCTAssertTrue(AudioCapabilities.resolve(inputs(mic: .notDetermined)).micCapture.isUsable)
        XCTAssertEqual(AudioCapabilities.resolve(inputs(mic: .denied)).micCapture, .denied(.microphone))
        XCTAssertNotNil(AudioCapabilities.resolve(inputs(mic: .denied)).micCapture.reason)
    }

    func testSystemCaptureNeedsProcessTaps() {
        XCTAssertEqual(AudioCapabilities.resolve(inputs(taps: false)).systemCapture,
                       .unsupportedOS(minimum: "14.2"))
    }

    // MARK: - Engine-backed classification

    func testTheEngineCapabilityTableIsReadable() throws {
        let engine = try XCTUnwrap(AudioCapabilities.engine)
        XCTAssertFalse(engine.speechToText.active, "no audio model family exists yet")
        XCTAssertNotNil(engine.speechToText.reason)
    }

    func testAttachmentAudioClassificationFollowsTheEngine() {
        XCTAssertEqual(AudioFileClass.classify(extension: "M4A"), .supported)
        XCTAssertEqual(AudioFileClass.classify(extension: "opus"), .refused)
        XCTAssertEqual(AudioFileClass.classify(extension: "pdf"), .notAudio)

        let clip = AppPromptAttachment(
            fileName: "memo.m4a", formatLabel: "Audio", extractedText: "",
            wasTruncatedDuringExtraction: false)
        XCTAssertTrue(clip.isAudio)
        XCTAssertFalse(clip.isImage)
        XCTAssertEqual(clip.symbolName, "waveform")
        // No source on disk: falls back to the text arm like images do.
        XCTAssertEqual(clip.previewKind, .text)
        XCTAssertNil(clip.thumbnailSourceURL)
        XCTAssertTrue(clip.detailText.contains("no transcript"))
        XCTAssertEqual(clip.promptBlockLabel, "audio transcript")
    }

    func testUntranscribedAudioIsRefusedAndTranscribedAudioIsNot() {
        let pending = AppPromptAttachment(
            fileName: "memo.wav", formatLabel: "Audio", extractedText: "",
            wasTruncatedDuringExtraction: false)
        var done = pending
        done.extractedText = "hello there"
        let document = AppPromptAttachment(
            fileName: "notes.txt", formatLabel: "Text", extractedText: "",
            wasTruncatedDuringExtraction: false)

        XCTAssertNotNil(AppPromptAttachment.untranscribedAudioRefusal(in: [pending]))
        XCTAssertNil(AppPromptAttachment.untranscribedAudioRefusal(in: [done]))
        // An empty DOCUMENT is not this guard's business.
        XCTAssertNil(AppPromptAttachment.untranscribedAudioRefusal(in: [document]))
        XCTAssertTrue(
            AppPromptAttachment.untranscribedAudioRefusal(in: [pending, pending])?
                .contains("2 audio attachments") ?? false)
    }

    // MARK: - Archive compatibility (Gotcha 13)

    func testAMessageWrittenBeforeAudioPathsStillDecodes() throws {
        let legacy = """
        {"id":"11111111-2222-3333-4444-555555555555","role":"user","content":"hi","imagePaths":[]}
        """.data(using: .utf8)!
        let message = try JSONDecoder().decode(AppChatMessage.self, from: legacy)
        XCTAssertEqual(message.audioPaths, [])
    }

    func testAudioPathsRoundTrip() throws {
        let message = AppChatMessage(role: .user, content: "x", audioPaths: ["turbospark-asset:abc"])
        let decoded = try JSONDecoder().decode(
            AppChatMessage.self, from: JSONEncoder().encode(message))
        XCTAssertEqual(decoded.audioPaths, ["turbospark-asset:abc"])
    }

    // MARK: - View arithmetic

    func testWaveformResampleKeepsBarCountAndPeaks() {
        XCTAssertEqual(WaveformMath.resample([0.1, 0.9, 0.2, 0.3], to: 2), [0.9, 0.3])
        XCTAssertEqual(WaveformMath.resample([0.5], to: 3), [0.5, 0.5, 0.5])
        XCTAssertEqual(WaveformMath.resample([], to: 2), [0, 0])
        XCTAssertEqual(WaveformMath.resample([1], to: 0), [])
    }

    func testDurationFormattingIsAscii() {
        XCTAssertEqual(WaveformMath.formatDuration(0), "0:00")
        XCTAssertEqual(WaveformMath.formatDuration(42.9), "0:42")
        XCTAssertEqual(WaveformMath.formatDuration(3_725), "1:02:05")
        XCTAssertEqual(WaveformMath.formatDuration(.nan), "0:00")
    }

    func testRateLabels() {
        XCTAssertEqual(WaveformMath.rateLabel(1), "1x")
        XCTAssertEqual(WaveformMath.rateLabel(1.5), "1.5x")
        XCTAssertEqual(WaveformMath.rateLabel(0.75), "0.75x")
    }

    func testAACBitRateIsClampedForLowRatesOnly() {
        XCTAssertEqual(AudioEngineBridge.clampedAACBitRate(128_000, sampleRate: 16_000, channels: 1), 64_000)
        XCTAssertEqual(AudioEngineBridge.clampedAACBitRate(128_000, sampleRate: 44_100, channels: 2), 128_000)
        XCTAssertEqual(AudioEngineBridge.clampedAACBitRate(256_000, sampleRate: 48_000, channels: 1), 192_000)
    }
}
