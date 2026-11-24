import Foundation
import TurboSpark
import XCTest
@testable import TurboSparkApp

/// Speech enhancement on the Clean Up page: the engine refuses a job whose
/// request names no `input_sample_rate`, and the page reads takes at 48 kHz
/// (`AudioMediaIO.readMono48kTake`), so the run request must say 48000 on the
/// wire. No other task gains the field.
final class AudioEnhancementRequestTests: XCTestCase {
    private func profile(task: AudioTask, family: String) -> AudioProfile {
        AudioProfile(
            identity: AudioProfileIdentity(task: task, alias: family + "-test", repository: "someone/" + family,
                                           revision: String(repeating: "0", count: 40), assetFingerprint: family),
            family: family, displayName: family,
            capabilities: AudioCapabilities(languages: [], operations: ["enhance"], backend: "portable_cpu",
                                            cancellation: "chunk", canRun: true),
            pcmFormat: AudioPCMFormat(sampleRate: 48_000, channels: 1), readiness: "implemented_unqualified")
    }

    private func wire(_ request: AudioRequest) throws -> [String: Any] {
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        return try XCTUnwrap(try JSONSerialization.jsonObject(with: encoder.encode(request)) as? [String: Any])
    }

    @MainActor func testEnhancementRunSendsTheRateItReadsAt() throws {
        XCTAssertTrue(AudioTask.enhancement.isRunnable)
        XCTAssertTrue(AudioWorkspacePage.tools.contains(.cleanup), "Clean Up appears once enhancement runs")
        let recipe = AudioRecipe()
        let enhancer = profile(task: .enhancement, family: "mossformer2_se")
        let request = AudioWorkspaceController.runRequest(recipe, task: .enhancement, profile: enhancer)
        let fields = try wire(request)
        XCTAssertEqual(fields["task"] as? String, "enhancement")
        XCTAssertEqual(fields["input_sample_rate"] as? Int, 48_000)
        XCTAssertNil(fields["reference_sample_rate"])

        let transcriber = profile(task: .speechToText, family: "whisper")
        let transcription = AudioWorkspaceController.runRequest(recipe, task: .speechToText, profile: transcriber)
        XCTAssertNil(try wire(transcription)["input_sample_rate"], "other tasks never send it")
    }
}
