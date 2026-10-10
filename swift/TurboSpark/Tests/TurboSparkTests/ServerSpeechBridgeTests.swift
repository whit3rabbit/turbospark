import Foundation
import XCTest

@testable import TurboSpark

final class ServerSpeechBridgeTests: XCTestCase {
    func testHTTPSpeechRequestCrossesSwiftBridge() async throws {
        let server = try TurboSparkServer.start(options: ServerOptions(apiKey: "speech-key"))
        defer { server.stop() }
        try server.attachAudioSpeechModel(id: "qwen3-tts-clone")
        let info = try server.info()
        XCTAssertTrue(info.audioModels.contains("qwen3-tts-clone"))

        let baseURL = try XCTUnwrap(info.baseURL)
        var request = URLRequest(url: baseURL.appendingPathComponent("v1/audio/speech"))
        request.httpMethod = "POST"
        request.setValue("Bearer speech-key", forHTTPHeaderField: "authorization")
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        let bodyJSON: [String: Any] = [
            "model": "qwen3-tts-clone",
            "input": "Hello from cloned voice",
            "voice": "clone",
            "speed": 1.25,
            "response_format": "wav",
            "language": "en",
            "reference_text": "Reference speech snippet",
            "instruction": "Speak warmly"
        ]
        request.httpBody = try JSONSerialization.data(withJSONObject: bodyJSON)

        let client = Task { try await URLSession.shared.data(for: request) }
        var received: (UInt64, ServerSpeechRequest)?
        for _ in 0..<100 where received == nil {
            for event in server.pollAudioSpeechEvents() {
                if case let .start(id, speechRequest) = event {
                    received = (id, speechRequest)
                }
            }
            if received == nil { try await Task.sleep(for: .milliseconds(20)) }
        }

        let (id, speechReq) = try XCTUnwrap(received)
        XCTAssertEqual(speechReq.model, "qwen3-tts-clone")
        XCTAssertEqual(speechReq.text, "Hello from cloned voice")
        XCTAssertEqual(speechReq.voice, "clone")
        XCTAssertEqual(speechReq.speed, 1.25)
        XCTAssertEqual(speechReq.language, "en")
        XCTAssertEqual(speechReq.referenceText, "Reference speech snippet")
        XCTAssertEqual(speechReq.instruction, "Speak warmly")

        let fakePCM: [Float] = [Float](repeating: 0.1, count: 2400)
        try server.completeAudioSpeechRequest(id: id, samples: fakePCM, sampleRate: 24_000)

        let (data, response) = try await client.value
        XCTAssertEqual((response as? HTTPURLResponse)?.statusCode, 200)
        XCTAssertGreaterThan(data.count, 44) // Valid WAV file header is at least 44 bytes
    }
}
