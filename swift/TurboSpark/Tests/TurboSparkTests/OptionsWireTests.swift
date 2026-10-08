import XCTest

@testable import TurboSpark

/// Option keys must reach Rust spelled exactly as `wire.rs` reads them; a
/// misspelled key is silently ignored by serde's `default`, so these assert
/// the encoded names and then drive the real C surface.
final class OptionsWireTests: XCTestCase {

    private func keys(_ value: some Encodable) throws -> [String: Any] {
        let data = try JSONEncoder().encode(value)
        return try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    func testSamplingKnobsEncodeUnderTheirWireNames() throws {
        var o = GenerateOptions()
        o.minP = 0.05
        o.presencePenalty = 0.5
        o.frequencyPenalty = -0.25
        let json = try keys(o)
        XCTAssertEqual(json["minP"] as? Double, 0.05)
        XCTAssertEqual(json["presencePenalty"] as? Double, 0.5)
        XCTAssertEqual(json["frequencyPenalty"] as? Double, -0.25)
    }

    func testIdleUnloadSecondsEncodesAndDefaultsToAbsent() throws {
        XCTAssertNil(try keys(ServerOptions())["idleUnloadSeconds"])
        let json = try keys(ServerOptions(idleUnloadSeconds: 600))
        XCTAssertEqual(json["idleUnloadSeconds"] as? Int, 600)
    }

    func testIdleUnloadReachesTheRunningServer() throws {
        let server = try TurboSparkServer.start(options: ServerOptions(idleUnloadSeconds: 120))
        defer { server.stop() }
        XCTAssertNoThrow(try server.setIdleUnload(after: 60))
        XCTAssertNoThrow(try server.setIdleUnload(after: nil))
    }

    func testIdleUnloadOnAStoppedServerThrows() throws {
        let server = try TurboSparkServer.start()
        server.stop()
        XCTAssertThrowsError(try server.setIdleUnload(after: 60))
    }
}

final class AudioStreamingTests: XCTestCase {
    private final class Sink: @unchecked Sendable {
        var chunks: [[Float]] = []
    }

    func testChunksReachTheLiveHandlerInOrderAndStillAccumulate() {
        let sink = Sink()
        let all = audioDeliverChunksForTesting(
            [[0.1, 0.2], [0.3], [0.4, 0.5]], onPCM: { sink.chunks.append($0) })
        XCTAssertEqual(sink.chunks, [[0.1, 0.2], [0.3], [0.4, 0.5]])
        XCTAssertEqual(all, [0.1, 0.2, 0.3, 0.4, 0.5], "the full audio must still be retained")
    }

    func testNoHandlerStillAccumulates() {
        XCTAssertEqual(audioDeliverChunksForTesting([[1], [2]], onPCM: nil), [1, 2])
    }
}

final class ErrorCodeTests: XCTestCase {
    func testNewCodesMapFromTheirWireValues() {
        XCTAssertEqual(TurboSparkError.Code(rawValue: 7), .cancelled)
        XCTAssertEqual(TurboSparkError.Code(rawValue: 8), .busy)
        XCTAssertEqual(TurboSparkError.Code(rawValue: 99) ?? .unknown, .unknown)
    }
}
