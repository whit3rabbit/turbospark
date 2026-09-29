import Foundation
import XCTest

@testable import TurboSpark

final class ServerImageBridgeTests: XCTestCase {
    func testHTTPImageRequestCrossesSwiftBridge() async throws {
        let server = try TurboSparkServer.start(options: ServerOptions(apiKey: "image-key"))
        defer { server.stop() }
        try server.attachImageModel(id: "z-test")
        XCTAssertEqual(try server.info().imageModels, ["z-test"])
        let baseURL = try XCTUnwrap(server.info().baseURL)
        var request = URLRequest(url: baseURL.appendingPathComponent("v1/images/generations"))
        request.httpMethod = "POST"
        request.setValue("Bearer image-key", forHTTPHeaderField: "authorization")
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = Data(#"{"model":"z-test","prompt":"a fox","size":"512x512","seed":7}"#.utf8)
        let client = Task { try await URLSession.shared.data(for: request) }
        var received: (UInt64, ServerImageRequest)?
        for _ in 0..<100 where received == nil {
            for event in server.pollImageEvents() {
                if case let .start(id, imageRequest) = event { received = (id, imageRequest) }
            }
            if received == nil { try await Task.sleep(for: .milliseconds(20)) }
        }
        let (id, imageRequest) = try XCTUnwrap(received)
        XCTAssertEqual(imageRequest.prompt, "a fox")
        XCTAssertEqual(imageRequest.seed, 7)
        XCTAssertEqual(imageRequest.width, 512)
        let fakePNG = Data([137, 80, 78, 71, 13, 10, 26, 10, 1])
        try server.completeImageRequest(id: id, png: fakePNG)
        let (data, response) = try await client.value
        XCTAssertEqual((response as? HTTPURLResponse)?.statusCode, 200)
        let document = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let encoded = (document?["data"] as? [[String: Any]])?.first?["b64_json"] as? String
        XCTAssertEqual(Data(base64Encoded: try XCTUnwrap(encoded)), fakePNG)
    }
}
