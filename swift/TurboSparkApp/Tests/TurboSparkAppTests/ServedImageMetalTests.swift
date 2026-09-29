import Foundation
import XCTest

@testable import TurboSparkApp

@MainActor
final class ServedImageMetalTests: XCTestCase {
    func testInstalledImageModelThroughHTTP() async throws {
        guard let alias = ProcessInfo.processInfo.environment["TURBOSPARK_TEST_SERVED_IMAGE_ALIAS"] else {
            throw XCTSkip("set TURBOSPARK_TEST_SERVED_IMAGE_ALIAS to an installed Z-Image or Qwen-Image alias")
        }
        let model = AppModel()
        defer { model.stopServer() }
        let installed = try XCTUnwrap(model.imageModels.first { $0.alias == alias })
        model.serverAPIKeyInput = "served-image-test-key"
        await model.performServerStart(attachChat: false)
        let server = try XCTUnwrap(model.server)
        model.stopServerPolling()
        model.attachImageModelToServer(installed)
        XCTAssertEqual(try server.info().imageModels, [alias])

        let baseURL = try XCTUnwrap(server.info().baseURL)
        var request = URLRequest(url: baseURL.appendingPathComponent("v1/images/generations"))
        request.httpMethod = "POST"
        request.timeoutInterval = 900
        request.setValue("Bearer served-image-test-key", forHTTPHeaderField: "authorization")
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = try JSONSerialization.data(withJSONObject: [
            "model": alias, "prompt": "A red fox in a snowy forest", "size": "\(installed.width)x\(installed.height)",
            "n": 1, "seed": 7
        ])
        let pump = Task { @MainActor in
            while !Task.isCancelled {
                model.handleServerImageEvents(server)
                try? await Task.sleep(for: .milliseconds(100))
            }
        }
        defer { pump.cancel() }
        let (data, response) = try await URLSession.shared.data(for: request)
        XCTAssertEqual((response as? HTTPURLResponse)?.statusCode, 200,
                       String(data: data.prefix(1024), encoding: .utf8) ?? "")
        let document = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let encoded = (document?["data"] as? [[String: Any]])?.first?["b64_json"] as? String
        let png = try XCTUnwrap(Data(base64Encoded: try XCTUnwrap(encoded)))
        XCTAssertEqual(Array(png.prefix(8)), [137, 80, 78, 71, 13, 10, 26, 10])
        XCTAssertGreaterThan(png.count, 1024)
    }
}
