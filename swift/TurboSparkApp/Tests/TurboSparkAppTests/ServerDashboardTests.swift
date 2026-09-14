import Foundation
import TurboSpark
import XCTest
@testable import TurboSparkApp

final class ServerDashboardTests: XCTestCase {
    func testRatesUseElapsedTimeAndResetDoesNotUnderflow() {
        var history = ServerLiveHistory()
        let date = Date(timeIntervalSince1970: 100)
        history.sample(date: date, received: 100, sent: 200, memory: 50)
        XCTAssertNil(history.points.last?.receivedPerSecond)
        history.sample(date: date.addingTimeInterval(2), received: 300, sent: 600, memory: 75)
        XCTAssertEqual(history.points.last?.receivedPerSecond, 100)
        XCTAssertEqual(history.points.last?.sentPerSecond, 200)
        XCTAssertEqual(history.points.last?.memoryBytes, 75)
        history.sample(date: date.addingTimeInterval(3), received: 0, sent: 0, memory: nil)
        XCTAssertNil(history.points.last?.sentPerSecond)
        XCTAssertNil(history.points.last?.memoryBytes)
        for i in 4...150 {
            history.sample(date: date.addingTimeInterval(Double(i)), received: nil, sent: nil, memory: nil)
        }
        XCTAssertEqual(history.points.count, 120)
        XCTAssertEqual(history.points.first?.date, date.addingTimeInterval(31))
    }

    func testFavoritesRoundTripAndOldSettingsDefaultToLoopback() throws {
        let favorite = ServerFavorite(name: "Local", host: "::1", port: 9090,
            modelPaths: ["/models/one.gturbo", "/models/two.gturbo"], contextTokens: 8192)
        var settings = MacAppSettings()
        settings.serverFavorites = [favorite]
        settings.serverHost = "::1"
        let decoded = try JSONDecoder().decode(MacAppSettings.self, from: JSONEncoder().encode(settings))
        XCTAssertEqual(decoded.serverFavorites, [favorite])
        XCTAssertEqual(decoded.serverHost, "::1")
        let old = try JSONDecoder().decode(MacAppSettings.self, from: Data("{}".utf8))
        XCTAssertEqual(old.serverHost, "127.0.0.1")
        XCTAssertEqual(old.serverFavorites, [])
    }

    func testCustomHostAndTrafficReachTheActualServer() async throws {
        let server = try TurboSparkServer.start(options: ServerOptions(host: "::1", captureText: true))
        defer { server.stop() }
        let info = try server.info()
        XCTAssertEqual(info.host, "::1")
        XCTAssertNotEqual(info.port, 0)
        var request = URLRequest(url: try XCTUnwrap(info.baseURL).appendingPathComponent("v1/chat/completions"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let body = Data(#"{"model":"missing","messages":[{"role":"user","content":"preview me"}]}"#.utf8)
        request.httpBody = body
        let (response, _) = try await URLSession.shared.data(for: request)
        let traffic = try XCTUnwrap(server.info().traffic)
        XCTAssertEqual(traffic.receivedBytes, UInt64(body.count))
        XCTAssertEqual(traffic.sentBytes, UInt64(response.count))
        XCTAssertTrue(traffic.previews.contains { $0.contains("preview me") })
    }

    func testInvalidHostAndUnauthenticatedNetworkBindAreRefused() {
        for host in ["not-an-ip", "0.0.0.0", "::"] {
            XCTAssertThrowsError(try TurboSparkServer.start(options: ServerOptions(host: host)))
        }
    }

    @MainActor
    func testFavoriteLoadsItsAddressAndClampsContext() async throws {
        let model = AppModel()
        defer { model.stopServer() }
        model.serverPortIsValid = false
        model.loadServerFavorite(ServerFavorite(name: "test", host: "::1", port: 0,
            modelPaths: [], contextTokens: -1))
        XCTAssertTrue(model.serverBusy)
        for _ in 0..<100 where model.serverBusy {
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        XCTAssertEqual(model.serverInfo?.host, "::1")
        XCTAssertEqual(model.maxContextTokens, 0)
        XCTAssertEqual(model.serverInfo?.models, [])
    }
}
