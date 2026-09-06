import XCTest
import TurboSpark
@testable import TurboSparkApp

final class ServerMenuBarTests: XCTestCase {

    func testMenuBarSettingsRoundTripThroughSettingsStore() {
        let original = MacAppSettingsFileStore.load()
        defer { MacAppSettingsFileStore.save(original) }

        var settings = MacAppSettings()
        settings.showMenuBarItem = false
        settings.serverAutoStartOnLaunch = true
        settings.keepServerRunningInBackground = false
        MacAppSettingsFileStore.save(settings)

        let loaded = MacAppSettingsFileStore.load()
        XCTAssertFalse(loaded.showMenuBarItem)
        XCTAssertTrue(loaded.serverAutoStartOnLaunch)
        XCTAssertFalse(loaded.keepServerRunningInBackground)
    }

    func testLegacySettingsJSONDecodesToDefaults() throws {
        let original = MacAppSettingsFileStore.load()
        let url = AppStorageRoot.file("settings.json")
        defer {
            try? FileManager.default.removeItem(at: url)
            MacAppSettingsFileStore.save(original)
        }

        try Data("{}".utf8).write(to: url)
        let loaded = MacAppSettingsFileStore.load()
        XCTAssertTrue(loaded.showMenuBarItem)
        XCTAssertFalse(loaded.serverAutoStartOnLaunch)
        XCTAssertTrue(loaded.keepServerRunningInBackground)
    }

    func testServerMetricsStoreTokenTotalsAndPPSpeed() {
        var store = ServerMetricsStore()
        XCTAssertEqual(store.totalTokensProcessed, 0)
        XCTAssertEqual(store.totalPromptTokens, 0)
        XCTAssertEqual(store.totalNewTokens, 0)
        XCTAssertNil(store.aggregatePromptTokensPerSecond)
        XCTAssertNil(store.aggregateTokensPerSecond)

        store.ingest(.requestStarted(id: 1, atMs: 1000, method: "POST", path: "/v1/chat/completions"))
        store.ingest(.generated(
            id: 1,
            model: "test-model",
            promptTokens: 100,
            newTokens: 50,
            prefillSeconds: 0.1,
            decodeSeconds: 0.5,
            stopReason: "stop"
        ))
        store.ingest(.requestFinished(id: 1, status: 200, durationMs: 650))

        XCTAssertEqual(store.totalTokensProcessed, 150)
        XCTAssertEqual(store.totalPromptTokens, 100)
        XCTAssertEqual(store.totalNewTokens, 50)
        XCTAssertEqual(store.totalRequests, 1)
        XCTAssertEqual(store.totalErrors, 0)

        if let pp = store.aggregatePromptTokensPerSecond {
            XCTAssertEqual(pp, 1000.0, accuracy: 0.01)
        } else {
            XCTFail("Expected non-nil aggregatePromptTokensPerSecond")
        }

        if let tg = store.aggregateTokensPerSecond {
            XCTAssertEqual(tg, 100.0, accuracy: 0.01)
        } else {
            XCTFail("Expected non-nil aggregateTokensPerSecond")
        }
    }

    func testServerEndpointCatalogFamilies() {
        let openAIEndpoints = ServerEndpointCatalog.endpoints(for: .openAI)
        XCTAssertTrue(openAIEndpoints.contains { $0.path == "/v1/chat/completions" })
        XCTAssertTrue(openAIEndpoints.contains { $0.path == "/v1/models" })

        let anthropicEndpoints = ServerEndpointCatalog.endpoints(for: .anthropic)
        XCTAssertTrue(anthropicEndpoints.contains { $0.path == "/v1/messages" })

        let ollamaEndpoints = ServerEndpointCatalog.endpoints(for: .ollama)
        XCTAssertTrue(ollamaEndpoints.contains { $0.path == "/api/chat" })
        XCTAssertTrue(ollamaEndpoints.contains { $0.path == "/api/tags" })

        let serviceEndpoints = ServerEndpointCatalog.endpoints(for: .turbospark)
        XCTAssertTrue(serviceEndpoints.contains { $0.path == "/health" })
    }
}
