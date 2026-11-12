import XCTest

@testable import TurboSparkApp

/// Tools in Chat. A Chat turn has no project, so the only thing keeping a
/// model from editing files or running a shell there is that the rootless
/// filter and the toggle both hold.
@MainActor
final class ChatToolsTests: XCTestCase {
    private func chatTools(enabled: Bool) -> TurnAvailableTools {
        AppToolCatalog.captureTurnAvailableTools(
            for: nil,
            chatToolsEnabled: enabled,
            globalMcpServers: [],
            contextTokens: nil,
            webToolsEnabled: true)
    }

    func testChatOffersRootlessToolsOnly() {
        let available = chatTools(enabled: true)
        let names = available.definitions.map { $0.function.name.lowercased() }
        XCTAssertFalse(names.isEmpty, "Chat tools on must offer something, or the toggle does nothing.")
        for name in names {
            XCTAssertFalse(
                AppToolRegistry.workspaceRootedToolNames.contains(name),
                "\(name) needs a workspace root and Chat has none.")
        }
        XCTAssertTrue(names.contains("websearch"), "The headline Chat tool must be offered.")
        XCTAssertTrue(available.deferredMcpTools.isEmpty, "MCP tools stay Projects-only.")
    }

    func testChatOffersNothingWhenTheToggleIsOff() {
        XCTAssertTrue(chatTools(enabled: false).definitions.isEmpty)
    }

    func testToolCallsAreParsedInChatOnlyWhileTheToggleIsOn() {
        let model = AppModel()
        model.interactionMode = .chat
        let reply = """
            <tool_call>
            <name>websearch</name>
            <arguments>{"query": "anything"}</arguments>
            </tool_call>
            """
        model.chatToolsEnabled = true
        XCTAssertEqual(model.extractToolCalls(from: reply, project: nil).count, 1)
        model.chatToolsEnabled = false
        XCTAssertTrue(model.extractToolCalls(from: reply, project: nil).isEmpty)
    }

    func testChatToolsDefaultOnAndSurviveARoundTrip() throws {
        XCTAssertTrue(MacAppSettings().chatToolsEnabled)
        var settings = MacAppSettings()
        settings.chatToolsEnabled = false
        let data = try JSONEncoder().encode(settings)
        let decoded = try JSONDecoder().decode(MacAppSettings.self, from: data)
        XCTAssertFalse(decoded.chatToolsEnabled)
    }

    func testSettingsWrittenBeforeTheToggleExistedDecodeAsOn() throws {
        let decoded = try JSONDecoder().decode(MacAppSettings.self, from: Data("{}".utf8))
        XCTAssertTrue(decoded.chatToolsEnabled)
    }
}
