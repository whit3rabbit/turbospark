import Foundation
import XCTest

@testable import TurboSparkApp

/// The composer tool picker shows two facts per tool: what the permission
/// settings typically do with a call, and roughly what its definition costs
/// in the prompt. Both are display-only; these tests pin that the permission
/// label never reads looser than the engine and that the estimate is the
/// documented chars-per-token rule.
@MainActor
final class ToolPickerFactsTests: XCTestCase {
    // MARK: Token estimate

    func testCharacterCountRoundsUpAtFourPerToken() {
        XCTAssertEqual(ToolPromptCostEstimate.tokens(forCharacterCount: 0), 0)
        XCTAssertEqual(ToolPromptCostEstimate.tokens(forCharacterCount: 1), 1)
        XCTAssertEqual(ToolPromptCostEstimate.tokens(forCharacterCount: 4), 1)
        XCTAssertEqual(ToolPromptCostEstimate.tokens(forCharacterCount: 5), 2)
        XCTAssertNil(ToolPromptCostEstimate.tokens(forCharacterCount: -1))
    }

    func testToolEstimateFollowsItsEncodedDefinition() throws {
        let short = OpenAITool.function(
            name: "probe", description: "Short.",
            parameters: JSONSchema(type: "object", properties: [:], required: []))
        var long = short
        long.function.description = String(repeating: "Much longer description. ", count: 20)
        let encoded = try JSONEncoder().encode(short)
        XCTAssertEqual(
            ToolPromptCostEstimate.tokens(for: short),
            ToolPromptCostEstimate.tokens(forCharacterCount: encoded.count))
        XCTAssertGreaterThan(
            ToolPromptCostEstimate.tokens(for: long) ?? 0,
            ToolPromptCostEstimate.tokens(for: short) ?? .max)
    }

    func testEveryListedToolCarriesAnEstimate() {
        let entries = AppToolCatalog.profileGroups().flatMap(\.entries)
        XCTAssertFalse(entries.isEmpty)
        for entry in entries {
            XCTAssertGreaterThan(entry.estimatedPromptTokens ?? 0, 0, entry.name)
        }
    }

    // MARK: Permission label

    private let openNetwork = AppNetworkPolicy(mode: .allowAll)

    private func label(
        _ tool: String, _ mode: AppPermissionMode,
        network: AppNetworkPolicy? = nil
    ) -> AppToolPermission {
        ToolPermissionSummary.typicalDecision(
            toolName: tool, permissions: .preset(for: mode),
            networkPolicy: network ?? openNetwork)
    }

    func testPresetsMapToTheirCategoryVerdicts() {
        XCTAssertEqual(label("read", .auto), .allow)
        XCTAssertEqual(label("bash", .auto), .ask)
        XCTAssertEqual(label("write", .auto), .ask)
        XCTAssertEqual(label("bash", .readOnly), .deny)
        XCTAssertEqual(label("write", .readOnly), .deny)
        XCTAssertEqual(label("read", .readOnly), .allow)
        XCTAssertEqual(label("bash", .permissive), .allow)
        XCTAssertEqual(label("bash", .fullAccess), .allow)
        XCTAssertEqual(label("bash", .ask), .ask)
        XCTAssertEqual(label("read", .ask), .allow)
    }

    func testCategoryDenyWinsEvenUnderFullAccess() {
        var permissions = AppProjectPermissions.preset(for: .fullAccess)
        permissions.terminal = .deny
        XCTAssertEqual(
            ToolPermissionSummary.typicalDecision(
                toolName: "bash", permissions: permissions, networkPolicy: openNetwork),
            .deny)
    }

    func testNetworkPolicyIsACeilingForWebTools() {
        XCTAssertEqual(label("WebFetch", .fullAccess), .allow)
        XCTAssertEqual(label("WebFetch", .fullAccess, network: AppNetworkPolicy(mode: .blockAll)), .deny)
        XCTAssertEqual(label("WebFetch", .permissive, network: AppNetworkPolicy(mode: .ask)), .ask)
        XCTAssertEqual(
            label("WebFetch", .permissive,
                  network: AppNetworkPolicy(mode: .custom, customUnlisted: .deny)),
            .deny)
        // The network policy never loosens a project's own deny.
        XCTAssertEqual(label("WebFetch", .readOnly), .deny)
        // And it does not touch tools that make no request.
        XCTAssertEqual(label("bash", .permissive, network: AppNetworkPolicy(mode: .blockAll)), .allow)
    }

    func testImageFetchAnswersToFileWritesToo() {
        var permissions = AppProjectPermissions.preset(for: .auto)
        permissions.web = .allow
        permissions.fileWrite = .deny
        XCTAssertEqual(
            ToolPermissionSummary.typicalDecision(
                toolName: "image_fetch", permissions: permissions, networkPolicy: openNetwork),
            .deny)
    }

    func testBrowserNeverReadsAsAllowedBecauseOriginsAreGrantedOneByOne() {
        XCTAssertEqual(label("browser_navigate", .fullAccess), .ask)
        var permissions = AppProjectPermissions.preset(for: .auto)
        permissions.browser = .deny
        XCTAssertEqual(
            ToolPermissionSummary.typicalDecision(
                toolName: "browser_navigate", permissions: permissions, networkPolicy: openNetwork),
            .deny)
    }

    /// The label mirrors the engine's ladder for an ordinary, low-risk call.
    /// If the engine changes, this fails before the picker starts telling
    /// the user something the gate does not do.
    func testLabelMatchesTheEngineForOrdinaryCalls() {
        let calls: [(String, [String: String])] = [
            ("read", ["path": "README.md"]),
            ("bash", ["command": "ls"]),
            ("write", ["path": "notes.txt", "content": "hi"]),
            ("WebFetch", ["url": "https://example.com"]),
            ("TaskList", [:]),
        ]
        for mode in AppPermissionMode.allCases {
            let permissions = AppProjectPermissions.preset(for: mode)
            let project = AppProject(name: "fixture", agentType: .coder, permissions: permissions)
            for (name, arguments) in calls {
                let call = AppToolCall(
                    name: name, arguments: arguments,
                    category: AppToolCatalog.category(for: name))
                let engine = AppToolPermissionEngine.evaluate(
                    call: call, project: project, globalServers: [],
                    networkPolicy: openNetwork)
                let engineKind: AppToolPermission
                switch engine {
                case .allow: engineKind = .allow
                case .ask: engineKind = .ask
                case .deny: engineKind = .deny
                }
                XCTAssertEqual(
                    ToolPermissionSummary.typicalDecision(
                        toolName: name, permissions: permissions, networkPolicy: openNetwork),
                    engineKind, "\(name) in \(mode)")
            }
        }
    }
}
