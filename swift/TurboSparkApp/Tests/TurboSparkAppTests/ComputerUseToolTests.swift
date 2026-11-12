import XCTest

import ComputerUseKit
@testable import TurboSparkApp

final class ComputerUseToolTests: XCTestCase {
    private let fullManifest = ComputerUseBackendManifest(
        backendIdentifier: "test", supportedCommands: Set(ComputerUseCommandKind.allCases))

    private func availability(
        enabled: Bool = true,
        manifest: ComputerUseBackendManifest? = nil,
        vision: Bool = true
    ) -> ComputerUseToolAvailability {
        ComputerUseToolAvailability(
            isEnabled: enabled,
            manifest: manifest ?? fullManifest,
            mediaCapability: vision ? .imageBearingToolResults() : .textOnly)
    }

    private func names(_ tools: [OpenAITool]) -> Set<String> {
        Set(tools.map(\.function.name))
    }

    // MARK: - Vision gate

    func testTextOnlyModelGetsNoComputerToolsAtAll() {
        XCTAssertTrue(ComputerUseToolDefinitions.tools(availableFor: availability(vision: false)).isEmpty)
    }

    func testVisionModelWithCurrentHelperGetsEveryTool() {
        XCTAssertEqual(
            names(ComputerUseToolDefinitions.tools(availableFor: availability())),
            ComputerUseToolDefinitions.toolNames)
    }

    func testDisabledSettingOrMissingOrStaleManifestHidesEveryTool() {
        XCTAssertTrue(ComputerUseToolDefinitions.tools(availableFor: availability(enabled: false)).isEmpty)
        XCTAssertTrue(ComputerUseToolDefinitions.tools(availableFor: .disabled).isEmpty)
        let noManifest = ComputerUseToolAvailability(
            isEnabled: true, manifest: nil, mediaCapability: .imageBearingToolResults())
        XCTAssertTrue(ComputerUseToolDefinitions.tools(availableFor: noManifest).isEmpty)
        let stale = ComputerUseBackendManifest(
            backendIdentifier: "old",
            commandSurfaceVersion: ComputerUseProtocol.currentVersion + 1,
            supportedCommands: Set(ComputerUseCommandKind.allCases))
        XCTAssertTrue(ComputerUseToolDefinitions.tools(availableFor: availability(manifest: stale)).isEmpty)
    }

    func testUnsupportedCommandHidesOnlyItsTool() {
        let partial = ComputerUseBackendManifest(backendIdentifier: "p", supportedCommands: [.listApps, .snapshot])
        XCTAssertEqual(
            names(ComputerUseToolDefinitions.tools(availableFor: availability(manifest: partial))),
            ["computer_list_apps", "computer_snapshot", "computer_request_access"])
    }

    // MARK: - Catalog

    func testTurnCaptureAddsComputerToolsOnlyWhenOffered() {
        let project = AppProject(name: "Computer")
        let without = AppToolCatalog.captureTurnAvailableTools(
            for: project, globalMcpServers: [], contextTokens: nil, webToolsEnabled: false)
        XCTAssertTrue(names(without.definitions).isDisjoint(with: ComputerUseToolDefinitions.toolNames))

        let with = AppToolCatalog.captureTurnAvailableTools(
            for: project, globalMcpServers: [], contextTokens: nil, webToolsEnabled: false,
            computerUseAvailability: availability())
        XCTAssertTrue(ComputerUseToolDefinitions.toolNames.isSubset(of: names(with.definitions)))
        XCTAssertTrue(ComputerUseToolDefinitions.toolNames.isSubset(of: names(with.promptDefinitions)))

        let textOnly = AppToolCatalog.captureTurnAvailableTools(
            for: project, globalMcpServers: [], contextTokens: nil, webToolsEnabled: false,
            computerUseAvailability: availability(vision: false))
        XCTAssertTrue(names(textOnly.definitions).isDisjoint(with: ComputerUseToolDefinitions.toolNames))
    }

    func testRootlessChatCaptureKeepsComputerTools() {
        let chat = AppToolCatalog.captureTurnAvailableTools(
            for: nil, chatToolsEnabled: true, globalMcpServers: [], contextTokens: nil,
            webToolsEnabled: false, computerUseAvailability: availability())
        XCTAssertTrue(ComputerUseToolDefinitions.toolNames.isSubset(of: names(chat.definitions)))
    }

    /// Subagents build their lists with `tools(for:)`, which must never add
    /// computer tools whatever the agent type.
    func testAgentProfileListsNeverIncludeComputerTools() {
        for type in [AppAgentType.coder, .researcher, .autonomous, .general] {
            XCTAssertTrue(
                names(AppToolCatalog.tools(for: type)).isDisjoint(with: ComputerUseToolDefinitions.toolNames),
                "\(type)")
        }
        XCTAssertTrue(names(AppToolCatalog.allTools).isDisjoint(with: ComputerUseToolDefinitions.toolNames))
    }

    func testComputerToolsHaveTheirOwnCategoryAndHandler() {
        for name in ComputerUseToolDefinitions.toolNames {
            XCTAssertEqual(AppToolCatalog.builtInCategory(for: name), .computerUse, name)
            XCTAssertTrue(AppToolRegistry.isImplemented(name), name)
        }
    }

    // MARK: - Permissions

    func testAccessRequestAlwaysAsksEvenInFullAccess() {
        let call = AppToolCall(
            name: "computer_request_access",
            arguments: ["apps": "com.apple.TextEdit", "reason": "Write a note"],
            category: .computerUse)
        for permissions in [AppProjectPermissions.fullAccess, .permissive, .standard] {
            let project = AppProject(name: "P", permissions: permissions)
            guard case .ask(let assessment, _) = AppToolPermissionEngine.evaluate(
                call: call, project: project, sessionApproved: true, globalServers: [])
            else {
                return XCTFail("\(permissions.mode) did not ask")
            }
            XCTAssertEqual(assessment.hardGated, true)
        }
    }

    func testReadOnlyModeAndDenyRefuseComputerUse() {
        let call = AppToolCall(name: "computer_screenshot", arguments: ["app": "com.apple.TextEdit"], category: .computerUse)
        for permissions in [
            AppProjectPermissions.readOnly,
            AppProjectPermissions(computerUse: .deny),
        ] {
            let project = AppProject(name: "P", permissions: permissions)
            guard case .deny = AppToolPermissionEngine.evaluate(call: call, project: project, globalServers: []) else {
                return XCTFail("expected deny")
            }
        }
    }

    func testObservingPassesOnAskButInputStillAsks() {
        let project = AppProject(name: "P", permissions: .standard)
        let read = AppToolCall(name: "computer_snapshot", arguments: ["app": "com.apple.TextEdit"], category: .computerUse)
        XCTAssertEqual(AppToolPermissionEngine.evaluate(call: read, project: project, globalServers: []), .allow)
        let input = AppToolCall(
            name: "computer_type", arguments: ["app": "com.apple.TextEdit", "text": "hi"], category: .computerUse)
        guard case .ask = AppToolPermissionEngine.evaluate(call: input, project: project, globalServers: []) else {
            return XCTFail("input must ask under the standard preset")
        }
    }

    func testPermissionDecodesAbsentKeyAsAskAndRoundTrips() throws {
        let legacy = Data(#"{"mode":"auto","browser":"allow"}"#.utf8)
        XCTAssertEqual(try JSONDecoder().decode(AppProjectPermissions.self, from: legacy).computerUse, .ask)
        let unknown = Data(#"{"computerUse":"sometimes"}"#.utf8)
        XCTAssertEqual(try JSONDecoder().decode(AppProjectPermissions.self, from: unknown).computerUse, .ask)
        let original = AppProjectPermissions(computerUse: .deny)
        let decoded = try JSONDecoder().decode(
            AppProjectPermissions.self, from: JSONEncoder().encode(original))
        XCTAssertEqual(decoded.computerUse, .deny)
        XCTAssertFalse(original.hasSameApprovalPolicy(as: AppProjectPermissions()))
    }

    // MARK: - App tiers

    func testTiers() {
        XCTAssertEqual(ComputerUseAppPolicy.tier(forBundleID: "com.apple.TextEdit"), .full)
        XCTAssertEqual(ComputerUseAppPolicy.tier(forBundleID: "com.apple.Safari"), .read)
        XCTAssertEqual(ComputerUseAppPolicy.tier(forBundleID: "com.apple.Terminal"), .click)
        XCTAssertEqual(ComputerUseAppPolicy.tier(forBundleID: "com.jetbrains.intellij"), .click)
        XCTAssertEqual(ComputerUseAppPolicy.tier(forBundleID: "com.apple.systempreferences"), .refused)
        XCTAssertEqual(ComputerUseAppPolicy.tier(forBundleID: "com.whit3rabbit.turbospark"), .refused)
        XCTAssertEqual(ComputerUseAppPolicy.tier(forBundleID: "com.whit3rabbit.turbospark.dev"), .refused)
    }

    func testTierActionMatrix() {
        let app = "x.y"
        let leftClick = ComputerUseCommand.click(bundleID: app, target: .reference("r"), button: .left, count: 1)
        let rightClick = ComputerUseCommand.click(bundleID: app, target: .reference("r"), button: .right, count: 1)
        let type = ComputerUseCommand.type(bundleID: app, text: "x")
        let shot = ComputerUseCommand.screenshot(bundleID: app)
        XCTAssertTrue(ComputerUseAppPolicy.allows(shot, tier: .read))
        XCTAssertFalse(ComputerUseAppPolicy.allows(leftClick, tier: .read))
        XCTAssertTrue(ComputerUseAppPolicy.allows(leftClick, tier: .click))
        XCTAssertFalse(ComputerUseAppPolicy.allows(rightClick, tier: .click))
        XCTAssertFalse(ComputerUseAppPolicy.allows(type, tier: .click))
        XCTAssertTrue(ComputerUseAppPolicy.allows(type, tier: .full))
        XCTAssertFalse(ComputerUseAppPolicy.allows(shot, tier: .refused))
    }

    // MARK: - Coordinates

    func testDownscaledScreenshotMapsToWindowPoints() throws {
        let frame = ComputerUseScreenshotFrame(
            screenshotID: 1, bundleID: "x.y",
            windowFrame: ComputerUseRect(x: 100, y: 200, width: 800, height: 600),
            imagePixelWidth: 400, imagePixelHeight: 300)
        let point = try XCTUnwrap(frame.screenPoint(x: 200, y: 150))
        XCTAssertEqual(point.x, 500)
        XCTAssertEqual(point.y, 500)
        XCTAssertNil(frame.screenPoint(x: 400, y: 0))
        XCTAssertNil(frame.screenPoint(x: -1, y: 0))
        XCTAssertNil(frame.screenPoint(x: .nan, y: 0))
    }

    // MARK: - Executor

    func testUngrantedAppIsDeniedBeforeReachingHelper() async {
        let port = RecordingPort()
        let outcome = await ComputerUseToolExecutor.execute(
            call: AppToolCall(name: "computer_snapshot", arguments: ["app": "com.apple.TextEdit"]),
            runtime: runtime(port: port))
        guard case .denied = outcome else { return XCTFail("\(outcome)") }
        let commands = await port.commands
        XCTAssertTrue(commands.isEmpty)
    }

    func testExtraArgumentIsRejected() async {
        let outcome = await ComputerUseToolExecutor.execute(
            call: AppToolCall(name: "computer_snapshot", arguments: ["app": "com.apple.TextEdit", "all": "true"]),
            runtime: runtime(port: RecordingPort()))
        guard case .invalidInput = outcome else { return XCTFail("\(outcome)") }
    }

    func testRequestAccessGrantsPolicyTierAndNeverRefusedApps() async {
        let session = ComputerUseSession()
        let outcome = await ComputerUseToolExecutor.execute(
            call: AppToolCall(
                name: "computer_request_access",
                arguments: ["apps": "com.apple.TextEdit, com.apple.Safari, com.apple.systempreferences", "reason": "r"]),
            runtime: runtime(port: RecordingPort(), session: session))
        guard case .completed(let output, _) = outcome else { return XCTFail("\(outcome)") }
        XCTAssertTrue(output.contains("Refused com.apple.systempreferences"))
        let granted = await session.grantedApps
        XCTAssertEqual(granted, ["com.apple.TextEdit": .full, "com.apple.Safari": .read])
    }

    func testPixelClickMapsThroughLatestScreenshotAndRejectsStaleID() async {
        let session = ComputerUseSession()
        _ = await session.grant(["com.apple.TextEdit"])
        let first = await session.recordScreenshot(
            bundleID: "com.apple.TextEdit", windowFrame: ComputerUseRect(x: 0, y: 0, width: 1000, height: 500),
            imagePixelWidth: 500, imagePixelHeight: 250)
        let latest = await session.recordScreenshot(
            bundleID: "com.apple.TextEdit", windowFrame: ComputerUseRect(x: 10, y: 20, width: 1000, height: 500),
            imagePixelWidth: 500, imagePixelHeight: 250)
        let port = RecordingPort()

        let stale = await ComputerUseToolExecutor.execute(
            call: AppToolCall(name: "computer_click", arguments: [
                "app": "com.apple.TextEdit", "x": "10", "y": "10", "screenshot_id": "\(first.screenshotID)",
            ]),
            runtime: runtime(port: port, session: session))
        guard case .invalidInput = stale else { return XCTFail("\(stale)") }

        let fresh = await ComputerUseToolExecutor.execute(
            call: AppToolCall(name: "computer_click", arguments: [
                "app": "com.apple.TextEdit", "x": "100", "y": "50", "screenshot_id": "\(latest.screenshotID)",
            ]),
            runtime: runtime(port: port, session: session))
        guard case .completed = fresh else { return XCTFail("\(fresh)") }
        let commands = await port.commands
        XCTAssertEqual(commands, [
            .click(bundleID: "com.apple.TextEdit", target: .point(x: 210, y: 120), button: .left, count: 1),
        ])
    }

    func testClickTierAppCannotType() async {
        let session = ComputerUseSession()
        _ = await session.grant(["com.apple.Terminal"])
        let port = RecordingPort()
        let outcome = await ComputerUseToolExecutor.execute(
            call: AppToolCall(name: "computer_type", arguments: ["app": "com.apple.Terminal", "text": "rm -rf ~"]),
            runtime: runtime(port: port, session: session))
        guard case .denied = outcome else { return XCTFail("\(outcome)") }
        let commands = await port.commands
        XCTAssertTrue(commands.isEmpty)
    }

    func testRevokeAllDropsGrants() async {
        let session = ComputerUseSession()
        _ = await session.grant(["com.apple.TextEdit"])
        await session.revokeAll()
        let tier = await session.tier(for: "com.apple.TextEdit")
        XCTAssertNil(tier)
    }

    private func runtime(
        port: RecordingPort,
        session: ComputerUseSession = ComputerUseSession()
    ) -> ComputerUseToolRuntime {
        ComputerUseToolRuntime(availability: availability(), session: session, port: port)
    }
}

private actor RecordingPort: ComputerUseControlPort {
    private(set) var commands: [ComputerUseCommand] = []

    func perform(_ command: ComputerUseCommand) async throws -> ComputerUseResultValue {
        commands.append(command)
        return .acknowledged("ok")
    }
}
