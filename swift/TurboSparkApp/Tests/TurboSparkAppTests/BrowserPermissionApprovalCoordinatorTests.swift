import XCTest
@testable import TurboSparkApp

final class BrowserPermissionApprovalCoordinatorTests: XCTestCase {
    func testRequestUsesCanonicalNavigationOriginAndOmitsTheFullURL() throws {
        let project = AppProject(name: "Browser approval")
        let call = AppToolCall(
            name: "browser_navigate",
            arguments: ["url": "HTTPS://Example.COM:443/account?token=private#section"],
            category: .browser
        )

        let request = try XCTUnwrap(BrowserPermissionApprovalRequest.make(
            for: call,
            project: project,
            currentOrigin: nil
        ))

        XCTAssertEqual(request.callID, call.id)
        XCTAssertEqual(request.projectID, project.id)
        XCTAssertEqual(request.origin.canonicalString, "https://example.com")
        XCTAssertFalse(request.origin.canonicalString.contains("token=private"))
    }

    @MainActor
    func testOneTimeApprovalRequiresTheSameCallProjectAndOrigin() throws {
        let project = AppProject(name: "Browser approval")
        let origin = try XCTUnwrap(BrowserOrigin(origin: "https://example.com"))
        let redirectedOrigin = try XCTUnwrap(BrowserOrigin(origin: "https://other.example"))
        let call = AppToolCall(
            name: "browser_click",
            arguments: ["reference": "opaque-ref"],
            category: .browser
        )
        let request = BrowserPermissionApprovalRequest(
            callID: call.id,
            projectID: project.id,
            toolName: call.name,
            origin: origin
        )
        let coordinator = BrowserPermissionApprovalCoordinator()
        coordinator.authorizeOnce(request)

        XCTAssertFalse(coordinator.allowsOnce(
            callID: UUID(), projectID: project.id, origin: origin))
        XCTAssertFalse(coordinator.allowsOnce(
            callID: call.id, projectID: UUID(), origin: origin))
        XCTAssertFalse(coordinator.allowsOnce(
            callID: call.id, projectID: project.id, origin: redirectedOrigin))
        XCTAssertTrue(coordinator.allowsOnce(
            callID: call.id, projectID: project.id, origin: origin))
    }

    @MainActor
    func testDenyCancelsTheTemporaryApproval() throws {
        let project = AppProject(name: "Browser approval")
        let origin = try XCTUnwrap(BrowserOrigin(origin: "https://example.com"))
        let call = AppToolCall(name: "browser_snapshot", category: .browser)
        let request = BrowserPermissionApprovalRequest(
            callID: call.id,
            projectID: project.id,
            toolName: call.name,
            origin: origin
        )
        let coordinator = BrowserPermissionApprovalCoordinator()
        coordinator.authorizeOnce(request)
        XCTAssertTrue(coordinator.allowsOnce(callID: call.id, projectID: project.id, origin: origin))

        coordinator.cancel(callID: call.id)

        XCTAssertFalse(coordinator.allowsOnce(callID: call.id, projectID: project.id, origin: origin))
    }

    // MARK: - The AppModel approval wrappers (the buttons on the card)

    @MainActor
    private func approveHarnessModel(
        permissions: AppProjectPermissions = AppProjectPermissions(browser: .allow)
    ) -> (AppModel, AppProject, AppToolCall) {
        let appModel = AppModel()
        let chat = AppChat(title: "browser approval")
        appModel.chats = [chat]
        appModel.selectedChatID = chat.id
        let project = AppProject(name: "Browser approval", permissions: permissions)
        appModel.projects = [project]
        let call = AppToolCall(
            name: "browser_navigate",
            arguments: ["url": "https://example.com/path"],
            category: .browser)
        appModel.pendingToolCall = call
        appModel.pendingToolCallChatID = chat.id
        appModel.pendingToolCallStep = 0
        appModel.pendingToolCallProject = project
        return (appModel, project, call)
    }

    @MainActor
    private func waitForMessages(_ appModel: AppModel) async throws {
        let deadline = Date().addingTimeInterval(5)
        while appModel.chats[0].messages.isEmpty && Date() < deadline {
            try await Task.sleep(nanoseconds: 20_000_000)
        }
    }

    @MainActor
    func testApprovePendingBrowserToolCallPersistsOriginGrantAndRunsTheCall() async throws {
        let (appModel, project, call) = approveHarnessModel()

        let result = appModel.approvePendingBrowserToolCall(id: call.id, alwaysAllowOrigin: true)
        XCTAssertEqual(result, .approved)
        XCTAssertNil(appModel.pendingToolCall, "approval consumes the pending call")

        let persisted = try XCTUnwrap(appModel.projects.first { $0.id == project.id })
        XCTAssertEqual(persisted.permissions.browserOriginAllowlist, ["https://example.com"])

        // The approved call dispatches even though no browser runtime is
        // attached; its turn lands in the chat the call was proposed in.
        try await waitForMessages(appModel)
        XCTAssertFalse(appModel.chats[0].messages.isEmpty)
    }

    @MainActor
    func testApprovingWhileAGenerationRunsOrWithAMismatchedIdIsUnavailableAndMutatesNothing() {
        let (appModel, _, call) = approveHarnessModel()
        appModel.generating = true

        XCTAssertEqual(
            appModel.approvePendingBrowserToolCall(id: call.id, alwaysAllowOrigin: true),
            .unavailable)
        XCTAssertNotNil(appModel.pendingToolCall)
        XCTAssertEqual(appModel.projects[0].permissions.browserOriginAllowlist, [])

        appModel.generating = false
        XCTAssertEqual(
            appModel.approvePendingBrowserToolCall(id: UUID(), alwaysAllowOrigin: true),
            .unavailable)
        XCTAssertNotNil(appModel.pendingToolCall, "a mismatched id must not consume the pending call")
        XCTAssertEqual(appModel.projects[0].permissions.browserOriginAllowlist, [])
    }

    @MainActor
    func testAGrantRefusalFailClosesWithoutPersistingOrExecuting() {
        let saturated = (0..<BrowserPermissionRuleStore.maximumGrantCount)
            .map { "https://site\($0).example" }
        let (appModel, project, call) = approveHarnessModel(
            permissions: AppProjectPermissions(
                browser: .allow,
                browserOriginAllowlist: saturated))

        let result = appModel.approvePendingBrowserToolCall(id: call.id, alwaysAllowOrigin: true)
        XCTAssertEqual(result, .grantRefused(.limitReached))
        XCTAssertNotNil(appModel.pendingToolCall, "a refused grant must not consume the pending call")
        let persisted = appModel.projects.first { $0.id == project.id }
        XCTAssertEqual(persisted?.permissions.browserOriginAllowlist, saturated)
        XCTAssertEqual(appModel.pendingToolCallProject?.id, project.id)
    }

    @MainActor
    func testDenyPendingBrowserToolCallIgnoresAMismatchedIdAndDeniesTheMatchingOne() async throws {
        let (appModel, _, call) = approveHarnessModel()

        appModel.denyPendingBrowserToolCall(id: UUID())
        XCTAssertNotNil(appModel.pendingToolCall, "a mismatched id must not deny the pending call")

        appModel.denyPendingBrowserToolCall(id: call.id)
        XCTAssertNil(appModel.pendingToolCall)

        try await waitForMessages(appModel)
        let denial = appModel.chats[0].messages.first?.toolResults.first
        XCTAssertEqual(denial?.callID, call.id)
        XCTAssertEqual(denial?.isError, true)
    }
}
