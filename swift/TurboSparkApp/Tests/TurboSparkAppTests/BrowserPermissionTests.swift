import Foundation
import XCTest
@testable import TurboSparkApp

final class BrowserPermissionTests: XCTestCase {
    private func context(
        _ origin: String,
        owner: BrowserPermissionOwner = .agent,
        currentActionApproved: Bool = false
    ) -> BrowserPermissionContext {
        BrowserPermissionContext(
            origin: BrowserOrigin(origin: origin),
            owner: owner,
            currentActionApproved: currentActionApproved
        )
    }

    private func browserCall(
        url: String = "https://docs.example/path",
        riskAssessment: ToolRiskAssessment? = nil
    ) -> AppToolCall {
        AppToolCall(
            name: "browser_navigate",
            arguments: ["url": url],
            category: .browser,
            riskAssessment: riskAssessment
        )
    }

    func testLegacyFullAccessArchiveDefaultsBrowserToAskWithoutOriginGrants() throws {
        let data = Data(#"{"mode":"fullAccess"}"#.utf8)
        let permissions = try JSONDecoder().decode(AppProjectPermissions.self, from: data)

        XCTAssertEqual(permissions.mode, .fullAccess)
        XCTAssertEqual(permissions.browser, .ask)
        XCTAssertEqual(permissions.browserOriginAllowlist, [])

        let project = AppProject(name: "Migrated", permissions: permissions)
        let decision = AppToolPermissionEngine.evaluate(
            call: browserCall(),
            project: project,
            browserContext: context("https://docs.example")
        )
        if case .ask = decision {} else {
            XCTFail("A legacy full-access project without browser state must ask, got \(decision)")
        }
    }

    func testOriginGrantAppliesOnlyToThatOriginInItsProject() {
        var firstPermissions = AppProjectPermissions.standard
        XCTAssertEqual(
            BrowserPermissionRuleStore.grant(origin: "https://docs.example", to: &firstPermissions),
            .added
        )
        let firstProject = AppProject(name: "First", permissions: firstPermissions)
        let secondProject = AppProject(name: "Second", permissions: .standard)
        let originContext = context("https://docs.example")

        XCTAssertEqual(
            AppToolPermissionEngine.evaluate(
                call: browserCall(), project: firstProject, browserContext: originContext),
            .allow
        )
        assertAsks(project: secondProject, origin: originContext)
        assertAsks(project: firstProject, origin: context("https://sub.docs.example"))
        assertAsks(project: firstProject, origin: context("http://docs.example"))
    }

    func testGrantCanBeRevoked() throws {
        var permissions = AppProjectPermissions.standard
        XCTAssertEqual(
            BrowserPermissionRuleStore.grant(origin: "https://docs.example", to: &permissions),
            .added
        )
        let origin = try XCTUnwrap(BrowserOrigin(origin: "https://docs.example"))

        XCTAssertTrue(BrowserPermissionRuleStore.revoke(origin: origin, from: &permissions))
        XCTAssertFalse(BrowserPermissionRuleStore.revoke(origin: origin, from: &permissions))
        XCTAssertTrue(permissions.browserOriginAllowlist.isEmpty)
        assertAsks(
            project: AppProject(name: "Revoked", permissions: permissions),
            origin: context("https://docs.example")
        )
    }

    func testMalformedOriginsAreRejectedAndNotPersisted() {
        var permissions = AppProjectPermissions.standard

        XCTAssertEqual(
            BrowserPermissionRuleStore.grant(origin: "javascript:alert(1)", to: &permissions),
            .invalidOrigin
        )
        XCTAssertEqual(
            BrowserPermissionRuleStore.grant(origin: "https://docs.example/path", to: &permissions),
            .invalidOrigin
        )
        XCTAssertEqual(
            BrowserPermissionRuleStore.grant(origin: "https://user@docs.example", to: &permissions),
            .invalidOrigin
        )
        XCTAssertTrue(permissions.browserOriginAllowlist.isEmpty)
    }

    func testGrantLimitRefuses257thGrantAndPreservesExistingEntries() {
        var permissions = AppProjectPermissions.standard
        for index in 0..<BrowserPermissionRuleStore.maximumGrantCount {
            XCTAssertEqual(
                BrowserPermissionRuleStore.grant(
                    origin: "https://site-\(index).example", to: &permissions),
                .added
            )
        }
        let existing = permissions.browserOriginAllowlist

        XCTAssertEqual(
            BrowserPermissionRuleStore.grant(origin: "https://overflow.example", to: &permissions),
            .limitReached
        )
        XCTAssertEqual(permissions.browserOriginAllowlist, existing)
        XCTAssertEqual(permissions.browserOriginAllowlist.count, 256)
    }

    func testCategoryDenyAndHighRiskRemainAuthoritativeForGrantedOrigin() {
        var deniedPermissions = AppProjectPermissions.standard
        deniedPermissions.browser = .deny
        XCTAssertEqual(
            BrowserPermissionRuleStore.grant(origin: "https://docs.example", to: &deniedPermissions),
            .added
        )
        let deniedProject = AppProject(name: "Denied", permissions: deniedPermissions)
        let grantedContext = context("https://docs.example")
        if case .deny = AppToolPermissionEngine.evaluate(
            call: browserCall(), project: deniedProject, browserContext: grantedContext
        ) {} else {
            XCTFail("An origin grant must not override the browser category deny")
        }

        var allowedPermissions = AppProjectPermissions.fullAccess
        XCTAssertEqual(
            BrowserPermissionRuleStore.grant(origin: "http://169.254.169.254", to: &allowedPermissions),
            .added
        )
        let highRisk = ToolRiskClassifier.assessRisk(
            name: "WebFetch",
            arguments: ["url": "http://169.254.169.254/latest/meta-data"]
        )
        let highRiskDecision = AppToolPermissionEngine.evaluate(
            call: browserCall(riskAssessment: highRisk),
            project: AppProject(name: "Risky", permissions: allowedPermissions),
            browserContext: context("http://169.254.169.254")
        )
        if case .ask(let assessment, _) = highRiskDecision {
            XCTAssertTrue(assessment.isHighRisk)
        } else {
            XCTFail("A project grant must not override the high-risk gate, got \(highRiskDecision)")
        }

        let agentHighRiskDecision = AppToolPermissionEngine.evaluate(
            call: browserCall(riskAssessment: highRisk),
            project: AppProject(name: "Risky agent", permissions: .agent),
            browserContext: context("http://169.254.169.254")
        )
        if case .ask(let assessment, _) = agentHighRiskDecision {
            XCTAssertTrue(assessment.isHighRisk)
            XCTAssertTrue(assessment.isHardGated)
        } else {
            XCTFail("Agent mode must hard-gate a high-risk browser action, got \(agentHighRiskDecision)")
        }

        var readOnlyPermissions = AppProjectPermissions.readOnly
        readOnlyPermissions.browser = .allow
        let readOnlyDecision = AppToolPermissionEngine.evaluate(
            call: browserCall(),
            project: AppProject(name: "Read only", permissions: readOnlyPermissions),
            sessionApproved: true,
            browserContext: context("https://docs.example", currentActionApproved: true)
        )
        if case .deny = readOnlyDecision {} else {
            XCTFail("Read-only mode must deny browser calls before typed approvals")
        }
    }

    func testGrantedPrivateBrowserOriginIsHighRiskWithoutSuppliedAssessment() {
        var permissions = AppProjectPermissions.fullAccess
        XCTAssertEqual(
            BrowserPermissionRuleStore.grant(origin: "http://169.254.169.254", to: &permissions),
            .added
        )

        let decision = AppToolPermissionEngine.evaluate(
            call: browserCall(url: "http://169.254.169.254/latest/meta-data"),
            project: AppProject(name: "Private browser destination", permissions: permissions),
            browserContext: context("http://169.254.169.254")
        )
        if case .ask(let assessment, _) = decision {
            XCTAssertTrue(assessment.isHighRisk)
        } else {
            XCTFail("An exact origin grant must not bypass private-network risk, got \(decision)")
        }
    }

    func testCurrentActionApprovalDoesNotGrantAnotherOrigin() {
        let project = AppProject(name: "One-time", permissions: .fullAccess)
        let approvedContext = context("https://docs.example", currentActionApproved: true)

        XCTAssertEqual(
            AppToolPermissionEngine.evaluate(
                call: browserCall(), project: project, browserContext: approvedContext),
            .allow
        )
        assertAsks(project: project, origin: context("https://other.example"))
        XCTAssertTrue(project.permissions.browserOriginAllowlist.isEmpty)
    }

    func testGenericSessionApprovalDoesNotAuthorizeAnUnlistedBrowserOrigin() {
        let project = AppProject(name: "Agent", permissions: .agent)
        let decision = AppToolPermissionEngine.evaluate(
            call: browserCall(),
            project: project,
            sessionApproved: true,
            browserContext: context("https://docs.example")
        )

        if case .ask(let assessment, _) = decision {
            XCTAssertTrue(assessment.isHardGated)
        } else {
            XCTFail("A generic session approval must not authorize a browser origin, got \(decision)")
        }
    }

    func testAgentModeBrowserAskIsHardGatedBeforeClassifierFallback() {
        let project = AppProject(name: "Agent", permissions: .agent)
        let decision = AppToolPermissionEngine.evaluate(
            call: browserCall(),
            project: project,
            browserContext: context("https://docs.example")
        )

        if case .ask(let assessment, _) = decision {
            XCTAssertTrue(assessment.isHardGated)
        } else {
            XCTFail("An ungranted browser origin in Agent mode must ask directly, got \(decision)")
        }
    }

    func testBrowserClassifierProjectionContainsOnlyToolName() {
        let call = AppToolCall(
            name: "browser_type",
            arguments: [
                "url": "https://docs.example/path?token=secret",
                "text": "private form value"
            ],
            category: .browser
        )

        XCTAssertEqual(ToolCallProjection.project(call), "browser action: browser_type")
    }

    func testUserOwnedNavigationBypassesAgentPermissionReadOnlyAndRiskGates() {
        let privateOrigin = "http://169.254.169.254"
        let standardProject = AppProject(name: "User", permissions: .standard)
        XCTAssertEqual(
            AppToolPermissionEngine.evaluate(
                call: browserCall(url: "\(privateOrigin)/latest/meta-data"),
                project: standardProject,
                browserContext: context(privateOrigin, owner: .user)
            ),
            .allow
        )

        let explicitHighRisk = ToolRiskAssessment(
            level: .high,
            category: .browser,
            reasons: ["private browser destination"],
            hardGated: true
        )
        XCTAssertEqual(
            AppToolPermissionEngine.evaluate(
                call: browserCall(
                    url: "\(privateOrigin)/latest/meta-data",
                    riskAssessment: explicitHighRisk
                ),
                project: standardProject,
                browserContext: context(privateOrigin, owner: .user)
            ),
            .allow
        )

        var deniedPermissions = AppProjectPermissions.standard
        deniedPermissions.browser = .deny
        let deniedDecision = AppToolPermissionEngine.evaluate(
            call: browserCall(),
            project: AppProject(name: "Denied user", permissions: deniedPermissions),
            browserContext: context("https://docs.example", owner: .user)
        )
        XCTAssertEqual(deniedDecision, .allow)

        var readOnlyPermissions = AppProjectPermissions.readOnly
        readOnlyPermissions.browser = .allow
        let readOnlyDecision = AppToolPermissionEngine.evaluate(
            call: browserCall(),
            project: AppProject(name: "Read-only user", permissions: readOnlyPermissions),
            browserContext: context("https://docs.example", owner: .user)
        )
        XCTAssertEqual(readOnlyDecision, .allow)
    }

    func testPermissionPresetsKeepBrowserDefaultsAndEmptyAllowlist() {
        for permissions in [
            AppProjectPermissions.auto,
            .standard,
            .agent,
            .alwaysAsk
        ] {
            XCTAssertEqual(permissions.browser, .ask)
            XCTAssertTrue(permissions.browserOriginAllowlist.isEmpty)
        }
        XCTAssertEqual(AppProjectPermissions.readOnly.browser, .deny)
        XCTAssertEqual(AppProjectPermissions.permissive.browser, .allow)
        XCTAssertEqual(AppProjectPermissions.fullAccess.browser, .allow)
        XCTAssertTrue(AppProjectPermissions.permissive.browserOriginAllowlist.isEmpty)
        XCTAssertTrue(AppProjectPermissions.fullAccess.browserOriginAllowlist.isEmpty)
    }

    func testCanonicalOriginNormalizesSchemeHostAndDefaultPort() throws {
        let implicitPort = try XCTUnwrap(BrowserOrigin(origin: "HTTPS://Docs.Example"))
        let explicitPort = try XCTUnwrap(BrowserOrigin(origin: "https://docs.example:443"))
        let ipv6ImplicitPort = try XCTUnwrap(BrowserOrigin(origin: "https://[::1]"))
        let ipv6ExplicitPort = try XCTUnwrap(BrowserOrigin(origin: "https://[::1]:443"))

        XCTAssertEqual(implicitPort, explicitPort)
        XCTAssertEqual(implicitPort.canonicalString, "https://docs.example")
        XCTAssertEqual(ipv6ImplicitPort, ipv6ExplicitPort)
        XCTAssertEqual(ipv6ImplicitPort.canonicalString, "https://[::1]")
    }

    private func assertAsks(project: AppProject, origin: BrowserPermissionContext, file: StaticString = #filePath, line: UInt = #line) {
        let decision = AppToolPermissionEngine.evaluate(
            call: browserCall(), project: project, browserContext: origin)
        if case .ask = decision {} else {
            XCTFail("Expected browser permission to ask, got \(decision)", file: file, line: line)
        }
    }
}
