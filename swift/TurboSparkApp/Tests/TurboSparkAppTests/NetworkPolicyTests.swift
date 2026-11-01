import XCTest
@testable import TurboSparkApp

/// Settings > Network: the policy table, its place in the permission engine,
/// the executor-side checks, and the Seatbelt profile for Block all.
final class NetworkPolicyTests: NetworkPolicyPinnedTestCase {
    override var pinnedNetworkPolicy: AppNetworkPolicy { AppNetworkPolicy() }

    private func isAllow(_ d: AppNetworkDecision) -> Bool { d == .allow }
    private func isAsk(_ d: AppNetworkDecision) -> Bool { if case .ask = d { return true }; return false }
    private func isDeny(_ d: AppNetworkDecision) -> Bool { if case .deny = d { return true }; return false }

    // MARK: - Host scopes

    func testHostScopes() {
        for host in ["localhost", "LOCALHOST.", "app.localhost", "127.0.0.1", "127.1.2.3", "0177.0.0.1",
                     "2130706433", "::1", "[::1]", "::ffff:127.0.0.1", "0::1"] {
            XCTAssertEqual(AppToolSandbox.hostScope(host), .loopback, host)
        }
        for host in ["169.254.169.254", "169.254.170.2", "metadata.google.internal", "metadata",
                     "instance-data", "fd00:ec2::254", "::ffff:169.254.169.254", "host.docker.internal"] {
            XCTAssertEqual(AppToolSandbox.hostScope(host), .metadata, host)
        }
        for host in ["10.0.0.1", "192.168.1.1", "172.16.0.1", "nas.local", "fe80::1", "fc00::1", "0.0.0.0", "::",
                     "not:an:ipv6"] {
            XCTAssertEqual(AppToolSandbox.hostScope(host), .privateNetwork, host)
        }
        for host in ["example.com", "93.184.216.34", "2606:4700::1111"] {
            XCTAssertEqual(AppToolSandbox.hostScope(host), .publicHost, host)
        }
    }

    // MARK: - Decision table

    func testMetadataIsDeniedInEveryModeWithEverythingAllowed() {
        for mode in AppNetworkMode.allCases {
            let policy = AppNetworkPolicy(
                mode: mode, allowLocalhost: true, allowLocalNetwork: true,
                allowedDomains: ["169.254.169.254", "metadata.google.internal"])
            XCTAssertTrue(isDeny(policy.decision(host: "169.254.169.254")), "\(mode)")
            XCTAssertTrue(isDeny(policy.decision(host: "metadata.google.internal")), "\(mode)")
        }
    }

    func testLoopbackFollowsTheToggleInEveryModeIncludingBlockAll() {
        for mode in AppNetworkMode.allCases {
            let on = AppNetworkPolicy(mode: mode, allowLocalhost: true, deniedDomains: ["localhost"])
            XCTAssertTrue(isAllow(on.decision(host: "localhost")), "\(mode)")
            XCTAssertTrue(isAllow(on.decision(host: "[::1]")), "\(mode)")
            let off = AppNetworkPolicy(mode: mode, allowLocalhost: false)
            XCTAssertTrue(isDeny(off.decision(host: "127.0.0.1")), "\(mode)")
        }
    }

    func testLocalNetworkNeedsItsToggleAndThenFollowsTheMode() {
        XCTAssertTrue(isDeny(AppNetworkPolicy(mode: .allowAll).decision(host: "192.168.1.10")))
        let lan = AppNetworkPolicy(mode: .allowAll, allowLocalNetwork: true)
        XCTAssertTrue(isAllow(lan.decision(host: "192.168.1.10")))
        let blocked = AppNetworkPolicy(mode: .blockAll, allowLocalNetwork: true)
        XCTAssertTrue(isDeny(blocked.decision(host: "192.168.1.10")))
        let asked = AppNetworkPolicy(mode: .ask, allowLocalNetwork: true)
        XCTAssertTrue(isAsk(asked.decision(host: "nas.local")))
    }

    func testModesOnPublicHosts() {
        XCTAssertTrue(isAsk(AppNetworkPolicy(mode: .ask).decision(host: "example.com")))
        XCTAssertTrue(isAllow(AppNetworkPolicy(mode: .ask, allowedDomains: ["example.com"]).decision(host: "example.com")))
        XCTAssertTrue(isAllow(AppNetworkPolicy(mode: .allowAll).decision(host: "example.com")))
        XCTAssertTrue(isDeny(AppNetworkPolicy(mode: .blockAll, allowedDomains: ["example.com"]).decision(host: "example.com")))
        XCTAssertTrue(isAsk(AppNetworkPolicy(mode: .custom).decision(host: "example.com")))
        XCTAssertTrue(isDeny(AppNetworkPolicy(mode: .custom, customUnlisted: .deny).decision(host: "example.com")))
        XCTAssertTrue(isAllow(AppNetworkPolicy(mode: .custom, allowedDomains: ["example.com"], customUnlisted: .deny)
            .decision(host: "api.example.com")))
    }

    func testDenylistBeatsAllowlist() {
        let policy = AppNetworkPolicy(mode: .allowAll, allowedDomains: ["example.com"], deniedDomains: ["example.com"])
        XCTAssertTrue(isDeny(policy.decision(host: "www.example.com")))
    }

    func testDomainMatching() {
        XCTAssertTrue(AppNetworkPolicy.matches("example.com", ["example.com"]))
        XCTAssertTrue(AppNetworkPolicy.matches("a.b.example.com", ["example.com"]))
        XCTAssertFalse(AppNetworkPolicy.matches("notexample.com", ["example.com"]))
        XCTAssertFalse(AppNetworkPolicy.matches("example.com", ["*.example.com"]))
        XCTAssertTrue(AppNetworkPolicy.matches("api.example.com", ["*.example.com"]))
        XCTAssertTrue(isAllow(AppNetworkPolicy(mode: .ask, allowedDomains: ["example.com"]).decision(host: "EXAMPLE.com.")))
    }

    func testReadOnlyMethodsApplyInCustomModeOnly() {
        let custom = AppNetworkPolicy(mode: .custom, allowedDomains: ["example.com"], readOnlyMethods: true)
        XCTAssertTrue(isAllow(custom.decision(host: "example.com", method: "get")))
        XCTAssertTrue(isAllow(custom.decision(host: "example.com", method: "HEAD")))
        XCTAssertTrue(isAllow(custom.decision(host: "example.com")))
        XCTAssertTrue(isDeny(custom.decision(host: "example.com", method: "POST")))
        let allowAll = AppNetworkPolicy(mode: .allowAll, readOnlyMethods: true)
        XCTAssertTrue(isAllow(allowAll.decision(host: "example.com", method: "POST")))
    }

    func testDomainEntryValidation() {
        XCTAssertEqual(AppNetworkPolicy.normalizedDomainEntry(" Example.COM. "), "example.com")
        XCTAssertEqual(AppNetworkPolicy.normalizedDomainEntry("*.example.com"), "*.example.com")
        XCTAssertEqual(AppNetworkPolicy.normalizedDomainEntry("10.0.0.5"), "10.0.0.5")
        for bad in ["", "https://example.com", "example.com/path", "example.com:8080", "ex ample.com",
                    "a.*.example.com", "*", "*.", "..com", "user@example.com"] {
            XCTAssertNil(AppNetworkPolicy.normalizedDomainEntry(bad), bad)
        }
    }

    func testPublicNameNeverEarnsTheLocalConnectionException() {
        let policy = AppNetworkPolicy(allowLocalhost: true, allowLocalNetwork: true)
        XCTAssertFalse(policy.permitsLocalConnection(urlHost: "evil.example", address: "127.0.0.1"))
        XCTAssertTrue(policy.permitsLocalConnection(urlHost: "localhost", address: "::1"))
        XCTAssertFalse(policy.permitsLocalConnection(urlHost: "localhost", address: "192.168.1.1"))
        XCTAssertFalse(policy.permitsLocalConnection(urlHost: "nas.local", address: "169.254.169.254"))
        XCTAssertFalse(policy.permitsLocalConnection(urlHost: "localhost", address: ""))
        XCTAssertFalse(AppNetworkPolicy(allowLocalhost: false)
            .permitsLocalConnection(urlHost: "localhost", address: "127.0.0.1"))
    }

    // MARK: - Persistence

    func testSettingsWithoutANetworkKeyDecodeToTheDefault() throws {
        let settings = try JSONDecoder().decode(MacAppSettings.self, from: Data("{}".utf8))
        XCTAssertEqual(settings.networkPolicy, AppNetworkPolicy())
        XCTAssertEqual(settings.networkPolicy.mode, .ask)
        XCTAssertTrue(settings.networkPolicy.allowLocalhost)
    }

    func testPolicyRoundTripsAndRejectsBadEntries() throws {
        let policy = AppNetworkPolicy(mode: .custom, allowedDomains: ["a.com"], deniedDomains: ["b.com"],
                                      customUnlisted: .deny, readOnlyMethods: true)
        let data = try JSONEncoder().encode(policy)
        XCTAssertEqual(try JSONDecoder().decode(AppNetworkPolicy.self, from: data), policy)

        let edited = #"{"mode":"custom","customUnlisted":"allow","allowedDomains":["ok.com","https://bad.com/x"]}"#
        let decoded = try JSONDecoder().decode(AppNetworkPolicy.self, from: Data(edited.utf8))
        XCTAssertEqual(decoded.customUnlisted, .ask, "an edited file must not turn Custom into Allow all")
        XCTAssertEqual(decoded.allowedDomains, ["ok.com"])
    }

    // MARK: - Permission engine

    private func fetch(_ url: String) -> AppToolCall {
        AppToolCall(name: "web_fetch", arguments: ["url": url], category: .web)
    }

    private func project(_ mode: AppPermissionMode) -> AppProject {
        AppProject(name: "net-\(mode.rawValue)", permissions: AppProjectPermissions.preset(for: mode))
    }

    func testBlockAllDeniesWebEvenInFullAccess() {
        let decision = AppToolPermissionEngine.evaluate(
            call: fetch("https://example.com"), project: project(.fullAccess),
            networkPolicy: AppNetworkPolicy(mode: .blockAll))
        guard case .deny = decision else { return XCTFail("got \(decision)") }

        let search = AppToolCall(name: "web_search", arguments: ["query": "swift"], category: .web)
        let searchDecision = AppToolPermissionEngine.evaluate(
            call: search, project: project(.fullAccess), networkPolicy: AppNetworkPolicy(mode: .blockAll))
        guard case .deny = searchDecision else { return XCTFail("got \(searchDecision)") }
    }

    func testAskTurnsAFullAccessAllowIntoAHardGatedAsk() {
        let decision = AppToolPermissionEngine.evaluate(
            call: fetch("https://example.com"), project: project(.fullAccess),
            networkPolicy: AppNetworkPolicy(mode: .ask))
        guard case .ask(let assessment, _) = decision else { return XCTFail("got \(decision)") }
        XCTAssertTrue(assessment.isHardGated)
    }

    func testHostApprovalLiftsAnAskButNotADeny() {
        let ask = AppToolPermissionEngine.evaluate(
            call: fetch("https://example.com"), project: project(.auto),
            networkPolicy: AppNetworkPolicy(mode: .ask), networkHostApproved: true)
        XCTAssertEqual(ask, .allow)

        let deny = AppToolPermissionEngine.evaluate(
            call: fetch("https://example.com"), project: project(.fullAccess),
            networkPolicy: AppNetworkPolicy(mode: .allowAll, deniedDomains: ["example.com"]),
            networkHostApproved: true)
        guard case .deny = deny else { return XCTFail("got \(deny)") }
    }

    func testAToolNameSessionGrantDoesNotCoverANewHost() {
        let decision = AppToolPermissionEngine.evaluate(
            call: fetch("https://other.example"), project: project(.auto), sessionApproved: true,
            networkPolicy: AppNetworkPolicy(mode: .ask))
        guard case .ask = decision else { return XCTFail("got \(decision)") }
    }

    func testProjectWebDenyStillDeniesUnderAllowAll() {
        var perms = AppProjectPermissions.standard
        perms.web = .deny
        let decision = AppToolPermissionEngine.evaluate(
            call: fetch("https://example.com"), project: AppProject(name: "deny-web", permissions: perms),
            networkPolicy: AppNetworkPolicy(mode: .allowAll))
        guard case .deny = decision else { return XCTFail("got \(decision)") }
    }

    func testLocalhostIsAnOrdinaryLowRiskCallWhenAllowed() {
        let risk = ToolRiskClassifier.assessRisk(name: "web_fetch", arguments: ["url": "http://localhost:3000/api"])
        XCTAssertFalse(risk.isHighRisk)
        let decision = AppToolPermissionEngine.evaluate(
            call: fetch("http://localhost:3000/api"), project: project(.auto),
            networkPolicy: AppNetworkPolicy(mode: .ask))
        XCTAssertEqual(decision, .allow)
    }

    // MARK: - Executors

    func testValidatorFollowsTheLocalhostToggle() throws {
        let url = try XCTUnwrap(URL(string: "http://localhost:9/"))
        XCTAssertNoThrow(try HttpRequestDestinationValidator.validate(url, policy: AppNetworkPolicy()))
        XCTAssertThrowsError(try HttpRequestDestinationValidator.validate(
            url, policy: AppNetworkPolicy(allowLocalhost: false)))
        let metadata = try XCTUnwrap(URL(string: "http://169.254.169.254/"))
        XCTAssertThrowsError(try HttpRequestDestinationValidator.validate(
            metadata, policy: AppNetworkPolicy(mode: .allowAll, allowLocalNetwork: true)))
    }

    func testHttpRequestReachesALocalServerOnlyWithLocalhostAllowed() async throws {
        let server = try BrowserAutomationHTTPFixtureServer()
        let url = server.url("/ok").absoluteString

        AppNetworkPolicyStore.current = AppNetworkPolicy(allowLocalhost: true)
        let output = try await HttpRequestExecutor.execute(url: url)
        XCTAssertTrue(output.contains("200"), output)

        AppNetworkPolicyStore.current = AppNetworkPolicy(allowLocalhost: false)
        do {
            _ = try await HttpRequestExecutor.execute(url: url)
            XCTFail("localhost must be refused when the toggle is off")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("Localhost access is turned off"),
                          error.localizedDescription)
        }
    }

    func testReadOnlyMethodsRefuseAWriteBeforeSendingIt() async throws {
        AppNetworkPolicyStore.current = AppNetworkPolicy(
            mode: .custom, allowedDomains: ["example.com"], readOnlyMethods: true)
        do {
            _ = try await HttpRequestExecutor.execute(url: "https://example.com/", method: "POST", body: "{}")
            XCTFail("POST must be refused")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("read-only"), error.localizedDescription)
        }
    }

    func testBlockAllRefusesAProviderSearchAtExecution() async {
        AppNetworkPolicyStore.current = AppNetworkPolicy(mode: .blockAll)
        // Another suite can leave the composer's web toggle provider set.
        AppToolRegistry.webToolsEnabledProvider = nil
        let call = AppToolCall(name: "web_search", arguments: ["query": "swift"], category: .web)
        let result = await AppToolRegistry.execute(call: call, in: nil)
        XCTAssertTrue(result.isError)
        XCTAssertTrue(result.output.contains("Block all"), result.output)
    }

    // MARK: - Seatbelt

    func testLaunchIsUnchangedOutsideBlockAll() throws {
        let zsh = URL(fileURLWithPath: "/bin/zsh")
        for mode in [AppNetworkMode.ask, .allowAll, .custom] {
            let launch = try ShellNetworkSandbox.launch(
                executable: zsh, arguments: ["-c", "true"], policy: AppNetworkPolicy(mode: mode))
            XCTAssertEqual(launch.executable, zsh)
            XCTAssertEqual(launch.arguments, ["-c", "true"])
        }
        let blocked = try ShellNetworkSandbox.launch(
            executable: zsh, arguments: ["-c", "true"], policy: AppNetworkPolicy(mode: .blockAll))
        XCTAssertEqual(blocked.executable.path, ShellNetworkSandbox.sandboxExecPath)
        XCTAssertEqual(Array(blocked.arguments.suffix(3)), ["/bin/zsh", "-c", "true"])
    }

    /// `(allow network* (local ip "localhost:*"))` let `curl http://1.1.1.1`
    /// straight through when this profile was first drafted.
    func testProfileNeverAllowsAllNetworkOnALocalAddress() {
        let profile = ShellNetworkSandbox.profile(allowLocalhost: true)
        XCTAssertFalse(profile.contains("(allow network* (local"), profile)
        XCTAssertTrue(profile.contains("(deny network*)"))
        XCTAssertFalse(ShellNetworkSandbox.profile(allowLocalhost: false).contains("allow network"))
    }

    private func runSandboxed(_ command: String, allowLocalhost: Bool) throws -> Int32 {
        let launch = try ShellNetworkSandbox.launch(
            executable: URL(fileURLWithPath: "/bin/zsh"), arguments: ["-c", command],
            policy: AppNetworkPolicy(mode: .blockAll, allowLocalhost: allowLocalhost))
        let process = Process()
        process.executableURL = launch.executable
        process.arguments = launch.arguments
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        try process.run()
        process.waitUntilExit()
        return process.terminationStatus
    }

    func testSandboxedShellReachesLoopbackOnlyWhenAllowed() throws {
        let server = try BrowserAutomationHTTPFixtureServer()
        let curl = "curl --noproxy '*' -sf -m 5 -o /dev/null '\(server.url("/ok").absoluteString)'"
        XCTAssertEqual(try runSandboxed(curl, allowLocalhost: true), 0)
        XCTAssertNotEqual(try runSandboxed(curl, allowLocalhost: false), 0)
        XCTAssertEqual(try runSandboxed("echo ok", allowLocalhost: false), 0)
    }

    func testSandboxedShellCannotReachAPublicAddress() throws {
        let curl = "curl --noproxy '*' -s -m 5 -o /dev/null http://1.1.1.1/"
        let control = Process()
        control.executableURL = URL(fileURLWithPath: "/bin/zsh")
        control.arguments = ["-c", curl]
        try control.run()
        control.waitUntilExit()
        guard control.terminationStatus == 0 else {
            throw XCTSkip("no outbound network on this host, so the sandbox cannot be shown to cut it")
        }
        XCTAssertNotEqual(try runSandboxed(curl, allowLocalhost: true), 0)
    }
}

extension NetworkPolicyTests {
    func testAgentBrowserNavigationToMetadataIsDeniedButUserNavigationIsNot() {
        var permissions = AppProjectPermissions.preset(for: .fullAccess)
        _ = BrowserPermissionRuleStore.grant(origin: "http://169.254.169.254", to: &permissions)
        let project = AppProject(name: "browser-metadata", permissions: permissions)
        let call = AppToolCall(
            name: "browser_navigate", arguments: ["url": "http://169.254.169.254/latest/meta-data"],
            category: .browser)
        let origin = BrowserOrigin(origin: "http://169.254.169.254")

        let agent = AppToolPermissionEngine.evaluate(
            call: call, project: project,
            browserContext: BrowserPermissionContext(origin: origin, owner: .agent))
        guard case .deny = agent else { return XCTFail("got \(agent)") }

        let user = AppToolPermissionEngine.evaluate(
            call: call, project: project,
            browserContext: BrowserPermissionContext(origin: origin, owner: .user))
        XCTAssertEqual(user, .allow)
    }
}
