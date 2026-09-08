import XCTest

@testable import TurboSparkApp

/// Agent mode (`swift/docs/SWIFT_AGENT_MODE.md`): the classifier contract
/// (parse, projections, hints), the fallback counters, the routing rules,
/// and the seams the mode adds to the engine, the risk assessment and the
/// subagent gate.
///
/// The classifier itself is never live in these tests: verdict routing runs
/// through `agentModeClassifierOverride`, and the subagent cases exercise
/// the nil-session (unavailable) path. Every counter case resets the shared
/// `AgentModeGate` for its session key first -- it is a process-wide
/// singleton, exactly like `SessionApprovalStore`.
final class AgentModeTests: XCTestCase {
    // MARK: - Verdict parsing

    func testAPlainAllowObjectParses() {
        XCTAssertEqual(
            LocalModelToolClassifier.parseVerdict(from: #"{"verdict":"allow"}"#),
            .allow)
    }

    func testAFencedBlockObjectParses() {
        let fenced = "```json\n{\"verdict\":\"block\",\"reason\":\"force push\"}\n```"
        XCTAssertEqual(
            LocalModelToolClassifier.parseVerdict(from: fenced),
            .block(reason: "force push"))
    }

    func testProseAroundTheObjectIsTolerated() {
        let noisy = "Sure! {\"verdict\":\"allow\"} hope that helps"
        XCTAssertEqual(LocalModelToolClassifier.parseVerdict(from: noisy), .allow)
    }

    func testAnUnknownVerdictIsNotAVerdict() {
        // Fail closed: nil means the caller retries once and then asks the
        // user. It must never mean allow.
        XCTAssertNil(
            LocalModelToolClassifier.parseVerdict(from: #"{"verdict":"maybe"}"#))
    }

    func testEmptyTextIsNotAVerdict() {
        XCTAssertNil(LocalModelToolClassifier.parseVerdict(from: ""))
    }

    func testABlockWithoutAReasonStillStatesOne() {
        let verdict = LocalModelToolClassifier.parseVerdict(from: #"{"verdict":"block"}"#)
        guard case .block(let reason) = verdict else {
            return XCTFail("expected block, got \(String(describing: verdict))")
        }
        XCTAssertFalse(reason.isEmpty, "a reasonless block leaves the sheet unexplained")
    }

    func testAnOverlongReasonIsClipped() {
        let long = String(repeating: "x", count: 5_000)
        let json = "{\"verdict\":\"block\",\"reason\":\"" + long + "\"}"
        let verdict = LocalModelToolClassifier.parseVerdict(from: json)
        guard case .block(let reason) = verdict else {
            return XCTFail("expected block, got \(String(describing: verdict))")
        }
        XCTAssertLessThanOrEqual(reason.utf8.count, 400)
    }

    // MARK: - Hint normalization

    func testHintCapsApplyAtPromptTime() {
        var hints = AgentModeHints(
            allow: (0..<60).map { "allow \($0)" },
            softDeny: [String(repeating: "s", count: 500)],
            hardDeny: ["", "   ", "real"],
            environment: (0..<25).map { "env \($0)" })
        hints = hints.normalized
        XCTAssertEqual(hints.allow.count, AgentModeHints.maxHintEntries)
        XCTAssertEqual(hints.softDeny.count, 1)
        XCTAssertEqual(
            hints.softDeny[0].utf8.count, AgentModeHints.maxEntryCharacters,
            "an over-long entry is clipped, not dropped")
        XCTAssertEqual(hints.hardDeny, ["real"], "blank entries are dropped")
        XCTAssertEqual(hints.environment.count, AgentModeHints.maxEnvironmentEntries)
    }

    // MARK: - Projections

    private func call(
        _ name: String, _ arguments: [String: String], _ category: AppToolCategory
    ) -> AppToolCall {
        AppToolCall(name: name, arguments: arguments, category: category)
    }

    func testAShellCommandIsForwardedInFull() {
        let projection = ToolCallProjection.projectedCall(
            call("run_command", ["command": "git push origin main"], .terminal))
        XCTAssertTrue(projection.contains("git push origin main"))
    }

    func testAWriteProjectionCarriesThePathAndShortPreviews() {
        let longContent = String(repeating: "a", count: 1_000)
        let projection = ToolCallProjection.project(
            call("write_file", ["path": "src/main.rs", "content": longContent], .fileWrite))
        XCTAssertTrue(projection.contains("src/main.rs"))
        XCTAssertFalse(projection.contains(String(repeating: "a", count: 400)),
            "content previews must be capped at \(ToolCallProjection.contentPreviewCharacters)")
    }

    func testAWebProjectionCarriesTheURLAndNotThePrompt() {
        let projection = ToolCallProjection.project(
            call("webfetch", ["url": "https://example.com", "prompt": "summarize this"], .web))
        XCTAssertTrue(projection.contains("https://example.com"))
        XCTAssertFalse(
            projection.contains("summarize this"),
            "the prompt field is not forwarded, qwen-code's rule")
    }

    func testMCPArgumentsAreBoundedAndMarked() {
        let bigValue = String(repeating: "v", count: 3_000)
        let projection = ToolCallProjection.project(
            call("mcp__fs__write", ["text": bigValue, "path": "x"], .mcp))
        XCTAssertTrue(projection.contains("[truncated"), "a cut must be marked in place")
        XCTAssertTrue(projection.contains("mcp server: fs"))
        XCTAssertTrue(projection.contains("mcp tool: write"))
    }

    // MARK: - Fallback counters

    private func resetGate(_ sessionID: String) async {
        await AgentModeGate.shared.reset(sessionID: sessionID)
    }

    func testThreeConsecutiveBlocksSkipTheClassifier() async {
        let session = "test-blocks"
        await resetGate(session)
        // Counts are hardcoded ON PURPOSE: deriving the loop from the
        // constant makes the test self-relative, and lowering the constant
        // to 2 would still pass it.
        await AgentModeGate.shared.recordBlock(sessionID: session)
        await AgentModeGate.shared.recordBlock(sessionID: session)
        let afterTwo = await AgentModeGate.shared.shouldSkipClassifier(sessionID: session)
        await AgentModeGate.shared.recordBlock(sessionID: session)
        let afterThree = await AgentModeGate.shared.shouldSkipClassifier(sessionID: session)
        XCTAssertFalse(afterTwo)
        XCTAssertTrue(afterThree, "the third consecutive block must skip the classifier")
    }

    func testAnAllowVerdictBreaksBothStreaks() async {
        let session = "test-allow-resets"
        await resetGate(session)
        // The streak must be AT the threshold first: a reset check below
        // it passes with or without the reset, which is exactly how the
        // first version of this test survived its mutation.
        for _ in 0..<AgentModeGate.maxConsecutiveBlocks {
            await AgentModeGate.shared.recordBlock(sessionID: session)
        }
        let skipping = await AgentModeGate.shared.shouldSkipClassifier(sessionID: session)
        await AgentModeGate.shared.recordAllow(sessionID: session)
        let after = await AgentModeGate.shared.shouldSkipClassifier(sessionID: session)
        XCTAssertTrue(skipping, "precondition: the streak had reached the skip threshold")
        XCTAssertFalse(after, "an allow resets both counters")
    }

    func testTwoConsecutiveUnavailableSkipTheClassifier() async {
        let session = "test-unavailable"
        await resetGate(session)
        await AgentModeGate.shared.recordUnavailable(sessionID: session)
        let first = await AgentModeGate.shared.shouldSkipClassifier(sessionID: session)
        await AgentModeGate.shared.recordUnavailable(sessionID: session)
        let second = await AgentModeGate.shared.shouldSkipClassifier(sessionID: session)
        XCTAssertFalse(first, "the FIRST unavailable already asks; it does not skip")
        XCTAssertTrue(second)
        let reason = await AgentModeGate.shared.skipReason(sessionID: session)
        XCTAssertTrue(reason?.contains("unavailable") ?? false)
    }

    func testRejectingAFallbackPreservesTheStreak() async {
        let session = "test-reject-preserves"
        await resetGate(session)
        for _ in 0..<AgentModeGate.maxConsecutiveUnavailable {
            await AgentModeGate.shared.recordUnavailable(sessionID: session)
        }
        let stillSkipping = await AgentModeGate.shared.shouldSkipClassifier(sessionID: session)
        XCTAssertTrue(stillSkipping, "only an approval or a mode switch ends the streak")
    }

    func testSuspensionSkipsTheClassifierAndSurvivesUntilReset() async {
        let session = "test-suspend"
        await resetGate(session)
        await AgentModeGate.shared.suspend(sessionID: session)
        let suspended = await AgentModeGate.shared.isSuspended(sessionID: session)
        XCTAssertTrue(suspended)
        await AgentModeGate.shared.reset(sessionID: session)
        let lifted = await AgentModeGate.shared.isSuspended(sessionID: session)
        XCTAssertFalse(lifted, "re-selecting Agent mode is the documented way out")
    }

    // MARK: - Routing rules (static half)

    private func routingDecision(
        _ call: AppToolCall, _ assessment: ToolRiskAssessment,
        suspended: Bool = false, skip: Bool = false
    ) -> AgentModeRouting.PreClassifierDecision {
        AgentModeRouting.preClassifierDecision(
            call: call, assessment: assessment,
            suspended: suspended, skipClassifier: skip)
    }

    func testAHardGatedAskNeverClassifies() {
        let assessment = ToolRiskAssessment(
            level: .high, category: .terminal, reasons: ["denylist"], hardGated: true)
        XCTAssertEqual(
            routingDecision(call("run_command", ["command": "sudo x"], .terminal), assessment),
            .manualCard)
    }

    func testASuspendedSessionNeverClassifies() {
        let assessment = ToolRiskAssessment(level: .low, category: .mcp, reasons: [])
        XCTAssertEqual(
            routingDecision(call("mcp__x__y", [:], .mcp), assessment, suspended: true),
            .manualCard)
    }

    func testMCPCallsAlwaysClassify() {
        // Even a `.low` MCP ask classifies: a tool name plus the server's
        // own annotations is exactly what a classifier exists to weigh.
        let assessment = ToolRiskAssessment(level: .low, category: .mcp, reasons: [])
        XCTAssertEqual(
            routingDecision(call("mcp__x__read", [:], .mcp), assessment),
            .classify)
    }

    func testALowRiskNonMCPAskTakesTheFastPath() {
        let assessment = ToolRiskAssessment(level: .low, category: .terminal, reasons: [])
        XCTAssertEqual(
            routingDecision(call("run_command", ["command": "cargo build"], .terminal), assessment),
            .fastAllow)
    }

    func testAHighRiskUnrecognizedCommandClassifies() {
        // `./deploy.sh` shape: single simple invocation, not on the
        // allowlist, no denylist match -- the population Agent mode exists
        // for.
        let assessment = ToolRiskAssessment(level: .high, category: .terminal, reasons: [])
        XCTAssertEqual(
            routingDecision(call("run_command", ["command": "./deploy.sh"], .terminal), assessment),
            .classify)
    }

    // MARK: - hardGated marking in the risk assessment

    func testADenylistCommandIsHardGated() {
        let assessment = ToolRiskClassifier.assessTerminalCommand("sudo rm /etc/hosts")
        XCTAssertEqual(assessment.level, .high)
        XCTAssertTrue(assessment.isHardGated, "a named denylist match is a deterministic guard")
    }

    func testAnUnrecognizedCommandIsSoft() {
        // The whole classifier band: refused by the allowlist, matched by
        // nothing written down.
        let assessment = ToolRiskClassifier.assessTerminalCommand("./deploy.sh")
        XCTAssertEqual(assessment.level, .high)
        XCTAssertFalse(assessment.isHardGated)
    }

    func testAnAllowlistedBuildCommandIsNotHardGated() {
        let assessment = ToolRiskClassifier.assessTerminalCommand("cargo build --release")
        XCTAssertEqual(assessment.level, .low)
        XCTAssertFalse(assessment.isHardGated)
    }

    func testASensitiveWriteIsHardGated() {
        let assessment = ToolRiskClassifier.assessRisk(
            name: "write_file", arguments: ["path": "/etc/hosts"])
        XCTAssertEqual(assessment.level, .high)
        XCTAssertTrue(assessment.isHardGated)
    }

    func testHardGatedSurvivesAnArchiveRoundTrip() {
        let assessment = ToolRiskAssessment(
            level: .high, category: .terminal, reasons: ["denylist"], hardGated: true)
        let data = try! JSONEncoder().encode(assessment)
        let decoded = try! JSONDecoder().decode(ToolRiskAssessment.self, from: data)
        XCTAssertTrue(decoded.isHardGated)
        // And an archive written before the field existed decodes soft.
        let legacy = try! JSONDecoder().decode(
            ToolRiskAssessment.self,
            from: JSONEncoder().encode(
                ToolRiskAssessment(level: .high, category: .terminal, reasons: ["x"])))
        XCTAssertFalse(legacy.isHardGated)
    }

    // MARK: - Engine seams

    func testAgentModePresetMatchesTheStandardMatrix() {
        let agent = AppProjectPermissions.preset(for: .agentAuto)
        let standard = AppProjectPermissions.preset(for: .auto)
        XCTAssertEqual(agent.mode, .agentAuto)
        XCTAssertEqual(agent.fileRead, standard.fileRead)
        XCTAssertEqual(agent.fileWrite, standard.fileWrite)
        XCTAssertEqual(agent.terminal, standard.terminal)
        XCTAssertEqual(agent.web, standard.web)
        XCTAssertEqual(agent.mcp, standard.mcp)
        XCTAssertEqual(agent.automation, standard.automation)
    }

    func testAgentAutoModeRoundTripsThroughProjectPermissions() throws {
        let original = AppProjectPermissions.preset(for: .agentAuto)
        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(AppProjectPermissions.self, from: data)
        XCTAssertEqual(decoded, original)
    }

    private func makeServer(
        _ name: String, autoApprove: Bool = false, sourcePath: String? = nil
    ) -> McpServerConfig {
        McpServerConfig(
            name: name,
            transport: .stdio(command: "/bin/echo"),
            isEnabled: true,
            autoApprove: autoApprove,
            sourcePath: sourcePath)
    }

    func testRepoImportedServerAsksUnderAgentModeToo() {
        // The 7b gate must fire in agent mode AND be hard: otherwise a
        // cloned .mcp.json's tools are judged by the classifier instead of
        // by the user.
        var project = AppProject(
            name: "p",
            permissions: AppProjectPermissions(mode: .agentAuto, mcp: .allow))
        project.mcpServers = [makeServer("fs", sourcePath: "/tmp/proj/.mcp.json")]

        let decision = AppToolPermissionEngine.evaluate(
            call: call("mcp__fs__read", [:], .mcp),
            project: project,
            globalServers: [])

        guard case .ask(let assessment, let reason) = decision else {
            return XCTFail("a repo-imported server must ask in agent mode, got \(decision)")
        }
        XCTAssertTrue(reason.contains("repository config"))
        XCTAssertTrue(assessment.isHardGated, "the import gate must not classify")
    }

    func testADenylistAskUnderAgentModeCarriesTheHardMark() {
        let project = AppProject(
            name: "p",
            permissions: AppProjectPermissions(mode: .agentAuto, terminal: .allow))
        let decision = AppToolPermissionEngine.evaluate(
            call: call("run_command", ["command": "rm -rf ~/Documents"], .terminal),
            project: project,
            globalServers: [])
        guard case .ask(let assessment, _) = decision else {
            return XCTFail("a denylist command must ask even in agent mode, got \(decision)")
        }
        XCTAssertTrue(assessment.isHardGated)
    }

    // MARK: - Subagent gate under Agent mode

    private func agentProject(
        terminal: AppToolPermission = .ask, mode: AppPermissionMode = .agentAuto
    ) -> AppProject {
        AppProject(
            name: "p", rootDirectoryPath: "/tmp",
            permissions: AppProjectPermissions(mode: mode, terminal: terminal))
    }

    func testASubagentSoftAskWithNoModelRefusesLikeAnAsk() async {
        let session = "subagent-agent-unavailable"
        await resetGate(session)
        // Nil session => classifier unavailable => the historical refusal.
        let refusal = await SubagentRunner.permissionRefusal(
            for: call("run_command", ["command": "./deploy.sh"], .terminal),
            project: agentProject())
        XCTAssertNotNil(refusal, "unavailable must fail closed for an unattended run")
    }

    func testASubagentAllowlistedCommandRunsUnderAgentMode() async {
        let session = "subagent-agent-fast"
        await resetGate(session)
        let refusal = await SubagentRunner.permissionRefusal(
            for: call("run_command", ["command": "cargo test"], .terminal),
            project: agentProject())
        XCTAssertNil(refusal, "the static fast path runs without a card, so a subagent runs it")
    }

    func testASubagentDenylistCommandStillRefusesUnderAgentMode() async {
        let session = "subagent-agent-hard"
        await resetGate(session)
        let refusal = await SubagentRunner.permissionRefusal(
            for: call("run_command", ["command": "sudo rm /etc/hosts"], .terminal),
            project: agentProject())
        XCTAssertNotNil(refusal, "hard rules win in agent mode too")
    }

    // MARK: - Main-loop routing (AppModel)

    @MainActor
    func testRouterDefersToTheCardOutsideAgentMode() async {
        let model = AppModel()
        model.agentModeClassifierOverride = ClassifierStub(verdict: .allow)
        let outcome = await model.resolveAskUnderAgentMode(
            call("run_command", ["command": "./deploy.sh"], .terminal),
            assessment: ToolRiskAssessment(level: .high, category: .terminal, reasons: []),
            chatID: UUID(),
            project: agentProject(mode: .auto))
        XCTAssertEqual(outcome, .parkCard(notice: nil))
    }

    @MainActor
    func testRouterRunsOnAClassifierAllowAndMarksIt() async {
        let model = AppModel()
        model.agentModeClassifierOverride = ClassifierStub(verdict: .allow)
        let outcome = await model.resolveAskUnderAgentMode(
            call("run_command", ["command": "./deploy.sh"], .terminal),
            assessment: ToolRiskAssessment(level: .high, category: .terminal, reasons: []),
            chatID: UUID(),
            project: agentProject())
        XCTAssertEqual(outcome, .run(byClassifier: true))
    }

    @MainActor
    func testRouterDeniesOnAClassifierBlockWithThePolicyMessage() async {
        let model = AppModel()
        model.agentModeClassifierOverride = ClassifierStub(verdict: .block(reason: "looks destructive"))
        let outcome = await model.resolveAskUnderAgentMode(
            call("run_command", ["command": "./deploy.sh"], .terminal),
            assessment: ToolRiskAssessment(level: .high, category: .terminal, reasons: []),
            chatID: UUID(),
            project: agentProject())
        guard case .denyWithReason(let reason) = outcome else {
            return XCTFail("expected a policy denial, got \(outcome)")
        }
        XCTAssertTrue(reason.contains("Blocked by Agent mode policy: looks destructive"))
        XCTAssertTrue(reason.contains("must not be completed"), "the guidance line rides along")
    }

    @MainActor
    func testRouterParksWithANoticeWhenTheClassifierIsUnavailable() async {
        let model = AppModel()
        model.agentModeClassifierOverride = ClassifierStub(verdict: .unavailable(reason: "no model"))
        let outcome = await model.resolveAskUnderAgentMode(
            call("run_command", ["command": "./deploy.sh"], .terminal),
            assessment: ToolRiskAssessment(level: .high, category: .terminal, reasons: []),
            chatID: UUID(),
            project: agentProject())
        guard case .parkCard(let notice) = outcome else {
            return XCTFail("expected a card, got \(outcome)")
        }
        XCTAssertTrue(notice?.contains("could not classify") ?? false)
    }

    @MainActor
    func testRouterFastAllowsALowRiskAskWithoutCallingTheClassifier() async {
        let model = AppModel()
        let stub = ClassifierStub(verdict: .allow)
        model.agentModeClassifierOverride = stub
        let outcome = await model.resolveAskUnderAgentMode(
            call("run_command", ["command": "cargo build"], .terminal),
            assessment: ToolRiskAssessment(level: .low, category: .terminal, reasons: []),
            chatID: UUID(),
            project: agentProject())
        XCTAssertEqual(outcome, .run(byClassifier: false))
        XCTAssertEqual(stub.callCount, 0, "the fast path must not spend a classifier call")
    }

    /// Counting stub: lets tests assert the classifier was (or was not)
    /// consulted. The lock lives behind sync methods because `lock()`
    /// itself is unavailable from an async context.
    private final class ClassifierStub: ToolCallClassifying {
        let verdict: ClassifierVerdict
        private let lock = NSLock()
        private var count = 0

        var callCount: Int {
            lock.lock()
            defer { lock.unlock() }
            return count
        }

        init(verdict: ClassifierVerdict) {
            self.verdict = verdict
        }

        func classify(_ request: ClassifierRequest) async -> ClassifierVerdict {
            record()
            return verdict
        }

        private func record() {
            lock.lock()
            count += 1
            lock.unlock()
        }
    }
}
