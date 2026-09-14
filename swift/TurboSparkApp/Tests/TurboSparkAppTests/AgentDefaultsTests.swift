import Foundation
import XCTest
@testable import TurboSparkApp

/// The built-in agents' Claude Code parity contract, the model-visible
/// agent listing, and the agent file write path (create/save/delete).
///
/// The file tests never touch the real `~/.turbospark/agents`: the
/// user-scope create goes through `directoryOverride` into a temp
/// directory (the storage page documents that the default cannot be
/// redirected after the fact), and project scope runs against a temp
/// project root, which discovery already treats as untrusted input.
final class AgentDefaultsTests: XCTestCase {
    private var tempDirURL: URL!

    override func setUp() {
        super.setUp()
        let uniqueName = "ts_agent_defaults_\(UUID().uuidString)"
        tempDirURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(uniqueName, isDirectory: true)
        try? FileManager.default.createDirectory(at: tempDirURL, withIntermediateDirectories: true)
    }

    override func tearDown() {
        if let tempDirURL {
            try? FileManager.default.removeItem(at: tempDirURL)
        }
        super.tearDown()
    }

    // MARK: - Built-in roster

    func testExploreBuiltInCarriesTheReadOnlyContract() {
        let explore = AgentManager.shared.builtInAgents.first { $0.name == "explore" }
        let prompt = explore?.systemPrompt ?? ""

        // The read-only prohibition block is the point of the agent: the
        // model must be told writes fail before it attempts one.
        XCTAssertTrue(prompt.contains("READ-ONLY MODE"), "prompt lost the read-only banner")
        XCTAssertTrue(prompt.contains("STRICTLY PROHIBITED"), "prompt lost the prohibition list")
        XCTAssertTrue(prompt.contains("redirect operators"), "prompt lost the redirect/heredoc prohibition")

        // Tool guidance must name the ADVERTISED wire names (what the
        // subagent's Available Tools listing prints), not legacy synonyms.
        for wireName in ["Glob", "Grep", "grep_search", "FileRead", "FileWrite", "FileEdit", "Bash"] {
            XCTAssertTrue(prompt.contains(wireName), "prompt does not mention \(wireName)")
        }
        XCTAssertTrue(prompt.contains("Syntext"), "prompt does not mention Syntext")
        XCTAssertTrue(prompt.contains("read-only operations"), "prompt lost the Bash read-only guidance")
        XCTAssertTrue(prompt.contains("thoroughness level"), "prompt lost the search-breadth guidance")
        XCTAssertTrue(prompt.contains("batch multiple tool calls"), "prompt lost the batching guidance")
        XCTAssertTrue(prompt.contains("absolute paths"), "prompt lost opencode's absolute-path reporting rule")
        XCTAssertTrue(prompt.contains("avoid using emojis"), "prompt lost opencode's no-emoji rule")

        // Read-only is ENFORCED by an allowlist, not only asked for in the
        // prompt (opencode's `"*": "deny"` + explicit allows). Every tool the
        // app gains later stays out by default.
        for allowed in ["FileRead", "Glob", "Grep", "grep_search", "Bash", "WebFetch", "WebSearch"] {
            XCTAssertTrue(explore?.isToolAllowed(allowed) ?? false, "explore lost \(allowed)")
        }
        for refused in ["FileWrite", "FileEdit", "apply_patch", "NotebookEdit", "notebook_edit", "agent", "TodoWrite", "memory", "skill", "exit_plan_mode"] {
            XCTAssertFalse(explore?.isToolAllowed(refused) ?? true, "explore must not gain \(refused)")
        }

        // The when-to-use description is what the model sees in the agent
        // listing; it must say what the agent is BAD at, not only good at.
        let description = explore?.agentDescription ?? ""
        XCTAssertTrue(description.lowercased().contains("do not use"), "description lost its anti-pattern guidance")
        XCTAssertTrue(description.lowercased().contains("breadth"), "description lost the breadth levels")
    }

    func testBuiltInDisallowedToolSetsAreUnchanged() {
        // These are the load-bearing ceilings: a project agent taking a
        // built-in's name is CONSTRAINED to this set (state#22), so a
        // silent change here silently rewrites every shadowing project
        // agent's permissions. Intentional changes belong here and in the
        // constraining test, never as drive-by edits.
        let agents = AgentManager.shared.builtInAgents
        let writeDenies: Set<String> = [
            "write_file", "save_file", "filewrite", "write",
            "edit_file", "fileedit", "edit",
            "apply_patch", "applypatch",
        ]

        let explore = agents.first { $0.name == "explore" }
        let expectedExploreDenies: Set<String> = writeDenies.union([
            "notebook_edit", "notebookedit",
            "agent", "subagent", "task",
            "enter_plan_mode", "enterplanmode",
            "exit_plan_mode", "exitplanmode",
            "todowrite", "todo_write",
            "enter_worktree", "enterworktree",
            "exit_worktree", "exitworktree"
        ])
        XCTAssertEqual(
            Set(explore?.disallowedTools ?? []),
            expectedExploreDenies,
            "explore's deny set moved")
        // The allowlist is the enforced half of explore's read-only
        // contract and, like the deny set, a ceiling: a project agent
        // shadowing `explore` has its own allowlist intersected with this
        // one, so widening it widens every shadowing override too.
        XCTAssertEqual(
            Set(explore?.tools ?? []),
            ["FileRead", "Glob", "Grep", "grep_search", "Bash", "WebFetch", "WebSearch"],
            "explore's allowlist moved")
        let plan = agents.first { $0.name == "plan" }
        XCTAssertEqual(
            Set(plan?.disallowedTools ?? []),
            expectedExploreDenies,
            "plan's deny set moved")
        XCTAssertEqual(
            Set(plan?.tools ?? []),
            ["FileRead", "Glob", "Grep", "grep_search", "Bash", "WebFetch", "WebSearch"],
            "plan's allowlist moved")

        let reviewer = agents.first { $0.name == "reviewer" }
        XCTAssertEqual(
            Set(reviewer?.disallowedTools ?? []),
            writeDenies,
            "reviewer's deny set moved")
        let generalPurpose = agents.first { $0.name == "general-purpose" }
        XCTAssertNil(generalPurpose?.disallowedTools, "general-purpose must stay unrestricted")
    }

    func testPlanBuiltInCarriesTheReadOnlyContract() {
        let plan = AgentManager.shared.builtInAgents.first { $0.name == "plan" }
        let prompt = plan?.systemPrompt ?? ""

        XCTAssertTrue(prompt.contains("CRITICAL: READ-ONLY MODE"), "prompt lost the critical read-only banner")
        XCTAssertTrue(prompt.contains("STRICTLY PROHIBITED"), "prompt lost the prohibition list")
        XCTAssertTrue(prompt.contains("redirect operators"), "prompt lost the redirect/heredoc prohibition")
        XCTAssertTrue(prompt.contains("Critical Files for Implementation"), "prompt lost the required Critical Files output section")

        for wireName in ["Glob", "Grep", "grep_search", "FileRead", "FileWrite", "FileEdit", "Bash"] {
            XCTAssertTrue(prompt.contains(wireName), "prompt does not mention \(wireName)")
        }
        XCTAssertTrue(prompt.contains("Syntext"), "prompt does not mention Syntext")
        XCTAssertTrue(prompt.contains("read-only operations"), "prompt lost the Bash read-only guidance")
        XCTAssertTrue(prompt.contains("avoid using emojis"), "prompt lost opencode's no-emoji rule")

        for allowed in ["FileRead", "Glob", "Grep", "grep_search", "Bash", "WebFetch", "WebSearch"] {
            XCTAssertTrue(plan?.isToolAllowed(allowed) ?? false, "plan lost \(allowed)")
        }
        for refused in ["FileWrite", "FileEdit", "apply_patch", "NotebookEdit", "notebook_edit", "agent", "TodoWrite", "memory", "skill", "exit_plan_mode"] {
            XCTAssertFalse(plan?.isToolAllowed(refused) ?? true, "plan must not gain \(refused)")
        }

        let description = plan?.agentDescription ?? ""
        XCTAssertTrue(description.lowercased().contains("architect"), "description lost architect guidance")
        XCTAssertTrue(description.lowercased().contains("implementation plans"), "description lost plans guidance")
    }

    func testGeneralPurposeBuiltInContract() {
        let gp = AgentManager.shared.builtInAgents.first { $0.name == "general-purpose" }
        XCTAssertNotNil(gp, "general-purpose agent must exist")
        XCTAssertNil(gp?.disallowedTools, "general-purpose must have no disallowedTools")
        XCTAssertNil(gp?.tools, "general-purpose must have no restricted tools allowlist")
        XCTAssertEqual(gp?.omitsProjectInstructions, false, "general-purpose must not omit project instructions")
        XCTAssertEqual(gp?.maxTurns, 6, "general-purpose maxTurns should be 6")

        let prompt = gp?.systemPrompt ?? ""
        XCTAssertTrue(prompt.contains("TurboSpark"), "prompt must mention TurboSpark")
        XCTAssertTrue(prompt.contains("Syntext"), "prompt must mention Syntext for search")
        XCTAssertTrue(prompt.contains("Searching for code"), "prompt must mention code search strength")
        XCTAssertTrue(prompt.contains("concise report"), "prompt must ask for concise report")

        let desc = gp?.agentDescription ?? ""
        XCTAssertTrue(desc.contains("researching complex questions"), "description lost complex questions")
        XCTAssertTrue(desc.contains("searching for code"), "description lost searching for code")
    }

    func testExploreBuiltInOmitsProjectInstructions() {
        let explore = AgentManager.shared.builtInAgents.first { $0.name == "explore" }
        XCTAssertEqual(explore?.omitsProjectInstructions, true, "explore must skip project instructions")
        for name in ["plan", "general-purpose", "reviewer"] {
            let agent = AgentManager.shared.builtInAgents.first { $0.name == name }
            XCTAssertEqual(
                agent?.omitsProjectInstructions, false,
                "\(name) must still see project instructions")
        }
    }

    func testParserRecognizesOmitClaudeMdFrontmatter() {
        let markdown = """
        ---
        name: custom-scout
        omit_claude_md: true
        ---
        Fast search.
        """
        let parsed = AgentParser.parseMarkdownContent(
            markdown, sourceURL: URL(fileURLWithPath: "/tmp/custom-scout.md"))
        XCTAssertTrue(parsed.omitsProjectInstructions, "omit_claude_md frontmatter was not parsed")
    }

    func testOmitFlagDropsOnlyTheProjectInstructionsSection() {
        let project = AppProject(
            name: "Demo",
            rootDirectoryPath: "/path/to/demo",
            customInstructions: "Use Swift 5.9 and swear off singletons")

        let omitting = AppAgentDefinition(
            name: "scout",
            agentDescription: "test",
            systemPrompt: "Search only.",
            omitsProjectInstructions: true)
        let omitted = SubagentRunner.buildSystemPrompt(for: omitting, project: project)
        XCTAssertTrue(omitted.contains("Root codebase directory: `/path/to/demo`"), "workspace root must survive the omit")
        XCTAssertFalse(omitted.contains("Use Swift 5.9"), "project instructions must be dropped")
        XCTAssertFalse(omitted.contains("## Project Specific Context"), "the section header must be gone with it")

        let keeping = AppAgentDefinition(
            name: "worker",
            agentDescription: "test",
            systemPrompt: "Do work.")
        let kept = SubagentRunner.buildSystemPrompt(for: keeping, project: project)
        XCTAssertTrue(kept.contains("Use Swift 5.9"), "the default agent must still see project instructions")
    }

    // MARK: - Model-visible agent listing

    func testAddendumListsTheAgentsItIsGiven() {
        let addendum = AppToolCatalog.systemPromptAddendum(
            for: .coder,
            availableAgents: [
                (name: "explore", whenToUse: "Fast read-only search agent."),
                (name: "my-deploy", whenToUse: "Ships the release."),
            ])

        XCTAssertTrue(addendum.contains("Available agent types for `subagent_type`"))
        XCTAssertTrue(addendum.contains("- `explore`: Fast read-only search agent."))
        XCTAssertTrue(addendum.contains("- `my-deploy`: Ships the release."))
        XCTAssertTrue(addendum.contains("falls back to `general-purpose`"))
    }

    func testAddendumWithoutAgentsKeepsItsPreviousShape() {
        let addendum = AppToolCatalog.systemPromptAddendum(for: .coder)
        XCTAssertFalse(
            addendum.contains("Available agent types"),
            "the default addendum grew an agent listing; existing callers and tests pinned its shape")
    }

    func testAddendumTruncatesLongDescriptions() {
        let long = String(repeating: "x", count: 400)
        let addendum = AppToolCatalog.systemPromptAddendum(
            for: .coder, availableAgents: [(name: "chatty", whenToUse: long)])
        guard let line = addendum
            .components(separatedBy: "\n")
            .first(where: { $0.hasPrefix("- `chatty`: ") }) else {
            return XCTFail("no listing line for chatty")
        }
        XCTAssertTrue(line.hasSuffix("..."), "a truncated line must say so")
        XCTAssertLessThan(line.count, 250, "one agent's listing line must stay bounded")
    }

    @MainActor
    func testSectionsListOnlyEnabledProjectAgents() throws {
        // End to end over the AppModel fold: a project agent created on a
        // temp root is advertised while enabled and VANISHES from the
        // listing when disabled -- the state#93 rule applied to the new
        // surface, where advertising a refusal would invite it.
        let appModel = AppModel()
        let projectRoot = tempDirURL.appendingPathComponent("proj", isDirectory: true)
        try FileManager.default.createDirectory(at: projectRoot, withIntermediateDirectories: true)
        let project = AppProject(name: "probe", rootDirectoryPath: projectRoot.path)

        _ = try makeAgentFile(name: "listing-probe", scope: .project, projectRoot: projectRoot)
        AgentManager.shared.invalidateResolutionCache()

        func toolsSection(for project: AppProject) -> String {
            appModel.buildSystemPromptSections(for: project)
                .first { $0.section == .tools }?.content ?? ""
        }
        XCTAssertTrue(
            toolsSection(for: project).contains("- `listing-probe`:"),
            "an enabled agent must be advertised to the model")

        AgentManager.shared.setAgentEnabled(false, name: "listing-probe", scope: .project)
        defer {
            AgentManager.shared.setAgentEnabled(true, name: "listing-probe", scope: .project)
        }
        XCTAssertFalse(
            toolsSection(for: project).contains("- `listing-probe`:"),
            "a disabled agent must not be advertised: the tool refuses it")
    }

    // MARK: - Agent file write path

    private func makeAgentFile(
        name: String = "deploy-auditor",
        scope: AppAgentScope = .userGlobal,
        projectRoot: URL? = nil,
        directory: URL? = nil
    ) throws -> AppAgentDefinition {
        try AgentManager.shared.createAgent(
            name: name,
            displayName: "Deploy Auditor",
            description: "Checks deploys before they ship.",
            systemPrompt: "Audit the deploy plan.",
            tools: nil,
            disallowedTools: ["FileWrite"],
            maxTurns: 7,
            scope: scope,
            projectRootURL: projectRoot,
            directoryOverride: directory ?? tempDirURL.appendingPathComponent("agents", isDirectory: true))
    }

    func testSerializedAgentRoundTripsThroughTheParser() throws {
        let agent = AppAgentDefinition(
            name: "round-trip",
            displayName: "Round Trip",
            agentDescription: "Verifies the writer against the reader.",
            systemPrompt: "Line one.\nLine two.",
            tools: ["FileRead", "Grep"],
            disallowedTools: ["FileWrite", "Bash"],
            maxTurns: 11,
            scope: .userGlobal)
        let serialized = AgentParser.serializeAgent(agent)
        let parsed = AgentParser.parseMarkdownContent(
            serialized,
            sourceURL: URL(fileURLWithPath: "/tmp/agents/round-trip.md"),
            scope: .userGlobal)

        XCTAssertEqual(parsed.name, "round-trip")
        XCTAssertEqual(parsed.displayName, "Round Trip")
        XCTAssertEqual(parsed.agentDescription, "Verifies the writer against the reader.")
        XCTAssertEqual(parsed.systemPrompt, "Line one.\nLine two.")
        XCTAssertEqual(parsed.tools, ["FileRead", "Grep"])
        XCTAssertEqual(parsed.disallowedTools, ["FileWrite", "Bash"])
        XCTAssertEqual(parsed.maxTurns, 11)
    }

    func testSerializerFoldsNewlinesOutOfFrontmatterValues() throws {
        // The frontmatter reader is line-oriented; a value carrying a
        // newline would re-enter the parser as key: value lines.
        let agent = AppAgentDefinition(
            name: "folded",
            agentDescription: "First line.\nSecond line.",
            systemPrompt: "Body.")
        let serialized = AgentParser.serializeAgent(agent)
        XCTAssertTrue(serialized.contains("description: First line. Second line."))

        let parsed = AgentParser.parseMarkdownContent(
            serialized,
            sourceURL: URL(fileURLWithPath: "/tmp/agents/folded.md"),
            scope: .userGlobal)
        XCTAssertEqual(parsed.agentDescription, "First line. Second line.")
    }

    func testCreateUserScopeAgentWritesAndParsesBack() throws {
        let agentsDir = tempDirURL.appendingPathComponent("agents", isDirectory: true)
        let agent = try makeAgentFile()

        let fileURL = agentsDir.appendingPathComponent("deploy-auditor.md")
        XCTAssertEqual(agent.filePath, fileURL.path)
        XCTAssertTrue(FileManager.default.fileExists(atPath: fileURL.path))

        let reread = try AgentParser.parseFile(at: fileURL, scope: .userGlobal)
        XCTAssertEqual(reread.name, "deploy-auditor")
        XCTAssertEqual(reread.displayName, "Deploy Auditor")
        XCTAssertEqual(reread.agentDescription, "Checks deploys before they ship.")
        XCTAssertEqual(reread.systemPrompt, "Audit the deploy plan.")
        XCTAssertEqual(reread.maxTurns, 7)
        XCTAssertEqual(reread.disallowedTools, ["FileWrite"])
        XCTAssertFalse(reread.isToolAllowed("FileWrite"))
    }

    func testCreateAgentSanitizesAndRefusesEscapingNames() throws {
        // `..` survives slash replacement and would resolve OUT of the
        // agents directory -- the createSkill footgun, refused here too.
        XCTAssertThrowsError(try makeAgentFile(name: ".."))
        XCTAssertThrowsError(try makeAgentFile(name: "."))
        XCTAssertThrowsError(try makeAgentFile(name: "  "))

        let sanitized = try makeAgentFile(name: "Team/Lead")
        XCTAssertEqual(
            sanitized.filePath, 
            tempDirURL.appendingPathComponent("agents/team-lead.md").path,
            "a slashed name must become a flat file name")

        // `../evil` sanitizes to `..-evil`, a dot-leading name the hidden
        // file skip would never discover, so it is refused outright rather
        // than written as an agent that silently does not exist.
        XCTAssertThrowsError(try makeAgentFile(name: "..//evil"))
        XCTAssertThrowsError(try makeAgentFile(name: ".hidden"))
    }

    func testCreateAgentRefusesToOverwriteAnExistingFile() throws {
        _ = try makeAgentFile()
        XCTAssertThrowsError(try makeAgentFile()) { error in
            guard case AgentManager.AgentFileError.destinationExists = error else {
                return XCTFail("expected destinationExists, got \(error)")
            }
        }
    }

    func testCreateAgentRefusesNonFileScopes() throws {
        for scope in [AppAgentScope.builtIn, .plugin] {
            XCTAssertThrowsError(try makeAgentFile(scope: scope)) { error in
                guard case AgentManager.AgentFileError.scopeNotAllowed = error else {
                    return XCTFail("expected scopeNotAllowed for \(scope), got \(error)")
                }
            }
        }
    }

    func testCreateProjectAgentNeedsARoot() {
        XCTAssertThrowsError(try makeAgentFile(scope: .project)) { error in
            guard case AgentManager.AgentFileError.projectRootRequired = error else {
                return XCTFail("expected projectRootRequired, got \(error)")
            }
        }
    }

    func testCreateProjectAgentWritesUnderTheRootAndIsDiscovered() throws {
        let projectRoot = tempDirURL.appendingPathComponent("proj", isDirectory: true)
        try FileManager.default.createDirectory(at: projectRoot, withIntermediateDirectories: true)

        _ = try makeAgentFile(scope: .project, projectRoot: projectRoot)

        let expected = projectRoot
            .appendingPathComponent(".turbospark/agents/deploy-auditor.md")
        XCTAssertTrue(FileManager.default.fileExists(atPath: expected.path))

        let discovered = AgentManager.shared.discoverProjectAgents(projectURL: projectRoot)
            .first { $0.name == "deploy-auditor" }
        XCTAssertNotNil(discovered, "a project agent must be discoverable right after create")
        XCTAssertEqual(discovered?.scope, .project)
        XCTAssertEqual(discovered?.maxTurns, 7)
    }

    func testSaveAgentRewritesTheFile() throws {
        var agent = try makeAgentFile()
        agent.agentDescription = "Updated description."
        agent.maxTurns = 9
        try AgentManager.shared.saveAgent(agent)

        let path = try XCTUnwrap(agent.filePath)
        let reread = try AgentParser.parseFile(
            at: URL(fileURLWithPath: path), scope: .userGlobal)
        XCTAssertEqual(reread.agentDescription, "Updated description.")
        XCTAssertEqual(reread.maxTurns, 9)
    }

    func testDeleteAgentRemovesTheFile() throws {
        let agent = try makeAgentFile()
        try AgentManager.shared.deleteAgent(agent)
        XCTAssertFalse(FileManager.default.fileExists(atPath: agent.filePath ?? ""))
    }

    func testSaveAndDeleteRefuseBuiltInAgents() {
        let builtInLike = AppAgentDefinition(
            name: "explore",
            agentDescription: "test",
            systemPrompt: "test",
            scope: .builtIn)
        XCTAssertThrowsError(try AgentManager.shared.saveAgent(builtInLike)) { error in
            guard case AgentManager.AgentFileError.scopeNotAllowed = error else {
                return XCTFail("expected scopeNotAllowed on save, got \(error)")
            }
        }
        XCTAssertThrowsError(try AgentManager.shared.deleteAgent(builtInLike)) { error in
            guard case AgentManager.AgentFileError.scopeNotAllowed = error else {
                return XCTFail("expected scopeNotAllowed on delete, got \(error)")
            }
        }
    }
}
