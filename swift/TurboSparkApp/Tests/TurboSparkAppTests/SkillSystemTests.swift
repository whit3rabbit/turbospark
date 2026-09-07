import Foundation
import XCTest
@testable import TurboSparkApp

final class SkillSystemTests: XCTestCase {
    private var tempDirURL: URL!

    override func setUp() {
        super.setUp()
        let uniqueName = "ts_skill_test_\(UUID().uuidString)"
        tempDirURL = FileManager.default.temporaryDirectory.appendingPathComponent(uniqueName, isDirectory: true)
        try? FileManager.default.createDirectory(at: tempDirURL, withIntermediateDirectories: true)
    }

    override func tearDown() {
        if let tempDirURL {
            try? FileManager.default.removeItem(at: tempDirURL)
        }
        super.tearDown()
    }

    // MARK: - Frontmatter Parsing Tests

    func testParseSkillWithFullFrontmatter() {
        let raw = """
        ---
        name: jupyter-skill
        description: View and edit Jupyter notebooks.
        allowed-tools:
          - Bash(python3:*)
          - Read
          - write_file
        argument-hint: path/to/notebook.ipynb
        arguments:
          - name: notebook_path
            placeholder: /path/to/file.ipynb
            default: main.ipynb
            description: Target notebook to inspect
        user-invocable: true
        disable-model-invocation: false
        model: gemma4
        context: inline
        paths:
          - "*.ipynb"
          - "notebooks/**/*.py"
        shell: bash
        ---

        # Jupyter Workflow Instructions
        Run `python3 ${CLAUDE_SKILL_DIR}/scripts/view.py ${notebook_path}` to view.
        """

        let skill = SkillParser.parseContent(rawText: raw, scope: .userGlobal, agentOrigin: .turboSpark)
        XCTAssertEqual(skill.name, "jupyter-skill")
        XCTAssertEqual(skill.manifest.description, "View and edit Jupyter notebooks.")
        XCTAssertEqual(skill.manifest.allowedTools, ["Bash(python3:*)", "Read", "write_file"])
        XCTAssertEqual(skill.manifest.argumentHint, "path/to/notebook.ipynb")
        XCTAssertEqual(skill.manifest.arguments.count, 1)
        XCTAssertEqual(skill.manifest.arguments.first?.name, "notebook_path")
        XCTAssertEqual(skill.manifest.arguments.first?.placeholder, "/path/to/file.ipynb")
        XCTAssertEqual(skill.manifest.arguments.first?.defaultValue, "main.ipynb")
        XCTAssertEqual(skill.manifest.paths, ["*.ipynb", "notebooks/**/*.py"])
        XCTAssertEqual(skill.manifest.context, .inline)
        XCTAssertEqual(skill.manifest.shell, .bash)
        XCTAssertTrue(skill.manifest.isToolAllowed(tool: "Bash", subTool: "python3:view.py"))
        XCTAssertTrue(skill.manifest.isToolAllowed(tool: "Read"))
        XCTAssertFalse(skill.manifest.isToolAllowed(tool: "delete_file"))
        XCTAssertTrue(skill.content.contains("# Jupyter Workflow Instructions"))
    }

    func testParseSkillWithoutFrontmatter() {
        let raw = """
        # Simple Skill
        This is plain instructions without YAML frontmatter.
        """

        let fileURL = tempDirURL.appendingPathComponent("custom-workflow.md")
        let skill = SkillParser.parseContent(rawText: raw, sourceURL: fileURL, scope: .userGlobal)
        XCTAssertEqual(skill.name, "custom-workflow")
        XCTAssertEqual(skill.content, raw.trimmingCharacters(in: .whitespacesAndNewlines))
        XCTAssertEqual(skill.manifest.allowedTools, [])
    }

    func testSerializeAndReParseSkill() {
        let manifest = SkillManifest(
            name: "git-commit-helper",
            description: "Format concise atomic git commit messages.",
            allowedTools: ["Bash(git:*)"],
            arguments: [SkillArgument(name: "branch", defaultValue: "main")],
            paths: ["*.rs", "*.swift"]
        )
        let skill = AppSkill(
            manifest: manifest,
            content: "Check staged diff and commit with standard prefixes.",
            sourceURL: tempDirURL.appendingPathComponent("git-commit-helper/SKILL.md"),
            scope: .userGlobal
        )

        let serialized = SkillParser.serializeSkill(skill)
        let reParsed = SkillParser.parseContent(rawText: serialized, scope: .userGlobal)

        XCTAssertEqual(reParsed.name, "git-commit-helper")
        XCTAssertEqual(reParsed.manifest.description, "Format concise atomic git commit messages.")
        XCTAssertEqual(reParsed.manifest.allowedTools, ["Bash(git:*)"])
        XCTAssertEqual(reParsed.manifest.paths, ["*.rs", "*.swift"])
        XCTAssertEqual(reParsed.content, "Check staged diff and commit with standard prefixes.")
    }

    // MARK: - Directory Scanning & Discovery Tests

    func testDiscoverDirectoryBasedSkillWithReferenceFiles() throws {
        let skillDir = tempDirURL.appendingPathComponent("code-reviewer", isDirectory: true)
        try FileManager.default.createDirectory(at: skillDir, withIntermediateDirectories: true)

        let skillMdURL = skillDir.appendingPathComponent("SKILL.md")
        let skillContent = """
        ---
        name: code-reviewer
        description: Review code style and patterns.
        ---
        Review the pull request diff carefully.
        """
        try skillContent.write(to: skillMdURL, atomically: true, encoding: .utf8)

        // Sidecar scripts
        let helperScript = skillDir.appendingPathComponent("review_linter.py")
        try "print('lint')".write(to: helperScript, atomically: true, encoding: .utf8)

        let manager = SkillManager()
        let discovered = manager.scanDirectory(tempDirURL, scope: .userGlobal, defaultAgent: .turboSpark)

        XCTAssertEqual(discovered.count, 1)
        let first = discovered.first!
        XCTAssertEqual(first.name, "code-reviewer")
        XCTAssertTrue(first.isDirectoryBased)
        XCTAssertEqual(first.referenceFiles, ["review_linter.py"])
    }

    func testDiscoverSingleFileSkill() throws {
        let fileURL = tempDirURL.appendingPathComponent("release-checklist.md")
        let content = """
        ---
        name: release-checklist
        description: Steps before cutting a release.
        ---
        1. Run test suite.
        2. Bump version.
        """
        try content.write(to: fileURL, atomically: true, encoding: .utf8)

        let manager = SkillManager()
        let discovered = manager.scanDirectory(tempDirURL, scope: .userGlobal, defaultAgent: .turboSpark)

        XCTAssertEqual(discovered.count, 1)
        XCTAssertEqual(discovered.first?.name, "release-checklist")
        XCTAssertFalse(discovered.first?.isDirectoryBased ?? true)
    }

    // MARK: - Multi-Agent Path Scanning Tests

    func testMultiAgentProjectDiscovery() throws {
        let claudeSkillsDir = tempDirURL.appendingPathComponent(".claude/skills/claude-skill", isDirectory: true)
        try FileManager.default.createDirectory(at: claudeSkillsDir, withIntermediateDirectories: true)
        try "---\nname: claude-skill\n---\nClaude instructions".write(
            to: claudeSkillsDir.appendingPathComponent("SKILL.md"),
            atomically: true,
            encoding: .utf8
        )

        let openCodeSkillsDir = tempDirURL.appendingPathComponent(".opencode/skills/opencode-skill", isDirectory: true)
        try FileManager.default.createDirectory(at: openCodeSkillsDir, withIntermediateDirectories: true)
        try "---\nname: opencode-skill\n---\nOpenCode instructions".write(
            to: openCodeSkillsDir.appendingPathComponent("SKILL.md"),
            atomically: true,
            encoding: .utf8
        )

        let piSkillsDir = tempDirURL.appendingPathComponent(".pi/skills/pi-skill", isDirectory: true)
        try FileManager.default.createDirectory(at: piSkillsDir, withIntermediateDirectories: true)
        try "---\nname: pi-skill\n---\nPi instructions".write(
            to: piSkillsDir.appendingPathComponent("SKILL.md"),
            atomically: true,
            encoding: .utf8
        )

        let antigravitySkillsDir = tempDirURL.appendingPathComponent(".agents/skills/agy-skill", isDirectory: true)
        try FileManager.default.createDirectory(at: antigravitySkillsDir, withIntermediateDirectories: true)
        try "---\nname: agy-skill\n---\nAntigravity instructions".write(
            to: antigravitySkillsDir.appendingPathComponent("SKILL.md"),
            atomically: true,
            encoding: .utf8
        )

        let manager = SkillManager()
        let projectSkills = manager.discoverProjectSkills(projectRootURL: tempDirURL)

        XCTAssertEqual(projectSkills.count, 4)
        let names = Set(projectSkills.map { $0.name })
        XCTAssertTrue(names.contains("claude-skill"))
        XCTAssertTrue(names.contains("opencode-skill"))
        XCTAssertTrue(names.contains("pi-skill"))
        XCTAssertTrue(names.contains("agy-skill"))
    }

    // MARK: - Precedence Tests

    func testProjectSkillOverridesUserSkillWithSameName() throws {
        let userDir = tempDirURL.appendingPathComponent("user_skills", isDirectory: true)
        let projectDir = tempDirURL.appendingPathComponent("project_skills", isDirectory: true)
        let projectTurbosparkSkills = projectDir.appendingPathComponent(".turbospark/skills/formatter", isDirectory: true)

        try FileManager.default.createDirectory(at: userDir.appendingPathComponent("formatter"), withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: projectTurbosparkSkills, withIntermediateDirectories: true)

        try "---\nname: formatter\ndescription: Global formatter\n---\nGlobal content".write(
            to: userDir.appendingPathComponent("formatter/SKILL.md"),
            atomically: true,
            encoding: .utf8
        )

        try "---\nname: formatter\ndescription: Project-specific formatter\n---\nProject content".write(
            to: projectTurbosparkSkills.appendingPathComponent("SKILL.md"),
            atomically: true,
            encoding: .utf8
        )

        let manager = SkillManager()
        let userSkills = manager.scanDirectory(userDir, scope: .userGlobal, defaultAgent: .turboSpark)
        let projectSkills = manager.discoverProjectSkills(projectRootURL: projectDir)

        var mergedMap: [String: AppSkill] = [:]
        for s in userSkills { mergedMap[s.name.lowercased()] = s }
        for s in projectSkills { mergedMap[s.name.lowercased()] = s }

        let resolved = mergedMap["formatter"]
        XCTAssertNotNil(resolved)
        XCTAssertEqual(resolved?.manifest.description, "Project-specific formatter")
        XCTAssertEqual(resolved?.content, "Project content")
        XCTAssertTrue(resolved?.scope.isProjectScope ?? false)
    }

    // MARK: - Argument Substitution Tests

    func testArgumentSubstitution() {
        let manager = SkillManager()
        let template = "Run ${tool_name} on ${file_path} in session ${SESSION_ID} at dir ${SKILL_DIR}."
        let fakeSkillDir = URL(fileURLWithPath: "/Users/test/.turbospark/skills/my-skill")

        let substituted = manager.substituteArguments(
            content: template,
            arguments: [
                "tool_name": "pytest",
                "file_path": "tests/test_app.py"
            ],
            skillDirectoryURL: fakeSkillDir,
            sessionID: "sess_12345"
        )

        XCTAssertEqual(
            substituted,
            "Run pytest on tests/test_app.py in session sess_12345 at dir /Users/test/.turbospark/skills/my-skill."
        )
    }

    // MARK: - Path Matching Tests

    func testPathMatchingTriggers() {
        let manager = SkillManager()
        let manifest = SkillManifest(
            name: "rust-tester",
            paths: ["*.rs", "crates/**/*.rs", "Cargo.toml"]
        )
        let skill = AppSkill(
            manifest: manifest,
            content: "Rust testing",
            sourceURL: tempDirURL.appendingPathComponent("SKILL.md"),
            scope: .userGlobal
        )

        XCTAssertTrue(manager.matchesPath(skill: skill, filePath: "main.rs"))
        XCTAssertTrue(manager.matchesPath(skill: skill, filePath: "crates/core/src/lib.rs"))
        XCTAssertTrue(manager.matchesPath(skill: skill, filePath: "Cargo.toml"))
        XCTAssertFalse(manager.matchesPath(skill: skill, filePath: "Package.swift"))
        XCTAssertFalse(manager.matchesPath(skill: skill, filePath: "src/index.js"))
    }

    // MARK: - Native Tool Execution Tests

    func testNativeSkillToolExecution() async throws {
        let skillDir = tempDirURL.appendingPathComponent(".turbospark/skills/db-migrate", isDirectory: true)
        try FileManager.default.createDirectory(at: skillDir, withIntermediateDirectories: true)
        let skillMd = skillDir.appendingPathComponent("SKILL.md")
        try """
        ---
        name: db-migrate
        description: Run database schema migrations.
        ---
        Execute `sqlx migrate run` on target DB ${db_name}.
        """.write(to: skillMd, atomically: true, encoding: .utf8)

        let project = AppProject(
            name: "TestProject",
            rootDirectoryPath: tempDirURL.path
        )

        let call = AppToolCall(
            name: "skill",
            arguments: [
                "name": "db-migrate",
                "db_name": "postgres_dev"
            ]
        )

        let result = await AppToolRegistry.execute(call: call, in: project)
        XCTAssertFalse(result.isError)
        XCTAssertTrue(result.output.contains("### Skill: db-migrate"))
        XCTAssertTrue(result.output.contains("Execute `sqlx migrate run` on target DB postgres_dev."))
    }

    // MARK: - Disable Persistence & Enforcement (state#12)

    func testDisablingASkillPersistsAcrossRediscoveryAndBlocksTheToolCall() async throws {
        // A unique name so this test's disabled-state write to
        // `SkillManager`'s shared (UserDefaults-backed) store cannot leak
        // into any other test's fixture.
        let skillName = "disable-test-\(UUID().uuidString.prefix(8))"
        defer {
            SkillManager.shared.setSkillEnabled(
                true, scope: .projectLocal(projectPath: tempDirURL.path), name: skillName)
        }

        let skillDir = tempDirURL.appendingPathComponent(".turbospark/skills/\(skillName)", isDirectory: true)
        try FileManager.default.createDirectory(at: skillDir, withIntermediateDirectories: true)
        try """
        ---
        name: \(skillName)
        description: A throwaway skill for the disable-persistence test.
        ---
        Do the thing.
        """.write(to: skillDir.appendingPathComponent("SKILL.md"), atomically: true, encoding: .utf8)

        // Freshly discovered, it is enabled (the parser's own default).
        let beforeDisable = SkillManager.shared.discoverProjectSkills(projectRootURL: tempDirURL)
        XCTAssertEqual(beforeDisable.first(where: { $0.name == skillName })?.isEnabled, true)

        // The scope is part of the key now: a project skill and a same-named
        // user skill are separate preferences (a shared key disabled both).
        let discoveredScope = beforeDisable.first(where: { $0.name == skillName })!.scope
        SkillManager.shared.setSkillEnabled(false, scope: discoveredScope, name: skillName)

        // The OLD bug: toggling only mutated an in-memory AppSkill copy, so
        // a fresh discovery (what reloadSkills() does on any project switch)
        // came back enabled again. Re-discovering here must see `false`.
        let afterDisable = SkillManager.shared.discoverProjectSkills(projectRootURL: tempDirURL)
        XCTAssertEqual(afterDisable.first(where: { $0.name == skillName })?.isEnabled, false)

        // The OTHER half of state#12: the `skill` tool executor must itself
        // refuse to run a disabled skill, not just report it as disabled.
        let project = AppProject(name: "TestProject", rootDirectoryPath: tempDirURL.path)
        let call = AppToolCall(name: "skill", arguments: ["name": skillName])
        let result = await AppToolRegistry.execute(call: call, in: project)
        XCTAssertTrue(result.isError, "Invoking a disabled skill must fail, not run it anyway.")
        XCTAssertTrue(result.output.contains("disabled"))

        // Re-enabling restores normal execution.
        SkillManager.shared.setSkillEnabled(true, scope: discoveredScope, name: skillName)
        let resultAfterReenable = await AppToolRegistry.execute(call: call, in: project)
        XCTAssertFalse(resultAfterReenable.isError)
        XCTAssertTrue(resultAfterReenable.output.contains("Do the thing."))
    }

    func testNativeSkillToolExecutionNotFound() async {
        let project = AppProject(
            name: "EmptyProject",
            rootDirectoryPath: tempDirURL.path
        )

        let call = AppToolCall(
            name: "skill",
            arguments: ["name": "non-existent-skill"]
        )

        let result = await AppToolRegistry.execute(call: call, in: project)
        XCTAssertFalse(result.isError)
        XCTAssertTrue(result.output.contains("Skill 'non-existent-skill' was not found."))
    }

    // MARK: - Marketplace & Budgeting Tests

    func testMarketplaceSourceEncodingAndDecoding() throws {
        let sources: [MarketplaceSource] = [
            .url(url: "https://example.com/marketplace.json", headers: ["Authorization": "Bearer token"]),
            .github(repo: "whit3rabbit/agent-skills", ref: "main", path: "skills/test", sparsePaths: ["skills"]),
            .git(url: "git@github.com:whit3rabbit/repo.git", ref: "v1.0", path: nil, sparsePaths: nil),
            .directory(path: "/local/path/to/skills")
        ]

        let encoder = JSONEncoder()
        let decoder = JSONDecoder()

        for src in sources {
            let data = try encoder.encode(src)
            let decoded = try decoder.decode(MarketplaceSource.self, from: data)
            XCTAssertEqual(src, decoded)
        }
    }

    func testSkillPromptBudgetingUnderAndOverLimit() {
        let manager = SkillManager.shared
        let normalSkills = [
            AppSkill(
                manifest: SkillManifest(name: "alpha", description: "Alpha description"),
                content: "content",
                sourceURL: tempDirURL.appendingPathComponent("alpha.md"),
                scope: .userGlobal
            ),
            AppSkill(
                manifest: SkillManifest(name: "beta", description: "Beta description"),
                content: "content",
                sourceURL: tempDirURL.appendingPathComponent("beta.md"),
                scope: .bundled
            )
        ]

        // Under limit (default 8000 chars at 200k tokens: 200000 * 4 * 1%)
        let formattedUnder = manager.formatSkillsWithinBudget(normalSkills, contextWindowTokens: 200_000)
        XCTAssertTrue(formattedUnder.contains("- alpha [User]: Alpha description"))
        XCTAssertTrue(formattedUnder.contains("- beta [Bundled]: Beta description"))

        // Extremely squeezed budget (tokens: 25 -> 25 * 4 * 0.01 = 1 character budget)
        let formattedSqueezed = manager.formatSkillsWithinBudget(normalSkills, contextWindowTokens: 25)
        // Bundled skill preserves description, non-bundled is reduced to name only
        XCTAssertTrue(formattedSqueezed.contains("- beta [Bundled]: Beta description"))
        XCTAssertTrue(formattedSqueezed.contains("- alpha"))
    }

    func testWhenToUseParsesSerializesAndAppearsInListings() {
        let raw = """
        ---
        name: deploy-staging
        description: Deploys the current branch to staging.
        when_to_use: Use when the user asks to deploy or push to staging
        ---
        Deploy steps here.
        """
        let skill = SkillParser.parseContent(rawText: raw, scope: .userGlobal)
        XCTAssertEqual(skill.manifest.whenToUse, "Use when the user asks to deploy or push to staging")

        // Round trip through the serializer.
        let reParsed = SkillParser.parseContent(
            rawText: SkillParser.serializeSkill(skill), scope: .userGlobal)
        XCTAssertEqual(reParsed.manifest.whenToUse, "Use when the user asks to deploy or push to staging")

        // The listing carries it, Claude Code style, appended to the description.
        let formatted = SkillManager.shared.formatSkillsWithinBudget([skill], contextWindowTokens: 200_000)
        XCTAssertTrue(
            formatted.contains("- deploy-staging [User]: Deploys the current branch to staging. - Use when the user asks to deploy or push to staging"),
            "listing was: \(formatted)")
    }

    func testClaudeCodeArgumentSpellingsSubstitute() {
        let manager = SkillManager()
        let template = "Run $0 against $ARGUMENTS[1] with everything: $ARGUMENTS"
        let substituted = manager.substituteArguments(
            content: template,
            arguments: ["arguments": "deploy \"prod us\" --dry-run"])

        // Quoted spans stay whole; the bare form is the raw string; the
        // indexed forms index the split.
        XCTAssertEqual(substituted, "Run deploy against prod us with everything: deploy \"prod us\" --dry-run")
    }

    func testRawArgumentsAppendWhenBodyHasNoPlaceholder() {
        let manager = SkillManager()
        let substituted = manager.substituteArguments(
            content: "Do the standard review workflow.",
            arguments: ["args": "focus on the parser"])

        XCTAssertTrue(substituted.hasSuffix("\n\nARGUMENTS: focus on the parser"))

        // A body that DID carry a placeholder (any supported kind) gets no
        // duplicate append, even when the placeholder is an unfilled one.
        let withPlaceholder = manager.substituteArguments(
            content: "Work in ${SKILL_DIR}.",
            arguments: ["args": "focus on the parser"])
        XCTAssertFalse(withPlaceholder.contains("ARGUMENTS:"))
    }

    func testListingExcludesBlockedAndUnactivatedConditionalSkills() {
        let manager = SkillManager.shared
        let skills = [
            AppSkill(
                manifest: SkillManifest(name: "open", description: "Plain skill"),
                content: "c",
                sourceURL: tempDirURL.appendingPathComponent("open.md"),
                scope: .userGlobal),
            AppSkill(
                manifest: SkillManifest(name: "off", description: "User disabled"),
                content: "c",
                sourceURL: tempDirURL.appendingPathComponent("off.md"),
                scope: .userGlobal,
                isEnabled: false),
            AppSkill(
                manifest: SkillManifest(name: "modelblocked", description: "Model may not call",
                    disableModelInvocation: true),
                content: "c",
                sourceURL: tempDirURL.appendingPathComponent("modelblocked.md"),
                scope: .userGlobal),
            AppSkill(
                manifest: SkillManifest(name: "conditional", description: "Needs a touch",
                    paths: ["**/*.swift"]),
                content: "c",
                sourceURL: tempDirURL.appendingPathComponent("conditional.md"),
                scope: .userGlobal),
        ]

        // Not activated: the conditional skill is withheld with the blocked ones.
        let before = manager.formatSkillsWithinBudget(
            skills, contextWindowTokens: 200_000, activatedSkillNames: [])
        XCTAssertTrue(before.contains("- open [User]: Plain skill"))
        XCTAssertFalse(before.contains("- off"))
        XCTAssertFalse(before.contains("- modelblocked"))
        XCTAssertFalse(before.contains("- conditional"))

        // Activated (the same set the session records on a matching touch):
        // the conditional skill joins the listing.
        let after = manager.formatSkillsWithinBudget(
            skills, contextWindowTokens: 200_000, activatedSkillNames: ["conditional"])
        XCTAssertTrue(after.contains("- conditional [User]: Needs a touch"))
    }

    func testForkSkillRoutesThroughTheSubagentPath() async throws {
        let skillDir = tempDirURL.appendingPathComponent(".turbospark/skills/forked-job", isDirectory: true)
        try FileManager.default.createDirectory(at: skillDir, withIntermediateDirectories: true)
        try """
        ---
        name: forked-job
        description: Runs isolated from the main conversation.
        context: fork
        ---
        Isolated body ${input} for the subagent.
        """.write(to: skillDir.appendingPathComponent("SKILL.md"), atomically: true, encoding: .utf8)

        // No session installed: SubagentRunner fails fast, which is the
        // observable that PROVES the fork route ran -- an inline expansion
        // would have returned the body as a successful result.
        let savedSession = AppToolRegistry.activeSessionProvider
        let savedPrompt = AppToolRegistry.userSystemPromptProvider
        AppToolRegistry.activeSessionProvider = nil
        AppToolRegistry.userSystemPromptProvider = nil
        defer {
            AppToolRegistry.activeSessionProvider = savedSession
            AppToolRegistry.userSystemPromptProvider = savedPrompt
        }

        let project = AppProject(name: "TestProject", rootDirectoryPath: tempDirURL.path)
        let call = AppToolCall(name: "skill", arguments: ["name": "forked-job", "input": "x"])
        let result = await AppToolRegistry.execute(call: call, in: project)
        XCTAssertTrue(result.isError, "fork without a session must surface the subagent's failure")
        XCTAssertTrue(result.output.contains("No active model session"))
    }

    func testConditionalSkillActivation() {
        let manager = SkillManager.shared
        manager.clearActivatedConditionalSkills()

        let conditionalSkill = AppSkill(
            manifest: SkillManifest(name: "swift-optimizer", description: "Optimize swift", paths: ["**/*.swift"]),
            content: "Optimize swift code.",
            sourceURL: tempDirURL.appendingPathComponent("swift-optimizer.md"),
            scope: .userGlobal
        )

        // Before touch
        XCTAssertFalse(manager.isSkillEligible(conditionalSkill, activatedSkillNames: manager.activatedConditionalSkillNames))

        // Touched a non-matching file
        let activated1 = manager.evaluateConditionalSkills(
            filePaths: ["docs/readme.md"],
            projectURL: tempDirURL,
            currentlyActivated: manager.activatedConditionalSkillNames
        )
        XCTAssertTrue(activated1.isEmpty)

        // Matches pattern directly
        XCTAssertTrue(manager.matchesPath(skill: conditionalSkill, filePath: "Sources/Core/Engine.swift"))
        XCTAssertFalse(manager.matchesPath(skill: conditionalSkill, filePath: "docs/readme.md"))
    }

    func testMarketplaceDirectoryInstallation() async throws {
        // Create a mock source directory with a valid SKILL.md
        let mockSourceDir = tempDirURL.appendingPathComponent("mock_source_skill", isDirectory: true)
        try FileManager.default.createDirectory(at: mockSourceDir, withIntermediateDirectories: true)
        let skillMd = """
        ---
        name: imported-mock-skill
        description: A mock skill installed via directory source.
        ---
        # Mock Skill Instructions
        Execute mock routine.
        """
        try skillMd.write(to: mockSourceDir.appendingPathComponent("SKILL.md"), atomically: true, encoding: .utf8)

        let entry = MarketplaceSkillEntry(
            name: "imported-mock-skill",
            description: "A mock skill installed via directory source.",
            source: .directory(path: mockSourceDir.path),
            category: "utilities"
        )

        let projectDir = tempDirURL.appendingPathComponent("dest_project", isDirectory: true)
        try FileManager.default.createDirectory(at: projectDir, withIntermediateDirectories: true)

        let installed = try await SkillMarketplaceManager.shared.installSkill(
            entry: entry,
            targetScope: .projectLocal(projectPath: projectDir.path),
            projectRootURL: projectDir
        )

        XCTAssertEqual(installed.name, "imported-mock-skill")
        XCTAssertTrue(FileManager.default.fileExists(atPath: projectDir.appendingPathComponent(".turbospark/skills/imported-mock-skill/SKILL.md").path))
    }

    func testProposeSkillsToProjectScope() throws {
        let result = try ProposeSkillsExecutor.execute(
            arguments: [
                "name": "auto-gen-skill",
                "description": "An auto-generated test skill",
                "skillMd": "---\nname: auto-gen-skill\ndescription: Test\n---\n# Test\nSteps here."
            ],
            projectRootURL: tempDirURL
        )

        XCTAssertTrue(result.contains("Successfully saved"))
        let expectedFile = tempDirURL.appendingPathComponent(".turbospark/skills/auto-gen-skill/SKILL.md")
        XCTAssertTrue(FileManager.default.fileExists(atPath: expectedFile.path))
    }
}
