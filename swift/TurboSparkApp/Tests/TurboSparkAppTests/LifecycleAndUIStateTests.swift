import XCTest

@testable import TurboSpark
@testable import TurboSparkApp

/// Tier 3: the permission ordering (state#46), the subagent gaps (state#47),
/// the skill/agent stores (state#48, #57), the model, server and project
/// races (state#50 to #55), the parsers (state#56, #58), the hook runner
/// (state#60) and the settings decode (state#59).
final class LifecycleAndUIStateTests: XCTestCase {

    // MARK: - state#46: permissive is not absolute

    private func terminalCall(_ command: String) -> AppToolCall {
        AppToolCall(name: "run_command", arguments: ["command": command], category: .terminal)
    }

    private func permissiveProject() -> AppProject {
        AppProject(
            name: "p", rootDirectoryPath: "/tmp/p",
            permissions: AppProjectPermissions(
                mode: .permissive, fileRead: .allow, fileWrite: .allow, terminal: .allow,
                web: .allow, mcp: .allow, automation: .allow))
    }

    /// `permissive` short-circuited at step 2, above category deny, above the
    /// high-risk gate and above the terminal allowlist -- so the mode a user
    /// picks to stop being asked was also the one that stopped asking about
    /// the class of call the engine exists for.
    func testPermissiveStillAsksBeforeAHighRiskCommand() {
        guard
            case .ask = AppToolPermissionEngine.evaluate(
                call: terminalCall("rm -rf ~/Documents"), project: permissiveProject(),
                globalServers: [])
        else {
            return XCTFail("A destructive command must reach the approval card in any mode.")
        }
    }

    /// An explicitly denied category is honoured under `permissive` too: the
    /// ordering comment always claimed deny had the first word.
    func testPermissiveStillHonoursAnExplicitCategoryDeny() {
        var project = permissiveProject()
        project.permissions.terminal = .deny
        guard
            case .deny = AppToolPermissionEngine.evaluate(
                call: terminalCall("ls"), project: project, globalServers: [])
        else {
            return XCTFail("An explicit deny must outrank the mode.")
        }
    }

    /// And it must still be permissive: a safe call runs with no prompt, or
    /// the mode means nothing.
    func testPermissiveStillRunsAnOrdinaryCallWithNoPrompt() {
        XCTAssertEqual(
            AppToolPermissionEngine.evaluate(
                call: AppToolCall(
                    name: "read_file", arguments: ["path": "a.txt"], category: .fileRead),
                project: permissiveProject(), globalServers: []),
            .allow)
    }

    // MARK: - state#56: the skill frontmatter scanner

    /// `|-` and `>-` are the markers most authors write, and neither was
    /// recognized: the marker itself became the description and its body was
    /// reparsed as `key: value` pairs -- so a `name:` line written inside a
    /// description RENAMED the skill, which is how a project skill silently
    /// shadows a user skill of that name.
    func testAStripChompedBlockScalarDoesNotRenameTheSkill() {
        let text = """
            ---
            name: deploy
            description: |-
              Ships the service.
              name: something-else
              Use with care.
            ---

            body
            """
        let skill = SkillParser.parseContent(rawText: text)
        XCTAssertEqual(skill.name, "deploy")
        XCTAssertTrue(skill.skillDescription.contains("Ships the service."))
        XCTAssertTrue(skill.skillDescription.contains("name: something-else"))
    }

    /// A blank line inside a block scalar is PART of the block. Ending there
    /// is what let the paragraph after it be reparsed as `key: value` pairs,
    /// and it is reachable on an ordinary LF file too.
    func testABlankLineInsideADescriptionDoesNotEndIt() {
        let text = """
            ---
            name: deploy
            description: |
              Ships the service.

              name: something-else
            ---

            body
            """
        let skill = SkillParser.parseContent(rawText: text)
        XCTAssertEqual(skill.name, "deploy")
        XCTAssertTrue(skill.skillDescription.contains("name: something-else"))
    }

    /// `components(separatedBy: .newlines)` splits on EACH character of the
    /// set, so a CRLF file yields an empty line between every real one --
    /// and the block loop treated the first of those as the block's end.
    ///
    /// What this really pins is the BLANK-LINE TOLERANCE above, which covers
    /// the CRLF case on its own; the `normalizedLines` fold is redundant for
    /// it and its mutation correctly survives. See that function's own note.
    func testACRLFFileParsesTheSameAsAnLFOne() {
        let lf = """
            ---
            name: deploy
            description: |
              Ships the service.
              name: something-else
            ---

            body
            """
        let crlf = lf.replacingOccurrences(of: "\n", with: "\r\n")
        let a = SkillParser.parseContent(rawText: lf)
        let b = SkillParser.parseContent(rawText: crlf)
        XCTAssertEqual(b.name, "deploy")
        XCTAssertEqual(a.name, b.name)
        XCTAssertEqual(
            a.skillDescription.trimmingCharacters(in: .whitespacesAndNewlines),
            b.skillDescription.trimmingCharacters(in: .whitespacesAndNewlines))
    }

    // MARK: - state#58: the bounded state is bounded

    /// `validate` checked TYPES and nothing else, so the "bounded" execution
    /// state grew without limit and reproduced the context exhaustion the
    /// whole feature was measured to avoid.
    func testAnOversizedCollectionIsRejected() {
        let tooMany = (0..<(AppSkillStatePatch.maxCollectionCount + 1)).map {
            AppJSONValue.string("fact \($0)")
        }
        let errors = AppSkillStatePatch.validate(["facts": .array(tooMany)])
        XCTAssertFalse(errors.isEmpty)
        XCTAssertTrue(errors.joined().contains("limit"))
    }

    func testAnOversizedStringIsRejected() {
        let huge = String(repeating: "x", count: AppSkillStatePatch.maxStringLength + 1)
        XCTAssertFalse(AppSkillStatePatch.validate(["goal": .string(huge)]).isEmpty)
    }

    /// A patch within every per-field bound is still accepted, or the bound
    /// is a refusal rather than a limit.
    func testAnOrdinaryPatchIsStillAccepted() {
        XCTAssertTrue(
            AppSkillStatePatch.validate([
                "goal": .string("ship it"),
                "facts": .array([.string("a"), .string("b")]),
            ]).isEmpty)
    }

    // MARK: - state#54: porcelain paths

    /// `R  old -> new` was taken literally, so the row named a file that does
    /// not exist and the `--numstat` lookup beside it missed -- reporting 0
    /// additions and 0 deletions for every rename.
    func testARenameReportsTheNewPath() {
        XCTAssertEqual(WorktreeModel.porcelainPath("old/name.swift -> new/name.swift"),
            "new/name.swift")
    }

    /// A path with a space, a quote or a non-ASCII byte arrives C-quoted.
    func testAQuotedPathIsUnescaped() {
        XCTAssertEqual(WorktreeModel.porcelainPath(#""My Project/a b.swift""#), "My Project/a b.swift")
        XCTAssertEqual(WorktreeModel.porcelainPath(#""caf\303\251.txt""#), "café.txt")
    }

    /// The overwhelmingly common case is an ordinary relative path, returned
    /// untouched.
    func testAnOrdinaryPathIsUnchanged() {
        XCTAssertEqual(WorktreeModel.porcelainPath("Sources/App.swift"), "Sources/App.swift")
    }

    // MARK: - state#60: the hook command string

    /// A path with a space split into two shell arguments; one with a `;` or
    /// a backtick executed whatever followed.
    func testAProjectPathIsShellQuoted() {
        XCTAssertEqual(
            AppHookExecutionEngine.shellQuoted("/Users/me/My Projects/app"),
            "'/Users/me/My Projects/app'")
        XCTAssertEqual(
            AppHookExecutionEngine.shellQuoted("/tmp/a'b; rm -rf ~"),
            "'/tmp/a'\\''b; rm -rf ~'")
    }

    // MARK: - state#59: one mistyped setting must not reset every preference

    /// `decodeIfPresent` tolerates an absent key and nothing else, so
    /// `"seed": -1` threw `typeMismatch` and the file was quarantined.
    func testAMistypedSettingFallsBackWithoutLosingTheOthers() throws {
        let json = #"{"seed": -1, "topK": "not a number", "temperature": 0.7}"#
        let settings = try JSONDecoder().decode(MacAppSettings.self, from: Data(json.utf8))
        XCTAssertEqual(settings.temperature, 0.7, "The readable fields must survive.")
        XCTAssertEqual(settings.seed, 0)
        XCTAssertEqual(settings.topK, 64)
    }

    /// JSON that will not parse at all is still a corrupt FILE, and is still
    /// quarantined rather than read as defaults.
    func testGenuinelyCorruptJsonStillThrows() {
        XCTAssertThrowsError(
            try JSONDecoder().decode(MacAppSettings.self, from: Data("{not json".utf8)))
    }
}

/// The `AppModel`-level half of Tier 3.
@MainActor
final class LifecycleAndUIStateModelTests: XCTestCase {

    // MARK: - state#49: clearing a conversation clears what it owned

    /// Skill state, a pending approval and the checklist all outlived
    /// `clearOutput`, so a cleared conversation came back carrying
    /// bookkeeping for a run the user could no longer read.
    func testClearingAConversationClearsItsCarriedState() {
        let model = AppModel()
        var chat = AppChat(title: "carrying")
        chat.messages = [AppChatMessage(role: .user, content: "hi")]
        chat.todos = [TodoItem(content: "a", activeForm: "doing a")]
        chat.skillState = AppSkillState(fields: ["goal": .string("ship")])
        model.chats = [chat]
        model.selectedChatID = chat.id
        model.pendingToolCall = AppToolCall(
            name: "read_file", arguments: ["path": "a"], category: .fileRead)
        model.pendingToolCallChatID = chat.id
        model.skillStateLastError = "an old rejection"

        model.clearOutput()

        XCTAssertTrue(model.chats[0].messages.isEmpty)
        XCTAssertTrue(model.chats[0].todos.isEmpty)
        XCTAssertNil(model.chats[0].skillState)
        XCTAssertNil(model.pendingToolCall)
        XCTAssertNil(model.skillStateLastError)
    }

    /// `createChat` had no `pendingToolCall` guard where `selectChat` did, so
    /// starting a new chat stranded the approval card on a conversation the
    /// user could no longer see.
    func testANewChatIsRefusedWhileACallAwaitsApproval() {
        let model = AppModel()
        var chat = AppChat(title: "waiting")
        // A message, or `createChat`'s own "reuse the empty selected chat"
        // arm returns the same id whatever the guard says and the case cannot
        // discriminate -- which is what the mutation check reported first
        // time round.
        chat.messages = [AppChatMessage(role: .user, content: "hi")]
        model.chats = [chat]
        model.selectedChatID = chat.id
        model.pendingToolCall = AppToolCall(
            name: "read_file", arguments: ["path": "a"], category: .fileRead)
        model.pendingToolCallChatID = chat.id

        let created = model.createChat()

        XCTAssertEqual(created, chat.id)
        XCTAssertEqual(model.chats.count, 1)
    }

    // MARK: - state#63: the draft chat is not lost on quit

    /// Draft text written before any chat existed went to a plain stored
    /// property that nothing published and the quit flush never walked.
    func testTheDraftChatIsMaterializedOnceItHasContent() {
        let model = AppModel()
        model.chats = []
        model.selectedChatID = UUID()
        XCTAssertNil(model.selectedChatIndex, "Precondition: nothing selected yet.")

        model.promptText = "something worth keeping"

        XCTAssertNotNil(
            model.selectedChatIndex,
            "A draft with content must join `chats`, which is what the flush walks.")
        XCTAssertEqual(model.selectedChat.draft, "something worth keeping")
    }

    /// An empty draft stays transient: a chat row that appears before the
    /// user types anything is noise.
    func testAnEmptyDraftIsNotMaterialized() {
        let model = AppModel()
        model.chats = []
        model.selectedChatID = UUID()
        model.promptText = ""
        XCTAssertTrue(model.chats.isEmpty)
    }

    // MARK: - state#50 / state#51: the model races

    /// Two overlapping opens each ran `defer { opening = false }`, so the
    /// first to finish cleared the flag for both and `selected` and `session`
    /// could name different models.
    func testASecondOpenIsRefusedWhileOneIsInFlight() async {
        let model = AppModel()
        model.opening = true
        XCTAssertFalse(model.canLoadModel)
        XCTAssertFalse(model.canReloadModel)

        let row = InstalledModel(
            alias: "a", repo: "r", revision: "main", path: "/tmp/a.gturbo",
            family: "gemma4", installBytes: 1, installedOn: "today")
        model.installed = [row]
        await model.open(row)
        XCTAssertTrue(model.opening, "The in-flight open still owns the flag.")
    }
}
