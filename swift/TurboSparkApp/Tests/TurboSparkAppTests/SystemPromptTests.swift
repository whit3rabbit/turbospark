import TurboSpark
import XCTest

@testable import TurboSparkApp

/// The user-authored system prompt: the app-wide default, the per-chat
/// override, and the one safety property the restructure had to preserve.
///
/// **`MacAppSettings`'s default is what these assert, never `AppModel`'s
/// declared one** (`swift/CLAUDE.md` Gotcha 39): `AppModel.init()` calls
/// `loadSettings()` as its first statement, so the declared default is
/// overwritten before any caller can observe it and a test of it would pass
/// or fail on whatever `settings.json` an earlier test in the process left.
final class SystemPromptTests: XCTestCase {

    // MARK: - Persistence

    func testTheDefaultSystemPromptLibrarySelectsTurboSparkAgent() {
        let settings = MacAppSettings()
        XCTAssertEqual(settings.systemPrompts, AppSystemPrompt.builtIns)
        XCTAssertEqual(settings.systemPrompts.map(\.name), [
            "TurboSpark Agent", "Compact Agent", "Code Reviewer"
        ])
        XCTAssertEqual(settings.activeSystemPromptID, AppSystemPrompt.builtIns[0].id.uuidString)
        XCTAssertEqual(settings.defaultSystemPrompt, AppSystemPrompt.builtIns[0].instructions)
    }

    func testTheDefaultSystemPromptRoundTrips() throws {
        let settings = MacAppSettings(defaultSystemPrompt: "Answer only in haiku.")
        let data = try JSONEncoder().encode(settings)
        let decoded = try JSONDecoder().decode(MacAppSettings.self, from: data)
        XCTAssertEqual(decoded.defaultSystemPrompt, "Answer only in haiku.")
        XCTAssertEqual(decoded.systemPrompts.map(\.name), ["Imported Default"])
        XCTAssertEqual(decoded.activeSystemPromptID, decoded.systemPrompts[0].id.uuidString)
    }

    /// Gotcha 13's rule: a settings file written before this field existed has
    /// to decode, or the store swallows the failure and discards every other
    /// preference with it.
    func testSettingsWrittenBeforeTheFieldExistedStillDecode() throws {
        let legacy = #"{"contextTokens":0,"temperature":0.2}"#
        let decoded = try JSONDecoder().decode(MacAppSettings.self, from: Data(legacy.utf8))
        XCTAssertEqual(decoded.defaultSystemPrompt, AppSystemPrompt.builtIns[0].instructions)
        XCTAssertEqual(decoded.systemPrompts, AppSystemPrompt.builtIns)
        XCTAssertEqual(decoded.activeSystemPromptID, AppSystemPrompt.builtIns[0].id.uuidString)
    }

    func testLegacyCustomDefaultMigratesIntoThePromptLibrary() throws {
        let legacy = #"{"defaultSystemPrompt":"LEGACY-PROMPT"}"#
        let decoded = try JSONDecoder().decode(MacAppSettings.self, from: Data(legacy.utf8))

        XCTAssertEqual(decoded.systemPrompts.map(\.name), ["Imported Default"])
        XCTAssertEqual(decoded.systemPrompts.map(\.instructions), ["LEGACY-PROMPT"])
        XCTAssertEqual(decoded.activeSystemPromptID, decoded.systemPrompts[0].id.uuidString)
        XCTAssertEqual(decoded.defaultSystemPrompt, "LEGACY-PROMPT")
    }

    /// The same rule on the chat archive, which is the store that actually
    /// lost a user's data once (Gotcha 13).
    func testAChatWrittenBeforeThePerChatPromptExistedStillDecodes() throws {
        let legacy = """
        {"id":"6C7E1E2E-0F2B-4C1E-9E7A-2B7A9D1C4E55","title":"Old","draft":"",
         "draftAttachments":[],"messages":[],"todos":[],
         "createdAt":0,"updatedAt":0}
        """
        let decoded = try JSONDecoder().decode(AppChat.self, from: Data(legacy.utf8))
        XCTAssertNil(decoded.systemPrompt)
    }

    // MARK: - Resolution: per-chat beats the default

    @MainActor
    func testAPerChatPromptOverridesTheDefault() {
        let model = AppModel()
        model.defaultSystemPrompt = "GLOBAL"
        model.chats = [AppChat(title: "c", systemPrompt: "PER CHAT")]
        XCTAssertEqual(model.resolvedUserSystemPrompt(chatIndex: 0), "PER CHAT")
    }

    /// A cleared editor is the way back to the default. Reading blank as
    /// "send nothing" would leave a user no route back from inside the chat.
    @MainActor
    func testAnEmptyPerChatPromptFallsBackToTheDefault() {
        let model = AppModel()
        model.defaultSystemPrompt = "GLOBAL"
        model.chats = [
            AppChat(title: "nil", systemPrompt: nil),
            AppChat(title: "blank", systemPrompt: "   "),
        ]
        XCTAssertEqual(model.resolvedUserSystemPrompt(chatIndex: 0), "GLOBAL")
        XCTAssertEqual(model.resolvedUserSystemPrompt(chatIndex: 1), "GLOBAL")
    }

    /// The draft chat is not in `chats`, so the index can be out of range on a
    /// real turn. It must resolve the default rather than trap.
    @MainActor
    func testAnOutOfRangeChatIndexResolvesTheDefault() {
        let model = AppModel()
        model.defaultSystemPrompt = "GLOBAL"
        model.chats = []
        XCTAssertEqual(model.resolvedUserSystemPrompt(chatIndex: 0), "GLOBAL")
    }

    @MainActor
    func testNoPromptAnywhereResolvesToEmpty() {
        let model = AppModel()
        model.defaultSystemPrompt = ""
        model.chats = [AppChat(title: "c")]
        XCTAssertEqual(model.resolvedUserSystemPrompt(chatIndex: 0), "")
    }

    @MainActor
    func testSelectedLibraryPromptResolvesAsTheAppDefault() {
        let prompt = AppSystemPrompt(
            id: UUID(uuidString: "F5C6F07A-AEF1-4551-A2B2-4F4D5D8E74B7")!,
            name: "Custom",
            instructions: "LIBRARY-PROMPT")
        let model = AppModel()
        model.systemPrompts = [prompt]
        model.selectedSystemPromptID = prompt.id
        model.defaultSystemPrompt = prompt.instructions
        model.chats = [AppChat(title: "c")]

        XCTAssertEqual(model.selectedSystemPrompt, prompt)
        XCTAssertEqual(model.resolvedUserSystemPrompt(chatIndex: 0), "LIBRARY-PROMPT")
    }

    // MARK: - Assembly, and the safety property

    /// Chat mode gets the user's prose. This is the behaviour change.
    @MainActor
    func testAProjectlessChatSendsTheUserPromptAndNothingElse() {
        let model = AppModel()
        let prompt = model.buildSystemPrompt(for: nil, userPrompt: "Speak plainly.")
        XCTAssertEqual(prompt, "Speak plainly.")
    }

    /// **THE LOAD-BEARING ONE.** Every project-derived section stays behind the
    /// nil-project guard, and one of them is the tool vocabulary. Emitting it
    /// without a project would advertise tools to a conversation with no root
    /// to run them against (state#82's hazard).
    ///
    /// Asserted against the real addendum rather than a guessed substring, so
    /// it cannot pass by looking for a word that vocabulary stopped using.
    @MainActor
    func testAProjectlessChatIsOfferedNoToolsEvenWithAUserPrompt() {
        let model = AppModel()
        let prompt = model.buildSystemPrompt(for: nil, userPrompt: "Speak plainly.")
        let toolVocabulary = AppToolRegistry.systemPromptAddendum(for: .general)
        XCTAssertFalse(
            toolVocabulary.isEmpty,
            "the fixture is meaningless if the vocabulary is empty")
        XCTAssertFalse(
            prompt.contains(toolVocabulary),
            "a projectless chat must not be told it has tools")
    }

    /// And with no user prompt it is empty, exactly as before this feature.
    @MainActor
    func testAProjectlessChatWithNoUserPromptIsStillEmpty() {
        let model = AppModel()
        XCTAssertEqual(model.buildSystemPrompt(for: nil, userPrompt: ""), "")
        XCTAssertEqual(model.buildSystemPrompt(for: nil), "")
    }

    /// The user's prompt LEADS, so a project's own rules read as refinements
    /// of it rather than as a competing set of instructions.
    @MainActor
    func testTheUserPromptLeadsAndProjectSectionsFollow() {
        let model = AppModel()
        let project = AppProject(
            name: "Demo", rootDirectoryPath: "/tmp/demo", customInstructions: "Use tabs")
        let prompt = model.buildSystemPrompt(for: project, userPrompt: "Speak plainly.")

        XCTAssertTrue(prompt.hasPrefix("Speak plainly."), "user prompt must come first")
        XCTAssertTrue(prompt.contains("/tmp/demo"), "project sections must survive")
        XCTAssertTrue(prompt.contains("Use tabs"))
    }

    @MainActor
    func testProjectPromptIncludesTheMacOSHarnessAndSubagentMatches() {
        let model = AppModel()
        let project = AppProject(name: "Demo", rootDirectoryPath: "/tmp/demo")
        let sections = model.buildSystemPromptSections(for: project, userPrompt: "USER-PROMPT")
        let environment = sections.first { $0.section == .environment }?.content

        XCTAssertEqual(
            environment,
            project.turboSparkEnvironmentPrompt())
        XCTAssertTrue(environment?.contains("TurboSpark macOS app") ?? false)
        XCTAssertTrue(environment?.contains("Tool calls perform work; prose does not.") ?? false)

        let agent = AgentManager.shared.findAgent(name: "general-purpose")
        XCTAssertNotNil(agent)
        if let agent {
            let subagentPrompt = SubagentRunner.buildSystemPrompt(for: agent, project: project)
            XCTAssertTrue(subagentPrompt.contains(project.turboSparkEnvironmentPrompt()))
        }
    }

    // MARK: - The prompt reaches the history as exactly one system turn

    /// Three of the five fallback renderers refuse a system message at any
    /// index but 0, so "exactly one, first" is a correctness requirement and
    /// not a stylistic one (state#32, state#74).
    @MainActor
    func testTheHistoryCarriesExactlyOneSystemMessageAndItIsFirst() {
        let model = AppModel()
        model.defaultSystemPrompt = "GLOBAL"
        var chat = AppChat(title: "c")
        chat.messages = [
            AppChatMessage(role: .user, content: "hi"),
            AppChatMessage(role: .assistant, content: "hello"),
            AppChatMessage(role: .user, content: "again"),
        ]
        model.chats = [chat]

        let history = model.buildAppendOnlyHistory(chatIndex: 0, project: nil)

        XCTAssertEqual(history.filter { $0.role == .system }.count, 1)
        XCTAssertEqual(history.first?.role, .system)
        XCTAssertEqual(history.first?.content, "GLOBAL")
    }

    /// The other half of the same property: no default anywhere means no
    /// system turn at all, rather than an empty one.
    @MainActor
    func testNoPromptMeansNoSystemMessageAtAll() {
        let model = AppModel()
        model.defaultSystemPrompt = ""
        var chat = AppChat(title: "c")
        chat.messages = [AppChatMessage(role: .user, content: "hi")]
        model.chats = [chat]

        let history = model.buildAppendOnlyHistory(chatIndex: 0, project: nil)
        XCTAssertTrue(history.allSatisfy { $0.role != .system })
    }

    @MainActor
    func testThePerChatPromptIsWhatReachesTheHistory() {
        let model = AppModel()
        model.defaultSystemPrompt = "GLOBAL"
        var chat = AppChat(title: "c", systemPrompt: "PER CHAT")
        chat.messages = [AppChatMessage(role: .user, content: "hi")]
        model.chats = [chat]

        let history = model.buildAppendOnlyHistory(chatIndex: 0, project: nil)
        XCTAssertEqual(history.first?.content, "PER CHAT")
    }

    // MARK: - The second assembler

    /// `SubagentRunner` shares no code with `AppModel.buildSystemPrompt`, so a
    /// prompt added to one and not the other applies to half this app's runs.
    func testASubagentPromptCarriesTheUserPromptFirst() {
        let agent = AgentManager.shared.findAgent(name: "general-purpose")
        XCTAssertNotNil(agent, "the built-in general-purpose agent should exist")
        guard let agent else { return }

        let prompt = SubagentRunner.buildSystemPrompt(
            for: agent, project: nil, userPrompt: "Speak plainly.")
        XCTAssertTrue(prompt.hasPrefix("Speak plainly."))
        XCTAssertTrue(prompt.contains("Subagent Role"), "the agent role must survive")

        let without = SubagentRunner.buildSystemPrompt(for: agent, project: nil)
        XCTAssertFalse(without.contains("Speak plainly."))
        XCTAssertTrue(without.hasPrefix("## Subagent Role"))
    }

    // MARK: - Shell quoting for the launch command

    /// The prompt is user-authored free text spliced into a command the user
    /// pastes into a shell. An apostrophe is the common case, not an exotic
    /// one, and unquoted it ends the argument.
    func testShellQuotingSurvivesAnApostrophe() {
        XCTAssertEqual(ShellQuote.single("don't"), "'don'\\''t'")
        XCTAssertEqual(ShellQuote.single("plain"), "'plain'")
    }

    /// Every character a shell would otherwise act on is inert inside single
    /// quotes, so the quoting adds nothing for them and must not mangle them.
    func testShellQuotingLeavesOtherMetacharactersAlone() {
        let hostile = "a; rm -rf / && echo $HOME `id` | tee \"x\"\nnewline"
        XCTAssertEqual(ShellQuote.single(hostile), "'\(hostile)'")
    }
}

// MARK: - The composer's context meter

extension SystemPromptTests {
    /// The estimate omitted the system message entirely before this feature, a
    /// small undercount while the prompt was project-derived and Chat mode had
    /// none. A user-authored default applies to every chat, so leaving it out
    /// would under-report the context fill by the whole prompt on every turn.
    ///
    /// Asserted on the ASSEMBLY rather than on `countTokens`, which needs a
    /// real session and a Metal device (`swift/CLAUDE.md` Gotcha 26).
    @MainActor
    func testTheEstimateAndTheSentHistoryStartTheSameWay() {
        let model = AppModel()
        model.defaultSystemPrompt = "GLOBAL"
        var chat = AppChat(title: "c")
        chat.messages = [AppChatMessage(role: .user, content: "hi")]
        model.chats = [chat]
        model.selectedChatID = chat.id

        let sent = model.buildAppendOnlyHistory(chatIndex: 0, project: nil)
        XCTAssertEqual(sent.first?.role, .system)
        XCTAssertEqual(
            sent.first?.content,
            model.buildSystemPrompt(
                for: model.turnProject(chatID: model.selectedChatID),
                userPrompt: model.resolvedUserSystemPrompt(chat: model.selectedChat)),
            "the estimate must price the same system message the turn sends")
    }

    /// The draft chat is not in `chats`, so the by-index form cannot reach it.
    /// The by-chat form is what `updateTokenEstimate` uses for that reason.
    @MainActor
    func testTheByChatResolverReachesAChatThatIsNotInTheList() {
        let model = AppModel()
        model.defaultSystemPrompt = "GLOBAL"
        model.chats = []
        let orphan = AppChat(title: "draft", systemPrompt: "PER CHAT")
        XCTAssertEqual(model.resolvedUserSystemPrompt(chat: orphan), "PER CHAT")
        XCTAssertEqual(model.resolvedUserSystemPrompt(chatIndex: 0), "GLOBAL")
    }
}
