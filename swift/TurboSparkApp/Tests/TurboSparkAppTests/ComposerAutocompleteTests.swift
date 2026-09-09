import Foundation
import XCTest

@testable import TurboSparkApp

/// Tests for the composer autocomplete: trigger detection, candidate
/// matching, controller accept/dismiss, and the built-in command registry's
/// coupling to the submit-time parser.
final class ComposerAutocompleteTests: XCTestCase {

    // MARK: - Trigger detection

    func testAtAloneTriggersEmptyMentionQuery() {
        let match = ComposerAutocompleteEngine.detectTrigger(in: "hello @")
        XCTAssertEqual(match?.trigger, .mention(query: ""))
        XCTAssertEqual(match?.tokenStartOffset, 6)
    }

    func testPartialPathTriggersMention() {
        let match = ComposerAutocompleteEngine.detectTrigger(in: "see @src/fo")
        XCTAssertEqual(match?.trigger, .mention(query: "src/fo"))
        XCTAssertEqual(match?.tokenStartOffset, 4)
    }

    func testSlashAloneTriggersEmptyCommandPrefix() {
        let match = ComposerAutocompleteEngine.detectTrigger(in: "/")
        XCTAssertEqual(match?.trigger, .slash(prefix: ""))
    }

    func testPartialCommandTriggersSlash() {
        XCTAssertEqual(
            ComposerAutocompleteEngine.detectTrigger(in: "run /pl")?.trigger,
            .slash(prefix: "pl"))
    }

    func testPlainTextTriggersNothing() {
        XCTAssertNil(ComposerAutocompleteEngine.detectTrigger(in: "no trigger here"))
    }

    func testTextEndingInWhitespaceTriggersNothing() {
        XCTAssertNil(ComposerAutocompleteEngine.detectTrigger(in: "hello @foo "))
    }

    func testDoubleAtIsEscapedAndTriggersNothing() {
        XCTAssertNil(ComposerAutocompleteEngine.detectTrigger(in: "literal @@ here"))
    }

    func testTokenInMiddleOfTextTriggersNothing() {
        // The trailing-token rule: only the LAST whitespace-delimited word
        // can trigger.
        XCTAssertNil(ComposerAutocompleteEngine.detectTrigger(in: "@foo bar"))
    }

    func testMentionAfterNewlineTriggers() {
        XCTAssertEqual(
            ComposerAutocompleteEngine.detectTrigger(in: "line\n@x")?.trigger,
            .mention(query: "x"))
    }

    func testUnterminatedQuotedPathKeepsTriggerAcrossSpaces() {
        let match = ComposerAutocompleteEngine.detectTrigger(in: "see @\"my partial path")
        XCTAssertEqual(match?.trigger, .mention(query: "my partial path"))
        XCTAssertEqual(match?.tokenStartOffset, 4)
    }

    func testAtQuoteAloneTriggersEmptyQuotedQuery() {
        XCTAssertEqual(
            ComposerAutocompleteEngine.detectTrigger(in: "@\"")?.trigger,
            .mention(query: ""))
    }

    func testClosedQuotedPathIsNoLongerATrigger() {
        XCTAssertNil(ComposerAutocompleteEngine.detectTrigger(in: "see @\"done path\""))
        XCTAssertNil(ComposerAutocompleteEngine.detectTrigger(in: "see @\"done path\" "))
    }

    func testQuotedTriggerRequiresTokenStart() {
        XCTAssertNil(ComposerAutocompleteEngine.detectTrigger(in: "x@\"partial"))
    }

    // MARK: - Slash candidates

    private func skill(
        _ name: String,
        enabled: Bool = true,
        userInvocable: Bool = true,
        description: String? = nil
    ) -> AppSkill {
        AppSkill(
            manifest: SkillManifest(
                name: name,
                description: description,
                userInvocable: userInvocable),
            content: "body",
            sourceURL: URL(fileURLWithPath: "/tmp/\(name).md"),
            isEnabled: enabled)
    }

    func testEmptyPrefixOffersAllBuiltIns() {
        let rows = ComposerAutocompleteEngine.slashCandidates(prefix: "", skills: [])
        // One row per table NAME (aliases never get rows), capped at
        // maxSuggestions -- the table has outgrown one screen, so the empty
        // prefix pins the cap and the dedupe, not the whole alphabet.
        let primaryNames = Set(BuiltInSlashCommand.all.map { "/" + $0.name })
        XCTAssertTrue(rows.map(\.title).allSatisfy(primaryNames.contains))
        XCTAssertEqual(
            rows.count,
            min(ComposerAutocompleteEngine.maxSuggestions, BuiltInSlashCommand.all.count),
            "the empty prefix still lists every command that fits the cap")
        // A typed prefix reaches the rows past the cap.
        XCTAssertTrue(ComposerAutocompleteEngine.slashCandidates(prefix: "theme", skills: [])
            .contains { $0.title == "/theme" })
        XCTAssertTrue(ComposerAutocompleteEngine.slashCandidates(prefix: "tools", skills: [])
            .contains { $0.title == "/tools" })
        XCTAssertTrue(ComposerAutocompleteEngine.slashCandidates(prefix: "btw", skills: [])
            .contains { $0.title == "/btw" })
    }

    func testDisabledAndModelOnlySkillsAreNotOffered() {
        let rows = ComposerAutocompleteEngine.slashCandidates(
            prefix: "",
            skills: [skill("off", enabled: false), skill("modelonly", userInvocable: false)])
        XCTAssertFalse(rows.contains { $0.title == "/off" })
        XCTAssertFalse(rows.contains { $0.title == "/modelonly" })
    }

    func testSkillShadowedByCommandNameIsNotOffered() {
        let rows = ComposerAutocompleteEngine.slashCandidates(
            prefix: "", skills: [skill("explore"), skill("reviewer")])
        XCTAssertEqual(rows.filter { $0.kind == .skill }.count, 0)
    }

    func testPrefixMatchesRankBeforeSubstringMatches() {
        let rows = ComposerAutocompleteEngine.slashCandidates(
            prefix: "plan", skills: [skill("myplan")])
        XCTAssertEqual(rows.first?.title, "/plan")
        XCTAssertEqual(rows.last?.title, "/myplan")
    }

    func testDescriptionMatchesOnlyForThreeOrMoreTypedCharacters() {
        let helper = skill("zzz", description: "A plan helper")
        // Two typed characters: the description bucket is closed, so the
        // helper is absent even though "pl" occurs in its description. The
        // built-ins still match by name ("/plan" prefix, "explore" contains).
        let short = ComposerAutocompleteEngine.slashCandidates(prefix: "pl", skills: [helper])
        XCTAssertEqual(short.map(\.title), ["/plan", "/explore"])
        // Three characters open the description bucket and the helper appears.
        let longer = ComposerAutocompleteEngine.slashCandidates(prefix: "pla", skills: [helper])
        XCTAssertEqual(longer.map(\.title), ["/plan", "/zzz"])
    }

    // MARK: - Mention candidates

    private let entries: [ProjectFileEntry] = [
        ProjectFileEntry(relativePath: "README.md", isDirectory: false),
        ProjectFileEntry(relativePath: "src", isDirectory: true),
        ProjectFileEntry(relativePath: "src/App.swift", isDirectory: false),
        ProjectFileEntry(relativePath: "src/deep/Helper.swift", isDirectory: false),
    ]

    func testEmptyQueryOffersTheWholeTreeShallowestFirst() {
        let rows = ComposerAutocompleteEngine.mentionCandidates(query: "", entries: entries)
        XCTAssertEqual(
            rows.map(\.title),
            ["README.md", "src", "src/App.swift", "src/deep/Helper.swift"])
    }

    func testFilenamePrefixRanksBeforePathMatch() {
        let rows = ComposerAutocompleteEngine.mentionCandidates(query: "app", entries: entries)
        XCTAssertEqual(rows.first?.title, "src/App.swift")
        // "src/deep/Helper.swift" does not match and is absent.
        XCTAssertEqual(rows.count, 1)
    }

    func testQuerySpanningDirectoriesMatchesTheRelativePath() {
        let rows = ComposerAutocompleteEngine.mentionCandidates(
            query: "deep/help", entries: entries)
        XCTAssertEqual(rows.map(\.title), ["src/deep/Helper.swift"])
    }

    func testFolderRowQuotesTokenWithSpacesAndMarksKind() {
        let folder = [ProjectFileEntry(relativePath: "my notes", isDirectory: true)]
        let row = ComposerAutocompleteEngine.mentionCandidates(query: "my", entries: folder).first
        XCTAssertEqual(row?.kind, .folder)
        XCTAssertEqual(row?.replacementToken, "@\"my notes\"")
    }

    func testFileWithoutSpacesNeedsNoQuotes() {
        let file = [ProjectFileEntry(relativePath: "src/App.swift", isDirectory: false)]
        let row = ComposerAutocompleteEngine.mentionCandidates(query: "app", entries: file).first
        XCTAssertEqual(row?.replacementToken, "@src/App.swift")
    }

    func testMentionCandidatesAreCapped() {
        let many = (0..<50).map {
            ProjectFileEntry(relativePath: String(format: "file%02d.txt", $0), isDirectory: false)
        }
        XCTAssertEqual(
            ComposerAutocompleteEngine.mentionCandidates(query: "", entries: many).count,
            ComposerAutocompleteEngine.maxSuggestions)
    }

    // MARK: - Controller

    @MainActor
    func testAcceptReplacesOnlyTheTriggerToken() {
        let controller = ComposerAutocompleteController()
        controller.textChanged(text: "say /pl", skills: [], projectRoot: nil)
        XCTAssertTrue(controller.isVisible)
        let accepted = controller.accept(in: "say /pl")
        XCTAssertEqual(accepted, "say /plan ")
    }

    @MainActor
    func testSelectionWrapsAtBothEnds() {
        let controller = ComposerAutocompleteController()
        controller.textChanged(text: "/compact", skills: [], projectRoot: nil)
        XCTAssertEqual(controller.selectedIndex, 0)
        controller.moveSelection(-1)
        XCTAssertEqual(controller.selectedIndex, controller.suggestions.count - 1)
        controller.moveSelection(1)
        XCTAssertEqual(controller.selectedIndex, 0)
    }

    @MainActor
    func testDismissStaysClosedWhileTheTokenIsUnchangedAndReopensWhenItGrows() {
        let controller = ComposerAutocompleteController()
        controller.textChanged(text: "/pl", skills: [], projectRoot: nil)
        XCTAssertTrue(controller.isVisible)
        controller.dismiss()
        XCTAssertFalse(controller.isVisible, "Escape must close the popup")
        XCTAssertNil(controller.accept(in: "/pl"), "Escape must also refuse acceptance")

        controller.textChanged(text: "/pla", skills: [], projectRoot: nil)
        XCTAssertTrue(controller.isVisible, "Typing past the dismissed token reopens")
    }

    @MainActor
    func testLosingFocusClearsThePopup() {
        let controller = ComposerAutocompleteController()
        controller.textChanged(text: "/pl", skills: [], projectRoot: nil)
        XCTAssertTrue(controller.isVisible)
        controller.focusChanged(isFocused: false)
        XCTAssertFalse(controller.isVisible)
    }

    @MainActor
    func testMentionWithoutAProjectShowsTheHintAndOffersNothing() {
        let controller = ComposerAutocompleteController()
        controller.textChanged(text: "@src", skills: [], projectRoot: nil)
        XCTAssertTrue(controller.isVisible, "The hint row still shows")
        XCTAssertTrue(controller.suggestions.isEmpty)
        XCTAssertNotNil(controller.hint)
        XCTAssertNil(controller.accept(in: "@src"))
    }

    // MARK: - Registry drift

    @MainActor
    private func parserRecognizes(_ name: String, in model: AppModel) -> Bool {
        // `/agent`'s target is its own first argument, so the drift probe
        // supplies a real agent name; the fixed commands take any rest.
        let draft = name == "agent" ? "/agent explore drift-probe" : "/\(name) drift-probe"
        return model.handleAgentSlashCommand(draft)
    }

    @MainActor
    func testEveryRegisteredAgentSpellingIsRecognizedByTheParser() {
        let model = AppModel()
        for name in BuiltInSlashCommand.recognizedAgentNames {
            XCTAssertTrue(
                parserRecognizes(name, in: model),
                "the parser no longer recognizes /\(name); the menu and popup offer it")
        }
    }

    @MainActor
    func testTheParserRecognizesTheRegisteredSpellingsByLiteral() {
        let model = AppModel()
        // LITERALS, deliberately not `recognizedAgentNames`: iterating the
        // registry's own list would shrink the probe when a row is deleted
        // and the deletion would pass unnoticed. A literal here reddens.
        for name in ["explore", "plan", "review", "reviewer", "agent"] {
            XCTAssertTrue(
                parserRecognizes(name, in: model),
                "/\(name) fell through as prose; the registry lost it or the parser broke")
        }
        XCTAssertFalse(
            model.handleAgentSlashCommand("/definitely-not-registered-xyz probe"),
            "the parser recognizes a command the registry never recorded")
    }

    func testMetaCommandsAreThePinnedSet() {
        XCTAssertEqual(
            BuiltInSlashCommand.all.filter { $0.kind == .metaCommand }.map(\.name),
            ["compact", "memory", "goal", "stats", "export", "help"])
        XCTAssertTrue(BuiltInSlashCommand.isMetaCommand("/compact"))
        XCTAssertTrue(BuiltInSlashCommand.isMetaCommand("/compact 10"))
        XCTAssertFalse(BuiltInSlashCommand.isMetaCommand("/compactify"))
        XCTAssertFalse(BuiltInSlashCommand.isMetaCommand("/compactify now"))
        // `/memory` is table-driven both at the gate and in `run()`'s
        // dispatcher: the row and the matcher below move together or a test
        // here reddens.
        XCTAssertTrue(BuiltInSlashCommand.isMemoryCommand("/memory"))
        XCTAssertTrue(BuiltInSlashCommand.isMemoryCommand("/memory extra"))
        XCTAssertFalse(BuiltInSlashCommand.isMemoryCommand("/memorize"))
        XCTAssertFalse(BuiltInSlashCommand.isMemoryCommand("/memories"))
        XCTAssertTrue(BuiltInSlashCommand.isMetaCommand("/memory"))

        // The LOCAL meta commands route through their own predicate and
        // their own `run()` dispatcher arm (qwen-code /stats /export
        // /help parity); they are session-less and must never fall
        // through as prose.
        XCTAssertTrue(BuiltInSlashCommand.isLocalMetaCommand("/stats"))
        XCTAssertTrue(BuiltInSlashCommand.isLocalMetaCommand("/export"))
        XCTAssertTrue(BuiltInSlashCommand.isLocalMetaCommand("/export json"))
        XCTAssertTrue(BuiltInSlashCommand.isLocalMetaCommand("/help"))
        XCTAssertFalse(BuiltInSlashCommand.isLocalMetaCommand("/statistical"))
        XCTAssertFalse(BuiltInSlashCommand.isLocalMetaCommand("/exports"))
        XCTAssertFalse(BuiltInSlashCommand.isMemoryCommand("/stats"))

        // `/goal` is table-driven the same way (swift/docs/SWIFT_GOALS.md):
        // its own predicate keys off the row, so the row and `run()`'s
        // dispatcher arm cannot drift apart either.
        XCTAssertTrue(BuiltInSlashCommand.isGoalCommand("/goal"))
        XCTAssertTrue(BuiltInSlashCommand.isGoalCommand("/goal all tests pass"))
        XCTAssertFalse(BuiltInSlashCommand.isGoalCommand("/goals"))
        XCTAssertFalse(BuiltInSlashCommand.isGoalCommand("/goalcheck now"))
        XCTAssertTrue(BuiltInSlashCommand.isMetaCommand("/goal"))
    }

    func testRegistryNamesAndAliasesAreUnique() {
        let names = BuiltInSlashCommand.recognizedNames
        XCTAssertEqual(names.count, Set(names).count, "a name or alias is registered twice")
    }
}
