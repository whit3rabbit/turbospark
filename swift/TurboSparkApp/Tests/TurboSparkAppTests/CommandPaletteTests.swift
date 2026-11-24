import XCTest
@testable import TurboSparkApp

/// The Cmd+K command center's engine (`PaletteProviders.swift`): prefix
/// parsing, the three providers, the All-scope merge, the history store and
/// the action routing. Everything here is pure; the view only renders what
/// these return.
@MainActor
final class CommandPaletteTests: XCTestCase {
    // MARK: - Fixtures

    private func makeChat(
        id: UUID = UUID(), title: String, text: String = "", updatedAt: TimeInterval,
        isGhost: Bool = false
    ) -> AppChat {
        AppChat(
            id: id, projectID: nil, title: title, draftAttachments: [],
            messages: text.isEmpty ? [] : [AppChatMessage(role: .user, content: text)],
            createdAt: Date(timeIntervalSince1970: updatedAt),
            updatedAt: Date(timeIntervalSince1970: updatedAt), isGhost: isGhost)
    }

    private func pluginSkill(_ name: String, enabled: Bool = true) -> AppSkill {
        AppSkill(
            manifest: SkillManifest(name: name, description: "Lint the diff"),
            content: "body", sourceURL: URL(fileURLWithPath: "/tmp/\(name).md"),
            scope: .plugin(pluginID: "demo@local"), isEnabled: enabled)
    }

    private func userSkill(_ name: String) -> AppSkill {
        AppSkill(
            manifest: SkillManifest(name: name, description: "mine"),
            content: "body", sourceURL: URL(fileURLWithPath: "/tmp/\(name).md"))
    }

    private func context(
        chats: [AppChat] = [], skills: [AppSkill] = [], files: [String]? = nil,
        hasProject: Bool = true, history: [String] = []
    ) -> PaletteContext {
        PaletteContext(
            documents: ChatSearch.buildDocuments(from: chats),
            commands: PaletteCommandCatalog.commands(skills: skills, canCheckForUpdates: true),
            files: files.map { $0.map { ProjectFileEntry(relativePath: $0, isDirectory: false) } },
            hasProject: hasProject, history: history)
    }

    private func sections(_ raw: String, tab: PaletteScope = .all, _ ctx: PaletteContext)
        -> [PaletteSection]
    {
        PaletteEngine.sections(query: .parse(raw, tab: tab), context: ctx)
    }

    private func commandTitles(_ sections: [PaletteSection]) -> [String] {
        sections.first(where: { $0.kind == .commands })?.rows.compactMap {
            if case .command(let command) = $0.payload { return command.title }
            return nil
        } ?? []
    }

    // MARK: - Prefix parsing

    func testGreaterThanPrefixSelectsTheCommandsScopeAndIsStripped() {
        let query = PaletteQuery.parse(">export")
        XCTAssertEqual(query.scope, .commands)
        XCTAssertEqual(query.text, "export")
        XCTAssertTrue(query.isPrefixed)
    }

    func testHashAndAtPrefixesSelectChatsAndFiles() {
        XCTAssertEqual(PaletteQuery.parse("#wet").scope, .chats)
        XCTAssertEqual(PaletteQuery.parse("@main").scope, .files)
        XCTAssertEqual(PaletteQuery.parse("@ main").text, "main")
    }

    func testOnlyTheFirstCharacterIsAPrefix() {
        let mid = PaletteQuery.parse("fix #123 in @main", tab: .all)
        XCTAssertEqual(mid.scope, .all)
        XCTAssertEqual(mid.text, "fix #123 in @main")
        XCTAssertFalse(mid.isPrefixed)
        // A leading space makes the character literal.
        XCTAssertEqual(PaletteQuery.parse(" >x").scope, .all)
    }

    func testAPrefixOverridesTheTabAndDeletingItReturnsToTheTab() {
        XCTAssertEqual(PaletteQuery.parse(">x", tab: .files).scope, .commands)
        XCTAssertEqual(PaletteQuery.parse("x", tab: .files).scope, .files)
    }

    func testScopesAdvanceAndWrap() {
        XCTAssertEqual(PaletteScope.all.advanced(by: 1), .commands)
        XCTAssertEqual(PaletteScope.files.advanced(by: 1), .all)
        XCTAssertEqual(PaletteScope.all.advanced(by: -1), .files)
    }

    // MARK: - Commands scope

    func testExportIsFoundByNameUnderTheCommandsPrefix() {
        let result = sections(">export", context())
        XCTAssertEqual(result.map(\.kind), [.commands])
        XCTAssertEqual(commandTitles(result).first, "/export")
    }

    func testCommandsScopeIsOnlyCommandsEvenWhenChatsAndFilesMatch() {
        let ctx = context(
            chats: [makeChat(title: "export notes", updatedAt: 10)], files: ["export.txt"])
        XCTAssertEqual(sections(">export", ctx).map(\.kind), [.commands])
    }

    func testNamePrefixOutranksSummaryMatches() {
        let titles = commandTitles(sections(">res", context()))
        // `/resume` and `/reset` start with "res"; anything matching only by
        // summary text must come after them.
        let firstTwo = Set(titles.prefix(2))
        XCTAssertEqual(firstTwo, ["/resume", "/reset"])
    }

    func testCatalogCoversBuiltInsSettingsNavigationAndUpdates() {
        let titles = Set(commandTitles(sections(">", context())))
        for command in BuiltInSlashCommand.all { XCTAssertTrue(titles.contains("/\(command.name)")) }
        for tab in AppSettingsView.SettingsTab.allCases { XCTAssertTrue(titles.contains(tab.title)) }
        for section in AppModel.AppNavigationSection.allCases {
            XCTAssertTrue(titles.contains(section.title))
        }
        XCTAssertTrue(titles.contains("Check for Updates\u{2026}"))
    }

    func testUpdatesCommandFollowsTheMenuItemsEnabledPredicate() {
        let off = PaletteCommandCatalog.commands(skills: [], canCheckForUpdates: false)
        XCTAssertFalse(off.contains { $0.action == .checkForUpdates })
    }

    func testNavigationRowsCarryTheirChord() {
        // The caller resolves chords from KeybindOverrideStore; the catalog
        // only carries what it is given, and a missing section has no chord.
        let commands = PaletteCommandCatalog.commands(
            skills: [], canCheckForUpdates: true, navigationBindings: [.chat: "\u{2318}1"])
        let chat = commands.first { $0.action == .navigate(.chat) }
        XCTAssertEqual(chat?.binding, "\u{2318}1")
        let other = commands.first {
            if case .navigate(let section) = $0.action { return section != .chat }
            return false
        }
        XCTAssertNotNil(other)
        XCTAssertNil(other?.binding)
    }

    func testPluginCommandsAppearNamespacedAndOnlyWhenInvocable() {
        let skills = [
            pluginSkill("demo:lint"), pluginSkill("demo:off", enabled: false),
            userSkill("personal"),
        ]
        let titles = commandTitles(sections(">", context(skills: skills)))
        XCTAssertTrue(titles.contains("/demo:lint"))
        XCTAssertFalse(titles.contains("/demo:off"), "a disabled plugin command is refused at send")
        XCTAssertFalse(titles.contains("/personal"), "user skills stay on the slash popup")
    }

    func testPluginCommandInsertsIntoTheComposerBecauseItTakesArguments() {
        XCTAssertEqual(
            PaletteActionRouter.slashAction("demo:lint"), .insertIntoComposer("/demo:lint "))
    }

    // MARK: - Chats scope and All

    func testChatsScopeKeepsChatSearchOrderAndMatching() {
        let old = makeChat(title: "old", text: "wetlands flood", updatedAt: 10)
        let recent = makeChat(title: "recent", text: "flood wetlands", updatedAt: 99)
        let ghost = makeChat(title: "ghost", text: "wetlands flood", updatedAt: 200, isGhost: true)
        let ctx = context(chats: [old, ghost, recent])
        let result = sections("#flood wetlands", ctx)
        let ids = result.first?.rows.compactMap { row -> UUID? in
            if case .chat(let hit) = row.payload { return hit.id }
            return nil
        }
        // Recency order, ghost excluded: exactly the old overlay's result.
        XCTAssertEqual(ids, [recent.id, old.id])
        XCTAssertEqual(ids, ChatSearch.hits(query: .parse("flood wetlands"), documents: ctx.documents).map(\.id))
    }

    func testPlainTextInAllBlendsChatsCommandsAndFilesChatsFirst() {
        let ctx = context(
            chats: [makeChat(title: "export plan", updatedAt: 5)], files: ["docs/export.md"])
        let result = sections("export", ctx)
        XCTAssertEqual(result.map(\.kind), [.chats, .commands, .files])
    }

    func testFirstRowInAllIsTheTopChatSoEnterOpensItLikeTheOldOverlay() {
        let newest = makeChat(title: "newest", updatedAt: 90)
        let ctx = context(chats: [makeChat(title: "older", updatedAt: 1), newest])
        let first = sections("", ctx).flatMap(\.rows).first
        guard case .chat(let hit)? = first?.payload else { return XCTFail("first row is not a chat") }
        XCTAssertEqual(hit.id, newest.id)
        XCTAssertEqual(PaletteActionRouter.action(for: first!), .openChat(newest.id))
    }

    func testAllScopeCapsEachSectionAndTheChatsTabDoesNot() {
        let chats = (0..<20).map { makeChat(title: "note \($0)", updatedAt: TimeInterval($0)) }
        let ctx = context(chats: chats)
        XCTAssertEqual(
            sections("note", ctx).first(where: { $0.kind == .chats })?.rows.count,
            PaletteEngine.allScopeSectionCap)
        XCTAssertEqual(sections("note", tab: .chats, ctx).first?.rows.count, 20)
    }

    func testHomeViewListsChatsCommandsAndRecentSearchesButNoFiles() {
        let ctx = context(
            chats: [makeChat(title: "a", updatedAt: 1)], files: ["a.txt"], history: ["alpha", "beta"])
        XCTAssertEqual(sections("", ctx).map(\.kind), [.chats, .commands, .history])
    }

    func testNothingMatchingYieldsNoSections() {
        XCTAssertTrue(sections("zzzzqqqq", context(chats: [makeChat(title: "a", updatedAt: 1)])).isEmpty)
    }

    // MARK: - Files scope

    func testFilesScopeRanksNamePrefixBeforePathMatches() {
        let ctx = context(files: ["src/zzz/main.swift", "main.rs", "notes/domain.md"])
        let rows = sections("@main", ctx).first?.rows ?? []
        let paths = rows.compactMap { row -> String? in
            if case .file(let entry) = row.payload { return entry.relativePath }
            return nil
        }
        XCTAssertEqual(paths.first, "main.rs")
        XCTAssertEqual(Set(paths), ["main.rs", "src/zzz/main.swift", "notes/domain.md"])
    }

    func testFilesScopeWithoutAListingOrProjectIsEmpty() {
        XCTAssertTrue(sections("@x", context(files: nil, hasProject: false)).isEmpty)
        XCTAssertTrue(sections("@x", context(files: nil)).isEmpty)
    }

    func testPathsWithAQuoteAreNeverOffered() {
        let ctx = context(files: ["we\"ird.txt", "fine.txt"])
        let rows = sections("@", ctx).first?.rows ?? []
        XCTAssertEqual(rows.count, 1)
    }

    func testFilesScopeIsCapped() {
        let ctx = context(files: (0..<200).map { "f\($0).txt" })
        XCTAssertEqual(sections("@", ctx).first?.rows.count, PaletteEngine.filesScopeCap)
    }

    // MARK: - Action routing and the composer handoff

    func testFileRowHandsOffTheExistingMentionTokenWithATrailingSpace() {
        let row = PaletteRow(payload: .file(ProjectFileEntry(relativePath: "src/App.swift", isDirectory: false)))
        XCTAssertEqual(PaletteActionRouter.action(for: row), .insertIntoComposer("@src/App.swift "))
        let spaced = PaletteRow(payload: .file(ProjectFileEntry(relativePath: "my notes.txt", isDirectory: false)))
        XCTAssertEqual(PaletteActionRouter.action(for: spaced), .insertIntoComposer("@\"my notes.txt\" "))
    }

    func testDraftInsertionAddsASeparatorOnlyWhenNeeded() {
        XCTAssertEqual(PaletteDraft.appending("@a ", to: ""), "@a ")
        XCTAssertEqual(PaletteDraft.appending("@a ", to: "look at"), "look at @a ")
        XCTAssertEqual(PaletteDraft.appending("@a ", to: "look at "), "look at @a ")
        XCTAssertEqual(PaletteDraft.appending("@a ", to: "line\n"), "line\n@a ")
    }

    func testExportRunsImmediatelyThroughTheLocalMetaPath() {
        XCTAssertEqual(PaletteActionRouter.slashAction("export"), .runLocalMetaCommand("/export"))
    }

    func testDestructiveAndArgumentCommandsAreInsertedNotRun() {
        for name in ["clear", "reset", "delete", "archive", "compact", "rename", "model", "theme", "fork"] {
            XCTAssertEqual(
                PaletteActionRouter.slashAction(name), .insertIntoComposer("/\(name) "),
                "/\(name) must not run from one Enter")
        }
    }

    func testEveryImmediateCommandIsARealLocalCommand() {
        // Drift guard: a renamed or removed command must not leave a stale
        // name that silently falls through to an insert.
        let local = Set(
            BuiltInSlashCommand.all.filter { $0.kind == .localCommand }.map(\.name))
        for name in PaletteActionRouter.runImmediately {
            XCTAssertTrue(local.contains(name), "/\(name) is not a local command")
            XCTAssertEqual(PaletteActionRouter.slashAction(name), .runLocalCommand("/\(name)"))
        }
    }

    func testEveryImmediateLocalCommandIsHandledByTheModelDispatcher() {
        // The dispatcher must recognize each name, or the palette would
        // report a run that did nothing.
        for name in PaletteActionRouter.runImmediately {
            XCTAssertTrue(BuiltInSlashCommand.isLocalCommand("/\(name)"))
        }
    }

    func testSettingsNavigationAndUpdateRowsRouteToTheirActions() {
        let commands = PaletteCommandCatalog.commands(skills: [], canCheckForUpdates: true)
        let settings = commands.first { $0.action == .settings(.plugins) }!
        XCTAssertEqual(
            PaletteActionRouter.action(for: PaletteRow(payload: .command(settings))),
            .openSettings(.plugins))
        let nav = commands.first { $0.action == .navigate(.audio) }!
        XCTAssertEqual(
            PaletteActionRouter.action(for: PaletteRow(payload: .command(nav))), .navigate(.audio))
        XCTAssertEqual(
            PaletteActionRouter.action(for: PaletteRow(payload: .history("export"))),
            .setQuery("export"))
    }

    func testAnActionPerformedThroughTheModelWritesTheComposerDraft() {
        let model = AppModel()
        model.promptText = "see"
        PaletteActionRouter.perform(.insertIntoComposer("@a.txt "), model: model)
        XCTAssertEqual(model.promptText, "see @a.txt ")
    }

    // MARK: - History

    func testHistoryIsMostRecentFirstDeduplicatedAndCapped() {
        var history = PaletteHistory()
        for i in 0..<30 { history.record("q\(i)", workspace: "w") }
        let entries = history.entries(workspace: "w")
        XCTAssertEqual(entries.count, PaletteHistory.cap)
        XCTAssertEqual(entries.first, "q29")
        XCTAssertEqual(entries.last, "q10")
        history.record("Q15", workspace: "w")
        XCTAssertEqual(history.entries(workspace: "w").first, "Q15")
        XCTAssertEqual(history.entries(workspace: "w").filter { $0.lowercased() == "q15" }.count, 1)
    }

    func testHistoryIgnoresBlankQueriesAndIsPerWorkspace() {
        var history = PaletteHistory()
        history.record("   ", workspace: "a")
        history.record("hello", workspace: "a")
        XCTAssertEqual(history.entries(workspace: "a"), ["hello"])
        XCTAssertTrue(history.entries(workspace: "b").isEmpty)
        history.clear(workspace: "a")
        XCTAssertTrue(history.entries(workspace: "a").isEmpty)
    }

    func testHistoryStorePersistsAcrossInstances() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("palette-history-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("h.json")
        PaletteHistoryStore(url: url).record(">export", workspace: "w")
        XCTAssertEqual(PaletteHistoryStore(url: url).entries(workspace: "w"), [">export"])
        PaletteHistoryStore(url: url).clear(workspace: "w")
        XCTAssertTrue(PaletteHistoryStore(url: url).entries(workspace: "w").isEmpty)
    }
}
