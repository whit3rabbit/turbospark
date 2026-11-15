import XCTest

import TurboSpark
@testable import TurboSparkApp

/// OC-09: the per-turn sources list is derived from archived tool calls and
/// results. These tests pin the parsing, the dedupe, the caps, the scheme
/// filter and the archive round trip.
final class AppChatTurnSourcesTests: XCTestCase {
    private func searchMessage(
        _ rows: [(String, String)], query: String = "q", isError: Bool = false, name: String = "WebSearch"
    ) -> AppChatMessage {
        let call = AppToolCall(
            name: name, arguments: ["query": query], status: .completed, category: .web)
        var lines = ["### Web Search Results for: \"\(query)\""]
        for (index, row) in rows.enumerated() {
            lines.append("")
            lines.append("\(index + 1). [\(row.0)](\(row.1))")
            lines.append("   a snippet")
        }
        return AppChatMessage(
            role: .assistant, content: "", toolCalls: [call],
            toolResults: [AppToolResult(
                callID: call.id, output: lines.joined(separator: "\n"), isError: isError)])
    }

    private func fetchMessage(_ url: String, isError: Bool = false, name: String = "WebFetch") -> AppChatMessage {
        let call = AppToolCall(
            name: name, arguments: ["url": url], status: .completed, category: .web)
        return AppChatMessage(
            role: .assistant, content: "", toolCalls: [call],
            toolResults: [AppToolResult(callID: call.id, output: "# page body", isError: isError)])
    }

    func testSearchRowsBecomeSourcesWithDomainAndTitle() {
        let sources = TurnSources.collect(in: [
            searchMessage([
                ("Swift Book", "https://www.swift.org/book/intro"),
                ("Apple Docs", "https://developer.apple.com/documentation/")
            ])
        ])
        XCTAssertEqual(sources.sources.map(\.title), ["Swift Book", "Apple Docs"])
        XCTAssertEqual(sources.sources.map(\.domain), ["swift.org", "developer.apple.com"])
        XCTAssertEqual(sources.sources[0].displayPath, "/book/intro")
        XCTAssertEqual(sources.sources[1].displayPath, "/documentation/")
        XCTAssertEqual(sources.totalCount, 2)
    }

    func testTitleWithBracketsAndUrlWithParenthesesParse() {
        let rows = TurnSources.searchRows(
            in: "1. [Array [Swift] docs](https://en.wikipedia.org/wiki/Array_(data_structure))")
        XCTAssertEqual(rows.count, 1)
        XCTAssertEqual(rows[0].title, "Array [Swift] docs")
        XCTAssertEqual(rows[0].url, "https://en.wikipedia.org/wiki/Array_(data_structure)")
    }

    /// Mutation check: replacing the dedupe in `add` with an unconditional
    /// append makes this fail (two searches naming one page list it twice).
    func testDuplicateUrlsAcrossSearchesCollapseToOneSource() {
        let sources = TurnSources.collect(in: [
            searchMessage([("Swift", "https://swift.org/")]),
            searchMessage([
                ("Swift again", "HTTPS://Swift.org"),
                ("Swift fragment", "https://swift.org/#install"),
                ("Other", "https://swift.org/blog")
            ])
        ])
        XCTAssertEqual(sources.sources.map(\.url), ["https://swift.org/", "https://swift.org/blog"])
        XCTAssertEqual(sources.sources[0].title, "Swift", "first position and title win")
    }

    func testQueryDistinguishesPages() {
        let sources = TurnSources.collect(in: [
            searchMessage([("A", "https://x.com/s?q=1"), ("B", "https://x.com/s?q=2")])
        ])
        XCTAssertEqual(sources.sources.count, 2)
    }

    func testFetchUsesCallArgumentAndUpgradesAMatchingSearchRow() {
        let sources = TurnSources.collect(in: [
            searchMessage([("Swift Book", "https://swift.org/book")]),
            fetchMessage("https://swift.org/book/"),
            fetchMessage("https://docs.example.com/guide")
        ])
        XCTAssertEqual(sources.sources.count, 2)
        XCTAssertEqual(sources.sources[0].kind, .fetchedPage, "read in full beats a search hit")
        XCTAssertEqual(sources.sources[0].title, "Swift Book", "the search title is kept")
        XCTAssertEqual(sources.sources[1].kind, .fetchedPage)
        XCTAssertEqual(sources.sources[1].title, "docs.example.com", "a bare fetch titles by domain")
    }

    func testFailedAndUnansweredCallsRecordNothing() {
        let unanswered = AppChatMessage(
            role: .assistant, content: "",
            toolCalls: [AppToolCall(name: "WebFetch", arguments: ["url": "https://a.com"], category: .web)])
        let sources = TurnSources.collect(in: [
            fetchMessage("https://failed.example", isError: true),
            searchMessage([("Nope", "https://err.example")], isError: true),
            unanswered
        ])
        XCTAssertTrue(sources.isEmpty)
    }

    func testOnlyHttpAndHttpsSchemesAreKept() {
        let sources = TurnSources.collect(in: [
            searchMessage([
                ("js", "javascript:alert(1)"),
                ("file", "file:///etc/passwd"),
                ("data", "data:text/html,hi"),
                ("ftp", "ftp://example.com/x"),
                ("ok", "http://example.com/ok")
            ]),
            fetchMessage("file:///Users/me/secret")
        ])
        XCTAssertEqual(sources.sources.map(\.url), ["http://example.com/ok"])
    }

    func testTavilyDirectAnswerIsNotASource() {
        let sources = TurnSources.collect(in: [
            searchMessage([
                ("Tavily Direct Answer", "https://tavily.com"),
                ("Real", "https://example.com/real")
            ])
        ])
        XCTAssertEqual(sources.sources.map(\.title), ["Real"])
    }

    func testToolNameAliasesAndNonWebToolsAreRecognized() {
        let sources = TurnSources.collect(in: [
            searchMessage([("A", "https://a.com")], name: "web_search"),
            fetchMessage("https://b.com", name: "fetch_url"),
            fetchMessage("https://c.com", name: "FileRead")
        ])
        XCTAssertEqual(sources.sources.map(\.domain), ["a.com", "b.com"])
    }

    func testCapListsFiftyAndCountsTheRest() {
        let rows = (0..<60).map { ("Page \($0)", "https://example.com/p\($0)") }
        var sources = TurnSources.collect(in: [searchMessage(rows)])
        XCTAssertEqual(sources.sources.count, TurnSources.maximumSources)
        XCTAssertEqual(sources.overflowCount, 10)
        XCTAssertEqual(sources.totalCount, 60)
        // A repeat of an overflowed page must not count twice.
        sources = TurnSources.collect(in: [searchMessage(rows), searchMessage([rows[55]])])
        XCTAssertEqual(sources.overflowCount, 10)
    }

    func testGroupsKeepFirstSeenDomainOrder() {
        let sources = TurnSources.collect(in: [
            searchMessage([
                ("1", "https://b.com/1"), ("2", "https://a.com/1"), ("3", "https://b.com/2")
            ])
        ])
        XCTAssertEqual(sources.groups.map(\.domain), ["b.com", "a.com"])
        XCTAssertEqual(sources.groups[0].sources.map(\.title), ["1", "3"])
    }

    func testNoWebCallMeansNoParsing() {
        let plain = AppChatMessage(role: .assistant, content: "hello")
        XCTAssertFalse(TurnSources.hasWebCall(in: [plain]))
        XCTAssertTrue(TurnSources.hasWebCall(in: [searchMessage([])]))
    }

    func testReplyRowIsTheFinalContentRowWithoutToolCalls() {
        let search = searchMessage([("A", "https://a.com")])
        let reply = AppChatMessage(role: .assistant, content: "Answer")
        XCTAssertEqual(TurnSources.replyRowID(in: [search, reply]), reply.id)
        XCTAssertNil(TurnSources.replyRowID(in: [search]), "a turn that ends on a tool call has no reply")
        XCTAssertNil(TurnSources.replyRowID(in: []))
    }

    /// CUJ 3: the chips survive a relaunch because nothing extra is stored:
    /// the same derivation over the decoded archive gives the same list.
    func testSourcesSurviveAnArchiveRoundTrip() throws {
        let messages = [
            AppChatMessage(role: .user, content: "find it"),
            searchMessage([("Swift", "https://swift.org"), ("Apple", "https://apple.com/a")]),
            fetchMessage("https://swift.org/install"),
            AppChatMessage(role: .assistant, content: "done")
        ]
        let before = TurnSources.collect(in: messages)
        let data = try JSONEncoder().encode(AppChat(title: "t", messages: messages))
        let decoded = try JSONDecoder().decode(AppChat.self, from: data)
        XCTAssertEqual(TurnSources.collect(in: decoded.messages), before)
        XCTAssertEqual(before.totalCount, 3)
    }

    func testTurnGroupingSeparatesTurnsSoSourcesDoNotLeak() {
        let messages = [
            AppChatMessage(role: .user, content: "one"),
            searchMessage([("A", "https://a.com")]),
            AppChatMessage(role: .assistant, content: "first"),
            AppChatMessage(role: .user, content: "two"),
            AppChatMessage(role: .assistant, content: "second")
        ]
        let byID = Dictionary(uniqueKeysWithValues: messages.map { ($0.id, $0) })
        let turns = TurnCollapseModel.turns(in: messages)
        XCTAssertEqual(turns.count, 2)
        let second = turns[1].messageIDs.compactMap { byID[$0] }
        XCTAssertTrue(TurnSources.collect(in: second).isEmpty)
        let first = turns[0].messageIDs.compactMap { byID[$0] }
        XCTAssertEqual(TurnSources.collect(in: first).totalCount, 1)
    }

    @MainActor
    func testOpenRefusedWhileBrowserIsOff() throws {
        let model = AppModel()
        defer { model.shutdown() }
        model.browserSettings.enabled = false
        let source = try XCTUnwrap(TurnSource(kind: .searchResult, title: "A", rawURL: "https://a.com"))
        XCTAssertFalse(model.canOpenSourcesInBrowserPane)
        XCTAssertFalse(model.openSourceInBrowserPane(source))
        XCTAssertFalse(model.browserPaneIsOpen)
    }

    @MainActor
    func testOpenCreatesAUserTabAndShowsThePane() throws {
        let model = AppModel()
        defer { model.shutdown() }
        model.browserSettings.enabled = true
        // A closed loopback port: the load fails at once, nothing leaves the Mac.
        let source = try XCTUnwrap(
            TurnSource(kind: .searchResult, title: "A", rawURL: "http://127.0.0.1:9/page"))
        XCTAssertTrue(model.openSourceInBrowserPane(source))
        XCTAssertTrue(model.browserPaneIsOpen)
        let store = model.browserAutomationCoordinator.engine.tabStore
        let active = try XCTUnwrap(store.activeTabID.flatMap { store.tab(id: $0) })
        XCTAssertEqual(active.owner, .user)
    }

    // MARK: Reveal in the tool card

    /// Mutation check: letting a later duplicate overwrite `origin` in `add`
    /// makes this fail (the reveal would jump to the second call).
    func testOriginIsTheFirstCallThatProducedTheSource() throws {
        let search = searchMessage([("Swift", "https://swift.org/a")])
        let again = searchMessage([("Swift", "https://swift.org/a/")], query: "q2")
        let fetch = fetchMessage("https://swift.org/a")
        let sources = TurnSources.collect(in: [search, again, fetch])
        XCTAssertEqual(sources.sources.count, 1)
        let source = try XCTUnwrap(sources.sources.first)
        XCTAssertEqual(source.kind, .fetchedPage)
        XCTAssertEqual(
            source.origin,
            TurnSourceOrigin(messageID: search.id, callID: search.toolCalls[0].id))
    }

    func testEachSourceKeepsItsOwnCall() throws {
        let first = searchMessage([("A", "https://a.com/x")])
        let second = fetchMessage("https://b.com/y")
        let sources = TurnSources.collect(in: [first, second])
        XCTAssertEqual(sources.sources.map { $0.origin?.callID },
                       [first.toolCalls[0].id, second.toolCalls[0].id])
        XCTAssertEqual(sources.sources.map { $0.origin?.messageID }, [first.id, second.id])
    }

    func testTurnAnchorFindsTheEnclosingTurn() {
        let user1 = AppChatMessage(role: .user, content: "one")
        let search = searchMessage([("A", "https://a.com")])
        let user2 = AppChatMessage(role: .user, content: "two")
        let later = fetchMessage("https://b.com")
        let all = [user1, search, user2, later]
        XCTAssertEqual(TurnSources.turnAnchor(containing: search.id, in: all), user1.id)
        XCTAssertEqual(TurnSources.turnAnchor(containing: later.id, in: all), user2.id)
        XCTAssertNil(TurnSources.turnAnchor(containing: UUID(), in: all))
    }

    @MainActor
    private func modelWithTurn() -> (AppModel, AppChatMessage, AppChatMessage, TurnSource) {
        let model = AppModel()
        let user = AppChatMessage(role: .user, content: "find it")
        let search = searchMessage([("A", "https://a.com/x")])
        let answer = AppChatMessage(role: .assistant, content: "Here.")
        var chat = AppChat(title: "reveal")
        chat.messages = [user, search, answer]
        model.chats = [chat]
        model.selectedChatID = chat.id
        let source = TurnSources.collect(in: [search]).sources[0]
        return (model, user, search, source)
    }

    @MainActor
    func testRevealOpensACollapsedTurnAndPublishesTheRequest() throws {
        let (model, user, search, source) = modelWithTurn()
        defer { model.shutdown() }
        model.collapsedTurnAnchors = [user.id]
        XCTAssertTrue(model.revealToolCard(for: source))
        XCTAssertFalse(model.collapsedTurnAnchors.contains(user.id))
        let reveal = try XCTUnwrap(model.toolCardReveal)
        XCTAssertEqual(reveal.messageID, search.id)
        XCTAssertEqual(reveal.callID, search.toolCalls[0].id)
        // A second reveal of the same call is a new request.
        XCTAssertTrue(model.revealToolCard(for: source))
        XCTAssertEqual(model.toolCardReveal?.token, reveal.token + 1)
    }

    /// Mutation check: dropping the `token > consumed` test in
    /// `consumeToolCardReveal` lets a recreated card re-expand, and this fails.
    @MainActor
    func testRevealIsConsumedOnceAndOnlyByTheNamedCard() throws {
        let (model, _, search, source) = modelWithTurn()
        defer { model.shutdown() }
        XCTAssertTrue(model.revealToolCard(for: source))
        XCTAssertFalse(model.consumeToolCardReveal(forCallID: UUID()))
        XCTAssertTrue(model.consumeToolCardReveal(forCallID: search.toolCalls[0].id))
        XCTAssertFalse(model.consumeToolCardReveal(forCallID: search.toolCalls[0].id))
        let reveal = try XCTUnwrap(model.toolCardReveal)
        XCTAssertFalse(model.isToolCardRevealPending(reveal))
        // A fresh request is pending again.
        XCTAssertTrue(model.revealToolCard(for: source))
        XCTAssertTrue(model.consumeToolCardReveal(forCallID: search.toolCalls[0].id))
    }

    @MainActor
    func testRevealRefusedWhenTheCallIsGone() throws {
        let (model, _, _, source) = modelWithTurn()
        defer { model.shutdown() }
        let stale = TurnSource(
            kind: .searchResult, title: "A", rawURL: "https://a.com/x",
            origin: TurnSourceOrigin(messageID: source.origin!.messageID, callID: UUID()))
        let originless = try XCTUnwrap(TurnSource(kind: .searchResult, title: "B", rawURL: "https://b.com"))
        XCTAssertFalse(model.revealToolCard(for: try XCTUnwrap(stale)))
        XCTAssertFalse(model.revealToolCard(for: originless))
        XCTAssertNil(model.toolCardReveal)
    }
}
