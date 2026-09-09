import XCTest
@testable import TurboSparkApp

/// The chat-history keyword search behind the Cmd+K dialog (`ChatSearch`).
///
/// Fixtures are built in code through the `AppChat`/`AppChatMessage` inits,
/// never decoded from JSON, so nothing here depends on archive encoding.
/// The load-bearing case is the ghost one: exclusion must come from the
/// `isGhost` FLAG, not from a ghost row happening to be empty, so the ghost
/// fixture deliberately carries messages.
final class ChatSearchTests: XCTestCase {
    // MARK: - Fixtures

    private func makeChat(
        id: UUID = UUID(),
        title: String = "Chat",
        projectID: UUID? = nil,
        messages: [AppChatMessage] = [],
        draftAttachments: [AppPromptAttachment] = [],
        updatedAt: Date = Date(timeIntervalSince1970: 1_000),
        isGhost: Bool = false
    ) -> AppChat {
        AppChat(
            id: id,
            projectID: projectID,
            title: title,
            draftAttachments: draftAttachments,
            messages: messages,
            createdAt: updatedAt,
            updatedAt: updatedAt,
            isGhost: isGhost)
    }

    private func makeDocument(from chat: AppChat) -> ChatSearch.Document {
        ChatSearch.buildDocuments(from: [chat])[0]
    }

    private var referenceNow: Date {
        Date(timeIntervalSince1970: 30 * 86_400 + 10 * 86_400)
    }

    // MARK: - Query parsing

    func testQueryParsingTrimsLowercasesAndDropsEmptyTokens() {
        XCTAssertEqual(ChatSearch.Query.parse("  Wetlands   FLOOD ").tokens, ["wetlands", "flood"])
        XCTAssertTrue(ChatSearch.Query.parse("   ").isEmpty)
        XCTAssertTrue(ChatSearch.Query.parse("\n\t").isEmpty)
    }

    // MARK: - Matching

    func testEveryTokenMustMatchSomewhereInOneEntry() {
        let document = makeDocument(from: makeChat(messages: [
            AppChatMessage(role: .user, content: "how do coastal wetlands work"),
        ]))
        let both = ChatSearch.Query.parse("wetlands coastal")
        let missing = ChatSearch.Query.parse("wetlands flood")
        XCTAssertTrue(both.matches(document.entries[0].haystack))
        // Tokens need not sit next to each other, but each must appear.
        XCTAssertFalse(missing.matches(document.entries[0].haystack))
    }

    func testMatchingIsCaseInsensitive() {
        let document = makeDocument(from: makeChat(messages: [
            AppChatMessage(role: .assistant, content: "Coastal Wetlands store carbon"),
        ]))
        XCTAssertTrue(ChatSearch.Query.parse("WETLANDS").matches(document.entries[0].haystack))
    }

    // MARK: - Documents and the ghost guarantee

    func testGhostChatsNeverProduceDocumentsEvenWhenTheyCarryMessages() {
        let ghost = makeChat(
            title: "Temporary Chat",
            messages: [
                AppChatMessage(role: .user, content: "secret project falcon blueprint"),
                AppChatMessage(role: .assistant, content: "understood, falcon it is"),
            ],
            isGhost: true)
        let documents = ChatSearch.buildDocuments(from: [
            makeChat(messages: [AppChatMessage(role: .user, content: "hello world")]),
            ghost,
        ])
        XCTAssertEqual(documents.count, 1)
        XCTAssertTrue(
            ChatSearch.hits(query: .parse("falcon"), documents: documents).isEmpty,
            "a temporary chat must not even reveal that it matches")
    }

    func testEverySearchableKindLandsInAnEntry() {
        let attachment = AppPromptAttachment(
            fileName: "notes.md", formatLabel: "Code",
            extractedText: "attachment body text", wasTruncatedDuringExtraction: false)
        let call = AppToolCall(
            name: "edit_file",
            arguments: ["path": "/tmp/wetlands.md"],
            rawInvocation: "edit_file(path: /tmp/wetlands.md)")
        let document = makeDocument(from: makeChat(
            title: "Wetland research",
            messages: [
                AppChatMessage(
                    role: .assistant, content: "the answer prose",
                    reasoning: "the reasoning trace",
                    toolCalls: [call],
                    toolResults: [AppToolResult(callID: call.id, output: "the tool output")]),
            ],
            draftAttachments: [attachment]))
        let kinds = document.entries.map(\.kind)
        XCTAssertEqual(kinds, ["message", "reasoning", "tool", "tool result", "attachment"])
        // One token per kind, so each entry is individually reachable.
        for token in ["prose", "trace", "wetlands.md", "tool output", "attachment body"] {
            XCTAssertTrue(
                document.entries.contains {
                    ChatSearch.Query.parse(token).matches($0.haystack)
                },
                "token \(token) should match some entry")
            XCTAssertFalse(
                ChatSearch.hits(query: .parse(token), documents: [document]).isEmpty,
                "token \(token) should produce a hit")
        }
    }

    // MARK: - Hits, ordering, counts

    func testEmptyQueryReturnsEveryDocumentAsSnippetlessRecencyRows() {
        let older = makeChat(
            title: "Older chat", updatedAt: Date(timeIntervalSince1970: 1_000))
        let newer = makeChat(
            title: "Newer chat", updatedAt: Date(timeIntervalSince1970: 2_000))
        let hits = ChatSearch.hits(
            query: .parse(""), documents: ChatSearch.buildDocuments(from: [older, newer]))
        XCTAssertEqual(hits.map(\.title), ["Newer chat", "Older chat"])
        XCTAssertTrue(hits.allSatisfy { $0.matchCount == 0 && $0.snippet == nil })
    }

    func testHitsStayInRecencyOrderRegardlessOfInputOrder() {
        let first = makeChat(
            title: "first wetlands", updatedAt: Date(timeIntervalSince1970: 1_000))
        let second = makeChat(
            title: "second wetlands", updatedAt: Date(timeIntervalSince1970: 2_000))
        let hits = ChatSearch.hits(
            query: .parse("wetlands"),
            documents: ChatSearch.buildDocuments(from: [first, second]))
        XCTAssertEqual(hits.map(\.title), ["second wetlands", "first wetlands"])
    }

    func testTitleAloneCanProduceAHitWithTheTitleAsSnippetSource() {
        let document = makeDocument(from: makeChat(
            title: "Flood policy",
            messages: [AppChatMessage(role: .user, content: "something unrelated")]))
        let hits = ChatSearch.hits(query: .parse("policy"), documents: [document])
        XCTAssertEqual(hits.count, 1)
        XCTAssertEqual(hits[0].matchCount, 1)
        XCTAssertEqual(hits[0].snippet?.plainText, "Flood policy")
        XCTAssertTrue(
            hits[0].snippet!.segments.contains(where: { $0.isMatch }),
            "the matched title word should be a highlighted segment")
    }

    func testMatchCountCountsMatchingEntriesPlusTheTitle() {
        let document = makeDocument(from: makeChat(
            title: "Wetlands",
            messages: [
                AppChatMessage(role: .user, content: "about wetlands again"),
                AppChatMessage(role: .assistant, content: "wetlands store carbon"),
                AppChatMessage(role: .assistant, content: "nothing here"),
            ]))
        let hits = ChatSearch.hits(query: .parse("wetlands"), documents: [document])
        XCTAssertEqual(hits[0].matchCount, 3, "title + two matching entries")
    }

    // MARK: - Snippets

    func testSnippetPrefersAContentMatchOverReasoningAndToolText() {
        let call = AppToolCall(
            name: "grep", arguments: ["pattern": "zanzibar"], rawInvocation: "grep zanzibar")
        // The FIRST matching entry in message order is a reasoning trace
        // (metadata); the content match arrives one message later. The
        // snippet must still come from the content, so this fixture fails
        // under a "first match wins" implementation.
        let document = makeDocument(from: makeChat(messages: [
            AppChatMessage(
                role: .assistant,
                content: "a first answer that mentions nothing relevant",
                reasoning: "zanzibar appears in my reasoning first"),
            AppChatMessage(
                role: .user,
                content: "The Zanzibar coast is tidal.",
                toolCalls: [call],
                toolResults: [AppToolResult(callID: call.id, output: "zanzibar in tool output")]),
        ]))
        let hits = ChatSearch.hits(query: .parse("zanzibar"), documents: [document])
        XCTAssertEqual(hits[0].matchCount, 4, "reasoning + content + call + result")
        XCTAssertEqual(hits[0].snippet?.plainText, "The Zanzibar coast is tidal.")
    }

    func testMetadataOnlyMatchSnippetsFromItsOwnText() {
        let document = makeDocument(from: makeChat(messages: [
            AppChatMessage(
                role: .assistant, content: "a plain answer",
                reasoning: "the quokka hypothesis"),
        ]))
        let hits = ChatSearch.hits(query: .parse("quokka"), documents: [document])
        XCTAssertEqual(hits.count, 1)
        XCTAssertEqual(hits[0].snippet?.plainText, "the quokka hypothesis")
    }

    func testSnippetWindowEllipsizesAndHighlightsOriginalCasing() {
        let filler = String(repeating: "context ", count: 30)
        let text = filler + "Coastal Wetlands reduce flood peaks." + filler
        let snippet = ChatSearch.makeSnippet(for: text, tokens: ["wetlands"])
        XCTAssertTrue(snippet.leadsWithEllipsis, "the match sits past the window start")
        XCTAssertTrue(snippet.trailsWithEllipsis, "the text continues past the window")
        XCTAssertEqual(
            snippet.segments.first?.text, "\u{2026}", "a leading ellipsis segment is emitted")
        XCTAssertEqual(snippet.segments.last?.text, "\u{2026}", "a trailing ellipsis segment is emitted")
        let matched = snippet.segments.filter(\.isMatch).map(\.text)
        XCTAssertEqual(matched, ["Wetlands"], "highlighted segments carry the ORIGINAL casing")
        let plain = snippet.plainText
        XCTAssertTrue(plain.contains("Coastal Wetlands"))
        XCTAssertFalse(plain.contains(filler + "Coastal"), "text before the window is cut")
    }

    func testSnippetWithoutARoomToEllipsizeReturnsTheWholeText() {
        let snippet = ChatSearch.makeSnippet(
            for: "short wetlands note", tokens: ["wetlands"])
        XCTAssertFalse(snippet.leadsWithEllipsis)
        XCTAssertFalse(snippet.trailsWithEllipsis)
        XCTAssertEqual(snippet.plainText, "short wetlands note")
    }

    func testSnippetKeepsTheMatchWhenTheLeadInHasNoWhitespace() {
        // A 40+-char URL before the match: the window walk finds no word
        // boundary before the anchor and stops ON it. A second step there
        // clipped the match's first character out of the window and the
        // highlight with it.
        let text = "see https://example.com/very/long/path/segments/aaaa?q=needle&x=1 for details"
        let snippet = ChatSearch.makeSnippet(for: text, tokens: ["needle"])
        XCTAssertTrue(snippet.plainText.contains("needle"), "the match survives the window cut")
        XCTAssertEqual(
            snippet.segments.filter(\.isMatch).map(\.text), ["needle"],
            "the match itself is highlighted")
    }

    func testSnippetForAMissFallsBackToTheWholeTextUnhighlighted() {
        let snippet = ChatSearch.makeSnippet(for: "nothing relevant", tokens: ["wetlands"])
        XCTAssertEqual(snippet.segments.count, 1)
        XCTAssertFalse(snippet.segments[0].isMatch)
    }

    // MARK: - Date buckets

    func testDateBucketBoundaries() {
        let now = referenceNow
        XCTAssertEqual(ChatSearch.dateBucket(now.addingTimeInterval(-60), now: now), "Today")
        XCTAssertEqual(
            ChatSearch.dateBucket(now.addingTimeInterval(60), now: now), "Today",
            "a future timestamp still reads Today")
        XCTAssertEqual(
            ChatSearch.dateBucket(now.addingTimeInterval(-2 * 86_400), now: now), "Past week")
        XCTAssertEqual(
            ChatSearch.dateBucket(now.addingTimeInterval(-8 * 86_400), now: now), "Past month")
        XCTAssertEqual(
            ChatSearch.dateBucket(now.addingTimeInterval(-31 * 86_400), now: now), "Older")
    }

    // MARK: - Carried fields

    func testHitCarriesTheProjectIDSoOpeningCanSwitchProjects() {
        let projectID = UUID()
        let document = makeDocument(from: makeChat(
            title: "wetlands", projectID: projectID))
        let hits = ChatSearch.hits(query: .parse("wetlands"), documents: [document])
        XCTAssertEqual(hits[0].projectID, projectID)
    }
}
