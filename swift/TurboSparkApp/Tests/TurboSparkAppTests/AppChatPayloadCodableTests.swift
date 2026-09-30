import XCTest

import TurboSpark
@testable import TurboSparkApp

final class AppChatPayloadCodableTests: XCTestCase {
    func testLegacyDefaultTitleInfersUnclaimedAndUnattemptedState() throws {
        let legacy = Data(#"{"title":"New Chat","messages":[]}"#.utf8)

        let chat = try JSONDecoder().decode(AppChat.self, from: legacy)

        XCTAssertEqual(chat.titleProvenance, .unclaimed)
        XCTAssertFalse(chat.titleGenerationAttempted)
        XCTAssertNil(chat.recoveryAnchor)
    }

    func testNewPayloadInitializersUseUnclaimedAndUnpinnedDefaults() {
        let chat = AppChat()
        let message = AppChatMessage(role: .user, content: "New message")

        XCTAssertEqual(chat.titleProvenance, .unclaimed)
        XCTAssertFalse(chat.titleGenerationAttempted)
        XCTAssertNil(chat.recoveryAnchor)
        XCTAssertFalse(message.isStandingInstruction)
    }

    func testLegacyCustomTitleInfersUserProvenanceAndCompletedAttempt() throws {
        let legacy = Data(#"{"title":"Existing title","messages":[]}"#.utf8)

        let chat = try JSONDecoder().decode(AppChat.self, from: legacy)

        XCTAssertEqual(chat.titleProvenance, .user)
        XCTAssertTrue(chat.titleGenerationAttempted)
    }

    func testLegacyMessageDefaultsStandingInstructionToFalse() throws {
        let legacy = Data(#"{"role":"user","content":"Keep this rule."}"#.utf8)

        let message = try JSONDecoder().decode(AppChatMessage.self, from: legacy)

        XCTAssertFalse(message.isStandingInstruction)
    }

    func testNewChatAndMessageStateRoundTrips() throws {
        let anchor = try XCTUnwrap(RecoveryAnchor(
            retainedMessageCount: 2,
            interruptedMessageID: UUID(uuidString: "3D71D674-9DDC-41B7-A68A-8F62A849A4ED")!,
            interruptedContentHash: "sha256:interrupted-row",
            continuationsUsed: 1,
            retriesUsed: 2,
            recordedAt: Date(timeIntervalSince1970: 1_700_000_000)))
        var message = AppChatMessage(role: .user, content: "Keep this rule.")
        message.isStandingInstruction = true
        var chat = AppChat(title: "Generated title")
        chat.titleProvenance = .generated
        chat.titleGenerationAttempted = true
        chat.recoveryAnchor = anchor
        chat.messages = [
            AppChatMessage(role: .user, content: "Earlier prompt"),
            AppChatMessage(role: .assistant, content: "Earlier response"),
            AppChatMessage(
                id: anchor.interruptedMessageID,
                role: .assistant,
                content: "Interrupted response"),
        ]
        chat.messages[0] = message

        let data = try JSONEncoder().encode(chat)
        let decoded = try JSONDecoder().decode(AppChat.self, from: data)

        XCTAssertEqual(decoded.titleProvenance, .generated)
        XCTAssertTrue(decoded.titleGenerationAttempted)
        XCTAssertEqual(decoded.recoveryAnchor, anchor)
        XCTAssertEqual(decoded.messages.first?.isStandingInstruction, true)
        XCTAssertEqual(decoded, chat)
    }

    func testRecoveryAnchorRejectsInvalidCountsAndEmptyHash() {
        let validID = "3D71D674-9DDC-41B7-A68A-8F62A849A4ED"
        let invalidAnchors = [
            #"{"retainedMessageCount":-1,"interruptedMessageID":"\#(validID)","interruptedContentHash":"sha256:row","continuationsUsed":0,"retriesUsed":0,"recordedAt":0}"#,
            #"{"retainedMessageCount":0,"interruptedMessageID":"\#(validID)","interruptedContentHash":"","continuationsUsed":0,"retriesUsed":0,"recordedAt":0}"#,
            #"{"retainedMessageCount":0,"interruptedMessageID":"\#(validID)","interruptedContentHash":"sha256:row","continuationsUsed":-1,"retriesUsed":0,"recordedAt":0}"#,
            #"{"retainedMessageCount":0,"interruptedMessageID":"\#(validID)","interruptedContentHash":"sha256:row","continuationsUsed":4,"retriesUsed":0,"recordedAt":0}"#,
            #"{"retainedMessageCount":0,"interruptedMessageID":"\#(validID)","interruptedContentHash":"sha256:row","continuationsUsed":0,"retriesUsed":-1,"recordedAt":0}"#,
            #"{"retainedMessageCount":0,"interruptedMessageID":"\#(validID)","interruptedContentHash":"sha256:row","continuationsUsed":0,"retriesUsed":4,"recordedAt":0}"#,
        ]

        for payload in invalidAnchors {
            XCTAssertThrowsError(try JSONDecoder().decode(RecoveryAnchor.self, from: Data(payload.utf8)))
        }
    }

    func testChatDropsRecoveryAnchorWithoutARowAfterTheRetainedPrefix() throws {
        for retainedMessageCount in [1, 2] {
            let payload = Data(
                #"{"title":"New Chat","messages":[{"role":"user","content":"hello"}],"recoveryAnchor":{"retainedMessageCount":\#(retainedMessageCount),"interruptedMessageID":"3D71D674-9DDC-41B7-A68A-8F62A849A4ED","interruptedContentHash":"sha256:row","continuationsUsed":0,"retriesUsed":0,"recordedAt":0}}"#.utf8)

            let chat = try JSONDecoder().decode(AppChat.self, from: payload)

            XCTAssertNil(chat.recoveryAnchor)
        }
    }

    func testChatInitializerDropsRecoveryAnchorWithoutARowAfterTheRetainedPrefix() throws {
        let anchor = try XCTUnwrap(RecoveryAnchor(
            retainedMessageCount: 1,
            interruptedMessageID: UUID(uuidString: "3D71D674-9DDC-41B7-A68A-8F62A849A4ED")!,
            interruptedContentHash: "sha256:row",
            continuationsUsed: 0,
            retriesUsed: 0,
            recordedAt: Date(timeIntervalSince1970: 1_700_000_000)))

        let chat = AppChat(
            messages: [AppChatMessage(role: .user, content: "hello")],
            recoveryAnchor: anchor)

        XCTAssertNil(chat.recoveryAnchor)
    }

    func testUnknownChatAndMessageFieldsAreIgnored() throws {
        let payload = Data(
            #"{"title":"Existing title","messages":[{"role":"user","content":"hello","futureMessageState":{"pinned":true}}],"futureChatState":{"version":2}}"#.utf8)

        let chat = try JSONDecoder().decode(AppChat.self, from: payload)

        XCTAssertEqual(chat.title, "Existing title")
        XCTAssertEqual(chat.titleProvenance, .user)
        XCTAssertTrue(chat.titleGenerationAttempted)
        XCTAssertEqual(chat.messages.map(\.content), ["hello"])
        XCTAssertFalse(chat.messages[0].isStandingInstruction)
    }
}
