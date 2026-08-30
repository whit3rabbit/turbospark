import Foundation
import XCTest
@testable import TurboSpark
@testable import TurboSparkApp

final class MessageInteractionsTests: XCTestCase {
    func testChatMessageCreatedAtTolerantDecoding() throws {
        // Given JSON representation of a message from older versions without createdAt
        let legacyJSON = """
        {
            "id": "11111111-2222-3333-4444-555555555555",
            "role": "user",
            "content": "Hello world"
        }
        """.data(using: .utf8)!

        let decoder = JSONDecoder()
        let message = try decoder.decode(AppChatMessage.self, from: legacyJSON)

        XCTAssertEqual(message.role, .user)
        XCTAssertEqual(message.content, "Hello world")
        // Should default to a valid Date
        XCTAssertNotNil(message.createdAt)
    }

    func testChatMessageCreatedAtEncodingAndDecoding() throws {
        let fixedDate = Date(timeIntervalSince1970: 1700000000)
        let message = AppChatMessage(
            role: .assistant,
            content: "Response text",
            createdAt: fixedDate
        )

        let encoder = JSONEncoder()
        let encodedData = try encoder.encode(message)

        let decoder = JSONDecoder()
        let decoded = try decoder.decode(AppChatMessage.self, from: encodedData)

        XCTAssertEqual(decoded.role, .assistant)
        XCTAssertEqual(decoded.content, "Response text")
        XCTAssertEqual(decoded.createdAt.timeIntervalSince1970, fixedDate.timeIntervalSince1970, accuracy: 1.0)
    }

    func testMessageTimestampFormatterRelativeString() {
        let now = Date()

        // Just now
        let justNow = Date(timeInterval: -10, since: now)
        XCTAssertEqual(MessageTimestampFormatter.relativeString(for: justNow, relativeTo: now), "Just now")

        // 5 minutes ago
        let fiveMinutesAgo = Date(timeInterval: -300, since: now)
        XCTAssertEqual(MessageTimestampFormatter.relativeString(for: fiveMinutesAgo, relativeTo: now), "5 minutes ago")

        // 1 hour ago
        let oneHourAgo = Date(timeInterval: -3600, since: now)
        XCTAssertEqual(MessageTimestampFormatter.relativeString(for: oneHourAgo, relativeTo: now), "1 hour ago")

        // 3 hours ago
        let threeHoursAgo = Date(timeInterval: -10800, since: now)
        XCTAssertEqual(MessageTimestampFormatter.relativeString(for: threeHoursAgo, relativeTo: now), "3 hours ago")
    }

    func testMessageTimestampFormatterExactString() {
        let fixedDate = Date(timeIntervalSince1970: 1700000000)
        let exact = MessageTimestampFormatter.exactString(for: fixedDate)
        XCTAssertFalse(exact.isEmpty)

        let standard = MessageTimestampFormatter.standardString(for: fixedDate)
        XCTAssertFalse(standard.isEmpty)
    }

    @MainActor
    func testSpeechSynthesizerInitialStateAndStop() {
        let speech = AppSpeechSynthesizer.shared
        XCTAssertFalse(speech.isSpeaking)
        XCTAssertNil(speech.speakingMessageID)

        speech.stop()
        XCTAssertFalse(speech.isSpeaking)
        XCTAssertNil(speech.speakingMessageID)
    }

    func testCollapsibleMessageContentViewInitialization() {
        let view = CollapsibleMessageContentView(text: "Short text", isUser: true, maxHeight: 200)
        XCTAssertEqual(view.text, "Short text")
        XCTAssertTrue(view.isUser)
        XCTAssertEqual(view.maxHeight, 200)
    }
}
