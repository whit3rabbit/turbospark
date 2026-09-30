import Foundation
import XCTest
@testable import TurboSparkApp

final class BrowserAutomationLoggerTests: XCTestCase {
    func testDecodingRejectsOriginWithPathQueryAndFragment() throws {
        let invalidOrigins = [
            "https://docs.example/private?token=query-secret#fragment-secret",
            "HTTPS://Docs.Example:443"
        ]

        for invalidOrigin in invalidOrigins {
            let data = try JSONSerialization.data(withJSONObject: [
                "actionType": "navigate",
                "canonicalOrigin": invalidOrigin,
                "outcome": "completed",
                "durationMilliseconds": 7
            ])

            XCTAssertThrowsError(
                try JSONDecoder().decode(BrowserAutomationDiagnosticEvent.self, from: data),
                "Expected to reject noncanonical origin: \(invalidOrigin)"
            )
        }
    }

    func testCanonicalEventEncodingAndDecodingPreserveTheAllowlistedShape() throws {
        let origin = try XCTUnwrap(BrowserOrigin(origin: "HTTPS://Docs.Example:443"))
        let event = BrowserAutomationDiagnosticEvent(
            actionType: .navigate,
            origin: origin,
            outcome: .completed,
            durationMilliseconds: 7
        )

        let encoded = try JSONEncoder().encode(event)
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        XCTAssertEqual(Set(json.keys), [
            "actionType", "canonicalOrigin", "outcome", "durationMilliseconds"
        ])
        XCTAssertEqual(json["canonicalOrigin"] as? String, "https://docs.example")

        let decoded = try JSONDecoder().decode(BrowserAutomationDiagnosticEvent.self, from: encoded)
        XCTAssertEqual(decoded, event)
        XCTAssertEqual(decoded.category, .browser)
    }

    func testTimedOutAndUnsupportedOutcomesStayTypedAndCodable() throws {
        let origin = try XCTUnwrap(BrowserOrigin(origin: "https://docs.example"))

        for outcome: BrowserAutomationOutcome in [.timedOut, .unsupported] {
            let event = BrowserAutomationDiagnosticEvent(
                actionType: .navigate,
                origin: origin,
                outcome: outcome,
                durationMilliseconds: 7
            )
            let encoded = try JSONEncoder().encode(event)
            let decoded = try JSONDecoder().decode(BrowserAutomationDiagnosticEvent.self, from: encoded)

            XCTAssertEqual(decoded.outcome, outcome)
            XCTAssertEqual(decoded.category, .browser)
        }
    }

    func testWritesOneCategorizedRecordPerTransitionWithOnlyAllowlistedFields() throws {
        let store = FakeBrowserAutomationLogSink()
        let logger = BrowserAutomationLogger(store: store)
        let origin = try XCTUnwrap(BrowserOrigin(origin: "HTTPS://Docs.Example:443"))

        logger.recordTransition(
            action: .navigate,
            origin: origin,
            outcome: .completed,
            durationMilliseconds: 24
        )
        logger.recordTransition(
            action: .click,
            origin: origin,
            outcome: .failed,
            durationMilliseconds: 11
        )

        XCTAssertEqual(store.events.count, 2)
        XCTAssertEqual(store.events.map(\.category), [.browser, .browser])
        XCTAssertEqual(store.levels, [.info, .info])
        XCTAssertEqual(store.events[0].actionType, .navigate)
        XCTAssertEqual(store.events[0].canonicalOrigin, "https://docs.example")
        XCTAssertEqual(store.events[0].outcome, .completed)
        XCTAssertEqual(store.events[0].durationMilliseconds, 24)
        XCTAssertEqual(store.events[1].actionType, .click)
        XCTAssertEqual(store.events[1].outcome, .failed)
        XCTAssertEqual(store.events[1].durationMilliseconds, 11)
    }

    func testOriginAndEncodedEventExcludePageAndUserContent() throws {
        let url = try XCTUnwrap(
            URL(string: "https://Docs.Example/private/page?token=query-secret#fragment-secret")
        )
        let origin = try XCTUnwrap(BrowserOrigin(url: url))
        let store = FakeBrowserAutomationLogSink()
        let logger = BrowserAutomationLogger(store: store)

        logger.recordTransition(
            action: .type,
            origin: origin,
            outcome: .completed,
            durationMilliseconds: 38
        )

        let event = try XCTUnwrap(store.events.first)
        XCTAssertEqual(event.canonicalOrigin, "https://docs.example")
        let encoded = try JSONEncoder().encode(event)
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        XCTAssertEqual(Set(json.keys), [
            "actionType", "canonicalOrigin", "outcome", "durationMilliseconds"
        ])

        let serialized = try XCTUnwrap(String(data: encoded, encoding: .utf8))
        for forbiddenValue in [
            "/private/page", "token", "query-secret", "fragment-secret",
            "typed-value", "dialog-text", "page-content", "secret-material"
        ] {
            XCTAssertFalse(serialized.contains(forbiddenValue))
        }
    }

    func testStoreWriteFailureIsCountedAndDoesNotBreakLaterTransitions() throws {
        let store = FakeBrowserAutomationLogSink()
        let logger = BrowserAutomationLogger(store: store)
        let origin = try XCTUnwrap(BrowserOrigin(origin: "https://example.com"))
        store.shouldFailWrites = true

        logger.recordTransition(
            action: .navigate,
            origin: origin,
            outcome: .failed,
            durationMilliseconds: 10
        )

        XCTAssertEqual(logger.writeFailureCount, 1)
        XCTAssertTrue(store.events.isEmpty)

        store.shouldFailWrites = false
        logger.recordTransition(
            action: .click,
            origin: origin,
            outcome: .completed,
            durationMilliseconds: 4
        )

        XCTAssertEqual(logger.writeFailureCount, 1)
        XCTAssertEqual(store.events.map(\.actionType), [.click])
    }
}

private final class FakeBrowserAutomationLogSink: BrowserAutomationLogSink {
    private(set) var events: [BrowserAutomationDiagnosticEvent] = []
    private(set) var levels: [BrowserAutomationLogLevel] = []
    var shouldFailWrites = false

    func record(
        _ event: BrowserAutomationDiagnosticEvent,
        level: BrowserAutomationLogLevel
    ) throws {
        if shouldFailWrites {
            throw FakeLogWriteError.failed
        }
        events.append(event)
        levels.append(level)
    }
}

private enum FakeLogWriteError: Error {
    case failed
}
