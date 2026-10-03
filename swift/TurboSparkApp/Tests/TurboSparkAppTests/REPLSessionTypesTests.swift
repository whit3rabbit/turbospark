import Foundation
import XCTest
@testable import TurboSparkApp

final class REPLSessionTypesTests: XCTestCase {
    func testDefaultLimitsMatchTheApprovedDesign() {
        let limits = REPLLimits()

        XCTAssertEqual(limits.defaultTimeout, 30)
        XCTAssertEqual(limits.maximumTimeout, 600)
        XCTAssertEqual(limits.maximumOutputCharacters, 30_000)
        XCTAssertEqual(limits.maximumImageBytes, 20 * 1_024 * 1_024)
        XCTAssertEqual(limits.footprintGrowthBudgetBytes, 512 * 1_024 * 1_024)
    }

    func testLimitsCanBeOverriddenByFocusedTests() {
        let limits = REPLLimits(
            defaultTimeout: 0.1,
            maximumTimeout: 0.25,
            maximumOutputCharacters: 64,
            maximumImageBytes: 128,
            footprintGrowthBudgetBytes: 256)

        XCTAssertEqual(limits.defaultTimeout, 0.1)
        XCTAssertEqual(limits.maximumTimeout, 0.25)
        XCTAssertEqual(limits.maximumOutputCharacters, 64)
        XCTAssertEqual(limits.maximumImageBytes, 128)
        XCTAssertEqual(limits.footprintGrowthBudgetBytes, 256)
    }

    func testSessionConfigurationCarriesPerChatArtifactDirectory() {
        let directory = URL(fileURLWithPath: "/tmp/repl-chat-artifacts", isDirectory: true)
        let configuration = REPLSessionConfiguration(artifactDirectory: directory)

        XCTAssertEqual(configuration.artifactDirectory, directory)
    }

    func testCallResultCarriesStatusOutputImagesAndLifecycleFlags() {
        let image = REPLEmittedImage(
            fileURL: URL(fileURLWithPath: "/tmp/repl-chat-artifacts/output.png"),
            label: "chart")
        let event = REPLTextOutputEvent(level: .warn, text: "recoverable warning")
        let result = REPLCallResult(
            status: .unsupportedSyntax,
            outputEvents: [event],
            consoleText: "",
            errorText: "top-level form is unsupported",
            completionText: nil,
            images: [image],
            truncated: true,
            sessionCreated: true,
            sessionReset: false)

        XCTAssertEqual(result.status.rawValue, "unsupportedSyntax")
        XCTAssertEqual(result.outputEvents, [event])
        XCTAssertEqual(result.consoleText, "")
        XCTAssertEqual(result.errorText, "top-level form is unsupported")
        XCTAssertNil(result.completionText)
        XCTAssertEqual(result.images, [image])
        XCTAssertTrue(result.truncated)
        XCTAssertTrue(result.sessionCreated)
        XCTAssertFalse(result.sessionReset)
        requireSendable(result)
    }

    func testCallStatusesHaveStableValuesForEveryOutcome() {
        let statuses: [REPLCallResult.Status] = [
            .completed,
            .failed,
            .parseError,
            .unsupportedSyntax,
            .timedOut,
            .cancelled,
            .memoryLimit,
            .busy
        ]

        XCTAssertEqual(statuses.map(\.rawValue), [
            "completed",
            "failed",
            "parseError",
            "unsupportedSyntax",
            "timedOut",
            "cancelled",
            "memoryLimit",
            "busy"
        ])
    }

    private func requireSendable<Value: Sendable>(_ value: Value) {}
}
