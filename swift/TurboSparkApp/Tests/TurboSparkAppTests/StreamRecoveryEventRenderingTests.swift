import AppKit
import SwiftUI
import TurboSpark
import Vision
import XCTest

@testable import TurboSparkApp

final class StreamRecoveryEventRenderingTests: XCTestCase {
    @MainActor
    func testNoticeRendersEveryRecoveryEventWithBoundedAnchorData() throws {
        let anchor = try XCTUnwrap(AppStreamRecovery.anchor(
            retainedMessageCount: 2,
            messageID: UUID(),
            persistedRowContent: "private partial transcript text",
            continuationsUsed: 1,
            retriesUsed: 2,
            recordedAt: Date(timeIntervalSince1970: 1_700_000_000)))
        let events: [StreamRecoveryEvent] = [
            .continued(attempt: 2, tokensSoFar: 384),
            .partialPreserved(rows: 1),
            .tailDiscarded(description: "incomplete continuation tail"),
            .tailDiscarded(description: "  \n "),
            .anchorRecorded(anchor),
            .retrySucceeded(attempt: 2),
            .retryConflict,
            .retryExhausted,
        ]
        let view = VStack(alignment: .leading, spacing: 8) {
            ForEach(Array(events.enumerated()), id: \.offset) { entry in
                StreamRecoveryEventNoticeView(event: entry.element)
            }
        }
        .frame(width: 1000, height: 540, alignment: .topLeading)
        .background(Color.white)
        .environment(\.locale, Locale(identifier: "en-US"))
        .environment(\.colorScheme, .light)

        let renderedText = try Self.recognizedText(in: view, width: 1000, height: 540)

        for expected in [
            "Generation continued",
            "Partial response preserved",
            "Generated tail discarded",
            "Recovery anchor recorded",
            "Recovery retry succeeded",
            "Recovery conflict",
            "Recovery retries exhausted",
        ] {
            XCTAssertTrue(
                renderedText.localizedCaseInsensitiveContains(expected),
                "Expected visible recovery text \(expected). OCR saw: \(renderedText)")
        }
        XCTAssertTrue(renderedText.contains("incomplete continuation tail"))
        XCTAssertTrue(renderedText.contains("unspecified reason"))
        XCTAssertTrue(renderedText.contains("2"))
        XCTAssertFalse(renderedText.contains("private partial transcript text"))
    }

    @MainActor
    func testTranscriptAndDiagnosticsShowTerminalRecoveryOutcomes() throws {
        let model = AppModel()
        model.stopCronScheduler()
        model.streamRecoveryEvents[model.selectedChatID] = [
            .retrySucceeded(attempt: 2),
            .retryConflict,
            .retryExhausted,
        ]

        let transcript = ChatTranscriptView(model: model)
            .frame(width: 1000, height: 600)
            .background(Color.white)
            .environment(\.locale, Locale(identifier: "en-US"))
            .environment(\.colorScheme, .light)
        let transcriptText = try Self.recognizedText(in: transcript, width: 1000, height: 600)
        XCTAssertTrue(transcriptText.localizedCaseInsensitiveContains("Recovery retry succeeded"), transcriptText)
        XCTAssertTrue(transcriptText.localizedCaseInsensitiveContains("Recovery conflict"), transcriptText)
        XCTAssertTrue(transcriptText.localizedCaseInsensitiveContains("Recovery retries exhausted"), transcriptText)

        let diagnostics = Form {
            RunnerDiagnosticsSection(
                diagnostics: nil,
                recoveryEvents: [.retrySucceeded(attempt: 2), .retryConflict, .retryExhausted])
        }
        .formStyle(.grouped)
        .frame(width: 1000, height: 600)
        .background(Color.white)
        .environment(\.locale, Locale(identifier: "en-US"))
        .environment(\.colorScheme, .light)
        let diagnosticsText = try Self.recognizedText(in: diagnostics, width: 1000, height: 600)
        XCTAssertTrue(diagnosticsText.localizedCaseInsensitiveContains("Recovery events"), diagnosticsText)
        XCTAssertTrue(diagnosticsText.localizedCaseInsensitiveContains("Recovery retry succeeded"), diagnosticsText)
        XCTAssertTrue(diagnosticsText.localizedCaseInsensitiveContains("Recovery conflict"), diagnosticsText)
        XCTAssertTrue(diagnosticsText.localizedCaseInsensitiveContains("Recovery retries exhausted"), diagnosticsText)
    }

    @MainActor
    private static func recognizedText<V: View>(in view: V, width: CGFloat, height: CGFloat) throws -> String {
        let host = NSHostingView(rootView: view)
        host.frame = NSRect(x: 0, y: 0, width: width, height: height)
        host.layoutSubtreeIfNeeded()

        let bitmap = try XCTUnwrap(host.bitmapImageRepForCachingDisplay(in: host.bounds))
        host.cacheDisplay(in: host.bounds, to: bitmap)
        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .accurate
        request.recognitionLanguages = ["en-US"]
        request.usesLanguageCorrection = false
        try VNImageRequestHandler(cgImage: try XCTUnwrap(bitmap.cgImage), options: [:])
            .perform([request])
        return (request.results ?? []).compactMap { $0.topCandidates(1).first?.string }
            .joined(separator: "\n")
    }
}
