import AppKit
import SwiftUI
import TurboSpark
import Vision
import XCTest

@testable import TurboSparkApp

final class MidTurnTranscriptRenderingTests: XCTestCase {
    private let body = "Continue with the requested change."

    func testOlderMessagePayloadsDecodeWithoutPresentationLabels() throws {
        let data = Data(#"{"role":"user","content":"legacy message"}"#.utf8)
        let message = try JSONDecoder().decode(AppChatMessage.self, from: data)

        XCTAssertNil(message.presentationLabel)
        XCTAssertEqual(message.role, .user)
        XCTAssertEqual(message.content, "legacy message")
    }

    func testPresentationLabelSurvivesMessageCodableRoundTrip() throws {
        let message = AppChatMessage(
            role: .user,
            content: "[Peer reply]\n\(body)",
            presentationLabel: MidTurnInputPresentation.peerReply.label)

        let decoded = try JSONDecoder().decode(
            AppChatMessage.self,
            from: JSONEncoder().encode(message))

        XCTAssertEqual(decoded.presentationLabel, "Peer reply")
        XCTAssertEqual(decoded.role, .user)
        XCTAssertEqual(decoded.content, message.content)
    }

    func testRowPresentationKeepsEachLabelAndTheWrappedUserContent() {
        for kind in MidTurnInputPresentation.allCases {
            let wrappedContent = kind.wrap(body)
            let message = AppChatMessage(
                role: kind.role,
                content: wrappedContent,
                presentationLabel: kind.label)
            let row = MidTurnTranscriptRowPresentation(message: message)

            XCTAssertEqual(row.label, kind.label)
            XCTAssertEqual(row.role, .user)
            XCTAssertEqual(row.content, wrappedContent)
        }
    }

    @MainActor
    func testHostedTranscriptChipUsesSelectedAppLocale() throws {
        let model = AppModel()
        model.stopCronScheduler()
        let message = AppChatMessage(
            role: .user,
            content: MidTurnInputPresentation.taskNotification.wrap(body),
            presentationLabel: MidTurnInputPresentation.taskNotification.label)
        let row = MessageRowView(model: model, message: message)
            .frame(width: 900, height: 360, alignment: .topLeading)
            .environment(\.locale, Locale(identifier: "fr"))
            .environment(\.colorScheme, .light)
        let host = NSHostingView(rootView: row)
        host.frame = NSRect(x: 0, y: 0, width: 900, height: 360)
        host.layoutSubtreeIfNeeded()

        let bitmap = try XCTUnwrap(host.bitmapImageRepForCachingDisplay(in: host.bounds))
        host.cacheDisplay(in: host.bounds, to: bitmap)
        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .accurate
        request.recognitionLanguages = ["fr-FR"]
        request.usesLanguageCorrection = false
        try VNImageRequestHandler(cgImage: try XCTUnwrap(bitmap.cgImage), options: [:])
            .perform([request])

        let renderedLines = (request.results ?? []).compactMap {
            $0.topCandidates(1).first?.string
        }
        let normalizedLines = renderedLines.map {
            $0.folding(options: [.diacriticInsensitive, .caseInsensitive], locale: Locale(identifier: "fr"))
        }
        let renderedText = normalizedLines.joined(separator: "\n")

        XCTAssertTrue(
            renderedText.contains("notification de tache"),
            "Expected the selected French locale in the transcript chip. OCR saw: \(renderedText)")
        XCTAssertFalse(
            normalizedLines.contains { line in
                !line.contains("[") && !line.contains("]")
                    && Self.ocrDistance(line, "Task notification") <= 1
            },
            "The chip should not fall back to the system-language label. OCR saw: \(renderedText)")
    }

    @MainActor
    func testHostedTranscriptRowsRenderPresentationLabelsSeparatelyFromTheirWrappers() throws {
        let model = AppModel()
        model.stopCronScheduler()

        let messages = MidTurnInputPresentation.allCases.map { kind in
            AppChatMessage(
                role: kind.role,
                content: kind.wrap(body),
                presentationLabel: kind.label)
        }
        let rows = VStack(alignment: .leading, spacing: 14) {
            ForEach(messages) { message in
                MessageRowView(model: model, message: message)
            }
        }
        .frame(width: 900, height: 900, alignment: .topLeading)
        .environment(\.locale, Locale(identifier: "en-US"))
        .environment(\.colorScheme, .light)
        let host = NSHostingView(rootView: rows)
        host.frame = NSRect(x: 0, y: 0, width: 900, height: 900)
        host.layoutSubtreeIfNeeded()

        let bitmap = try XCTUnwrap(host.bitmapImageRepForCachingDisplay(in: host.bounds))
        host.cacheDisplay(in: host.bounds, to: bitmap)
        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .accurate
        request.recognitionLanguages = ["en-US"]
        request.usesLanguageCorrection = false
        try VNImageRequestHandler(cgImage: try XCTUnwrap(bitmap.cgImage), options: [:])
            .perform([request])

        let renderedLines = (request.results ?? []).compactMap {
            $0.topCandidates(1).first?.string
        }
        let renderedText = renderedLines.joined(separator: "\n")

        for kind in MidTurnInputPresentation.allCases {
            XCTAssertTrue(
                renderedLines.contains { line in
                    !line.contains("[") && !line.contains("]")
                        && Self.ocrDistance(line, kind.label) <= 1
                },
                "Expected a separate visible \(kind.label) chip. OCR saw: \(renderedText)")
        }
        XCTAssertTrue(
            renderedText.localizedCaseInsensitiveContains(body),
            "Expected the wrapped user content to remain visible. OCR saw: \(renderedText)")
        XCTAssertTrue(messages.allSatisfy { $0.role == .user })
        XCTAssertTrue(messages.allSatisfy { $0.content.contains(body) })
    }

    private static func ocrDistance(_ lhs: String, _ rhs: String) -> Int {
        let lhs = Array(lhs.lowercased().filter(\.isLetter))
        let rhs = Array(rhs.lowercased().filter(\.isLetter))
        var previous = Array(0...rhs.count)

        for (lhsIndex, lhsCharacter) in lhs.enumerated() {
            var current = [lhsIndex + 1]
            for (rhsIndex, rhsCharacter) in rhs.enumerated() {
                let substitutionCost = lhsCharacter == rhsCharacter ? 0 : 1
                current.append(min(
                    previous[rhsIndex + 1] + 1,
                    current[rhsIndex] + 1,
                    previous[rhsIndex] + substitutionCost))
            }
            previous = current
        }

        return previous[rhs.count]
    }
}
