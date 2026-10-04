import AppKit
import SwiftUI
import Vision
import XCTest

@testable import TurboSparkApp

final class AppChatInstructionPinActionTests: XCTestCase {
    func testEligibilityRequiresStandaloneNonemptyUserText() {
        XCTAssertTrue(
            AppChatInstructionPin.canPin(
                AppChatMessage(role: .user, content: "Keep this preference.")))
        XCTAssertFalse(
            AppChatInstructionPin.canPin(AppChatMessage(role: .user, content: "")))
        XCTAssertFalse(
            AppChatInstructionPin.canPin(AppChatMessage(role: .user, content: " \n\t ")))
        XCTAssertFalse(
            AppChatInstructionPin.canPin(
                AppChatMessage(role: .user, content: "Describe this image.", imagePaths: ["/image.png"])))
        XCTAssertFalse(
            AppChatInstructionPin.canPin(
                AppChatMessage(
                    role: .user,
                    content: "Summarize this.\n\n--- Attachment: report.pdf (PDF) ---\nprivate file text\n--- End of report.pdf ---")))
        XCTAssertFalse(
            AppChatInstructionPin.canPin(
                AppChatMessage(role: .assistant, content: "Assistant text.")))
        XCTAssertFalse(
            AppChatInstructionPin.canPin(
                AppChatMessage(role: .system, content: "System text.")))
    }

    @MainActor
    func testPinAndUnpinPersistThroughTheMessageMarker() throws {
        AppChatFileStore.save(.empty())
        let message = AppChatMessage(role: .user, content: "Keep answers concise.")
        let chat = AppChat(messages: [message])
        let model = AppModel()
        model.stopCronScheduler()
        model.chats = [chat]
        model.selectedChatID = chat.id
        model.persistChats()

        model.setMessageStandingInstruction(id: message.id, pinned: true, chatID: chat.id)

        XCTAssertTrue(model.turnMessages(for: chat.id).first?.isStandingInstruction == true)
        XCTAssertTrue(
            AppChatFileStore.load().chats.first?.messages.first?.isStandingInstruction == true)

        model.setMessageStandingInstruction(id: message.id, pinned: false, chatID: chat.id)

        XCTAssertFalse(model.turnMessages(for: chat.id).first?.isStandingInstruction == true)
        XCTAssertFalse(
            AppChatFileStore.load().chats.first?.messages.first?.isStandingInstruction == true)
    }

    @MainActor
    func testHostedActionBarRendersPinAndUnpinLabels() throws {
        let unpinned = MessageActionBarView(
            text: "Keep this preference.",
            messageID: UUID(),
            date: Date(timeIntervalSince1970: 1_700_000_000),
            standingInstructionAction: {},
            isStandingInstruction: false)
            .frame(width: 900, height: 80, alignment: .leading)
            .background(Color.white)
            .environment(\.locale, Locale(identifier: "en-US"))
            .environment(\.colorScheme, .light)
        let unpinnedText = try Self.recognizedText(in: unpinned, width: 900, height: 80)
        XCTAssertTrue(unpinnedText.localizedCaseInsensitiveContains("Pin instruction"), unpinnedText)

        let pinned = MessageActionBarView(
            text: "Keep this preference.",
            messageID: UUID(),
            date: Date(timeIntervalSince1970: 1_700_000_000),
            standingInstructionAction: {},
            isStandingInstruction: true)
            .frame(width: 900, height: 80, alignment: .leading)
            .background(Color.white)
            .environment(\.locale, Locale(identifier: "en-US"))
            .environment(\.colorScheme, .light)
        let pinnedText = try Self.recognizedText(in: pinned, width: 900, height: 80)
        XCTAssertTrue(pinnedText.localizedCaseInsensitiveContains("Unpin instruction"), pinnedText)
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
