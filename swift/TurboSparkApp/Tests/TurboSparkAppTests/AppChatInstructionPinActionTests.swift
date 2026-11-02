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

    /// The message actions are icon-only (`MessageIconActionStyle`), so
    /// the pin state is spelled out only in the action's name, its VoiceOver
    /// label and tooltip. A headless render exposes neither, so the name is
    /// asserted where the view reads it.
    func testPinActionIsNamedForWhatItWillDo() {
        XCTAssertEqual(MessageActionBarView.standingInstructionName(isPinned: false), "Pin instruction")
        XCTAssertEqual(MessageActionBarView.standingInstructionName(isPinned: true), "Unpin instruction")
    }
}
