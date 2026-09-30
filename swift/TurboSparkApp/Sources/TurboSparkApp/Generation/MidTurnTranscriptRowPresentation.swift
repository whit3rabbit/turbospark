import Foundation
import TurboSpark

/// Separates a transcript source tag from the unchanged model-facing content.
struct MidTurnTranscriptRowPresentation: Equatable {
    let label: String?
    let role: ChatMessage.Role
    let content: String

    init(message: AppChatMessage) {
        label = message.presentationLabel
        role = message.role
        content = message.content
    }
}
