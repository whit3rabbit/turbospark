import Foundation
import TurboSpark

/// Local title generation isolated from the chat's visible turn output.
enum AppTitleGeneration {
    static let titleCharacterLimit = 80
    static let maxNewTokens: UInt32 = 64

    typealias Completion = ([ChatMessage], GenerateOptions) async throws -> String

    /// Runs a small plain-text completion on the already-loaded local session.
    /// Failures and unusable output are auxiliary and therefore return nil.
    static func generateTitle(
        session: TurboSparkSession,
        firstUserMessage: String
    ) async -> String? {
        await generateTitle(firstUserMessage: firstUserMessage) { messages, options in
            var response = ""
            for try await event in session.generate(messages, options: options) {
                try Task.checkCancellation()
                if case .content(let chunk) = event {
                    response += chunk
                }
            }
            return response
        }
    }

    /// Completion injection exercises prompt settings and failure handling
    /// without loading a model. Production calls use the local-session overload.
    static func generateTitle(
        firstUserMessage: String,
        completion: Completion
    ) async -> String? {
        let messages = [
            ChatMessage.system("""
                Create a short, descriptive title for this conversation.
                Return only the title as plain text on one line. Do not add quotes or explanation.
                """),
            ChatMessage.user(firstUserMessage),
        ]
        var options = GenerateOptions()
        options.reasoning = .off
        options.temperature = 0.2
        options.maxNewTokens = maxNewTokens

        do {
            let reply = try await completion(messages, options)
            return candidateTitle(from: firstUserMessage, replyHint: reply)
        } catch {
            return nil
        }
    }

    /// Returns a normalized, bounded candidate unless it is empty or merely
    /// repeats the entire first user message.
    static func candidateTitle(from firstUserMessage: String, replyHint: String?) -> String? {
        guard let replyHint else { return nil }

        let normalizedMessage = normalizedLine(firstUserMessage)
        let normalizedReply = normalizedLine(replyHint)
        guard normalizedReply.caseInsensitiveCompare(normalizedMessage) != .orderedSame else {
            return nil
        }

        guard let firstLine = replyHint
            .components(separatedBy: .newlines)
            .first(where: { !$0.split(whereSeparator: \.isWhitespace).isEmpty })
        else { return nil }

        let candidate = normalizedLine(firstLine)
        guard !candidate.isEmpty else { return nil }

        guard candidate.caseInsensitiveCompare(normalizedMessage) != .orderedSame else {
            return nil
        }

        return String(candidate.prefix(titleCharacterLimit))
    }

    private static func normalizedLine(_ text: String) -> String {
        text.split(whereSeparator: \.isWhitespace).joined(separator: " ")
    }
}
