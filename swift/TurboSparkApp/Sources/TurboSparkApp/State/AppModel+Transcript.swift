import Foundation
import SwiftUI
import TurboSpark

// MARK: - Transcript & Presentation Calculations

extension AppModel {
    /// Live decode throughput in tokens per second.
    public var liveTokensPerSecond: Double {
        liveElapsedDecodeSeconds > 0 ? Double(liveTokenCount) / liveElapsedDecodeSeconds : 0
    }

    /// Peak resident process memory footprint in bytes.
    public var currentProcessMemoryBytes: UInt64? {
        TurboSparkSession.peakFootprintBytes
    }

    /// Whether there is any conversation history or live output to display.
    public var hasOutputTranscript: Bool {
        !selectedChat.messages.isEmpty || !outputText.isEmpty || !outputReasoningText.isEmpty
    }

    /// Rough token count of everything that would be sent on the next turn.
    ///
    /// A four-characters-per-token approximation over the committed transcript
    /// plus the live estimate of the draft. It is a STATUS reading and never a
    /// budget: the real window fit is decided by `fitConversationWindow` on the
    /// engine side against the real tokenizer.
    public var estimatedContextTokens: Int {
        let transcriptCharacters = selectedChat.messages.reduce(0) { $0 + $1.content.count }
        let attachmentCharacters = promptAttachments.reduce(0) { $0 + $1.characterCount }
        return transcriptCharacters / 4 + attachmentCharacters / 4 + estimatedPromptTokens
    }

    /// Resolved context token limit when in automatic mode.
    public var resolvedContextTokens: Int {
        if let info = info {
            return Int(info.maxContext)
        }
        return 4096
    }

    /// Whether starter prompt examples should be displayed in place of transcript.
    public var showsPromptExamples: Bool {
        promptText.isEmpty && promptAttachments.isEmpty && !hasOutputTranscript
    }

    /// Plain text of the latest assistant output.
    public var outputResponsePlainText: String {
        if !outputText.isEmpty {
            return outputText
        }
        return selectedChat.messages.last(where: { $0.role == .assistant })?.content ?? ""
    }

    /// Full plain text transcript of the active chat conversation.
    public var outputConversationPlainText: String {
        var transcriptLines: [String] = []
        for message in selectedChat.messages {
            let label = message.role == .user ? "You" : "Assistant"
            transcriptLines.append("\(label):\n\(message.content)")
        }
        if !outputText.isEmpty {
            transcriptLines.append("Assistant:\n\(outputText)")
        }
        return transcriptLines.joined(separator: "\n\n")
    }

    /// History messages in the active chat conversation.
    public var transcriptBaseMessages: [AppChatMessage] {
        selectedChat.messages
    }
}
