import Darwin
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

    /// Current process CPU usage percentage across all active threads.
    public var currentProcessCPUUsage: Double? {
        var threadsList: thread_act_array_t?
        var threadsCount: mach_msg_type_number_t = 0
        let kernReturn = withUnsafeMutablePointer(to: &threadsList) {
            $0.withMemoryRebound(to: thread_act_array_t?.self, capacity: 1) {
                task_threads(mach_task_self_, $0, &threadsCount)
            }
        }
        guard kernReturn == KERN_SUCCESS, let threads = threadsList else { return nil }
        defer {
            let size = vm_size_t(threadsCount * UInt32(MemoryLayout<thread_t>.stride))
            vm_deallocate(mach_task_self_, vm_address_t(UInt(bitPattern: threads)), size)
        }

        var totalUsage: Double = 0.0
        for i in 0..<Int(threadsCount) {
            var threadInfo = thread_basic_info()
            var threadInfoCount = mach_msg_type_number_t(THREAD_INFO_MAX)
            let infoReturn = withUnsafeMutablePointer(to: &threadInfo) {
                $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                    thread_info(threads[i], thread_flavor_t(THREAD_BASIC_INFO), $0, &threadInfoCount)
                }
            }
            guard infoReturn == KERN_SUCCESS else { continue }
            if threadInfo.flags & TH_FLAGS_IDLE == 0 {
                totalUsage += Double(threadInfo.cpu_usage) / Double(TH_USAGE_SCALE) * 100.0
            }
        }
        return totalUsage
    }

    /// Whether there is any conversation history or live output to display.
    public var hasOutputTranscript: Bool {
        !selectedTurnMessages.isEmpty
            || !outputText.isEmpty
            || !outputReasoningText.isEmpty
            || isRunning
            || pendingToolCall != nil
            || !liveSubagentRuns.isEmpty
            || backgroundAgentRuns.values.contains { $0.chatID == nil || $0.chatID == selectedChatID }
    }

    /// Token count of everything that would be sent on the next turn.
    ///
    /// `estimatedPromptTokens` is the EXACT count of the assembled prompt
    /// (system message, transcript after the compaction boundary, summary,
    /// draft), so the only thing it misses is attachments: a render emits
    /// one marker per image, and their page-token expansion happens in the
    /// engine's splice. Those are priced at the four-characters-per-token
    /// approximation, which is the whole sum here.
    ///
    /// **THE TRANSCRIPT'S OWN chars/4 TERM IS GONE, AND IT WAS DOUBLE
    /// COUNTING (state#116).** `estimatedPromptTokens` already included every
    /// transcript
    /// row exactly; adding the approximation on top overstated the fill by
    /// roughly a quarter of the transcript, which pushed this meter -- and
    /// now the context ring -- toward yellow and red well before the real
    /// trigger. It is a STATUS reading and never a budget: the real window
    /// fit is decided by `fitConversationWindow` on the engine side against
    /// the real tokenizer.
    public var estimatedContextTokens: Int {
        let attachmentCharacters = promptAttachments.reduce(0) { $0 + $1.characterCount }
        return attachmentCharacters / 4 + estimatedPromptTokens
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
        return selectedTurnMessages.last(where: { $0.role == .assistant })?.content ?? ""
    }

    /// Full plain text transcript of the active chat conversation.
    public var outputConversationPlainText: String {
        var transcriptLines: [String] = []
        for message in selectedTurnMessages {
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
        selectedTurnMessages
    }
}
