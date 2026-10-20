import AppKit
import SwiftUI
import TurboSpark

/// The strip of background subagent runs for the selected chat, running or
/// finished. It renders OUTSIDE the streaming row so a background agent is
/// visible while nothing else is happening in the conversation -- which is
/// most of a background run's life.
struct BackgroundAgentsStripView: View {
    @ObservedObject var model: AppModel

    private var runsForChat: [SubagentRunState] {
        model.backgroundAgentRuns.values
            .filter { $0.chatID == nil || $0.chatID == model.selectedChatID }
            .sorted { $0.startedAt < $1.startedAt }
    }

    var body: some View {
        let runs = runsForChat
        if !runs.isEmpty {
            VStack(alignment: .leading, spacing: 8) {
                ForEach(runs) { state in
                    SubagentLiveCardView(
                        state: state,
                        onStop: state.status == "running" ? { stop(state.id) } : nil,
                        onDismiss: state.status != "running" && state.isRecordedComplete
                            ? { model.dismissBackgroundAgent(state.id) } : nil)
                }
            }
        }
    }

    private func stop(_ id: String) {
        Task { @MainActor in
            do {
                let message = try await model.stopBackgroundAgent(id)
                model.showToast(message, style: .info)
            } catch {
                model.showToast(error.localizedDescription, style: .error)
            }
        }
    }
}
