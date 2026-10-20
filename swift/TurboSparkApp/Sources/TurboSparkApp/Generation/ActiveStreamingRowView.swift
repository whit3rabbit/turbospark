import AppKit
import SwiftUI
import TurboSpark

/// View rendering live streaming output and prefill animations.
struct ActiveStreamingRowView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel
    let output: String
    let reasoning: String
    let isRunning: Bool

    /// When this reply started, for the elapsed time on its activity line.
    /// Reset when a new run starts while the row stays mounted.
    @State private var startedAt = Date()
    /// Live reasoning starts closed: the activity line previews its latest
    /// line, and the chevron reveals the rest.
    @State private var showsReasoning = false

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            activityLine
                .padding(.bottom, -6)

            if showsReasoning && !reasoning.isEmpty {
                ReasoningTraceView(reasoning: reasoning)
            }
            if let pending = model.pendingToolCall {
                ToolCallCardView(model: model, call: pending, result: nil)
            }
            // Foreground subagent runs proposed by THIS turn stream
            // here, the same place the approval card goes up. The run's
            // own chat is what filters, never a selection made later.
            ForEach(liveRunsForChat) { state in
                SubagentLiveCardView(state: state)
            }
            if !output.isEmpty {
                ChatMessageMarkdownView(output)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .onChange(of: isRunning) { wasRunning, running in
            if running && !wasRunning { startedAt = Date() }
        }
    }

    private var status: ChatActivityStatus {
        var inputs = ChatActivityStatus.Inputs()
        inputs.isCancellationPending = model.isCancellationPending
        inputs.isEscCancelArmed = model.isEscCancelArmed
        inputs.pendingToolName = model.pendingToolCall?.name
        if model.imageModeEnabled, let job = model.imageJob {
            inputs.imageJob = (
                job.status == .waiting, job.stage, job.completed, job.total,
                model.imageProgressFraction)
        }
        inputs.isPrefill = model.phase == .prefill
        inputs.prefillDone = model.livePrefillDone
        inputs.prefillTotal = model.livePrefillTotal
        inputs.runningToolName = model.selectedTurnMessages.last?.toolCalls
            .first(where: { $0.status == .running })?.name
        inputs.activeTask = model.activeTaskDescription
        inputs.reasoning = reasoning
        inputs.hasOutput = !output.isEmpty
        return ChatActivityStatus.resolve(inputs)
    }

    /// The reply's one-line header. Once the run ends (the row lingers
    /// only until the message commits) it drops back to the model name.
    private var activityLine: some View {
        let modelName = model.selected?.alias ?? "TurboSpark"
        let canExpand = !reasoning.isEmpty
        guard isRunning else {
            return ChatActivityLine(
                modelName: modelName,
                label: canExpand ? Text("Thought process", bundle: .module) : nil,
                reasoningExpanded: canExpand ? $showsReasoning : nil)
        }

        var line = ChatActivityLine(
            modelName: modelName, isLive: true, startedAt: startedAt,
            reasoningExpanded: canExpand ? $showsReasoning : nil)
        line.counts = model.liveTokenCount > 0
            ? Text("\(model.liveTokenCount) tokens", bundle: .module) : nil
        switch status {
        case .stopping:
            line.label = Text("Stopping", bundle: .module)
        case .escArmed:
            line.label = Text("Press Esc again to stop", bundle: .module)
            line.tone = .attention
        case .needsConfirmation(let tool):
            line.label = Text("Needs confirmation for \(tool)", bundle: .module)
            line.tone = .attention
        case .imageWaiting:
            line.label = Text("Waiting for image slot", bundle: .module)
        case let .imageStage(stage, completed, total, fraction):
            line.label = stage.map(GenerateControl.imageStageText)
                ?? Text("Generating image", bundle: .module)
            line.progress = .some(fraction)
            line.counts = total > 0 ? Text(verbatim: "\(completed)/\(total)") : nil
        case let .readingPrompt(done, total):
            line.label = Text("Reading prompt...", bundle: .module)
            if total > 0 {
                line.progress = .some(min(1, Double(done) / Double(total)))
                line.counts = Text("\(done.formatted()) / \(total.formatted()) tokens", bundle: .module)
            } else {
                line.progress = .some(nil)
                line.counts = nil
            }
        case .runningTool(let name):
            line.label = Text("Running tools...", bundle: .module)
            line.detail = name
            line.detailIsCode = true
        case .task(let task):
            line.label = Text(verbatim: task)
        case .thinking(let latestLine):
            line.label = Text("Thinking...", bundle: .module)
            line.detail = latestLine
        case .generating:
            line.label = Text("Generating response...", bundle: .module)
        }
        return line
    }

    /// Live foreground subagent runs belonging to this chat, oldest first.
    private var liveRunsForChat: [SubagentRunState] {
        model.liveSubagentRuns.values
            .filter { $0.chatID == nil || $0.chatID == model.selectedChatID }
            .sorted { $0.startedAt < $1.startedAt }
    }
}
