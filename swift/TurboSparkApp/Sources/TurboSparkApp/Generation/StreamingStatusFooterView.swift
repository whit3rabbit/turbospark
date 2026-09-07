import SwiftUI

/// Animated footer status row displayed during model generation and tool execution.
/// Features a slowly pulsing/breathing TurboSpark logo and live telemetry (elapsed time, token count, status).
public struct StreamingStatusFooterView: View {
    @ObservedObject var model: AppModel

    @State private var isBreathing: Bool = false
    @State private var startTime: Date? = nil
    @State private var elapsedSeconds: Int = 0

    private let timer = Timer.publish(every: 1.0, on: .main, in: .common).autoconnect()

    public init(model: AppModel) {
        self.model = model
    }

    public var body: some View {
        HStack(spacing: 8) {
            TaskProgressFlameIcon(size: 15)

            HStack(spacing: 6) {
                if elapsedSeconds > 0 {
                    Text(formattedElapsedTime)
                        .themedFont(points: 11, weight: .regular)
                        .foregroundStyle(.secondary)
                    dotSeparator
                }

                if model.liveTokenCount > 0 {
                    Text("\(model.liveTokenCount) tokens", bundle: .module)
                        .themedFont(points: 11, weight: .regular)
                        .foregroundStyle(.secondary)
                    dotSeparator
                }

                if let pending = model.pendingToolCall {
                    Text("Needs confirmation for \(pending.name)", bundle: .module)
                        .themedFont(points: 11, weight: .medium)
                        .foregroundStyle(Color.orange)
                } else {
                    Text(currentStatusText)
                        .themedFont(points: 11, weight: .regular)
                        .foregroundStyle(.secondary)
                }
            }
        }
        .padding(.vertical, 6)
        .padding(.horizontal, 4)
        .onAppear {
            startTime = Date()
        }
        .onReceive(timer) { _ in
            if let start = startTime {
                elapsedSeconds = max(0, Int(Date().timeIntervalSince(start)))
            }
        }
        .onChange(of: model.isRunning) { wasRunning, isRunning in
            if isRunning && !wasRunning {
                startTime = Date()
                elapsedSeconds = 0
            }
        }
    }

    private var dotSeparator: some View {
        Circle()
            .fill(Color.secondary.opacity(0.4))
            .frame(width: 2.5, height: 2.5)
            .accessibilityHidden(true)
    }

    private var formattedElapsedTime: String {
        if elapsedSeconds < 60 {
            return "\(elapsedSeconds)s"
        }
        let minutes = elapsedSeconds / 60
        let seconds = elapsedSeconds % 60
        return "\(minutes)m \(seconds)s"
    }

    private var currentStatusText: String {
        if let activeTask = model.activeTaskDescription {
            return activeTask
        }
        if !model.outputReasoningText.isEmpty && model.outputText.isEmpty {
            return "Thinking..."
        }
        if model.selectedTurnMessages.last?.toolCalls.contains(where: { $0.status == .running }) == true {
            return "Running tools..."
        }
        if model.isRunning {
            return "Generating response..."
        }
        return "Idle"
    }
}
