import SwiftUI

struct StreamRecoveryEventNoticeView: View {
    let event: StreamRecoveryEvent
    private let occurrence: Int

    init(event: StreamRecoveryEvent, occurrence: Int = 0) {
        self.event = event
        self.occurrence = occurrence
    }

    private var presentation: StreamRecoveryEventPresentation {
        StreamRecoveryEventPresentation(event: event)
    }

    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: presentation.symbolName)
                .foregroundStyle(presentation.isTerminal ? Color.orange : Color.secondary)
                .accessibilityHidden(true)
            message
                .themedFont(.small)
                .foregroundStyle(.primary)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.quaternary, in: RoundedRectangle(cornerRadius: 10))
        .accessibilityElement(children: .combine)
        .accessibilityIdentifier("\(presentation.accessibilityIdentifier)-\(occurrence)")
    }

    @ViewBuilder
    private var message: some View {
        switch presentation {
        case .continued(let attempt, let tokensSoFar):
            Text(
                "Generation continued. Attempt: \(attempt), tokens so far: \(tokensSoFar).",
                bundle: .module)
        case .partialPreserved(let rows):
            Text("Partial response preserved. Rows: \(rows).", bundle: .module)
        case .tailDiscarded(let reason):
            if let reason {
                Text("Generated tail discarded: \(reason).", bundle: .module)
            } else {
                Text("Generated tail was discarded for an unspecified reason.", bundle: .module)
            }
        case .anchorRecorded(let retainedMessages, let retriesUsed):
            Text(
                "Recovery anchor recorded. Retained messages: \(retainedMessages), retries used: \(retriesUsed).",
                bundle: .module)
        case .retrySucceeded(let attempt):
            Text("Recovery retry succeeded. Attempt: \(attempt).", bundle: .module)
        case .retryConflict:
            Text(
                "Recovery conflict: the saved partial response no longer matches the transcript.",
                bundle: .module)
        case .retryExhausted(let limit):
            Text("Recovery retries exhausted. Limit: \(limit) attempts.", bundle: .module)
        }
    }
}
