import Foundation

/// Projects browser tool results into the closed diagnostics shape accepted by
/// support reports. The transcript output and image references are never read.
enum BrowserAutomationSupportReportAdapter {
    static func project(
        call: AppToolCall,
        result: AppToolResult
    ) -> BrowserAutomationDiagnosticEvent? {
        guard let action = BrowserToolDefinitions.commandKind(for: call.name.lowercased()),
              let originString = result.browserCardMetadata?.canonicalOrigin,
              let origin = BrowserOrigin(origin: originString),
              origin.canonicalString == originString,
              result.durationSeconds.isFinite,
              result.durationSeconds >= 0
        else { return nil }

        let outcome: BrowserAutomationOutcome
        switch ToolPresentation.status(call: call, result: result) {
        case .completed: outcome = .completed
        case .denied: outcome = .denied
        case .failed: outcome = .failed
        case .running, .pendingApproval: return nil
        }

        let milliseconds = min(
            result.durationSeconds * 1_000,
            Double(UInt32.max)
        )
        return BrowserAutomationDiagnosticEvent(
            actionType: action,
            origin: origin,
            outcome: outcome,
            durationMilliseconds: UInt32(milliseconds.rounded(.down)))
    }
}
