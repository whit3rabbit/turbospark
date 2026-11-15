import Foundation
import TurboSpark

/// Trajectory logging (ZP-03): one local log line per finished model call and
/// per finished tool execution, written after the fact by `RolloutStore`.
///
/// Observation only. Nothing here changes a prompt, a sample or a tool
/// result, and nothing is sent anywhere. Off by default.
extension AppModel {
    /// Whether this chat's calls may be logged. Pure so the rule is tested
    /// without a model: the setting must be on, and the chat must be neither
    /// a Ghost chat (it exists only in memory) nor in a passphrase-protected
    /// profile (nothing readable while locked).
    nonisolated static func trajectoryEligible(
        enabled: Bool, isGhost: Bool, profileProtected: Bool
    ) -> Bool {
        enabled && !isGhost && !profileProtected
    }

    func trajectoryEligible(chatID: UUID) -> Bool {
        guard let chat = chats.first(where: { $0.id == chatID }) else { return false }
        return Self.trajectoryEligible(
            enabled: trajectoryLoggingEnabled, isGhost: chat.isGhost,
            profileProtected: ProfileVaultCoordinator.shared.isProtected)
    }

    /// Called from `trajectoryLoggingEnabled`'s `didSet`. Turning logging off
    /// deletes every stored log: they are derived records, and "off" has to
    /// mean there is nothing left on disk.
    func trajectoryLoggingDidChange(from previous: Bool) {
        if previous && !trajectoryLoggingEnabled { RolloutStore.shared.deleteAll() }
    }

    /// At launch: a log left by a crash between turning logging off and the
    /// delete is removed, and expired logs are pruned.
    func reconcileTrajectoryStoreAtLoad() {
        if trajectoryLoggingEnabled {
            RolloutStore.shared.pruneExpired()
        } else {
            RolloutStore.shared.deleteAll()
        }
    }

    /// One finished model request (each continuation segment is its own).
    /// Latency is the engine's own prefill plus decode time.
    func logTrajectoryModelCall(result: GenerationResult, step: Int, chatID: UUID) {
        guard trajectoryEligible(chatID: chatID) else { return }
        let entry = RolloutEntry(
            kind: .modelCall, role: "assistant", step: step,
            model: sessionModelPath.map { ($0 as NSString).lastPathComponent },
            promptTokens: result.promptTokens, completionTokens: result.newTokens,
            cachedTokens: result.reusedPrefixTokens,
            latencyMs: Int(((result.prefillSeconds + result.decodeSeconds) * 1_000).rounded()),
            finishReason: result.stopReason.rawValue,
            args: nil,
            result: result.content.isEmpty ? nil : result.content)
        RolloutStore.shared.append(entry, chatID: chatID)
    }

    /// One finished tool execution, from the single funnel every outcome
    /// passes through (`appendToolExecutionTurn`).
    func logTrajectoryToolExec(call: AppToolCall, result: AppToolResult, chatID: UUID) {
        guard trajectoryEligible(chatID: chatID) else { return }
        let entry = RolloutEntry(
            kind: .toolExec, role: "tool",
            latencyMs: Int((result.durationSeconds * 1_000).rounded()),
            finishReason: result.isError ? "error" : "ok",
            name: call.name, args: call.arguments,
            result: result.isError ? nil : result.output,
            error: result.isError ? result.output : nil)
        RolloutStore.shared.append(entry, chatID: chatID)
    }
}
