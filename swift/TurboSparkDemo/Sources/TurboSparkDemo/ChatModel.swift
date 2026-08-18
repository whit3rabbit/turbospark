import Foundation
import SwiftUI
import TurboSpark

/// The demo's whole state.
///
/// `@MainActor` throughout: every property here drives a view, and the
/// binding's own threading is already handled a layer down (the session owns
/// a serial queue and streams events back). So this type never touches a
/// background thread itself, which is the arrangement a real app should copy.
@MainActor
final class ChatModel: ObservableObject {

    struct Turn: Identifiable, Equatable {
        let id = UUID()
        var role: ChatMessage.Role
        var content: String
        /// Shown in a collapsible block. Kept separate from `content`
        /// because it must NOT be sent back as history: Harmony drops
        /// prior-turn analysis and Qwen's template drops prior-turn
        /// `<think>` blocks, so replaying it sends the model something it
        /// was never trained to read.
        var reasoning: String = ""
        var stopReason: GenerationResult.StopReason?
    }

    // Model selection
    @Published var installed: [InstalledModel] = []
    @Published var catalog: [CatalogEntry] = []
    @Published var selected: InstalledModel?
    @Published var session: TurboSparkSession?
    @Published var opening = false

    // Conversation
    @Published var turns: [Turn] = []
    @Published var draft = ""
    @Published var generating = false
    @Published var prefill: (done: Int, total: Int)?

    // Status
    @Published var lastResult: GenerationResult?
    @Published var peakFootprint: UInt64?
    @Published var error: String?

    // Settings
    @Published var reasoning: GenerateOptions.Reasoning = .off
    @Published var temperature: Double = 0.2
    @Published var maxNewTokens: Double = 512

    private var task: Task<Void, Never>?

    var info: SessionInfo? { session?.info }

    /// True when the open checkpoint's template cannot express a reasoning
    /// level, so the picker should be disabled rather than silently ignored.
    var reasoningAvailable: Bool {
        info?.reasoningSupport != SessionInfo.ReasoningSupport.none
    }

    func refreshModels() {
        do {
            installed = try TurboSparkCatalog.installed()
            catalog = try TurboSparkCatalog.available()
            if selected == nil { selected = installed.first }
        } catch {
            self.error = "\(error)"
        }
    }

    func open(_ model: InstalledModel) async {
        opening = true
        error = nil
        // Drop the old session first: each one pins gigabytes, and holding
        // two while the second opens is the easy way to run a machine out of
        // memory during a model switch.
        session = nil
        turns = []
        defer { opening = false }
        do {
            // Everything automatic. A GUI is exactly the caller that SHOULD
            // let Low Power Mode pick the power profile and let the machine
            // pick the expert-cache slots; a measurement harness is the one
            // that must not.
            session = try await TurboSparkSession(modelPath: model.path)
            selected = model
        } catch {
            self.error = "\(error)"
        }
    }

    func send() {
        guard let session, !draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return
        }
        let text = draft
        draft = ""
        turns.append(Turn(role: .user, content: text))
        turns.append(Turn(role: .assistant, content: ""))
        generating = true
        error = nil
        prefill = nil

        // Only `.content` becomes history. See `Turn.reasoning`.
        let history = turns.compactMap { turn -> ChatMessage? in
            guard !turn.content.isEmpty || turn.role == .user else { return nil }
            return ChatMessage(role: turn.role, content: turn.content)
        }

        var options = GenerateOptions()
        options.reasoning = reasoning
        options.temperature = temperature
        options.maxNewTokens = UInt32(maxNewTokens)

        task = Task {
            do {
                for try await event in session.generate(history, options: options) {
                    switch event {
                    case .prefill(let done, let total):
                        prefill = (done, total)
                    case .content(let chunk):
                        prefill = nil
                        turns[turns.count - 1].content += chunk
                    case .reasoning(let chunk):
                        prefill = nil
                        turns[turns.count - 1].reasoning += chunk
                    case .finished(let result):
                        turns[turns.count - 1].stopReason = result.stopReason
                        lastResult = result
                        peakFootprint = TurboSparkSession.peakFootprintBytes
                    }
                }
            } catch {
                self.error = "\(error)"
            }
            generating = false
            prefill = nil
        }
    }

    /// Stop. Reaches the engine immediately rather than queueing behind the
    /// turn it is stopping -- the whole reason the binding is shaped the way
    /// it is.
    func stop() {
        session?.cancel()
    }

    func clear() {
        turns = []
        lastResult = nil
    }
}

/// Bytes as something a person reads.
func humanBytes(_ bytes: UInt64) -> String {
    let units = ["B", "KiB", "MiB", "GiB", "TiB"]
    var value = Double(bytes)
    var unit = 0
    while value >= 1024, unit < units.count - 1 {
        value /= 1024
        unit += 1
    }
    return unit == 0 ? "\(bytes) B" : String(format: "%.1f %@", value, units[unit])
}
