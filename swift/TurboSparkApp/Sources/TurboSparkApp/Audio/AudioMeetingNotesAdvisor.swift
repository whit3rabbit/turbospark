import Foundation
import TurboSpark

/// Recommends and configures chat models for meeting summarization and note generation.
///
/// Phase 2:
/// Offers any installed chat model to the user, while highlighting and recommending
/// lightweight, high-speed models with large context windows (specifically Qwen 3.5 1.5B
/// and Qwen 1.5B families) that run efficiently on local Apple Silicon.
enum AudioMeetingNotesAdvisor {

    /// Whether a model qualifies as recommended for meeting summarization.
    /// Prioritizes Qwen 3.5 1.5B / Qwen 1.5B and lightweight models (<= 4B parameters or Qwen family).
    static func isRecommended(model: InstalledModel) -> Bool {
        let alias = model.alias.lowercased()
        let family = model.family.lowercased()

        // Explicit recommendation: Qwen 3.5 1.5B or Qwen 1.5B
        if alias.contains("qwen") && (alias.contains("1.5") || alias.contains("1_5") || alias.contains("3.5") || alias.contains("35")) {
            return true
        }

        // Lightweight Qwen variants
        if alias.contains("qwen") && (alias.contains("0.5") || alias.contains("1b") || alias.contains("2b") || alias.contains("3b") || alias.contains("4b") || alias.contains("7b")) {
            return true
        }

        // Qwen family trunks
        if family.contains("qwen35") || family.contains("qwen3") || family.contains("qwen2") {
            return true
        }

        return false
    }

    /// User-facing recommendation badge reason.
    static func recommendationBadge(for model: InstalledModel) -> String? {
        let alias = model.alias.lowercased()
        if alias.contains("1.5") || alias.contains("1_5") || alias.contains("3.5") || alias.contains("35") {
            return "Recommended: Fast, 32k context"
        }
        if isRecommended(model: model) {
            return "Recommended: High speed"
        }
        return nil
    }

    /// Numeric recommendation tier (higher is better).
    static func recommendationTier(for model: InstalledModel) -> Int {
        let alias = model.alias.lowercased()
        if alias.contains("qwen") && (alias.contains("1.5") || alias.contains("1_5") || alias.contains("3.5") || alias.contains("35")) {
            return 2
        }
        if isRecommended(model: model) {
            return 1
        }
        return 0
    }

    /// Sorts models so recommended models appear at the top.
    static func prioritizedModels(_ models: [InstalledModel]) -> [InstalledModel] {
        models.sorted { lhs, rhs in
            let lhsTier = recommendationTier(for: lhs)
            let rhsTier = recommendationTier(for: rhs)
            if lhsTier != rhsTier {
                return lhsTier > rhsTier
            }
            return lhs.alias.localizedStandardCompare(rhs.alias) == .orderedAscending
        }
    }

    /// Selects the best initial model for meeting summarization from available models.
    static func defaultModel(from models: [InstalledModel], currentSelected: InstalledModel?) -> InstalledModel? {
        let textModels = models.filter { $0.modality == "text" }
        guard !textModels.isEmpty else { return nil }

        // 1. Look for Qwen 3.5 1.5B or Qwen 1.5B
        if let match = textModels.first(where: {
            let a = $0.alias.lowercased()
            return a.contains("qwen") && (a.contains("1.5") || a.contains("1_5") || a.contains("3.5") || a.contains("35"))
        }) {
            return match
        }

        // 2. Look for any recommended model
        if let match = textModels.first(where: { isRecommended(model: $0) }) {
            return match
        }

        // 3. Keep current selection if valid
        if let currentSelected, textModels.contains(where: { $0.id == currentSelected.id }) {
            return currentSelected
        }

        return textModels.first
    }
}
