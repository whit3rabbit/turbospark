import Foundation
import TurboSpark

/// A chat-model load that did not happen, typed so the strip above the
/// composer can say why and offer the fix that matches (`ModelLoadFailureBannerView`).
///
/// **THE CAUSE IS NEVER GUESSED.** The engine reports an open failure as free
/// text (`TurboSparkError.Code.open` covers every reason), so a specific kind is
/// chosen only from a sentence the Rust side writes verbatim. Anything else is
/// `.other` and the strip shows the engine's own words. A vague "memory" or
/// "not found" in an unrelated message would otherwise send the user to the
/// wrong fix.
struct ModelLoadFailure: Identifiable, Equatable {
    let id = UUID()
    /// The row the load was for, kept so Retry asks for the same model.
    let model: InstalledModel
    let kind: Kind
    /// The engine's message, or empty for a refusal made before the engine ran.
    let detail: String
    /// The exact `model.error` text set alongside this failure, if any.
    /// `error` is still set because other paths and tests read it; the
    /// banner hides only for this exact text so the user sees one notice.
    var errorText: String? = nil

    /// True when `error` is the same failure this strip already shows.
    func covers(error: String?) -> Bool {
        guard let errorText, let error else { return false }
        return errorText == error
    }

    enum Kind: Equatable {
        /// The model's drive is away (`LoadGate.notConnected`).
        case notConnected(libraryName: String?)
        /// The folder or one of its required files is gone.
        case missingFiles
        /// The install's checksum or index does not match its manifest.
        case damaged
        /// The context window or the load guard's ceiling does not fit memory.
        case notEnoughMemory
        /// This build cannot read the install's family, version or format.
        case unsupported
        case other

        /// A refusal made before the engine ran. Nil for a gate that loads.
        init?(gate: LoadGate) {
            switch gate {
            case .notConnected(let libraryName): self = .notConnected(libraryName: libraryName)
            case .missing: self = .missingFiles
            case .open, .warn: return nil
            }
        }

        /// Classifies the engine's open message. Each marker is a fixed part of
        /// a `Display` in `crates/model-io/src/error.rs` or
        /// `crates/model-io/src/context_policy.rs`; if that text changes, the
        /// failure falls back to `.other`, which is safe.
        static func classify(engineMessage message: String) -> Kind {
            // First: an unknown family arrives wrapped as "resident index is
            // corrupt", and reinstalling would not help it.
            let unsupported = [
                "unknown arch.family",
                "is not supported by the text-model loader",
                "is not supported (this build reads",
                "magic does not equal \"GTURBO\"",
            ]
            if unsupported.contains(where: message.contains) { return .unsupported }
            let memory = [
                "of KV cache;",
                "--load-guard custom would allocate",
                "was not met: `auto` resolved",
            ]
            if memory.contains(where: message.contains) { return .notEnoughMemory }
            let missing = ["is missing manifest.json", "is missing required file"]
            if missing.contains(where: message.contains) { return .missingFiles }
            let damaged = [
                "does not match manifest.files[",
                "resident index is corrupt",
                "no IndexEntry named",
            ]
            if damaged.contains(where: message.contains) { return .damaged }
            return .other
        }

        /// The strip's actions, the most useful first (it is the primary
        /// button). Retry appears only where trying again can change the
        /// answer: a drive that may be back, memory that may have been freed,
        /// or a cause nobody could name.
        var actions: [Action] {
            switch self {
            case .notConnected, .other: return [.retry, .chooseModel]
            case .notEnoughMemory: return [.chooseModel, .retry, .engineSettings]
            case .missingFiles, .damaged, .unsupported: return [.chooseModel]
            }
        }
    }

    enum Action: Equatable {
        case retry
        /// Opens Installed Models, where another model is chosen or this one
        /// reinstalled or removed.
        case chooseModel
        /// Opens Engine settings, which holds the context size and the memory
        /// load guard.
        case engineSettings
    }

    func title(language: AppLanguage = AppStrings.currentLanguage) -> String {
        AppStrings.localized("Could not load \(model.alias)", language: language)
    }

    /// One line naming the cause. `.other` shows the engine's words, since
    /// nothing better is known.
    func cause(language: AppLanguage = AppStrings.currentLanguage) -> String {
        switch kind {
        case .notConnected(let libraryName):
            return LoadGateText.statement(for: .notConnected(libraryName: libraryName), language: language) ?? detail
        case .missingFiles:
            return LoadGateText.statement(for: .missing, language: language) ?? detail
        case .damaged:
            return AppStrings.localized(
                "This model's files are damaged. Reinstall it or choose another model.", language: language)
        case .notEnoughMemory:
            return AppStrings.localized(
                "There is not enough memory for this model at its current context size. Choose a smaller model or lower the context in Engine settings.",
                language: language)
        case .unsupported:
            return AppStrings.localized(
                "This version of TurboSpark cannot open this model. Update TurboSpark or choose another model.",
                language: language)
        case .other:
            return detail
        }
    }
}
