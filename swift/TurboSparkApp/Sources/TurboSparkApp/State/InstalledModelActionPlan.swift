import Foundation

/// What the action bar of an installed model offers, per state (requirements 3.2,
/// 3.7, 5.4 and 5.5, design.md "Library UI set").
///
/// **IT IS PURE.** The bar draws this and nothing else, so a test pins each state
/// without a view.
///
/// - open: Load and Start Chat, as before.
/// - not connected: Load is REPLACED by a statement naming the drive to connect,
///   and there is no Start Chat. The model offers no removal either (the engine
///   forgets a row only with its library connected).
/// - missing: the statement that the files are missing, no Start Chat, and the
///   bar's own Remove from list.
/// - warned: the warning stays on screen, the load action becomes "Load anyway"
///   (the same load Load runs), and Start Chat is disabled with the warning as its
///   reason until the model is loaded, so Start Chat cannot skip the warning.
/// - loaded: Unload and Start Chat, and no warning: the model is in memory.
///
/// **A STATEMENT AND A WARNING ARE A GLYPH AND A SENTENCE.** The glyph is
/// decoration (VoiceOver skips it); the sentence is what is read, before the
/// buttons.
struct InstalledModelActionPlan: Equatable {
    enum LoadControl: Equatable {
        case unload
        case load
        case loadAnyway
        /// No load control at all: a statement stands where it was.
        case none
    }

    enum StartChat: Equatable {
        case hidden
        case enabled
        case disabled(reason: String)
    }

    struct Notice: Equatable {
        enum Kind: Equatable {
            /// Replaces the load action: "Connect T7 to load".
            case statement
            /// Accompanies "Load anyway".
            case warning
        }

        let kind: Kind
        let text: String
        let glyph: String
    }

    let loadControl: LoadControl
    let notice: Notice?
    let startChat: StartChat
    /// Reveal in Finder needs files that can be reached.
    let revealEnabled: Bool

    init(rowModel: InstalledModelRowModel, isCurrentlyLoaded: Bool) {
        revealEnabled = rowModel.status == nil

        if isCurrentlyLoaded {
            loadControl = .unload
            notice = nil
            startChat = .enabled
            return
        }
        switch rowModel.gate {
        case .open:
            loadControl = .load
            notice = nil
            startChat = .enabled
        case .warn:
            let text = rowModel.warningText ?? ""
            loadControl = .loadAnyway
            notice = Notice(kind: .warning, text: text, glyph: LoadGateText.warningGlyph)
            startChat = .disabled(reason: text)
        case .notConnected, .missing:
            loadControl = .none
            notice = Notice(
                kind: .statement, text: rowModel.loadBlockedReason ?? "", glyph: rowModel.statusGlyph ?? "")
            startChat = .hidden
        }
    }
}
