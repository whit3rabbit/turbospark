import Foundation
import TurboSpark

/// `AppModel`'s own nested vocabulary: the phase a turn is in, the section
/// the window is showing, and the mode the app is being used in.
///
/// Split out of the base file, which the package's own convention says is
/// for published state and core lifecycle (`swift/CLAUDE.md`) -- three enum
/// declarations with their display tables are neither, and they are the part
/// of that file most likely to be read on its own. Still NESTED, so every
/// `AppModel.AppNavigationSection` spelling in the views and the tests is
/// unchanged.
extension AppModel {
    /// The current execution phase of text generation.
    public enum GenerationPhase: Equatable, Sendable {
        /// No active generation.
        case idle
        /// Evaluating prompt tokens into key-value cache.
        case prefill
        /// Generating new output tokens sequentially or speculatively.
        case decode
    }

    /// Primary top-level navigation destination in the application.
    public enum AppNavigationSection: String, CaseIterable, Identifiable, Sendable {
        case chat
        case images
        case files
        case modelManager
        case modelHub
        case server
        case audio
        case humanize

        public var id: String { rawValue }
        public var title: String {
            switch self {
            case .chat: return "Chat"
            case .images: return "Image Generation"
            case .files: return "Files"
            case .modelManager: return "Installed"
            case .modelHub: return "Discover"
            case .server: return "API"
            case .audio: return "Audio"
            case .humanize: return "Humanize"
            }
        }
        public var systemImage: String {
            switch self {
            case .chat: return "bubble.left.and.bubble.right"
            case .images: return "photo.on.rectangle"
            case .files: return "folder"
            case .modelManager: return "internaldrive"
            case .modelHub: return "shippingbox"
            case .server: return "server.rack"
            case .audio: return "waveform"
            case .humanize: return "text.badge.checkmark"
            }
        }
        /// Filled variant used when the section is the active one.
        public var selectedSystemImage: String {
            switch self {
            case .chat: return "bubble.left.and.bubble.right.fill"
            case .images: return "photo.on.rectangle.angled"
            case .files: return "folder.fill"
            case .modelManager: return "internaldrive.fill"
            case .modelHub: return "shippingbox.fill"
            case .server: return "server.rack"
            case .audio: return "waveform"
            case .humanize: return "text.badge.checkmark"
            }
        }
        /// Full-bleed sections that replace the chat window, so the chat's
        /// artifact, output and browser panes stay hidden while one is active.
        public var hidesChatPanes: Bool {
            switch self {
            case .images, .audio, .humanize: return true
            case .chat, .files, .modelManager, .modelHub, .server: return false
            }
        }
        /// Keyboard shortcut character shown in the rail tooltip.
        public var shortcutKey: Character {
            switch self {
            case .chat: return "1"
            case .images: return "2"
            case .files: return "3"
            case .modelManager: return "4"
            case .modelHub: return "5"
            case .server: return "6"
            case .audio: return "7"
            case .humanize: return "8"
            }
        }
    }

    /// Primary user interface interaction mode.
    public enum AppInteractionMode: String, Codable, CaseIterable, Identifiable, Sendable {
        case chat
        case projects

        public var id: String { rawValue }
        public var title: String {
            switch self {
            case .chat: return "Chat"
            case .projects: return "Projects"
            }
        }
        public var systemImage: String {
            switch self {
            case .chat: return "bubble.left.and.bubble.right"
            case .projects: return "chevron.left.forwardslash.chevron.right"
            }
        }
    }
}
