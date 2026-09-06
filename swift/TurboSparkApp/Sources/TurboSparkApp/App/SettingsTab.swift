import SwiftUI

extension AppSettingsView {
    public enum SettingsTab: String, CaseIterable, Identifiable {
        case general = "General"
        case profiles = "Profiles"
        case appearance = "Appearance"
        case shortcuts = "Keyboard shortcuts"
        case permissions = "Files & Permissions"
        case models = "Models & Storage"
        case engine = "Engine"
        case safety = "Safety & Steering"
        case mcp = "MCP Servers"
        case skills = "Skills"
        case agents = "Agents & Subagents"
        case plugins = "Plugins"
        case hooks = "Hooks & Lifecycle"

        public var id: String { rawValue }

        public var title: String { rawValue }

        public var systemImage: String {
            switch self {
            case .general: return "gearshape"
            case .profiles: return "person.crop.circle"
            case .appearance: return "sun.max"
            case .shortcuts: return "keyboard"
            case .permissions: return "folder.badge.gearshape"
            case .models: return "cylinder.split.1x2"
            case .engine: return "cpu"
            case .safety: return "dial.medium"
            case .mcp: return "server.rack"
            case .skills: return "wand.and.stars"
            case .agents: return "person.2.badge.gearshape"
            case .plugins: return "puzzlepiece.extension"
            case .hooks: return "link.badge.plus"
            }
        }

        public var category: String {
            switch self {
            case .general, .profiles, .appearance, .shortcuts, .permissions:
                return "Personal"
            case .models, .engine, .safety, .mcp, .skills, .agents, .plugins, .hooks:
                return "Engine & Coding"
            }
        }

        public var keywords: [String] {
            switch self {
            case .general:
                return ["language", "locale", "localization", "keyboard", "readability", "text size"]
            case .profiles:
                return ["profiles", "users", "accounts", "switch user", "multi user"]
            case .appearance:
                return ["theme", "font", "size", "color", "accent", "contrast", "dark", "light", "display", "motion"]
            case .shortcuts:
                return ["keyboard", "keys", "shortcuts", "hotkeys", "commands"]
            case .permissions:
                return ["files", "privacy", "access", "filesystem", "security"]
            case .models:
                return ["storage", "lm studio", "downloads", "folders", "cache", "context"]
            case .engine:
                return ["system prompt", "temperature", "tokens", "sampling", "guardrails", "speculation", "server", "reasoning"]
            case .safety:
                return ["steering", "vectors", "safety", "direction", "control"]
            case .mcp:
                return ["mcp", "servers", "tools", "protocols"]
            case .skills:
                return ["skills", "custom tools", "instructions"]
            case .agents:
                return ["agents", "subagents", "personas"]
            case .plugins:
                return ["plugins", "marketplace", "extensions", "addons"]
            case .hooks:
                return ["hooks", "lifecycle", "events", "scripts"]
            }
        }
    }
}
