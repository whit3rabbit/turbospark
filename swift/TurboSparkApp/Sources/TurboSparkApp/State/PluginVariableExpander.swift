import Foundation

/// The plugin variable substitution contract, matching Claude Code's:
///
/// - `${CLAUDE_PLUGIN_ROOT}`: the plugin's version-scoped install directory
///   (it changes on update, which is the point -- a hardcoded absolute path
///   would survive an update only by pointing at the old version).
/// - `${CLAUDE_PLUGIN_DATA}`: the persistent per-plugin data directory.
/// - `${user_config.KEY}`: a saved option value. Sensitive keys are
///   substituted ONLY by callers that opt in (`preserveSensitive: true`,
///   used for hook and MCP environments); skill and agent prose get a
///   placeholder instead, because content is shown, copied and sent to
///   places the user cannot see. Unknown keys stay literal, also Claude
///   Code's behavior.
public enum PluginVariableExpander {
    public static let sensitivePlaceholder =
        "[sensitive option not available in content]"

    static let userConfigPattern = #"\$\{user_config\.([A-Za-z_][A-Za-z0-9_]*)\}"#

    public static func expand(
        _ input: String,
        pluginRoot: String?,
        pluginData: String?,
        optionValue: (String) -> String?,
        sensitiveKeys: Set<String> = [],
        preserveSensitive: Bool = false
    ) -> String {
        var result = input
        if let pluginRoot {
            result = result.replacingOccurrences(
                of: "${CLAUDE_PLUGIN_ROOT}", with: pluginRoot)
        }
        if let pluginData {
            result = result.replacingOccurrences(
                of: "${CLAUDE_PLUGIN_DATA}", with: pluginData)
        }

        for key in referencedOptionKeys(in: result) {
            let placeholder = "${user_config.\(key)}"
            if sensitiveKeys.contains(key), !preserveSensitive {
                result = result.replacingOccurrences(of: placeholder, with: sensitivePlaceholder)
            } else if let value = optionValue(key) {
                result = result.replacingOccurrences(of: placeholder, with: value)
            }
            // An unknown key stays literal: a plugin author reading their
            // own prose can see the name they misspelled.
        }
        return result
    }

    /// The option keys a piece of text references, in order of appearance.
    static func referencedOptionKeys(in text: String) -> [String] {
        guard let regex = try? NSRegularExpression(pattern: userConfigPattern) else { return [] }
        let range = NSRange(location: 0, length: (text as NSString).length)
        return regex.matches(in: text, range: range).compactMap { match in
            Range(match.range(at: 1), in: text).map { String(text[$0]) }
        }
    }
}
