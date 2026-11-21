import Foundation

/// Display-only masking of MCP server environment values.
///
/// Nothing here touches what is stored, persisted, synced, or handed to the
/// child process: the pane and the editor call these on text they are about
/// to draw. Env values are where people put API keys, and they also leak into
/// the command line (`--token $VALUE` pasted literally) and into error text a
/// server echoes back, so every place the pane shows such text scrubs it.
enum McpEnvMasking {
    /// Fixed width on purpose: a bullet per character would tell an onlooker
    /// how long the secret is.
    static let mask = String(repeating: "\u{2022}", count: 8)

    /// Values shorter than this are left visible in scrubbed text. "1",
    /// "true" or "prod" are common non-secret values, and masking them would
    /// blank ordinary words out of every command line and error.
    static let minimumScrubbedLength = 6

    /// The mask for a single value shown on its own. An empty value has
    /// nothing to hide and shows as empty, so "unset" stays distinguishable.
    static func maskedValue(_ value: String) -> String {
        value.isEmpty ? "" : mask
    }

    /// `text` with every occurrence of every non-trivial `secrets` value
    /// replaced by `mask`.
    ///
    /// Matches are collected on the original text for all values and merged
    /// before anything is replaced. Replacing one value at a time would break
    /// an overlapping value's match (or a longer value containing a shorter
    /// one) and leave part of it on screen.
    static func scrub(_ text: String, secrets: [String]) -> String {
        let candidates = Set(secrets.filter { $0.count >= minimumScrubbedLength })
        guard !text.isEmpty, !candidates.isEmpty else { return text }

        let source = text as NSString
        var ranges: [NSRange] = []
        for secret in candidates {
            var search = NSRange(location: 0, length: source.length)
            while search.length > 0 {
                let found = source.range(of: secret, options: .literal, range: search)
                guard found.location != NSNotFound else { break }
                ranges.append(found)
                // Step one unit, not past the match, so self-overlapping
                // occurrences ("aaaaaa" in "aaaaaaaa") are all covered.
                let next = found.location + 1
                search = NSRange(location: next, length: source.length - next)
            }
        }
        guard !ranges.isEmpty else { return text }

        ranges.sort { $0.location < $1.location }
        var merged: [NSRange] = []
        for range in ranges {
            if let last = merged.last, range.location <= NSMaxRange(last) {
                let end = max(NSMaxRange(last), NSMaxRange(range))
                merged[merged.count - 1] = NSRange(location: last.location, length: end - last.location)
            } else {
                merged.append(range)
            }
        }

        let result = NSMutableString(string: source)
        for range in merged.reversed() {
            result.replaceCharacters(in: range, with: mask)
        }
        return result as String
    }
}

extension McpServerConfig {
    /// The env values to scrub from this server's displayed text. Remote
    /// servers have no env block.
    var envSecretValues: [String] {
        guard case .stdio(_, _, let env, _, _) = transport else { return [] }
        return Array(env.values)
    }
}

/// The editor's env block as rows, over the same `KEY=VALUE` per line text
/// the sheet already parses into the saved dictionary.
///
/// The text stays the single source of truth so `buildConfig` (and so the
/// saved config) is unchanged. Parsing here is deliberately looser than
/// `buildConfig`: it keeps spaces and empty keys so a row being typed into
/// does not lose characters or vanish; `buildConfig` still trims and drops
/// empty keys at save time.
enum McpEnvRows {
    struct Row: Equatable {
        var key: String
        var value: String
    }

    static func parse(_ text: String) -> [Row] {
        text.components(separatedBy: .newlines).compactMap { line in
            guard !line.trimmingCharacters(in: .whitespaces).isEmpty else { return nil }
            guard let eq = line.firstIndex(of: "=") else { return Row(key: line, value: "") }
            return Row(key: String(line[..<eq]), value: String(line[line.index(after: eq)...]))
        }
    }

    /// A key may not hold "=" (it would move the split point) and neither
    /// side may hold a line break (it would split the row in two), so both
    /// are dropped rather than written through.
    static func serialize(_ rows: [Row]) -> String {
        rows.map { row in
            let key = row.key.filter { $0 != "=" && !$0.isNewline }
            let value = row.value.filter { !$0.isNewline }
            return "\(key)=\(value)"
        }
        .joined(separator: "\n")
    }
}
