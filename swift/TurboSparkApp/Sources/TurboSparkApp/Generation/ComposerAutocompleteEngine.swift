import Foundation

/// What the trailing token of the composer's text is currently triggering.
enum ComposerTrigger: Equatable {
    /// The trailing token reads `/something`.
    case slash(prefix: String)
    /// The trailing token reads `@something`, or the text ends inside an
    /// unterminated `@"...` quoted path.
    case mention(query: String)
}

/// A detected trigger and where its token starts, so accepting a suggestion
/// can replace exactly the token.
struct ComposerTriggerMatch: Equatable {
    var trigger: ComposerTrigger
    /// Character offset (count of `Character`s from the start of the text)
    /// of the token's first character. An Int rather than a String.Index so
    /// a match made against one String instance applies to an equal one.
    var tokenStartOffset: Int
}

/// One row the autocomplete popup offers.
struct ComposerSuggestion: Identifiable, Equatable {
    enum Kind: Equatable {
        case builtInCommand
        case skill
        case file
        case folder
    }

    let kind: Kind
    /// Full replacement token, trigger character included: `/explore`,
    /// `/my-skill`, `@src/App.swift`, `@"my notes.txt"`.
    let replacementToken: String
    let title: String
    let subtitle: String?
    let iconName: String

    var id: String { replacementToken }
}

/// Pure trigger detection and candidate matching for the composer
/// autocomplete. No model, no view, no I/O -- everything here is testable
/// without an `AppModel`.
enum ComposerAutocompleteEngine {
    /// How many rows the popup offers at most.
    static let maxSuggestions = 36

    // MARK: - Trigger detection

    /// Detects a trigger from the text's trailing token.
    ///
    /// **TRAILING TOKEN ONLY.** SwiftUI's `TextEditor` exposes no caret, so
    /// there is nothing to anchor a mid-text detection to: a trigger typed
    /// anywhere but the end of the draft does not open the popup, and
    /// editing an earlier line never re-opens one. The quoted form exists
    /// because it is the one case the trailing-token rule would otherwise
    /// break outright -- a path with spaces stops being a single token the
    /// moment its first space is typed.
    static func detectTrigger(in text: String) -> ComposerTriggerMatch? {
        if let quoteMatch = detectQuotedMention(in: text) {
            return quoteMatch
        }

        guard let tokenStart = startOfTrailingToken(in: text) else { return nil }
        let token = String(text[tokenStart...])
        // `@@` is an escaped literal @, never a trigger.
        guard !token.hasPrefix("@@") else { return nil }
        if token.hasPrefix("@") {
            return ComposerTriggerMatch(
                trigger: .mention(query: String(token.dropFirst())),
                tokenStartOffset: text.distance(from: text.startIndex, to: tokenStart))
        }
        if token.hasPrefix("/") {
            return ComposerTriggerMatch(
                trigger: .slash(prefix: String(token.dropFirst())),
                tokenStartOffset: text.distance(from: text.startIndex, to: tokenStart))
        }
        return nil
    }

    /// `foo @"my partial path` keeps its trigger across the spaces a quoted
    /// path needs. The quote must OPEN a token (start of text or whitespace
    /// before the `@`) and stay unclosed; a closed `@"..."` is a complete
    /// token and no longer a trigger.
    private static func detectQuotedMention(in text: String) -> ComposerTriggerMatch? {
        guard let quoteStart = text.range(of: "@\"", options: .backwards) else {
            return nil
        }
        let afterQuote = text[quoteStart.upperBound...]
        guard !afterQuote.contains("\"") else { return nil }
        let before = text[..<quoteStart.lowerBound]
        guard before.isEmpty || before.last?.isWhitespace == true || before.last?.isNewline == true else {
            return nil
        }
        return ComposerTriggerMatch(
            trigger: .mention(query: String(afterQuote)),
            tokenStartOffset: text.distance(from: text.startIndex, to: quoteStart.lowerBound))
    }

    private static func startOfTrailingToken(in text: String) -> String.Index? {
        guard !text.isEmpty else { return nil }
        let afterWhitespace = text.lastIndex(where: { $0.isWhitespace || $0.isNewline })
            .map { text.index(after: $0) }
        return afterWhitespace ?? text.startIndex
    }

    // MARK: - Slash candidates

    /// Commands and skills matching a typed `/prefix`, best first.
    ///
    /// A skill whose name collides with a built-in command (or alias) is
    /// dropped rather than shown: the agent parser runs first at submit
    /// time, so such a skill is unreachable via slash and offering it would
    /// promise a dispatch that never happens.
    static func slashCandidates(
        prefix: String,
        skills: [AppSkill],
        builtIns: [BuiltInSlashCommand] = BuiltInSlashCommand.all
    ) -> [ComposerSuggestion] {
        let clean = prefix.lowercased()
        let commandRows = builtIns.map { ComposerSuggestion(builtIn: $0) }
        let takenNames = Set(builtIns.flatMap { [$0.name] + $0.aliases })
        let skillRows = skills
            .filter { $0.isEnabled && $0.manifest.userInvocable }
            .filter { !takenNames.contains($0.name.lowercased()) }
            .map { ComposerSuggestion(skill: $0) }

        var ranked: [(bucket: Int, suggestion: ComposerSuggestion)] = []
        for row in commandRows + skillRows {
            // The row title carries the trigger character; matching runs on
            // the BARE name, or a typed prefix is never an actual prefix
            // ("/plan".hasPrefix("pl") is false).
            let bareName = row.title.hasPrefix("/") ? String(row.title.dropFirst()) : row.title
            if let bucket = matchBucket(for: clean, name: bareName, description: row.subtitle) {
                ranked.append((bucket, row))
            }
        }
        return ranked
            .sorted {
                $0.bucket != $1.bucket ? $0.bucket < $1.bucket : $0.suggestion.title < $1.suggestion.title
            }
            .prefix(maxSuggestions)
            .map(\.suggestion)
    }

    // MARK: - Mention candidates

    /// Project paths matching a typed `@query`, best first.
    ///
    /// An empty query offers the whole tree alphabetically rather than
    /// nothing: `@` alone is the browse gesture. A query may span directories
    /// (`src/deep/he`) and is matched as a substring of the relative path.
    static func mentionCandidates(
        query: String,
        entries: [ProjectFileEntry]
    ) -> [ComposerSuggestion] {
        let clean = query.lowercased()
        var ranked: [(bucket: Int, entry: ProjectFileEntry)] = []
        for entry in entries {
            // The mention grammar has no escape sequence for a quote inside a
            // quoted path. Do not serialize a filename that could terminate
            // its token and inject another mention.
            guard !entry.relativePath.contains("\"") else { continue }
            let path = entry.relativePath.lowercased()
            let fileName = (path as NSString).lastPathComponent
            let bucket: Int
            if clean.isEmpty {
                bucket = 0
            } else if fileName.hasPrefix(clean) {
                bucket = 0
            } else if fileName.contains(clean) {
                bucket = 1
            } else if path.contains(clean) {
                bucket = 2
            } else {
                continue
            }
            ranked.append((bucket, entry))
        }
        return ranked
            .sorted {
                if $0.bucket != $1.bucket { return $0.bucket < $1.bucket }
                return $0.entry.relativePath.lowercased() < $1.entry.relativePath.lowercased()
            }
            .prefix(maxSuggestions)
            .map { ComposerSuggestion(entry: $0.entry) }
    }

    /// 0 = name prefix, 1 = name contains, 2 = description contains, else
    /// no match. The buckets ARE the ordering: a stable prefix-first rank
    /// beats any fuzzy score for a list this short.
    private static func matchBucket(for clean: String, name: String, description: String?) -> Int? {
        let lowerName = name.lowercased()
        if lowerName.hasPrefix(clean) { return 0 }
        if lowerName.contains(clean) { return 1 }
        if let description, clean.count >= 3, description.lowercased().contains(clean) { return 2 }
        return nil
    }
}

extension ComposerSuggestion {
    init(builtIn: BuiltInSlashCommand) {
        self.init(
            kind: .builtInCommand,
            replacementToken: "/\(builtIn.name)",
            title: "/\(builtIn.name)",
            subtitle: builtIn.summary,
            iconName: builtIn.iconName)
    }

    init(skill: AppSkill) {
        self.init(
            kind: .skill,
            replacementToken: "/\(skill.name)",
            title: "/\(skill.name)",
            subtitle: skill.skillDescription,
            iconName: "wand.and.stars")
    }

    init(entry: ProjectFileEntry) {
        let needsQuotes = entry.relativePath.contains(where: { $0.isWhitespace })
        let token = needsQuotes ? "@\"\(entry.relativePath)\"" : "@\(entry.relativePath)"
        self.init(
            kind: entry.isDirectory ? .folder : .file,
            replacementToken: token,
            title: entry.relativePath,
            subtitle: entry.isDirectory ? "Folder" : nil,
            iconName: entry.isDirectory ? "folder.fill" : "doc")
    }
}
