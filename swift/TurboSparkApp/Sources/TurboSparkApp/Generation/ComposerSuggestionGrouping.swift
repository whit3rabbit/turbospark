import Foundation

/// A section of the composer autocomplete popup. The raw value IS the fixed
/// display order, so a group never moves between keystrokes.
enum ComposerSuggestionGroup: Int, CaseIterable, Equatable {
    // Typed references lead: `@` alone is also the menu of schemes, and the
    // engine already ranked them ahead of paths before grouping existed.
    case references
    case files
    case folders
    case commands
    case skills

    init(_ kind: ComposerSuggestion.Kind) {
        switch kind {
        case .reference: self = .references
        case .file: self = .files
        case .folder: self = .folders
        case .builtInCommand: self = .commands
        case .skill: self = .skills
        }
    }

    /// The header word, a catalog key. The word (plus the row glyph) carries
    /// the kind; the glyph tint is secondary.
    var title: String {
        switch self {
        case .references: return "References"
        case .files: return "Files"
        case .folders: return "Folders"
        case .commands: return "Commands"
        case .skills: return "Skills"
        }
    }
}

/// One group's rows and where they start in the flattened list.
struct ComposerSuggestionSection: Equatable {
    let group: ComposerSuggestionGroup
    /// Index of `rows[0]` in `ComposerSuggestionGrouping.flattened`, the list
    /// the controller's `selectedIndex` points into.
    let startIndex: Int
    let rows: [ComposerSuggestion]
}

/// Pure grouping for the popup. Keyboard navigation stays a single flat
/// index: the controller stores the flattened order, so arrows walk the menu
/// in the order it reads and accept reads the same index.
enum ComposerSuggestionGrouping {
    /// Ranked rows split into groups in `ComposerSuggestionGroup` order,
    /// empty groups dropped. Order inside a group is the input order, so the
    /// engine's ranking survives.
    static func sections(_ ranked: [ComposerSuggestion]) -> [ComposerSuggestionSection] {
        var buckets: [ComposerSuggestionGroup: [ComposerSuggestion]] = [:]
        for row in ranked {
            buckets[ComposerSuggestionGroup(row.kind), default: []].append(row)
        }
        var sections: [ComposerSuggestionSection] = []
        var offset = 0
        for group in ComposerSuggestionGroup.allCases {
            guard let rows = buckets[group], !rows.isEmpty else { continue }
            sections.append(ComposerSuggestionSection(group: group, startIndex: offset, rows: rows))
            offset += rows.count
        }
        return sections
    }

    /// The rows in grouped display order.
    static func flattened(_ ranked: [ComposerSuggestion]) -> [ComposerSuggestion] {
        sections(ranked).flatMap(\.rows)
    }
}

// MARK: - Slash menu filter chips

extension ComposerSuggestionGrouping {
    /// The kinds the `/` menu's filter chips offer, in fixed group order.
    /// Empty when fewer than two kinds match: a chip that cannot narrow the
    /// list is noise.
    static func filterGroups(_ ranked: [ComposerSuggestion]) -> [ComposerSuggestionGroup] {
        let present = Set(ranked.map { ComposerSuggestionGroup($0.kind) })
        guard present.count > 1 else { return [] }
        return ComposerSuggestionGroup.allCases.filter(present.contains)
    }

    /// `ranked` narrowed to one group, ranking kept. Nil ("All"), or a group
    /// with no rows (the query moved on), returns the flat list unchanged,
    /// so a stale chip never empties the menu.
    static func filtered(
        _ ranked: [ComposerSuggestion], to group: ComposerSuggestionGroup?
    ) -> [ComposerSuggestion] {
        guard let group else { return ranked }
        let rows = ranked.filter { ComposerSuggestionGroup($0.kind) == group }
        return rows.isEmpty ? ranked : rows
    }
}
