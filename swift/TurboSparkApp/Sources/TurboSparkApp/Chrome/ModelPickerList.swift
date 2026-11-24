import Foundation
import TurboSpark

/// One row of the chat model picker: an installed model or an enabled remote
/// provider model, with the facts the row shows.
///
/// A value type so the picker's filter, ranking and highlight can be tested
/// without a view. Facts the app does not know are `nil`, never a guess.
struct ModelPickerRow: Equatable, Identifiable {
    enum Source: Equatable {
        case local(alias: String)
        case remote(providerID: UUID, modelID: String)
    }

    let source: Source
    /// The alias, or `Provider: model` for a remote row. Identifiers, never
    /// translated.
    let title: String
    /// The family for a local row. Matched by the filter as well as shown.
    let detail: String?
    /// The GGUF quantization label of a variant install.
    let quant: String?
    /// Size on disk. `nil` when the install never recorded one: a 0 there is
    /// a missing figure, and printing "Zero KB" would read as a measurement.
    let installBytes: UInt64?
    let isSelected: Bool

    var id: String {
        switch source {
        case .local(let alias): return "local:\(alias)"
        case .remote(let provider, let model): return "remote:\(provider.uuidString):\(model)"
        }
    }

    /// The alias the catalog's fit rows are keyed by. Remote rows have no
    /// memory fit on this Mac.
    var fitAlias: String? {
        if case .local(let alias) = source { return alias }
        return nil
    }

    static func local(_ model: InstalledModel, selectedAlias: String?) -> ModelPickerRow {
        ModelPickerRow(
            source: .local(alias: model.alias),
            title: model.alias,
            detail: model.family.isEmpty ? nil : model.family,
            quant: model.variant.flatMap { $0.isEmpty ? nil : $0 },
            installBytes: model.installBytes > 0 ? model.installBytes : nil,
            isSelected: model.alias == selectedAlias)
    }

    // Same "Provider: model" spelling the composer's Remote pill uses.
    static func remote(
        providerID: UUID, providerName: String, modelID: String, displayName: String,
        selected: RemoteModelSelection?
    ) -> ModelPickerRow {
        ModelPickerRow(
            source: .remote(providerID: providerID, modelID: modelID),
            title: "\(providerName): \(displayName)",
            detail: nil,
            quant: nil,
            installBytes: nil,
            isSelected: selected?.providerID == providerID && selected?.modelID == modelID)
    }
}

/// A titled group of picker rows. Local comes before Remote: local is every
/// chat's default, and remote use is a deliberate choice.
struct ModelPickerSection: Equatable, Identifiable {
    enum Group: Equatable {
        case local
        case remote
    }

    let group: Group
    let rows: [ModelPickerRow]

    var id: Group { group }
}

enum ModelPickerList {
    /// Filters and ranks both groups by `query`, dropping a group with no rows.
    ///
    /// Every whitespace-separated word must appear in the row's title, family
    /// or quant (case and diacritic insensitive). With a query, rows rank
    /// alias prefix first, then the words inside the title, then a match on
    /// another field only; ties keep their source order, so an empty query
    /// shows the lists exactly as the app orders them.
    static func sections(
        local: [ModelPickerRow], remote: [ModelPickerRow], query: String
    ) -> [ModelPickerSection] {
        let words = query
            .split(whereSeparator: \.isWhitespace)
            .map { fold(String($0)) }
        let groups: [(ModelPickerSection.Group, [ModelPickerRow])] = [(.local, local), (.remote, remote)]
        return groups.compactMap { group, rows in
            let kept = filtered(rows, words: words)
            return kept.isEmpty ? nil : ModelPickerSection(group: group, rows: kept)
        }
    }

    /// The rows in display order, the space the keyboard highlight moves in.
    static func flattened(_ sections: [ModelPickerSection]) -> [ModelPickerRow] {
        sections.flatMap(\.rows)
    }

    /// Where the highlight starts: on the current model, so Return on open
    /// keeps the choice, else on the first row.
    static func initialHighlight(_ rows: [ModelPickerRow]) -> Int? {
        guard !rows.isEmpty else { return nil }
        return rows.firstIndex(where: \.isSelected) ?? 0
    }

    /// Moves the highlight by `delta`, clamped to the rows. Clamped rather
    /// than wrapping, like a native list: holding Down stops at the end.
    static func moveHighlight(from index: Int?, by delta: Int, count: Int) -> Int? {
        guard count > 0 else { return nil }
        guard let index else { return 0 }
        return min(max(min(index, count - 1) + delta, 0), count - 1)
    }

    // MARK: - Private

    private static func filtered(_ rows: [ModelPickerRow], words: [String]) -> [ModelPickerRow] {
        guard !words.isEmpty else { return rows }
        let ranked: [(rank: Int, offset: Int, row: ModelPickerRow)] = rows.enumerated().compactMap { offset, row in
            let title = fold(row.title)
            let haystack = [title, row.detail.map(fold), row.quant.map(fold)].compactMap { $0 }
            guard words.allSatisfy({ word in haystack.contains { $0.contains(word) } }) else { return nil }
            let rank: Int
            if title.hasPrefix(words[0]) {
                rank = 0
            } else if words.allSatisfy({ title.contains($0) }) {
                rank = 1
            } else {
                rank = 2
            }
            return (rank, offset, row)
        }
        return ranked
            .sorted { ($0.rank, $0.offset) < ($1.rank, $1.offset) }
            .map(\.row)
    }

    private static func fold(_ text: String) -> String {
        text.folding(options: [.caseInsensitive, .diacriticInsensitive], locale: nil)
    }
}
