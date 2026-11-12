import Foundation
import TurboSpark

/// Filtering and sorting for the Model Hub list.
///
/// Pure and free of SwiftUI so it can be unit tested: the previous version
/// lived inline in the view and could only be checked by looking at the app.
///
/// EVERY OPTION HERE IS DERIVED FROM DATA THE CARD ALSO DISPLAYS. The version
/// this replaces matched alias substrings, which drifts silently and was
/// already wrong: `.conversational` fell through to `break` and returned the
/// whole catalog, `museGlimmer` was listed under both "Reasoning" and "Dense",
/// and `gemma` was listed under "Coding" and "MoE" by hand. Format and
/// capability now come from `ModelFamilyVisuals`, the same table the badges
/// read, and fit comes from `ModelRecommendation.verdict`, which is measured.
struct ModelHubFilter: Equatable {
    /// Which half of the catalog to show.
    enum Tab: String, CaseIterable, Identifiable {
        case recommended = "Recommended"
        case discover = "Discover"
        case audio = "Audio"
        case onDevice = "On Device"
        var id: String { rawValue }
    }

    /// Ordering of the resulting list.
    enum SortOption: String, CaseIterable, Identifiable {
        case recommended = "Best fit"
        case name = "Name"
        case size = "Size"
        var id: String { rawValue }
    }

    /// A dropdown value that is either "no filter" or one concrete choice.
    ///
    /// Modelled as a string rather than an enum because the concrete choices
    /// are read off the catalog at runtime; a fixed enum is exactly what went
    /// stale before.
    typealias Selection = String?

    var tab: Tab = .recommended
    var searchText: String = ""
    var format: Selection = nil
    var capability: Selection = nil
    var fit: Selection = nil
    var sort: SortOption = .recommended

    /// Whether anything other than the tab and sort is narrowing the list.
    var isNarrowed: Bool {
        !searchText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            || format != nil
            || capability != nil
            || fit != nil
    }

    /// Clears every narrowing filter, leaving the tab and sort alone.
    mutating func clearNarrowing() {
        searchText = ""
        format = nil
        capability = nil
        fit = nil
    }

    // MARK: - Option lists

    /// Format labels present in the catalog, in display order.
    static func formatOptions(for entries: [CatalogEntry]) -> [String] {
        sortedUnique(entries.map { visuals(for: $0).formatLabel })
    }

    /// Capability tags present in the catalog, in display order.
    static func capabilityOptions(for entries: [CatalogEntry]) -> [String] {
        sortedUnique(entries.flatMap { visuals(for: $0).capabilities })
    }

    /// Fit labels present, given whatever recommendations have loaded.
    ///
    /// Driven by what is actually in the map rather than by the full verdict
    /// enum: offering "Too large" on a machine where no row is refused is a
    /// filter that can only ever return nothing.
    static func fitOptions(
        for entries: [CatalogEntry],
        recommendations: [String: ModelRecommendation]
    ) -> [String] {
        let labels = entries.compactMap { recommendations[$0.alias].map { fitLabel($0.verdict) } }
        let present = Set(labels)
        return orderedFitLabels.filter { present.contains($0) }
    }

    /// Display label for a fit verdict, as this filter's menu groups by.
    /// The mapping lives in `ModelFitPresentation`; this is a thin forward so
    /// the badge and the filter cannot disagree about what a verdict is
    /// called (swift Gotcha 22).
    static func fitLabel(_ verdict: ModelRecommendation.FitVerdict) -> String {
        ModelFitPresentation.filterLabel(verdict)
    }

    private static let orderedFitLabels = [
        "Fits in memory", "Streams", "Tight fit", "Unknown", "Too large",
    ]

    private static func sortedUnique(_ values: [String]) -> [String] {
        Array(Set(values)).sorted()
    }

    private static func visuals(for entry: CatalogEntry) -> ModelFamilyVisuals {
        ModelFamilyVisuals.resolve(alias: entry.alias, family: entry.family, name: entry.name)
    }

    // MARK: - Application

    /// Applies the tab, search, filters and sort to a catalog.
    func apply(
        to catalog: [CatalogEntry],
        installedAliases: Set<String>,
        recommendations: [String: ModelRecommendation]
    ) -> [CatalogEntry] {
        // Audio profiles have their own catalog and installer. They are
        // rendered by AudioDiscoverCatalogView, never treated as text rows.
        guard tab != .audio else { return [] }
        var list = catalog

        if tab == .onDevice {
            list = list.filter { installedAliases.contains($0.alias) || $0.installed }
        } else if tab == .recommended {
            list = list.filter { entry in
                guard let rec = recommendations[entry.alias] else { return false }
                return rec.runs && rec.verdict != .refused
            }
        }

        let query = searchText.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if !query.isEmpty {
            list = list.filter { entry in
                entry.alias.lowercased().contains(query)
                    || entry.name.lowercased().contains(query)
                    || entry.family.lowercased().contains(query)
                    || (entry.notes?.lowercased().contains(query) ?? false)
            }
        }

        if let format {
            list = list.filter { Self.visuals(for: $0).formatLabel == format }
        }

        if let capability {
            list = list.filter { Self.visuals(for: $0).capabilities.contains(capability) }
        }

        if let fit {
            list = list.filter { entry in
                guard let verdict = recommendations[entry.alias]?.verdict else { return false }
                return Self.fitLabel(verdict) == fit
            }
        }

        switch sort {
        case .recommended:
            list.sort { lhs, rhs in
                let lhsRec = recommendations[lhs.alias]
                let rhsRec = recommendations[rhs.alias]

                if tab == .recommended {
                    return Self.recommendedPrecedes(
                        lhs, rhs, lhsVerdict: lhsRec?.verdict, rhsVerdict: rhsRec?.verdict)
                }

                let lhsRank = Self.verdictRank(lhsRec?.verdict)
                let rhsRank = Self.verdictRank(rhsRec?.verdict)
                if lhsRank != rhsRank { return lhsRank < rhsRank }
                return lhs.alias < rhs.alias
            }
        case .name:
            list.sort { $0.alias.localizedCaseInsensitiveCompare($1.alias) == .orderedAscending }
        case .size:
            list.sort { $0.installBytes < $1.installBytes }
        }

        return list
    }

    // MARK: - Recommended tab order

    /// The order inside the Recommended tab, which is the order its sections
    /// are drawn in.
    ///
    /// **FIT FIRST, BECAUSE THAT IS WHAT THE SECTION HEADERS SAY.** The order
    /// this replaces ranked five hidden tiers (optimal MLX, optimal MoE,
    /// optimal dense, ...) that no row or header ever named, so a streaming MoE
    /// could sit above a model that fits entirely in memory for no reason a
    /// reader could see. Now a row's section is its fit verdict, and inside a
    /// section the keys are ones the card shows: how much evidence there is
    /// (verified before runs before caveat), then MLX before GGUF (the format
    /// native to this hardware), then name so a family stays together.
    static func recommendedPrecedes(
        _ lhs: CatalogEntry,
        _ rhs: CatalogEntry,
        lhsVerdict: ModelRecommendation.FitVerdict?,
        rhsVerdict: ModelRecommendation.FitVerdict?
    ) -> Bool {
        let lhsFit = verdictRank(lhsVerdict)
        let rhsFit = verdictRank(rhsVerdict)
        if lhsFit != rhsFit { return lhsFit < rhsFit }
        let lhsStatus = statusRank(lhs.status)
        let rhsStatus = statusRank(rhs.status)
        if lhsStatus != rhsStatus { return lhsStatus < rhsStatus }
        let lhsMlx = isMlx(lhs)
        let rhsMlx = isMlx(rhs)
        if lhsMlx != rhsMlx { return lhsMlx }
        return lhs.alias < rhs.alias
    }

    /// How much a catalog row has been taken on real hardware, best first.
    static func statusRank(_ status: String) -> Int {
        switch status {
        case "verified": return 0
        case "runs": return 1
        case "caveat": return 2
        default: return 3
        }
    }

    private static func isMlx(_ entry: CatalogEntry) -> Bool {
        visuals(for: entry).formatLabel.localizedCaseInsensitiveContains("mlx")
    }

    /// One visible group of the Recommended tab.
    struct RecommendedSection: Equatable, Identifiable {
        let verdict: ModelRecommendation.FitVerdict
        let entries: [CatalogEntry]
        var id: String { "\(verdict)" }
    }

    /// Splits an ALREADY-SORTED Recommended list into its fit sections,
    /// keeping each section's order. Empty sections are not drawn, so a Mac
    /// where nothing is tight shows no "Tight" header over nothing.
    static func recommendedSections(
        _ sorted: [CatalogEntry],
        recommendations: [String: ModelRecommendation]
    ) -> [RecommendedSection] {
        var sections: [RecommendedSection] = []
        for entry in sorted {
            // A row with no assessment cannot be on this tab, but a caller
            // that passes one gets it under "unknown" rather than a crash.
            let verdict = recommendations[entry.alias]?.verdict ?? .unknown
            if let last = sections.last, last.verdict == verdict {
                sections[sections.count - 1] = RecommendedSection(
                    verdict: verdict, entries: last.entries + [entry])
            } else {
                sections.append(RecommendedSection(verdict: verdict, entries: [entry]))
            }
        }
        return sections
    }

    /// Identifies models that use Apple Silicon native MLX formats or MoE architectures.
    static func isMlxOrMoe(
        entry: CatalogEntry,
        recommendation: ModelRecommendation? = nil
    ) -> Bool {
        let visuals = visuals(for: entry)
        return visuals.formatLabel.localizedCaseInsensitiveContains("mlx")
            || visuals.capabilities.contains("MoE")
            || recommendation?.verdict == .streams
            || entry.name.localizedCaseInsensitiveContains("moe")
            || entry.family.localizedCaseInsensitiveContains("moe")
    }

    /// Sort order for the "Best fit" option: what runs best comes first.
    static func verdictRank(_ verdict: ModelRecommendation.FitVerdict?) -> Int {
        switch verdict {
        case .resident: return 0
        case .streams: return 1
        case .tight: return 2
        case .unknown, .none: return 3
        case .refused: return 4
        }
    }
}
