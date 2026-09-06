import XCTest
import TurboSpark
@testable import TurboSparkApp

/// Covers the Model Hub's filtering and sorting.
///
/// The rows are built by DECODING the catalog's own snake_case JSON rather
/// than by a memberwise init, for two reasons: `CatalogEntry`'s memberwise
/// init is internal to the binding and unreachable from here, and the wire
/// shape is the thing worth exercising (`swift/CLAUDE.md` Gotcha 5).
final class ModelHubFilterTests: XCTestCase {
    /// Decodes a fixture catalog entry from JSON with specified fields.
    private func entry(
        alias: String,
        name: String,
        family: String,
        status: String = "verified",
        downloadBytes: UInt64 = 1_000,
        notes: String? = nil,
        installed: Bool = false
    ) throws -> CatalogEntry {
        let notesField = notes.map { "\"notes\": \"\($0)\"," } ?? "\"notes\": null,"
        let json = """
        {
          "alias": "\(alias)",
          "name": "\(name)",
          "family": "\(family)",
          "download_bytes": \(downloadBytes),
          "install_bytes": \(downloadBytes),
          "status": "\(status)",
          \(notesField)
          "installed": \(installed)
        }
        """
        return try JSONDecoder().decode(CatalogEntry.self, from: Data(json.utf8))
    }

    /// Decodes a fixture model recommendation from JSON with alias and verdict.
    private func recommendation(
        alias: String,
        verdict: String
    ) throws -> ModelRecommendation {
        let json = """
        {
          "alias": "\(alias)",
          "name": "\(alias)",
          "family": null,
          "verdict": "\(verdict)",
          "verdictSummary": "",
          "runs": true,
          "countedBytes": 0,
          "countedSource": "estimated",
          "installBytes": 0,
          "slotCacheSlots": 16,
          "largestContext": 4096,
          "notes": [],
          "toksPerSecondMin": null,
          "toksPerSecondMax": null
        }
        """
        return try JSONDecoder().decode(ModelRecommendation.self, from: Data(json.utf8))
    }

    /// Returns fixture catalog entries covering multiple families and formats.
    private func catalog() throws -> [CatalogEntry] {
        [
            try entry(alias: "gemma4", name: "Gemma 4 26B-A4B (MLX INT4, group 64)", family: "gemma4", downloadBytes: 13_000),
            try entry(alias: "gemma4-gguf", name: "Gemma 4 26B-A4B (GGUF Q4_K_M)", family: "gemma4", downloadBytes: 25_000),
            try entry(alias: "mistral7b", name: "Mistral 7B (GGUF Q4_K_M)", family: "llama", status: "runs", downloadBytes: 4_000),
        ]
    }

    // MARK: - Options come from the data

    /// Tests that format filter options are dynamically derived from catalog rows.
    func testFormatOptionsAreDerivedFromTheCatalogRatherThanHardcoded() throws {
        let options = ModelHubFilter.formatOptions(for: try catalog())
        XCTAssertTrue(options.contains("MLX INT4"), "expected an MLX row, got \(options)")
        XCTAssertTrue(options.contains(where: { $0.hasPrefix("GGUF") }), "expected a GGUF row, got \(options)")
        XCTAssertEqual(options, options.sorted(), "options should be display-ordered")
    }

    /// Tests that capability options contain no duplicate entries.
    func testCapabilityOptionsAreDeduplicatedAcrossRows() throws {
        let options = ModelHubFilter.capabilityOptions(for: try catalog())
        XCTAssertEqual(Set(options).count, options.count, "duplicates leaked: \(options)")
        XCTAssertFalse(options.isEmpty)
    }

    /// The previous version offered every verdict unconditionally, so on a
    /// machine where nothing is refused "Too large" was a filter that could
    /// only ever return an empty list.
    func testFitOptionsOnlyOfferVerdictsThatArePresent() throws {
        let rows = try catalog()
        let recommendations = [
            "gemma4": try recommendation(alias: "gemma4", verdict: "streams"),
            "mistral7b": try recommendation(alias: "mistral7b", verdict: "resident"),
        ]
        let options = ModelHubFilter.fitOptions(for: rows, recommendations: recommendations)
        XCTAssertEqual(options, ["Fits in memory", "Streams"])
        XCTAssertFalse(options.contains("Too large"))
    }

    /// Tests that fit filter options are empty before recommendations are loaded.
    func testFitOptionsAreEmptyBeforeRecommendationsLoad() throws {
        XCTAssertEqual(ModelHubFilter.fitOptions(for: try catalog(), recommendations: [:]), [])
    }

    // MARK: - Application

    /// Tests that an unconstrained filter preserves all catalog rows.
    func testUnfilteredListKeepsEveryRow() throws {
        var filter = ModelHubFilter()
        filter.sort = .name
        let result = filter.apply(to: try catalog(), installedAliases: [], recommendations: [:])
        XCTAssertEqual(result.map(\.alias), ["gemma4", "gemma4-gguf", "mistral7b"])
    }

    /// Tests that the On-Device tab limits visible models to installed aliases.
    func testOnDeviceTabKeepsOnlyInstalledRows() throws {
        var filter = ModelHubFilter()
        filter.tab = .onDevice
        let result = filter.apply(
            to: try catalog(),
            installedAliases: ["mistral7b"],
            recommendations: [:])
        XCTAssertEqual(result.map(\.alias), ["mistral7b"])
    }

    /// Tests that search queries match against alias, name, family, and notes.
    func testSearchMatchesAliasNameFamilyAndNotes() throws {
        let rows = [try entry(alias: "a1", name: "Alpha", family: "gemma4", notes: "streams experts")]
        for query in ["a1", "alpha", "GEMMA4", "experts"] {
            var filter = ModelHubFilter()
            filter.searchText = query
            XCTAssertEqual(
                filter.apply(to: rows, installedAliases: [], recommendations: [:]).count,
                1,
                "query \(query) should match")
        }
    }

    /// Tests that search queries containing only whitespace do not filter rows.
    func testSearchIgnoresSurroundingWhitespace() throws {
        var filter = ModelHubFilter()
        filter.searchText = "   "
        XCTAssertEqual(
            filter.apply(to: try catalog(), installedAliases: [], recommendations: [:]).count,
            3,
            "a whitespace-only query must not filter anything out")
    }

    /// The bug this replaces: `.conversational` fell through to `break` and
    /// returned the whole catalog, so a selected filter looked applied and did
    /// nothing. Every option must now actually narrow or genuinely match all.
    func testEveryCapabilityOptionSelectsASubsetThatReallyHasIt() throws {
        let rows = try catalog()
        for option in ModelHubFilter.capabilityOptions(for: rows) {
            var filter = ModelHubFilter()
            filter.capability = option
            let result = filter.apply(to: rows, installedAliases: [], recommendations: [:])
            XCTAssertFalse(result.isEmpty, "\(option) matched nothing")
            for row in result {
                let visuals = ModelFamilyVisuals.resolve(
                    alias: row.alias, family: row.family, name: row.name)
                XCTAssertTrue(
                    visuals.capabilities.contains(option),
                    "\(row.alias) was kept by \(option) without carrying it")
            }
        }
    }

    /// Tests that format filters keep only entries matching the specified format.
    func testFormatFilterKeepsOnlyThatFormat() throws {
        var filter = ModelHubFilter()
        filter.format = "MLX INT4"
        let result = filter.apply(to: try catalog(), installedAliases: [], recommendations: [:])
        XCTAssertEqual(result.map(\.alias), ["gemma4"])
    }

    /// Tests that fit filtering drops rows that lack recommendation entries.
    func testFitFilterDropsRowsWithNoRecommendation() throws {
        var filter = ModelHubFilter()
        filter.fit = "Streams"
        let result = filter.apply(
            to: try catalog(),
            installedAliases: [],
            recommendations: ["gemma4": try recommendation(alias: "gemma4", verdict: "streams")])
        XCTAssertEqual(result.map(\.alias), ["gemma4"])
    }

    /// Tests that format and search filters compose conjunctively.
    func testFiltersCompose() throws {
        var filter = ModelHubFilter()
        filter.format = "MLX INT4"
        filter.searchText = "mistral"
        let result = filter.apply(to: try catalog(), installedAliases: [], recommendations: [:])
        XCTAssertTrue(result.isEmpty, "an MLX-only Mistral row does not exist in this fixture")
    }

    // MARK: - Sorting

    /// Tests that sorting by size orders rows ascending by download size.
    func testSizeSortIsAscendingByDownloadBytes() throws {
        var filter = ModelHubFilter()
        filter.sort = .size
        let result = filter.apply(to: try catalog(), installedAliases: [], recommendations: [:])
        XCTAssertEqual(result.map(\.alias), ["mistral7b", "gemma4", "gemma4-gguf"])
    }

    /// Tests that best-fit sorting orders resident before streams before refused.
    func testBestFitSortPutsResidentBeforeStreamsBeforeRefused() throws {
        var filter = ModelHubFilter()
        filter.sort = .recommended
        let result = filter.apply(
            to: try catalog(),
            installedAliases: [],
            recommendations: [
                "gemma4": try recommendation(alias: "gemma4", verdict: "refused"),
                "gemma4-gguf": try recommendation(alias: "gemma4-gguf", verdict: "streams"),
                "mistral7b": try recommendation(alias: "mistral7b", verdict: "resident"),
            ])
        XCTAssertEqual(result.map(\.alias), ["mistral7b", "gemma4-gguf", "gemma4"])
    }

    /// Tests that best-fit sorting breaks verdict ties by sorting on alias.
    func testBestFitSortIsStableOnAliasWhenVerdictsTie() throws {
        var filter = ModelHubFilter()
        filter.sort = .recommended
        let result = filter.apply(to: try catalog(), installedAliases: [], recommendations: [:])
        XCTAssertEqual(result.map(\.alias), ["gemma4", "gemma4-gguf", "mistral7b"])
    }

    /// Tests that unprobed models rank ahead of refused models in recommendations.
    func testUnknownRanksAheadOfRefusedSoAnUnprobedRowIsNotBuriedLast() {
        XCTAssertLessThan(
            ModelHubFilter.verdictRank(nil),
            ModelHubFilter.verdictRank(.refused))
    }

    // MARK: - Format labels

    /// The name shapes actually shipped in `crates/catalog/src/models.json`.
    ///
    /// Pinned as a table because the label used to be hand-written per family
    /// branch and was wrong for most of these: a GGUF Mistral and a GGUF
    /// TinyLlama both rendered "MLX INT4", and a Q8_0 Gemma rendered "Q4_K_M".
    /// A family and a quantization are independent, so nothing about the
    /// family may decide this.
    func testFormatLabelIsReadOffTheCatalogName() {
        let cases: [(alias: String, name: String, expected: String)] = [
            ("gemma4", "Gemma 4 26B-A4B Instruct (MLX INT4, group 64)", "MLX INT4"),
            ("qwen3moe", "Qwen3-30B-A3B (GGUF Q4_K_M)", "GGUF Q4_K_M"),
            ("gptoss-20b", "gpt-oss 20B (GGUF MXFP4)", "GGUF MXFP4"),
            ("ternary27b", "Ternary-Bonsai 27B (MLX affine 2-bit, group 128)", "MLX affine 2-bit"),
            ("bonsai27b", "Bonsai 27B (MLX affine 1-bit, group 128)", "MLX affine 1-bit"),
            ("gemma4-gguf", "Gemma 4 26B-A4B Instruct (GGUF Q8_0)", "GGUF Q8_0"),
            ("gemma4-iq3", "Gemma 4 26B-A4B Instruct (GGUF UD-Q3_K_M, IQ3_XXS/IQ4_NL experts)", "GGUF UD-Q3_K_M"),
            ("mistral7b", "Mistral 7B Instruct v0.3 (GGUF Q4_K_M, dense)", "GGUF Q4_K_M"),
            ("tinyllama", "TinyLlama 1.1B Chat v1.0 (GGUF Q6_K, dense)", "GGUF Q6_K"),
            ("ornith9b", "Ornith-1.5 9B (GGUF Q8_0)", "GGUF Q8_0"),
        ]
        for row in cases {
            XCTAssertEqual(
                ModelFamilyVisuals.formatLabel(alias: row.alias, name: row.name),
                row.expected,
                "\(row.alias)")
        }
    }

    /// The badge and the filter must agree, or selecting a format hides rows
    /// that display it.
    func testResolvedVisualsUseTheDerivedFormatLabel() {
        let name = "Mistral 7B Instruct v0.3 (GGUF Q4_K_M, dense)"
        let visuals = ModelFamilyVisuals.resolve(alias: "mistral7b", family: "llama", name: name)
        XCTAssertEqual(visuals.formatLabel, "GGUF Q4_K_M")
        XCTAssertEqual(
            visuals.formatLabel,
            ModelFamilyVisuals.formatLabel(alias: "mistral7b", name: name))
    }

    /// Tests fallback format labels when model names lack parenthesized formats.
    func testFormatLabelFallsBackToCoarseSignalWithoutParentheses() {
        XCTAssertEqual(ModelFamilyVisuals.formatLabel(alias: "x-gguf", name: "Some Model"), "GGUF")
        XCTAssertEqual(ModelFamilyVisuals.formatLabel(alias: "x", name: "Some Model"), "MLX")
    }

    /// Tests that non-format parenthesized groups like previews are ignored.
    func testFormatLabelIgnoresParenthesesThatAreNotAFormat() {
        // The LAST parenthesised group is the format by catalog convention.
        XCTAssertEqual(
            ModelFamilyVisuals.formatLabel(
                alias: "q", name: "Qwen 3.6 (preview) 35B-A3B (MLX INT4, group 64)"),
            "MLX INT4")
    }

    // MARK: - Narrowing state

    /// Tests that active tab and sorting mode do not mark the filter as narrowed.
    func testIsNarrowedIgnoresTabAndSort() throws {
        var filter = ModelHubFilter()
        filter.tab = .onDevice
        filter.sort = .size
        XCTAssertFalse(filter.isNarrowed, "tab and sort are views, not filters")

        filter.capability = "MoE"
        XCTAssertTrue(filter.isNarrowed)
    }

    /// Tests that clearing narrowing resets filters while preserving tab and sort.
    func testClearNarrowingLeavesTabAndSortAlone() {
        var filter = ModelHubFilter()
        filter.tab = .onDevice
        filter.sort = .size
        filter.searchText = "q"
        filter.format = "MLX INT4"
        filter.capability = "MoE"
        filter.fit = "Streams"

        filter.clearNarrowing()

        XCTAssertFalse(filter.isNarrowed)
        XCTAssertEqual(filter.tab, .onDevice)
        XCTAssertEqual(filter.sort, .size)
    }
}
