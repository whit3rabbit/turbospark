import XCTest

@testable import TurboSparkApp

/// The `@` menu groups its rows by kind. These pin the pure grouping: stable
/// kind order, ranking kept inside each group, empty groups dropped, and a
/// flattened index that keyboard navigation and accept both read.
final class ComposerSuggestionGroupingTests: XCTestCase {

    private func row(_ kind: ComposerSuggestion.Kind, _ token: String) -> ComposerSuggestion {
        ComposerSuggestion(kind: kind, replacementToken: token, title: token, subtitle: nil, iconName: "doc")
    }

    func testGroupsFollowTheFixedKindOrderAndKeepRankingWithinEachGroup() {
        // Ranked input interleaves kinds the way mentionCandidates does.
        let ranked = [
            row(.reference, "@diff"),
            row(.file, "@src/b.swift"),
            row(.folder, "@src"),
            row(.file, "@src/a.swift"),
            row(.reference, "@staged"),
            row(.folder, "@docs"),
        ]
        let sections = ComposerSuggestionGrouping.sections(ranked)
        XCTAssertEqual(sections.map(\.group), [.references, .files, .folders])
        XCTAssertEqual(sections[0].rows.map(\.replacementToken), ["@diff", "@staged"])
        XCTAssertEqual(sections[1].rows.map(\.replacementToken), ["@src/b.swift", "@src/a.swift"])
        XCTAssertEqual(sections[2].rows.map(\.replacementToken), ["@src", "@docs"])
    }

    func testEmptyGroupsAreDropped() {
        let sections = ComposerSuggestionGrouping.sections([
            row(.folder, "@src"), row(.file, "@a.swift"),
        ])
        XCTAssertEqual(sections.map(\.group), [.files, .folders])
        XCTAssertTrue(ComposerSuggestionGrouping.sections([]).isEmpty)
    }

    func testStartIndexIsTheFlattenedOffsetOfEachGroup() {
        let ranked = [
            row(.file, "@a"), row(.reference, "@diff"), row(.folder, "@d"), row(.file, "@b"),
        ]
        let sections = ComposerSuggestionGrouping.sections(ranked)
        XCTAssertEqual(sections.map(\.startIndex), [0, 1, 3])
        let flat = ComposerSuggestionGrouping.flattened(ranked)
        XCTAssertEqual(flat.map(\.replacementToken), ["@diff", "@a", "@b", "@d"])
        // Every row's flattened index is its section start plus its offset,
        // which is the index the popup highlights and accept reads.
        for section in sections {
            for (offset, suggestion) in section.rows.enumerated() {
                XCTAssertEqual(flat[section.startIndex + offset], suggestion)
            }
        }
    }

    func testSlashKindsGroupAsCommandsThenSkills() {
        let sections = ComposerSuggestionGrouping.sections([
            row(.skill, "/my-skill"), row(.builtInCommand, "/plan"),
        ])
        XCTAssertEqual(sections.map(\.group), [.commands, .skills])
    }

    // MARK: - Controller wiring

    @MainActor
    func testMentionControllerOrdersRowsByGroupSoArrowsWalkTheMenuAsItReads() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("ts-group-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(
            at: root.appendingPathComponent("fdir"), withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try "x".write(to: root.appendingPathComponent("fdir/f.swift"), atomically: true, encoding: .utf8)
        try "x".write(to: root.appendingPathComponent("a.swift"), atomically: true, encoding: .utf8)

        let controller = ComposerAutocompleteController()
        // "@f" ranks the folder `fdir` (name prefix) ahead of `fdir/f.swift`
        // and also offers the `@file:`/`@folder:` reference rows.
        controller.textChanged(text: "@f", skills: [], projectRoot: root)
        for _ in 0..<200 where !controller.suggestions.contains(where: { $0.kind == .folder }) {
            try await Task.sleep(nanoseconds: 25_000_000)
        }
        let kinds = controller.suggestions.map(\.kind)
        XCTAssertEqual(
            kinds, kinds.sorted { ComposerSuggestionGroup($0).rawValue < ComposerSuggestionGroup($1).rawValue },
            "rows must be contiguous by group: \(controller.suggestions.map(\.replacementToken))")
        XCTAssertEqual(controller.sections.map(\.group), [.references, .files, .folders])
        XCTAssertTrue(controller.showsGroupHeaders)

        // Arrow navigation crosses group boundaries in reading order.
        let firstFile = controller.sections[1].startIndex
        controller.select(firstFile - 1)
        controller.moveSelection(1)
        XCTAssertEqual(controller.suggestions[controller.selectedIndex].kind, .file)
    }

    @MainActor
    func testSlashMenuKeepsItsFlatRankingAndShowsNoHeaders() {
        let controller = ComposerAutocompleteController()
        controller.textChanged(text: "/pl", skills: [], projectRoot: nil)
        XCTAssertTrue(controller.isVisible)
        XCTAssertFalse(controller.showsGroupHeaders)
        XCTAssertEqual(
            controller.suggestions,
            ComposerAutocompleteEngine.slashCandidates(prefix: "pl", skills: []))
    }
}
