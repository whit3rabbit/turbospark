import XCTest

@testable import TurboSparkApp

/// The `/` menu's kind filter chips (All / Commands / Skills). "All" is the
/// flat-ranked menu unchanged; a chip narrows it without reordering, and
/// arrows plus accept keep reading one flat index.
final class ComposerSlashFilterTests: XCTestCase {

    private func row(_ kind: ComposerSuggestion.Kind, _ token: String) -> ComposerSuggestion {
        ComposerSuggestion(kind: kind, replacementToken: token, title: token, subtitle: nil, iconName: "doc")
    }

    private func skill(_ name: String) -> AppSkill {
        AppSkill(
            manifest: SkillManifest(name: name, description: nil, userInvocable: true),
            content: "body",
            sourceURL: URL(fileURLWithPath: "/tmp/\(name).md"),
            isEnabled: true)
    }

    // MARK: - Pure

    func testChipsOnlyWhenTwoKindsArePresent() {
        XCTAssertEqual(ComposerSuggestionGrouping.filterGroups([row(.builtInCommand, "/a")]), [])
        XCTAssertEqual(ComposerSuggestionGrouping.filterGroups([]), [])
        XCTAssertEqual(
            ComposerSuggestionGrouping.filterGroups([
                row(.skill, "/s"), row(.builtInCommand, "/a"), row(.skill, "/t"),
            ]),
            [.commands, .skills])
    }

    func testFilterKeepsRankingAndNilIsAll() {
        let ranked = [row(.skill, "/s1"), row(.builtInCommand, "/c1"), row(.skill, "/s2")]
        XCTAssertEqual(ComposerSuggestionGrouping.filtered(ranked, to: nil), ranked)
        XCTAssertEqual(
            ComposerSuggestionGrouping.filtered(ranked, to: .skills).map(\.replacementToken),
            ["/s1", "/s2"])
        XCTAssertEqual(
            ComposerSuggestionGrouping.filtered(ranked, to: .commands).map(\.replacementToken),
            ["/c1"])
    }

    func testFilterForAnAbsentGroupFallsBackToAll() {
        let ranked = [row(.builtInCommand, "/c1"), row(.builtInCommand, "/c2")]
        XCTAssertEqual(ComposerSuggestionGrouping.filtered(ranked, to: .skills), ranked)
    }

    // MARK: - Controller

    @MainActor
    func testAllIsTheFlatRankedMenuAndChipsNarrowIt() throws {
        let skills = [skill("planner")]
        let controller = ComposerAutocompleteController()
        controller.textChanged(text: "/pl", skills: skills, projectRoot: nil)
        let all = ComposerAutocompleteEngine.slashCandidates(prefix: "pl", skills: skills)
        XCTAssertEqual(controller.suggestions, all)
        XCTAssertEqual(controller.slashFilterGroups, [.commands, .skills])
        XCTAssertNil(controller.slashFilter)

        controller.setSlashFilter(.skills)
        XCTAssertEqual(controller.suggestions.map(\.kind), [.skill])
        XCTAssertEqual(controller.selectedIndex, 0)
        XCTAssertEqual(controller.accept(in: "/pl"), "/planner ")

        controller.setSlashFilter(.commands)
        XCTAssertTrue(controller.suggestions.allSatisfy { $0.kind == .builtInCommand })
        controller.moveSelection(-1)
        XCTAssertEqual(controller.selectedIndex, controller.suggestions.count - 1)

        controller.setSlashFilter(nil)
        XCTAssertEqual(controller.suggestions, all)
    }

    @MainActor
    func testFilterResetsWhenThePopupCloses() {
        let skills = [skill("planner")]
        let controller = ComposerAutocompleteController()
        controller.textChanged(text: "/pl", skills: skills, projectRoot: nil)
        controller.setSlashFilter(.skills)
        controller.textChanged(text: "hello", skills: skills, projectRoot: nil)
        XCTAssertNil(controller.slashFilter)
        XCTAssertEqual(controller.slashFilterGroups, [])
    }

    @MainActor
    func testAMentionMenuShowsNoSlashChips() {
        let controller = ComposerAutocompleteController()
        controller.textChanged(text: "@", skills: [], projectRoot: nil)
        XCTAssertEqual(controller.slashFilterGroups, [])
    }
}
