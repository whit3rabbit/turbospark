import XCTest
@testable import TurboSparkApp

/// The Installed | Discover switcher mapping and the Filter button's count.
@MainActor
final class ModelWorkspaceControlsTests: XCTestCase {
    func testModeMapsBetweenSections() {
        XCTAssertEqual(ModelWorkspaceMode(section: .modelHub), .discover)
        XCTAssertEqual(ModelWorkspaceMode(section: .modelManager), .installed)
        XCTAssertEqual(ModelWorkspaceMode.discover.section, .modelHub)
        XCTAssertEqual(ModelWorkspaceMode.installed.section, .modelManager)
    }

    private func installedCount(
        favorites: Bool = false, available: Bool = false,
        arch: ModelManagerView.ArchitectureFilter = .all,
        drafter: ModelManagerView.DrafterFilter = .all,
        source: ModelManagerView.SourceFilter = .all,
        tag: String? = nil
    ) -> Int {
        ModelManagerView.activeFilterCount(
            showFavoritesOnly: favorites, availableOnly: available,
            architectureFilter: arch, drafterFilter: drafter,
            sourceFilter: source, selectedTag: tag)
    }

    func testInstalledCountIsZeroAtDefaults() {
        XCTAssertEqual(installedCount(), 0)
    }

    func testInstalledCountCountsEachNarrowingFilterOnce() {
        XCTAssertEqual(installedCount(favorites: true), 1)
        XCTAssertEqual(installedCount(favorites: true, available: true, arch: .moeOnly), 3)
        XCTAssertEqual(
            installedCount(favorites: true, available: true, arch: .denseOnly,
                           drafter: .mtp, source: .lmStudio, tag: "work"), 6)
    }

    func testHubCountIgnoresSearchTabAndSort() {
        var filter = ModelHubFilter()
        filter.searchText = "qwen"
        filter.tab = .discover
        filter.sort = .name
        XCTAssertEqual(filter.activeFilterCount, 0)
        filter.format = "MLX INT4"
        filter.fit = "Tight fit"
        XCTAssertEqual(filter.activeFilterCount, 2)
    }

    func testHubClearFiltersKeepsSearchTabAndSort() {
        var filter = ModelHubFilter()
        filter.searchText = "qwen"
        filter.tab = .discover
        filter.sort = .size
        filter.format = "GGUF"
        filter.capability = "Reasoning"
        filter.fit = "Streams"
        filter.clearFilters()
        XCTAssertEqual(filter.activeFilterCount, 0)
        XCTAssertEqual(filter.searchText, "qwen")
        XCTAssertEqual(filter.tab, .discover)
        XCTAssertEqual(filter.sort, .size)
    }
}
