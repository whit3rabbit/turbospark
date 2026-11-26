import XCTest

@testable import TurboSparkApp

final class SidebarLayoutTests: XCTestCase {
    typealias Section = AppModel.AppNavigationSection

    private func flat(_ groups: [[Section]]) -> [Section] { groups.flatMap { $0 } }

    func testDefaultGroupsPutTheMostUsedFirst() {
        let groups = SidebarLayout.default.groups(includesChat: true)
        XCTAssertEqual(
            groups,
            [[.chat, .humanize, .server, .modelManager], [.images, .audio], [.files]])
    }

    func testChatIsOmittedWhenTheSwitcherOwnsIt() {
        let groups = SidebarLayout.default.groups(includesChat: false)
        XCTAssertEqual(groups.first, [.humanize, .server, .modelManager])
    }

    func testDiscoverIsNeverARow() {
        XCTAssertFalse(flat(SidebarLayout.default.groups(includesChat: true)).contains(.modelHub))
        XCTAssertEqual(Section.modelHub.navigationRow, .modelManager)
    }

    func testPinMovesASectionToTheTopGroupAfterChat() {
        var layout = SidebarLayout.default
        layout.pin(.audio)
        let groups = layout.groups(includesChat: true)
        XCTAssertEqual(groups[0], [.chat, .audio, .humanize, .server, .modelManager])
        XCTAssertEqual(groups[1], [.images])
        layout.unpin(.audio)
        XCTAssertEqual(layout, .default)
    }

    func testTopGroupSectionsCannotBePinned() {
        var layout = SidebarLayout.default
        XCTAssertFalse(layout.canPin(.humanize))
        layout.pin(.humanize)
        XCTAssertTrue(layout.pinned.isEmpty)
    }

    func testChatAndModelsCannotBeHidden() {
        var layout = SidebarLayout.default
        layout.hide(.chat)
        layout.hide(.modelManager)
        XCTAssertTrue(layout.hidden.isEmpty)
        XCTAssertFalse(SidebarLayout.isHideable(.chat))
        XCTAssertFalse(SidebarLayout.isHideable(.modelManager))
    }

    func testHideAndShowRoundTrip() {
        var layout = SidebarLayout.default
        layout.hide(.files)
        XCTAssertFalse(flat(layout.groups(includesChat: true)).contains(.files))
        XCTAssertEqual(layout.hiddenSections, [.files])
        layout.show(.files)
        XCTAssertEqual(layout, .default)
    }

    func testTheSectionBeingShownStaysListedEvenIfHidden() {
        var layout = SidebarLayout.default
        layout.hide(.images)
        XCTAssertFalse(flat(layout.groups(includesChat: true)).contains(.images))
        XCTAssertTrue(flat(layout.groups(includesChat: true, keeping: .images)).contains(.images))
    }

    func testHidingAPinnedSectionUnpinsIt() {
        var layout = SidebarLayout.default
        layout.pin(.files)
        layout.hide(.files)
        XCTAssertFalse(layout.isPinned(.files))
    }

    func testMoveUpAndDownSwapNeighboursInAGroup() {
        var layout = SidebarLayout.default
        XCTAssertTrue(layout.canMove(.audio, by: -1))
        XCTAssertFalse(layout.canMove(.audio, by: 1))
        layout.move(.audio, by: -1)
        XCTAssertEqual(layout.groups(includesChat: true)[1], [.audio, .images])
        layout.move(.audio, by: 1)
        XCTAssertEqual(layout.groups(includesChat: true)[1], [.images, .audio])
    }

    func testChatNeverMovesAndOtherRowsStayBelowIt() {
        var layout = SidebarLayout.default
        XCTAssertFalse(layout.canMove(.chat, by: 1))
        XCTAssertFalse(layout.canMove(.humanize, by: -1), "Chat holds the first slot")
        layout.move(.modelManager, by: -1)
        layout.move(.modelManager, by: -1)
        XCTAssertEqual(layout.groups(includesChat: true)[0], [.chat, .modelManager, .humanize, .server])
    }

    func testPinsReorderAmongPinsOnly() {
        var layout = SidebarLayout.default
        layout.pin(.files)
        layout.pin(.audio)
        XCTAssertEqual(layout.groups(includesChat: true)[0].prefix(3), [.chat, .files, .audio])
        layout.move(.audio, by: -1)
        XCTAssertEqual(layout.groups(includesChat: true)[0].prefix(3), [.chat, .audio, .files])
        XCTAssertFalse(layout.canMove(.audio, by: -1), "the first pin cannot pass into other rows")
    }

    func testDropBeforeReordersWithinARegion() {
        var layout = SidebarLayout.default
        layout.move(.audio, before: .images)
        XCTAssertEqual(layout.groups(includesChat: true)[1], [.audio, .images])
        layout.move(.audio, before: .files)
        XCTAssertEqual(
            layout.groups(includesChat: true)[1], [.audio, .images],
            "a drop onto another group is ignored")
    }

    func testResetRestoresTheDefault() {
        var layout = SidebarLayout.default
        layout.pin(.audio)
        layout.hide(.files)
        layout.move(.server, by: -1)
        layout.reset()
        XCTAssertTrue(layout.isDefault)
    }

    @MainActor
    func testStorePersistsAndFallsBackOnCorruptData() throws {
        let suite = "SidebarLayoutTests.\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }

        let store = SidebarLayoutStore(defaults: defaults)
        XCTAssertTrue(store.layout.isDefault)
        store.update { $0.pin(.audio); $0.hide(.files) }

        let reloaded = SidebarLayoutStore(defaults: defaults)
        XCTAssertEqual(reloaded.layout.pinned, [Section.audio.rawValue])
        XCTAssertEqual(reloaded.layout.hidden, [Section.files.rawValue])

        reloaded.update { $0.reset() }
        XCTAssertNil(defaults.data(forKey: SidebarLayoutStore.defaultsKey))

        defaults.set(Data("not json".utf8), forKey: SidebarLayoutStore.defaultsKey)
        XCTAssertTrue(SidebarLayoutStore(defaults: defaults).layout.isDefault)
    }
}
