import Foundation
import XCTest

@testable import TurboSparkApp

/// ZP-06 host: the page cap, eviction, teardown, opt-in and lifecycle.
/// Sessions here are real (so teardown really frees a web view) but nothing
/// waits on a page load: the host's rules do not depend on one.
@MainActor
final class PluginUiHostTests: XCTestCase {
    private var roots: [URL] = []
    private var clock = Date(timeIntervalSince1970: 10_000)

    override func tearDown() {
        for root in roots { try? FileManager.default.removeItem(at: root) }
        roots = []
        super.tearDown()
    }

    private func makePlugin(surfaces: Int = 1, name: String = "demo") throws -> LoadedPlugin {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("tshost-\(UUID().uuidString)", isDirectory: true)
        roots.append(root)
        try FileManager.default.createDirectory(at: root.appendingPathComponent("ui"), withIntermediateDirectories: true)
        var list: [PluginUiSurface] = []
        for i in 0..<surfaces {
            try "<!doctype html><title>p\(i)</title>".write(to: root.appendingPathComponent("ui/p\(i).html"), atomically: true, encoding: .utf8)
            list.append(PluginUiSurface(id: "s\(i)", title: "Page \(i)", resourceUri: "ui/p\(i).html", capabilities: [.widgetState]))
        }
        return LoadedPlugin(
            name: name, manifest: PluginManifest(name: name, uiSurfaces: list),
            directoryURL: root.resolvingSymlinksInPath(), origin: .local)
    }

    private func makeHost(plugins: [LoadedPlugin], optedIn: Bool = true) -> (PluginUiHost, PluginUiOptInStore, SidePaneBox) {
        let optIn = PluginUiOptInStore(fileURL: nil)
        if optedIn { for p in plugins { optIn.setEnabled(true, pluginID: p.id) } }
        let box = SidePaneBox()
        let host = PluginUiHost(PluginUiHost.Dependencies(
            optIn: optIn,
            plugin: { id in plugins.first { $0.id == id } },
            invoker: { _, _ in { name, _ in name } },
            draftSink: { _ in },
            onSidePaneChange: { box.id = $0 },
            now: { [unowned self] in self.clock }))
        return (host, optIn, box)
    }

    private final class SidePaneBox { var id: UUID? }

    func testNothingOpensWithoutThePerPluginOptIn() throws {
        let plugin = try makePlugin()
        let (host, optIn, _) = makeHost(plugins: [plugin], optedIn: false)
        XCTAssertEqual(host.open(pluginID: plugin.id, surfaceID: "s0", sessionID: "c").failureValue, .notOptedIn)
        XCTAssertTrue(host.pages.isEmpty)
        optIn.setEnabled(true, pluginID: plugin.id)
        XCTAssertNotNil(host.open(pluginID: plugin.id, surfaceID: "s0", sessionID: "c").successValue)
        XCTAssertEqual(host.open(pluginID: plugin.id, surfaceID: "missing", sessionID: "c").failureValue, .noSuchSurface)
        optIn.setEnabled(true, pluginID: "ghost@local")
        XCTAssertEqual(host.open(pluginID: "ghost@local", surfaceID: "s0", sessionID: "c").failureValue, .pluginUnavailable)
        host.closeAll()
    }

    func testTheCapEvictsTheLeastRecentlyVisiblePageAndFreesItsSlot() throws {
        let plugin = try makePlugin(surfaces: PluginUiHost.maxLivePages + 2)
        let (host, _, _) = makeHost(plugins: [plugin])
        var opened: [PluginPageSession] = []
        for i in 0..<PluginUiHost.maxLivePages {
            opened.append(try XCTUnwrap(host.open(pluginID: plugin.id, surfaceID: "s\(i)", sessionID: "c").successValue))
        }
        XCTAssertEqual(host.pages.count, PluginUiHost.maxLivePages)
        // One past the cap: the least recently visible (the first) goes.
        let extra = try XCTUnwrap(host.open(pluginID: plugin.id, surfaceID: "s\(PluginUiHost.maxLivePages)", sessionID: "c").successValue)
        XCTAssertEqual(host.pages.count, PluginUiHost.maxLivePages)
        XCTAssertTrue(opened[0].isTornDown, "the evicted page is torn down, not just hidden")
        XCTAssertFalse(host.pages.contains { $0 === opened[0] })
        XCTAssertTrue(host.pages.contains { $0 === extra })
        // Re-showing a page makes it most recent, so the next eviction skips it.
        host.show(opened[1])
        _ = host.open(pluginID: plugin.id, surfaceID: "s\(PluginUiHost.maxLivePages + 1)", sessionID: "c")
        XCTAssertFalse(opened[1].isTornDown)
        XCTAssertTrue(opened[2].isTornDown)
        host.closeAll()
    }

    func testAnInteractingPageIsNeverEvictedAndAFullHostRefuses() throws {
        let plugin = try makePlugin(surfaces: PluginUiHost.maxLivePages + 1)
        let (host, _, _) = makeHost(plugins: [plugin])
        for i in 0..<PluginUiHost.maxLivePages {
            let page = try XCTUnwrap(host.open(pluginID: plugin.id, surfaceID: "s\(i)", sessionID: "c").successValue)
            page.bridge.lastUserGestureAt = clock  // every page was just used
        }
        XCTAssertEqual(
            host.open(pluginID: plugin.id, surfaceID: "s\(PluginUiHost.maxLivePages)", sessionID: "c").failureValue,
            .pageLimitReached)
        XCTAssertEqual(host.pages.count, PluginUiHost.maxLivePages, "nothing was evicted for the refused page")
        // After the interaction window passes the oldest page can go again.
        clock = clock.addingTimeInterval(PluginUiHost.interactingWindowSeconds + 1)
        XCTAssertNotNil(host.open(pluginID: plugin.id, surfaceID: "s\(PluginUiHost.maxLivePages)", sessionID: "c").successValue)
        host.closeAll()
    }

    func testCloseTearsDownClearsWidgetStateAndTheSidePane() throws {
        let plugin = try makePlugin()
        let (host, _, box) = makeHost(plugins: [plugin])
        let page = try XCTUnwrap(host.open(pluginID: plugin.id, surfaceID: "s0", sessionID: "c").successValue)
        XCTAssertEqual(box.id, page.id)
        host.widgetStates.set(page.identity, json: "{\"a\":1}")
        host.close(page)
        XCTAssertTrue(page.isTornDown)
        XCTAssertNil(host.widgetStates.get(page.identity))
        XCTAssertNil(host.sidePaneID)
        XCTAssertNil(box.id)
        XCTAssertTrue(host.pages.isEmpty)
    }

    func testPageIdentityIsPluginSurfaceAndSession() throws {
        let plugin = try makePlugin()
        let (host, _, _) = makeHost(plugins: [plugin])
        let a = try XCTUnwrap(host.open(pluginID: plugin.id, surfaceID: "s0", sessionID: "chat-a").successValue)
        let again = try XCTUnwrap(host.open(pluginID: plugin.id, surfaceID: "s0", sessionID: "chat-a").successValue)
        let b = try XCTUnwrap(host.open(pluginID: plugin.id, surfaceID: "s0", sessionID: "chat-b").successValue)
        XCTAssertTrue(a === again, "the same identity reuses its page")
        XCTAssertFalse(a === b, "another chat gets its own page")
        XCTAssertNotEqual(a.token, b.token, "and its own origin")
        host.closePages(exceptSession: "chat-b")
        XCTAssertTrue(a.isTornDown)
        XCTAssertFalse(b.isTornDown, "a chat switch closes the other chat's pages")
        host.closeAll()
    }

    func testDisablingOrOptingOutClosesThePluginsPages() throws {
        let plugin = try makePlugin()
        var available = [plugin]
        let optIn = PluginUiOptInStore(fileURL: nil)
        optIn.setEnabled(true, pluginID: plugin.id)
        let host = PluginUiHost(PluginUiHost.Dependencies(
            optIn: optIn, plugin: { id in available.first { $0.id == id } },
            invoker: { _, _ in { n, _ in n } }, draftSink: { _ in }, onSidePaneChange: { _ in }))
        let page = try XCTUnwrap(host.open(pluginID: plugin.id, surfaceID: "s0", sessionID: "c").successValue)
        available = []  // the plugin was disabled
        host.reconcile()
        XCTAssertTrue(page.isTornDown)
        available = [plugin]
        let second = try XCTUnwrap(host.open(pluginID: plugin.id, surfaceID: "s0", sessionID: "c").successValue)
        optIn.setEnabled(false, pluginID: plugin.id)
        host.reconcile()
        XCTAssertTrue(second.isTornDown)
    }

    func testOptInPersistsAndACorruptFileMeansNothingIsOptedIn() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("tsoptin-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let file = dir.appendingPathComponent("plugin_ui_optin.json")
        let store = PluginUiOptInStore(fileURL: file)
        store.setEnabled(true, pluginID: "demo@local")
        XCTAssertTrue(PluginUiOptInStore(fileURL: file).isEnabled("demo@local"))
        try "{ not json".write(to: file, atomically: true, encoding: .utf8)
        XCTAssertFalse(PluginUiOptInStore(fileURL: file).isEnabled("demo@local"))
    }

    func testRightColumnClaimantTreatsAPluginPageAsAPreviewPane() {
        let id = UUID()
        let claimant = AppRightColumnClaimant.resolve(
            openArtifactID: nil, previewAttachmentID: nil, isInspectorVisible: true, pluginPageID: id)
        XCTAssertEqual(claimant, .pluginPage(id))
        XCTAssertTrue(claimant.isPreviewPane, "the inspector shortcut must close it")
        XCTAssertEqual(AppChromeLayout.rightColumnWidth(claimant), AppChromeLayout.artifactPanelWidth)
        // An artifact the person just opened still outranks it, and with no
        // page the inspector is back.
        XCTAssertEqual(
            AppRightColumnClaimant.resolve(openArtifactID: id, previewAttachmentID: nil, isInspectorVisible: true, pluginPageID: UUID()),
            .artifact(id))
        XCTAssertEqual(
            AppRightColumnClaimant.resolve(openArtifactID: nil, previewAttachmentID: nil, isInspectorVisible: true),
            .inspector)
    }

    func testThePageCapIsTheDocumentedConstant() {
        XCTAssertEqual(PluginUiHost.maxLivePages, 8)
    }
}

private extension Result {
    var failureValue: Failure? { if case .failure(let e) = self { return e }; return nil }
    var successValue: Success? { if case .success(let v) = self { return v }; return nil }
}
