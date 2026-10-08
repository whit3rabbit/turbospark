import XCTest

@testable import TurboSparkApp

/// Low-severity UI/safety fixes from the Swift review (App, Browser, Files,
/// Generation sections). Each test pins one behavior the fix changed.
final class LowRoundUIFixesATests: XCTestCase {
    // MARK: Settings search catalog

    func testSearchCatalogTitlesHaveNoLiteralBackslashEscapes() {
        // "Add Folder\\u{2026}" was emitted verbatim, so the entry matched no control.
        for entry in SettingsControlCatalog.entries {
            XCTAssertFalse(entry.title.contains("\\"), "literal escape in \(entry.title)")
        }
        XCTAssertTrue(SettingsControlCatalog.entries.contains { $0.title == "Add Folder\u{2026}" })
    }

    // MARK: HF mirror

    func testMirrorEndpointRequiresHttpsExceptLoopback() {
        XCTAssertEqual(HfEndpointResolution.effectiveEndpoint(from: "https://hf-mirror.com"), "https://hf-mirror.com")
        XCTAssertEqual(HfEndpointResolution.effectiveEndpoint(from: "http://localhost:8080"), "http://localhost:8080")
        XCTAssertEqual(HfEndpointResolution.effectiveEndpoint(from: "http://127.0.0.1:9"), "http://127.0.0.1:9")
        XCTAssertNil(HfEndpointResolution.effectiveEndpoint(from: "http://hf-mirror.example"))
        XCTAssertNil(HfEndpointResolution.effectiveEndpoint(from: "hf-mirror.example"))
        XCTAssertNil(HfEndpointResolution.effectiveEndpoint(from: "ftp://hf-mirror.example"))
        XCTAssertTrue(HfEndpointResolution.isRejectedInput("http://hf-mirror.example"))
        XCTAssertFalse(HfEndpointResolution.isRejectedInput(""))
        XCTAssertFalse(HfEndpointResolution.isRejectedInput("https://huggingface.co"))
    }

    // MARK: Tool presentation

    func testWebLinkAcceptsOnlyHttpWithHost() {
        XCTAssertNotNil(ToolPresentation.webLink("https://example.com/a"))
        XCTAssertNotNil(ToolPresentation.webLink("http://example.com"))
        XCTAssertNil(ToolPresentation.webLink("file:///Applications/Utilities/Terminal.app"))
        XCTAssertNil(ToolPresentation.webLink("vscode://file/etc/passwd"))
        XCTAssertNil(ToolPresentation.webLink("x-apple.systempreferences:com.apple.preference"))
        XCTAssertNil(ToolPresentation.webLink("https://"))
    }

    // MARK: Approval dropdown

    func testApprovalDropdownAlwaysListsTheActiveMode() {
        for mode in AppPermissionMode.allCases {
            XCTAssertTrue(ToolApprovalMenuPopover.displayModes(including: mode).contains(mode), "\(mode)")
        }
        XCTAssertEqual(
            ToolApprovalMenuPopover.displayModes(including: .ask), [.ask, .auto, .permissive, .fullAccess])
    }

    // MARK: Composer autocomplete

    @MainActor
    func testNarrowingTheQueryResetsHighlightToTheTopMatch() {
        let controller = ComposerAutocompleteController()
        controller.textChanged(text: "/", skills: [], projectRoot: nil)
        XCTAssertGreaterThan(controller.suggestions.count, 1)
        controller.moveSelection(-1)
        XCTAssertEqual(controller.selectedIndex, controller.suggestions.count - 1)

        controller.textChanged(text: "/co", skills: [], projectRoot: nil)
        XCTAssertEqual(controller.selectedIndex, 0)
    }

    // MARK: Files

    private func makeTempDir() throws -> URL {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("lowround-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    func testFolderImportSkipsCredentialFilesAndKeepsGoingPastOversizedFiles() throws {
        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        try "SECRET=1".write(to: dir.appendingPathComponent("secrets.env.md"), atomically: true, encoding: .utf8)
        try "k".write(to: dir.appendingPathComponent(".env.md"), atomically: true, encoding: .utf8)
        // Two sparse 25 MB files: only one fits the 40 MB budget.
        for name in ["a1.md", "a2.md"] {
            let url = dir.appendingPathComponent(name)
            FileManager.default.createFile(atPath: url.path, contents: nil)
            let handle = try FileHandle(forWritingTo: url)
            try handle.truncate(atOffset: 25 * 1024 * 1024)
            try handle.close()
        }
        for name in ["z1.md", "z2.md", "z3.md"] {
            try "hi".write(to: dir.appendingPathComponent(name), atomically: true, encoding: .utf8)
        }

        let names = AttachmentImporter.collectFolderURLs(from: dir, maxFiles: 50).map(\.lastPathComponent)
        XCTAssertFalse(names.contains(".env.md"), "\(names)")
        XCTAssertTrue(names.contains("secrets.env.md"), "only dotenv-style names are refused: \(names)")
        XCTAssertEqual(names.filter { $0.hasPrefix("a") }.count, 1, "second big file skipped: \(names)")
        XCTAssertEqual(names.filter { $0.hasPrefix("z") }.count, 3, "walk must continue: \(names)")
    }

    func testProjectFileIndexOmitsFilesTheResolverRefuses() throws {
        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        try "x".write(to: dir.appendingPathComponent(".env"), atomically: true, encoding: .utf8)
        try "x".write(to: dir.appendingPathComponent(".env.local"), atomically: true, encoding: .utf8)
        try "x".write(to: dir.appendingPathComponent("readme.md"), atomically: true, encoding: .utf8)

        let paths = ProjectFileIndex.scan(root: dir, maxEntries: 100, maxDepth: 4).map(\.relativePath)
        XCTAssertEqual(paths, ["readme.md"])
    }

    // MARK: Browser

    @MainActor
    func testRetryOnCrashedAgentTabTakesTheTabOverFirst() throws {
        let server = try BrowserAutomationHTTPFixtureServer()
        let store = BrowserTabStore()
        let engine = WebKitBrowserEngine(
            tabStore: store, authorizeNavigation: { _ in .allow }, onUserTakeover: { _ in })
        let tab = engine.createTab(owner: .agent)
        try engine.navigate(to: server.url("/ok"), in: tab)
        let webView = try XCTUnwrap(engine.webView(for: tab))
        engine.webViewWebContentProcessDidTerminate(webView)
        XCTAssertEqual(store.tab(id: tab)?.loadState, .crashed)

        let pane = BrowserPaneModel(engine: engine)
        pane.selectTab(tab)
        XCTAssertTrue(pane.retryOrRecover())
        XCTAssertEqual(store.tab(id: tab)?.owner, .user)
    }
}

extension LowRoundUIFixesATests {
    func testInteractiveSnapshotRemapsParentIndicesToRetainedAncestors() {
        func node(_ role: String, parent: Int?, ref: String?) -> DOMSnapshotNode {
            DOMSnapshotNode(role: role, name: role, parentIndex: parent, bounds: nil, reference: ref)
        }
        // 0 html, 1 body, 2 form, 3 input(ref), 4 button(ref), 5 span(ref, child of button)
        let nodes = [
            node("html", parent: nil, ref: nil),
            node("body", parent: 0, ref: nil),
            node("form", parent: 1, ref: nil),
            node("input", parent: 2, ref: "e1"),
            node("button", parent: 2, ref: "e2"),
            node("span", parent: 4, ref: "e3"),
        ]
        let kept = WebKitBrowserControlPort.retainingReferencedNodes(nodes)
        XCTAssertEqual(kept.map(\.role), ["input", "button", "span"])
        XCTAssertEqual(kept.map(\.parentIndex), [nil, nil, 1])
        for node in kept {
            if let parent = node.parentIndex { XCTAssertTrue(kept.indices.contains(parent)) }
        }
    }
}

extension LowRoundUIFixesATests {
    func testHookOptionEnvSuffixMatchesTheExportedVariable() {
        XCTAssertEqual(AppHookExecutionEngine.optionEnvSuffix(for: "api-key"), "API_KEY")
        XCTAssertTrue(AppHookExecutionEngine.blockingEvents.contains(.preToolUse))
        XCTAssertFalse(AppHookExecutionEngine.blockingEvents.contains(.postToolUse))
    }
}

extension LowRoundUIFixesATests {
    func testArtifactFileNavigationIsScopedToTheFolderAndDeniedUnderANetworkGrant() {
        let folder = URL(fileURLWithPath: "/x/artifacts", isDirectory: true)
        let inside = URL(fileURLWithPath: "/x/artifacts/other.html")
        let sibling = URL(fileURLWithPath: "/x/artifacts-secret/a.html")
        XCTAssertTrue(ArtifactWebView.allowsFileNavigation(to: inside, readFolder: folder, networkAllowed: false))
        XCTAssertFalse(ArtifactWebView.allowsFileNavigation(to: sibling, readFolder: folder, networkAllowed: false))
        XCTAssertFalse(ArtifactWebView.allowsFileNavigation(to: inside, readFolder: nil, networkAllowed: false))
        XCTAssertFalse(ArtifactWebView.allowsFileNavigation(to: inside, readFolder: folder, networkAllowed: true))
    }

    @MainActor
    func testArtifactContentProcessTerminationReloadsAtMostOnce() {
        let coordinator = ArtifactWebView.Coordinator()
        XCTAssertTrue(coordinator.shouldReloadAfterTermination())
        XCTAssertFalse(coordinator.shouldReloadAfterTermination())
        XCTAssertFalse(coordinator.shouldReloadAfterTermination())
    }
}
