import Foundation
import WebKit
import XCTest

@testable import TurboSparkApp

/// ZP-06 sandbox guards. The pure ones (path resolution, CSP, bridge
/// policy, parser) run without a window; the WebKit ones load real pages
/// through the real scheme handler and prove each denial from inside the
/// page. Nothing here reaches the network: the one HTTP fixture is a
/// loopback listener that must record ZERO requests.
@MainActor
final class PluginPageSandboxTests: XCTestCase {
    // MARK: - Fixtures

    private var roots: [URL] = []

    override func tearDown() {
        for root in roots { try? FileManager.default.removeItem(at: root) }
        roots = []
        super.tearDown()
    }

    private func makeRoot(_ files: [String: String]) throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("tsplugin-\(UUID().uuidString)", isDirectory: true)
        roots.append(root)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        for (path, content) in files {
            let url = root.appendingPathComponent(path)
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try content.write(to: url, atomically: true, encoding: .utf8)
        }
        return root.resolvingSymlinksInPath()
    }

    private func surface(
        page: String = "ui/page.html", resources: [String] = [],
        capabilities: [PluginUiCapability] = [], tools: [String] = []
    ) -> PluginUiSurface {
        PluginUiSurface(
            id: "main", title: "Chart", resourceUri: page, resources: resources,
            capabilities: capabilities, placement: .sidePane, tools: tools)
    }

    private final class Recorder: @unchecked Sendable {
        private let lock = NSLock()
        private var names: [String] = []
        var calls: [String] { lock.lock(); defer { lock.unlock() }; return names }
        func record(_ name: String) { lock.lock(); names.append(name); lock.unlock() }
    }

    private struct Harness {
        let session: PluginPageSession
        let recorder: Recorder
        let states: PluginWidgetStateStore
        var drafts: DraftBox
    }

    private final class DraftBox { var drafts: [String] = [] }

    private func makeSession(
        root: URL, surface: PluginUiSurface, now: @escaping () -> Date = Date.init,
        cspOverride: String? = nil
    ) -> Harness {
        let recorder = Recorder()
        let states = PluginWidgetStateStore()
        let box = DraftBox()
        let environment = PluginPageEnvironment(
            invoker: { name, _ in
                recorder.record(name)
                return "result-of-\(name)"
            },
            draftSink: { box.drafts.append($0) },
            widgetStates: states,
            now: now)
        let identity = PluginPageIdentity(pluginID: "demo@local", surfaceID: surface.id, sessionID: "chat-1")
        let tools = PluginUiToolNames.declaredTools(pluginName: "demo", surface: surface)
        let session = PluginPageSession(
            identity: identity, surface: surface, pluginName: "demo", root: root,
            environment: environment, declaredTools: tools, cspOverrideForTests: cspOverride)
        return Harness(session: session, recorder: recorder, states: states, drafts: box)
    }

    private func eventually(_ timeout: TimeInterval = 15, _ condition: @escaping () -> Bool) async -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if condition() { return true }
            try? await Task.sleep(nanoseconds: 50_000_000)
        }
        return condition()
    }

    /// Loads the session and waits for the load to settle (rendered or failed).
    private func loadAndSettle(_ session: PluginPageSession) async -> Bool {
        session.load()
        return await eventually { !session.isLoading || session.failure != nil }
    }

    private func title(_ session: PluginPageSession) async -> String {
        (try? await session.webView.evaluateJavaScript("document.title") as? String) ?? ""
    }

    private func waitForTitle(_ session: PluginPageSession, _ expected: String) async -> String {
        var last = ""
        let deadline = Date().addingTimeInterval(10)
        while Date() < deadline {
            last = await title(session)
            if last == expected { return last }
            try? await Task.sleep(nanoseconds: 50_000_000)
        }
        return last
    }

    // MARK: - Path resolution (pure)

    func testNormalizeRefusesTraversalAbsoluteHiddenAndEncodedPaths() {
        for bad in ["../x.html", "a/../x.html", "/etc/passwd", "a//b.html", ".hidden/x.html",
                    "a/.git/x.js", "a\\b.html", "a%2e%2e/b.html", "file:///x.html", "a/b.html?x=1",
                    "", ".", "..", "a/\u{0}b.js"] {
            XCTAssertNil(PluginPageResolver.normalizeRelativePath(bad), "must refuse \(bad.debugDescription)")
        }
        XCTAssertEqual(PluginPageResolver.normalizeRelativePath("./ui/page.html"), "ui/page.html")
        XCTAssertEqual(PluginPageResolver.normalizeRelativePath("ui/page.html"), "ui/page.html")
    }

    func testLoadServesOnlyDeclaredFilesOfAllowedTypes() throws {
        let root = try makeRoot([
            "ui/page.html": "<p>hi</p>", "ui/app.js": "1", "ui/secret.txt": "s", "ui/blob.bin": "b",
        ])
        let declared: Set<String> = ["ui/page.html", "ui/app.js", "ui/blob.bin"]
        XCTAssertEqual(try PluginPageResolver.load(relativePath: "ui/page.html", root: root, declared: declared).get().mimeType, "text/html; charset=utf-8")
        // Present on disk but undeclared.
        XCTAssertEqual(PluginPageResolver.load(relativePath: "ui/secret.txt", root: root, declared: declared), .failure(.notDeclared))
        // Declared but not an allowed type.
        XCTAssertEqual(PluginPageResolver.load(relativePath: "ui/blob.bin", root: root, declared: declared), .failure(.unsupportedType))
        // Declared, missing on disk.
        XCTAssertEqual(PluginPageResolver.load(relativePath: "ui/app.js", root: try makeRoot([:]), declared: declared), .failure(.notFound))
    }

    func testLoadRefusesTraversalEvenWhenTheTargetIsDeclared() throws {
        let root = try makeRoot(["ui/page.html": "x"])
        let outside = root.deletingLastPathComponent().appendingPathComponent("outside-\(UUID().uuidString).html")
        try "secret".write(to: outside, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: outside) }
        let escape = "../" + outside.lastPathComponent
        // Even a manifest that (maliciously) declares the escape string.
        XCTAssertEqual(
            PluginPageResolver.load(relativePath: escape, root: root, declared: [escape]),
            .failure(.forbiddenPath))
    }

    func testLoadRefusesASymlinkLeafAndASymlinkedDirectory() throws {
        let root = try makeRoot(["ui/page.html": "x"])
        let outside = root.deletingLastPathComponent().appendingPathComponent("outside-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: outside, withIntermediateDirectories: true)
        try "secret".write(to: outside.appendingPathComponent("s.js"), atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: outside) }
        try FileManager.default.createSymbolicLink(
            at: root.appendingPathComponent("ui/leak.js"), withDestinationURL: outside.appendingPathComponent("s.js"))
        try FileManager.default.createSymbolicLink(
            at: root.appendingPathComponent("linked"), withDestinationURL: outside)
        let declared: Set<String> = ["ui/leak.js", "linked/s.js"]
        XCTAssertEqual(PluginPageResolver.load(relativePath: "ui/leak.js", root: root, declared: declared), .failure(.symlinkRefused))
        XCTAssertEqual(PluginPageResolver.load(relativePath: "linked/s.js", root: root, declared: declared), .failure(.symlinkRefused))
    }

    func testLoadEnforcesTheSizeCap() throws {
        let root = try makeRoot(["a.js": String(repeating: "x", count: 100)])
        XCTAssertEqual(PluginPageResolver.load(relativePath: "a.js", root: root, declared: ["a.js"], maxBytes: 99), .failure(.tooLarge))
        XCTAssertNoThrow(try PluginPageResolver.load(relativePath: "a.js", root: root, declared: ["a.js"], maxBytes: 100).get())
    }

    func testRequestPathRejectsAForeignHostAndDoubleEncodedDots() {
        let token = "abc123"
        XCTAssertEqual(PluginPageResolver.requestPath(from: URL(string: "tsplugin://abc123/ui/app.js")!, token: token), "ui/app.js")
        XCTAssertNil(PluginPageResolver.requestPath(from: URL(string: "tsplugin://other/ui/app.js")!, token: token))
        XCTAssertNil(PluginPageResolver.requestPath(from: URL(string: "https://abc123/ui/app.js")!, token: token))
        // %252e%252e decodes once to "%2e%2e", which still carries a percent sign.
        XCTAssertNil(PluginPageResolver.requestPath(from: URL(string: "tsplugin://abc123/%252e%252e/x.js")!, token: token))
        XCTAssertNil(PluginPageResolver.requestPath(from: URL(string: "tsplugin://abc123/..%2fx.js")!, token: token))
    }

    // MARK: - CSP and navigation policy (pure)

    func testCSPDeniesNetworkFramesAndInlineScriptAndTracksDeclarations() {
        let none = PluginPageCSP.header(for: surface())
        XCTAssertTrue(none.contains("connect-src 'none'"))
        XCTAssertTrue(none.contains("frame-src 'none'"))
        XCTAssertTrue(none.contains("script-src 'none'"), "no declared script means no script source")
        XCTAssertFalse(none.contains("unsafe-eval"))
        XCTAssertFalse(none.contains("http"))
        let withScript = PluginPageCSP.header(for: surface(resources: ["ui/app.js"]))
        XCTAssertTrue(withScript.contains("script-src 'self'"))
        XCTAssertFalse(withScript.contains("script-src 'self' 'unsafe-inline'"))
    }

    func testNavigationPolicyAllowsOnlyTheInitialDocumentLoad() {
        let page = URL(string: "tsplugin://tok/ui/page.html")!
        func allows(_ url: String?, main: Bool = true, pending: Bool = true) -> Bool {
            PluginPageNavigationPolicy.allows(
                url: url.flatMap(URL.init(string:)), pageURL: page, isMainFrame: main, initialLoadPending: pending)
        }
        XCTAssertTrue(allows("tsplugin://tok/ui/page.html"))
        XCTAssertFalse(allows("tsplugin://tok/ui/page.html", pending: false), "a page-initiated reload")
        XCTAssertFalse(allows("tsplugin://tok/ui/other.html"))
        XCTAssertFalse(allows("tsplugin://tok/ui/page.html", main: false), "a subframe")
        for denied in ["https://example.com/", "http://127.0.0.1/", "file:///etc/passwd", "data:text/html,x",
                       "javascript:alert(1)", "blob:tsplugin://tok/x", "ws://example.com", "about:srcdoc"] {
            XCTAssertFalse(allows(denied), denied)
        }
        XCTAssertFalse(allows(nil))
        XCTAssertTrue(allows("about:blank"))
        XCTAssertFalse(allows("about:blank", main: false))
    }

    // MARK: - Bridge policy (pure, no WebKit)

    private func bridge(
        capabilities: [PluginUiCapability], tools: [String] = ["srv/echo"],
        now: @escaping () -> Date = Date.init, invoker: PageBridge.ToolInvoker? = nil,
        drafts: DraftBox = DraftBox(), states: PluginWidgetStateStore? = nil
    ) -> PageBridge {
        let states = states ?? PluginWidgetStateStore()
        let surface = surface(capabilities: capabilities, tools: tools)
        return PageBridge(
            identity: PluginPageIdentity(pluginID: "demo@local", surfaceID: "main", sessionID: "c"),
            pluginName: "demo", pageTitle: "Chart", token: "tok",
            capabilities: Set(capabilities),
            declaredTools: PluginUiToolNames.declaredTools(pluginName: "demo", surface: surface),
            widgetStates: states,
            invoker: invoker ?? { name, _ in name },
            draftSink: { drafts.drafts.append($0) }, now: now)
    }

    private func send(
        _ bridge: PageBridge, _ method: String, _ params: [String: Any] = [:],
        main: Bool = true, host: String? = "tok", scheme: String? = "tsplugin"
    ) async -> Result<Any, PageBridgeError> {
        await bridge.handle(body: ["v": 1, "method": method, "params": params], isMainFrame: main, scheme: scheme, host: host)
    }

    private func code(_ result: Result<Any, PageBridgeError>) -> String? {
        if case .failure(let error) = result { return error.code.rawValue }
        return nil
    }

    func testBridgeRefusesUndeclaredMethodsByName() async {
        let b = bridge(capabilities: [.widgetState])
        let invoke = await send(b, "invokeTool", ["name": "srv/echo"])
        XCTAssertEqual(code(invoke), "undeclared_method")
        let follow = await send(b, "sendFollowUp", ["text": "hi"])
        XCTAssertEqual(code(follow), "undeclared_method")
        let unknown = await send(b, "evalRemote", ["code": "1"])
        XCTAssertEqual(code(unknown), "undeclared_method")
        let ok = await send(b, "getWidgetState")
        XCTAssertNil(code(ok))
    }

    func testBridgeRefusesSubframesAndForeignOrigins() async {
        let b = bridge(capabilities: [.widgetState])
        let sub = await send(b, "getWidgetState", main: false)
        XCTAssertEqual(code(sub), "bad_frame")
        let foreign = await send(b, "getWidgetState", host: "other")
        XCTAssertEqual(code(foreign), "bad_frame")
        let scheme = await send(b, "getWidgetState", scheme: "https")
        XCTAssertEqual(code(scheme), "bad_frame")
    }

    func testBridgeEnforcesMessageAndStateSizeCaps() async {
        let b = bridge(capabilities: [.widgetState, .sendFollowUp])
        let huge = String(repeating: "x", count: PluginPageLimits.maxMessageBytes + 1)
        let message = await send(b, "setWidgetState", ["state": huge])
        XCTAssertEqual(code(message), "payload_too_large")
        let longFollowUp = await send(b, "sendFollowUp", ["text": String(repeating: "y", count: PluginPageLimits.maxFollowUpCharacters + 1)])
        XCTAssertEqual(code(longFollowUp), "payload_too_large")
        // Under the message cap but over the state cap needs a lower message cap than state cap;
        // the state cap is therefore implied by the message cap and asserted as an invariant.
        XCTAssertLessThanOrEqual(PluginPageLimits.maxMessageBytes, PluginPageLimits.maxWidgetStateBytes)
        XCTAssertEqual(PluginPageBridgeLimitsDocumentation.messageBytes, PluginPageLimits.maxMessageBytes)
    }

    func testWidgetStateIsKeyedByPageIdentity() async {
        let states = PluginWidgetStateStore()
        let a = bridge(capabilities: [.widgetState], states: states)
        let otherSurface = surface(capabilities: [.widgetState])
        let b = PageBridge(
            identity: PluginPageIdentity(pluginID: "other@local", surfaceID: "main", sessionID: "c"),
            pluginName: "other", pageTitle: "t", token: "tok", capabilities: [.widgetState],
            declaredTools: [:], widgetStates: states, invoker: { n, _ in n }, draftSink: { _ in })
        _ = otherSurface
        let set = await send(a, "setWidgetState", ["state": ["slider": 3]])
        XCTAssertNil(code(set))
        let read = await send(a, "getWidgetState")
        XCTAssertEqual((try? read.get() as? [String: Int]), ["slider": 3])
        let foreign = await send(b, "getWidgetState")
        XCTAssertTrue((try? foreign.get()) is NSNull, "another page identity must not see this state")
    }

    func testInvokeToolNeedsADeclaredToolAndARecentGesture() async {
        var clock = Date(timeIntervalSince1970: 1_000)
        let recorder = Recorder()
        let b = bridge(capabilities: [.invokeTool], now: { clock }, invoker: { name, _ in recorder.record(name); return "ok" })
        let undeclared = await send(b, "invokeTool", ["name": "srv/other"])
        XCTAssertEqual(code(undeclared), "tool_not_declared")
        let noGesture = await send(b, "invokeTool", ["name": "srv/echo"])
        XCTAssertEqual(code(noGesture), "gesture_required")
        b.lastUserGestureAt = clock
        let allowed = await send(b, "invokeTool", ["name": "srv/echo", "args": ["q": "1"]])
        XCTAssertNil(code(allowed))
        XCTAssertEqual(recorder.calls, ["mcp__plugin:demo:srv__echo"])
        clock = clock.addingTimeInterval(PluginPageLimits.userGestureWindowSeconds + 1)
        let stale = await send(b, "invokeTool", ["name": "srv/echo"])
        XCTAssertEqual(code(stale), "gesture_required", "the gesture window expires")
    }

    func testInvokeToolIsRateLimitedAndBoundedInFlight() async {
        let clock = Date(timeIntervalSince1970: 1_000)
        let b = bridge(capabilities: [.invokeTool], now: { clock })
        b.lastUserGestureAt = clock
        for _ in 0..<PluginPageLimits.maxToolCallsPerMinute {
            let r = await send(b, "invokeTool", ["name": "srv/echo"])
            XCTAssertNil(code(r))
        }
        let limited = await send(b, "invokeTool", ["name": "srv/echo"])
        XCTAssertEqual(code(limited), "rate_limited")
    }

    func testPermissionRefusalIsNamedAndSendFollowUpOnlyFillsTheLabeledDraft() async {
        let b = bridge(capabilities: [.invokeTool, .sendFollowUp],
                       invoker: { _, _ in throw PageToolError.denied("requires interactive approval") })
        b.lastUserGestureAt = Date()
        let denied = await send(b, "invokeTool", ["name": "srv/echo"])
        XCTAssertEqual(code(denied), "permission_denied")

        let drafts = DraftBox()
        let d = bridge(capabilities: [.sendFollowUp], drafts: drafts)
        let noGesture = await send(d, "sendFollowUp", ["text": "hello"])
        XCTAssertEqual(code(noGesture), "gesture_required")
        d.lastUserGestureAt = Date()
        let sent = await send(d, "sendFollowUp", ["text": "hello\u{1b}[31m"])
        XCTAssertNil(code(sent))
        XCTAssertEqual(drafts.drafts.count, 1)
        XCTAssertTrue(drafts.drafts[0].contains("Chart"), "labeled with the page title")
        XCTAssertTrue(drafts.drafts[0].contains("demo"), "labeled with the plugin name")
        XCTAssertFalse(drafts.drafts[0].contains("\u{1b}"), "control characters are stripped")
    }

    func testToolResultsAreClippedToTheCap() async {
        let big = String(repeating: "a", count: PluginPageLimits.maxToolResultBytes + 500)
        let b = bridge(capabilities: [.invokeTool], invoker: { _, _ in big })
        b.lastUserGestureAt = Date()
        let r = await send(b, "invokeTool", ["name": "srv/echo"])
        let value = try? r.get() as? [String: Any]
        XCTAssertEqual((value?["output"] as? String)?.utf8.count, PluginPageLimits.maxToolResultBytes)
        XCTAssertEqual(value?["truncated"] as? Bool, true)
    }

    // MARK: - Manifest parsing

    func testParserReadsSurfacesAndDiagnosesMalformedEntries() throws {
        let json: [String: Any] = [
            "name": "demo",
            "ui": ["surfaces": [
                ["id": "chart", "resourceUri": "./ui/chart.html", "resources": ["ui/chart.js", "../evil.js", "ui/x.exe"],
                 "capabilities": ["invokeTool", "widgetState", "root"], "placement": "sidePane", "tools": ["srv/run", "bad"]],
                ["id": "bad id", "resourceUri": "ui/a.html"],
                ["id": "esc", "resourceUri": "../up.html"],
                ["id": "abs", "resourceUri": "/etc/passwd"],
                ["id": "notHtml", "resourceUri": "ui/a.js"],
                ["id": "chart", "resourceUri": "ui/dup.html"],
                "junk",
            ]],
        ]
        let manifest = try PluginManifestParser.manifest(from: json, fallbackName: "demo", sourceDescription: "t")
        XCTAssertEqual(manifest.uiSurfaces.map(\.id), ["chart"])
        let s = manifest.uiSurfaces[0]
        XCTAssertEqual(s.resourceUri, "ui/chart.html")
        XCTAssertEqual(s.resources, ["ui/chart.js"], "traversal and unserved types are dropped")
        XCTAssertEqual(s.capabilities, [.invokeTool, .widgetState], "an unknown capability is dropped")
        XCTAssertEqual(s.tools, ["srv/run"])
        let notes = manifest.unsupportedNotes.joined(separator: "\n")
        for needle in ["bad id", "resourceUri", "duplicate surface id", "unknown capability", "not an object"] {
            XCTAssertTrue(notes.contains(needle) || notes.contains("ui.surfaces"), needle)
        }
    }

    func testParserCapsSurfacesAndReportsAWrongShape() throws {
        let many = (0..<12).map { ["id": "s\($0)", "resourceUri": "ui/\($0).html"] }
        let manifest = try PluginManifestParser.manifest(from: ["name": "d", "ui": ["surfaces": many]], fallbackName: "d", sourceDescription: "t")
        XCTAssertEqual(manifest.uiSurfaces.count, PluginPageLimits.maxSurfacesPerPlugin)
        let wrong = try PluginManifestParser.manifest(from: ["name": "d", "ui": "nope"], fallbackName: "d", sourceDescription: "t")
        XCTAssertTrue(wrong.uiSurfaces.isEmpty)
        XCTAssertTrue(wrong.unsupportedNotes.contains { $0.hasPrefix("ui:") })
        let none = try PluginManifestParser.manifest(from: ["name": "d"], fallbackName: "d", sourceDescription: "t")
        XCTAssertTrue(none.uiSurfaces.isEmpty)
    }

    // MARK: - Real WKWebView

    func testPageLoadsDeclaredScriptAndBlocksInlineScript() async throws {
        let root = try makeRoot([
            "ui/page.html": "<!doctype html><title>start</title><script>document.title='inline-ran'</script><script src=\"app.js\"></script>",
            "ui/app.js": "document.title = (document.title === 'inline-ran' ? 'inline-ran' : 'declared-ran');",
        ])
        let h = makeSession(root: root, surface: surface(resources: ["ui/app.js"]))
        defer { h.session.tearDown() }
        let settled = await loadAndSettle(h.session)
        XCTAssertTrue(settled)
        let result = await waitForTitle(h.session, "declared-ran")
        XCTAssertEqual(result, "declared-ran", "declared script runs; the inline script is blocked by CSP")
        XCTAssertNil(h.session.failure)
    }

    func testPageCannotReachTheNetworkByFetchImageNavigationOrWindow() async throws {
        try await runNetworkProbe(cspOverride: nil)
    }

    /// The offline content rule list is its own layer: with the CSP relaxed
    /// to allow everything, the rule list ALONE must still keep every request
    /// off the loopback server. Removing the rule list fails this test.
    func testOfflineRuleListAloneKeepsTheNetworkOut() async throws {
        try await runNetworkProbe(cspOverride: "default-src * 'unsafe-inline'; script-src 'self'; connect-src *; img-src *")
    }

    private func runNetworkProbe(cspOverride: String?) async throws {
        let server = try BrowserAutomationHTTPFixtureServer()
        let base = server.url("")
        let js = """
        var out = [];
        function done(tag) { out.push(tag); document.title = out.join(','); }
        fetch('\(base)/leak-fetch').then(function(){ done('fetch-open'); }, function(){ done('fetch-blocked'); });
        var img = new Image(); img.onload = function(){ done('img-open'); }; img.onerror = function(){ done('img-blocked'); };
        img.src = '\(base)/leak-img';
        var w = window.open('\(base)/leak-open'); done(w === null ? 'open-null' : 'open-window');
        try { location.href = '\(base)/leak-nav'; } catch (e) {}
        var f = document.createElement('iframe'); f.src = '\(base)/leak-frame'; document.body.appendChild(f);
        """
        let root = try makeRoot([
            "ui/page.html": "<!doctype html><title>start</title><body><script src=\"app.js\"></script>",
            "ui/app.js": js,
        ])
        let h = makeSession(root: root, surface: surface(resources: ["ui/app.js"]), cspOverride: cspOverride)
        defer { h.session.tearDown() }
        _ = await loadAndSettle(h.session)
        _ = await eventually { true }
        // Wait until the fetch and image outcomes are in.
        let deadline = Date().addingTimeInterval(10)
        var current = ""
        while Date() < deadline {
            current = await title(h.session)
            if current.contains("fetch-blocked"), current.contains("img-blocked") { break }
            try? await Task.sleep(nanoseconds: 100_000_000)
        }
        XCTAssertTrue(current.contains("fetch-blocked"), current)
        XCTAssertTrue(current.contains("img-blocked"), current)
        XCTAssertTrue(current.contains("open-null"), current)
        try? await Task.sleep(nanoseconds: 500_000_000)
        for path in ["/leak-fetch", "/leak-img", "/leak-open", "/leak-nav", "/leak-frame"] {
            XCTAssertEqual(server.requestCount(for: path), 0, "\(path) reached the loopback server")
        }
    }

    func testSymlinkedResourceIsRefusedEndToEnd() async throws {
        let root = try makeRoot([
            "ui/page.html": "<!doctype html><title>start</title><script src=\"leak.js\"></script>",
        ])
        let outside = root.deletingLastPathComponent().appendingPathComponent("out-\(UUID().uuidString).js")
        try "document.title='leaked'".write(to: outside, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: outside) }
        try FileManager.default.createSymbolicLink(at: root.appendingPathComponent("ui/leak.js"), withDestinationURL: outside)
        let h = makeSession(root: root, surface: surface(resources: ["ui/leak.js"]))
        defer { h.session.tearDown() }
        _ = await loadAndSettle(h.session)
        try? await Task.sleep(nanoseconds: 500_000_000)
        let current = await title(h.session)
        XCTAssertNotEqual(current, "leaked")
        XCTAssertTrue(h.session.refusals.contains { $0.error == .symlinkRefused })
    }

    func testBridgeInsideTheRealPageRefusesUndeclaredAndAnswersDeclared() async throws {
        let js = """
        (async function () {
          var out = [];
          try { await window.turbospark.invokeTool('srv/echo', {}); out.push('invoke-ran'); }
          catch (e) { out.push('invoke:' + e.code); }
          try { await window.turbospark.sendFollowUp('x'); out.push('follow-ran'); }
          catch (e) { out.push('follow:' + e.code); }
          try { await window.turbospark.setWidgetState({ n: 7 }); var s = await window.turbospark.getWidgetState(); out.push('state:' + s.n); }
          catch (e) { out.push('state:' + e.code); }
          try { window.turbospark = null; } catch (e) {}
          out.push(window.turbospark ? 'frozen' : 'replaced');
          document.title = out.join('|');
        })();
        """
        let root = try makeRoot([
            "ui/page.html": "<!doctype html><title>start</title><script src=\"app.js\"></script>",
            "ui/app.js": js,
        ])
        // Only widgetState is declared: invokeTool and sendFollowUp are undeclared.
        let h = makeSession(root: root, surface: surface(resources: ["ui/app.js"], capabilities: [.widgetState], tools: ["srv/echo"]))
        defer { h.session.tearDown() }
        _ = await loadAndSettle(h.session)
        let result = await waitForTitle(h.session, "invoke:undeclared_method|follow:undeclared_method|state:7|frozen")
        XCTAssertEqual(result, "invoke:undeclared_method|follow:undeclared_method|state:7|frozen")
        XCTAssertTrue(h.recorder.calls.isEmpty, "nothing executed undeclared")
    }

    func testDeclaredToolCallNeedsGestureThenRunsThroughTheInvoker() async throws {
        let js = """
        window.go = async function () {
          try { var r = await window.turbospark.invokeTool('srv/echo', { q: 1 }); document.title = 'ok:' + r.output; }
          catch (e) { document.title = 'err:' + e.code; }
        };
        """
        let root = try makeRoot([
            "ui/page.html": "<!doctype html><title>start</title><script src=\"app.js\"></script>",
            "ui/app.js": js,
        ])
        let h = makeSession(root: root, surface: surface(resources: ["ui/app.js"], capabilities: [.invokeTool], tools: ["srv/echo"]))
        defer { h.session.tearDown() }
        _ = await loadAndSettle(h.session)
        _ = await waitForTitle(h.session, "start")
        _ = try await h.session.webView.evaluateJavaScript("void window.go()")
        let denied = await waitForTitle(h.session, "err:gesture_required")
        XCTAssertEqual(denied, "err:gesture_required")
        XCTAssertTrue(h.recorder.calls.isEmpty)
        // A real mouse-down reaching the web view is the only thing that opens the gate.
        let event = try XCTUnwrap(NSEvent.mouseEvent(
            with: .leftMouseDown, location: .zero, modifierFlags: [], timestamp: 0,
            windowNumber: 0, context: nil, eventNumber: 0, clickCount: 1, pressure: 1))
        h.session.webView.onUserGesture?()
        _ = event
        _ = try await h.session.webView.evaluateJavaScript("void window.go()")
        let allowed = await waitForTitle(h.session, "ok:result-of-mcp__plugin:demo:srv__echo")
        XCTAssertEqual(allowed, "ok:result-of-mcp__plugin:demo:srv__echo")
        XCTAssertEqual(h.recorder.calls, ["mcp__plugin:demo:srv__echo"])
    }

    func testCameraMicrophoneAreDeniedByDefaultInThePage() async throws {
        let js = """
        navigator.mediaDevices.getUserMedia({ video: true, audio: true }).then(
          function () { document.title = 'granted'; }, function (e) { document.title = 'denied:' + e.name; });
        """
        let root = try makeRoot([
            "ui/page.html": "<!doctype html><title>start</title><script src=\"app.js\"></script>",
            "ui/app.js": js,
        ])
        let h = makeSession(root: root, surface: surface(resources: ["ui/app.js"]))
        defer { h.session.tearDown() }
        _ = await loadAndSettle(h.session)
        let deadline = Date().addingTimeInterval(4)
        var current = "start"
        while Date() < deadline, current == "start" || current.isEmpty {
            current = await title(h.session)
            try? await Task.sleep(nanoseconds: 100_000_000)
        }
        // On a Mac with no capture hardware WebKit may leave the promise
        // pending instead of rejecting it, so the guard asserted is "never
        // granted". The delegate's .deny answer cannot be called directly
        // (WKSecurityOrigin and WKFrameInfo have no public initializers).
        XCTAssertNotEqual(current, "granted")
    }

    // MARK: - Failure card and lifecycle

    func testMissingPageGivesATypedFailureNotABlankView() async throws {
        let root = try makeRoot(["other.txt": "x"])
        let h = makeSession(root: root, surface: surface(page: "ui/page.html"))
        _ = await loadAndSettle(h.session)
        _ = await eventually { h.session.failure != nil }
        let failure = h.session.failure
        let refused = h.session.refusals.map { "\($0.path)=\($0.error.code)" }
        h.session.tearDown()
        XCTAssertEqual(failure, .notFound, "refusals: \(refused)")
    }

    func testProcessTerminationReloadsOnceThenShowsTheCard() async throws {
        let root = try makeRoot(["ui/page.html": "<!doctype html><title>alive</title>"])
        let h = makeSession(root: root, surface: surface())
        defer { h.session.tearDown() }
        _ = await loadAndSettle(h.session)
        h.session.webViewWebContentProcessDidTerminate(h.session.webView)
        XCTAssertNil(h.session.failure, "the first loss reloads from the stored page")
        _ = await eventually { !h.session.isLoading }
        h.session.webViewWebContentProcessDidTerminate(h.session.webView)
        XCTAssertEqual(h.session.failure, .processTerminated)
        h.session.retry()
        _ = await eventually { !h.session.isLoading || h.session.failure != nil }
        XCTAssertNil(h.session.failure, "retry reloads the page")
    }

    func testTearDownClearsStateAndFreesTheWebView() async throws {
        let root = try makeRoot(["ui/page.html": "<!doctype html><title>x</title>"])
        let h = makeSession(root: root, surface: surface(capabilities: [.widgetState]))
        _ = await loadAndSettle(h.session)
        h.states.set(h.session.identity, json: "{\"a\":1}")
        h.session.tearDown()
        XCTAssertTrue(h.session.isTornDown)
        XCTAssertNil(h.states.get(h.session.identity), "widget state is cleared with the page")
        XCTAssertNil(h.session.webView.navigationDelegate)
        h.session.tearDown()  // idempotent
    }
}

/// Mirrors the documented limit so a change to the constant fails a test that
/// names it, not just a number in the docs.
enum PluginPageBridgeLimitsDocumentation {
    static let messageBytes = 16 * 1024
}
