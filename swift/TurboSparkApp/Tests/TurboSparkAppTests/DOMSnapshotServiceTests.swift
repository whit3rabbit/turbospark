import Foundation
import XCTest
import WebKit
import DOMSnapshotFixtures

@testable import TurboSparkApp

@MainActor
final class DOMSnapshotServiceTests: XCTestCase {
    func testSameNodeKeepsItsReferenceAfterBenignUpdate() async throws {
        let fixture = try fixtureURL("dynamic-rerender")
        let page = try await makePage(fixture: fixture)
        defer { page.service.invalidate(); page.webView.stopLoading() }

        let initial = try await page.service.snapshot()
        let reference = try XCTUnwrap(node(named: "Stable node", in: initial).reference)
        try await page.webView.evaluateJavaScript("document.querySelector('#stable').textContent = 'Updated node'")

        let updated = try await page.service.snapshot()
        XCTAssertEqual(node(named: "Updated node", in: updated).reference, reference)
    }

    func testReplacingNodeAtSamePathMakesTheOldReferenceStale() async throws {
        let fixture = try fixtureURL("dynamic-rerender")
        let page = try await makePage(fixture: fixture)
        defer { page.service.invalidate(); page.webView.stopLoading() }

        let initial = try await page.service.snapshot()
        let oldReference = try XCTUnwrap(node(named: "Original node", in: initial).reference)
        try await page.webView.evaluateJavaScript("document.querySelector('#replace-me').outerHTML = '<button id=\"replace-me\" type=\"button\">Replacement node</button>'")

        let updated = try await page.service.snapshot()
        let newReference = try XCTUnwrap(node(named: "Replacement node", in: updated).reference)
        XCTAssertNotEqual(newReference, oldReference)
        do {
            try await page.service.resolve(reference: oldReference, generation: initial.generation)
            XCTFail("Expected the replaced node reference to be stale")
        } catch let error as BrowserControlError {
            XCTAssertEqual(error, .staleReference(reference: oldReference))
        }
    }

    func testNavigationInvalidatesReferencesFromThePreviousGeneration() async throws {
        let firstFixture = try fixtureURL("dynamic-rerender")
        let page = try await makePage(fixture: firstFixture)
        defer { page.service.invalidate(); page.webView.stopLoading() }

        let initial = try await page.service.snapshot()
        let oldReference = try XCTUnwrap(node(named: "Stable node", in: initial).reference)
        let nextFixture = try fixtureURL("dialogs")
        let loaded = expectation(description: "Second fixture committed")
        page.navigationObserver.onCommit = { loaded.fulfill() }
        page.webView.loadFileURL(nextFixture, allowingReadAccessTo: nextFixture.deletingLastPathComponent())
        await fulfillment(of: [loaded], timeout: 10)

        do {
            try await page.service.resolve(reference: oldReference, generation: initial.generation)
            XCTFail("Expected a reference from the previous document to be stale")
        } catch let error as BrowserControlError {
            XCTAssertEqual(error, .staleReference(reference: oldReference))
        }
    }

    func testCredentialValuesAreNeverPresentInTheSnapshot() async throws {
        let fixture = try fixtureURL("credential-fields")
        let page = try await makePage(fixture: fixture)
        defer { page.service.invalidate(); page.webView.stopLoading() }

        let snapshot = try await page.service.snapshot()
        let serialized = String(decoding: try JSONEncoder().encode(snapshot), as: UTF8.self)
        for secret in [
            "alice-private", "super-secret-password", "secret-recovery-code",
            "unlabeled-secret-value", "self-labeled-secret-value"
        ] {
            XCTAssertFalse(serialized.contains(secret), "Snapshot leaked a form value")
        }
        XCTAssertTrue(serialized.contains("Password field"))
        XCTAssertTrue(snapshot.nodes.contains { $0.role == "textbox" && $0.reference != nil })
    }

    func testSnapshotWrapsRolesNamesBoundsAndActionableReferences() async throws {
        let fixture = try fixtureURL("dialogs")
        let page = try await makePage(fixture: fixture)
        defer { page.service.invalidate(); page.webView.stopLoading() }

        let snapshot = try await page.service.snapshot()
        let save = try XCTUnwrap(snapshot.nodes.first { $0.name == "Save" }, "Snapshot names: \(snapshot.nodes.map(\.name))")
        XCTAssertEqual(save.role, "button")
        XCTAssertNotNil(save.bounds)
        XCTAssertNotNil(save.reference)
        XCTAssertTrue(snapshot.nodes.contains { $0.role == "dialog" && $0.name == "Confirm changes" })
    }

    func testOversizeSnapshotHasAnExplicitTruncationMarker() async throws {
        let fixture = try fixtureURL("forms")
        let page = try await makePage(fixture: fixture, maximumNodes: 3, maximumBytes: 2_000)
        defer { page.service.invalidate(); page.webView.stopLoading() }

        let snapshot = try await page.service.snapshot()
        XCTAssertTrue(snapshot.truncated)
        XCTAssertLessThanOrEqual(snapshot.nodes.count, 3)
        XCTAssertEqual(snapshot.version, "ts_snapshot_v1")
    }

    func testPageTextIsReturnedAsDataAndCannotExecuteAsInstructions() async throws {
        let fixture = try fixtureURL("credential-fields")
        let page = try await makePage(fixture: fixture)
        defer { page.service.invalidate(); page.webView.stopLoading() }

        let snapshot = try await page.service.snapshot()
        XCTAssertTrue(snapshot.nodes.contains { $0.name == "<script>window.__pagePayloadRan=true</script>" })
        let pageScriptRan = try await page.webView.evaluateJavaScript("window.__pagePayloadRan === true") as? Bool
        XCTAssertNotEqual(pageScriptRan, true)
    }

    func testPageWorldCannotReachTheIsolatedSnapshotHandler() async throws {
        let fixture = try fixtureURL("forms")
        let page = try await makePage(fixture: fixture)
        defer { page.service.invalidate(); page.webView.stopLoading() }

        let pageWorldHandler = try await page.webView.evaluateJavaScript(
            "typeof window.webkit?.messageHandlers?.turboSparkDOMSnapshot"
        ) as? String
        XCTAssertEqual(pageWorldHandler, "undefined")
    }

    func testSnapshotRejectsUnexpectedOriginAndInactiveGeneration() async throws {
        let fixture = try fixtureURL("forms")
        let page = try await makePage(fixture: fixture, expectedOrigin: "https://unexpected.invalid")
        defer { page.service.invalidate(); page.webView.stopLoading() }

        do {
            _ = try await page.service.snapshot()
            XCTFail("Expected a message from the wrong origin to be rejected")
        } catch let error as DOMSnapshotServiceError {
            XCTAssertEqual(error, .rejectedMessage)
        }

        page.service.navigationStarted()
        do {
            _ = try await page.service.snapshot()
            XCTFail("Expected an inactive document generation to be rejected")
        } catch let error as DOMSnapshotServiceError {
            XCTAssertEqual(error, .inactiveDocument)
        }
    }

    func testInvalidateDisconnectsTheBridgeHandler() async throws {
        let fixture = try fixtureURL("forms")
        let page = try await makePage(fixture: fixture)
        defer { page.webView.stopLoading() }

        _ = try await page.service.snapshot()
        page.service.invalidate()
        do {
            _ = try await page.service.snapshot()
            XCTFail("Expected an invalidated bridge to reject further snapshots")
        } catch let error as DOMSnapshotServiceError {
            XCTAssertEqual(error, .unavailable)
        }
    }

    func testActiveGenerationGuardRejectsNavigationAndInvalidation() {
        let configuration = WKWebViewConfiguration()
        let webView = WKWebView(frame: .zero, configuration: configuration)
        let service = DOMSnapshotService(webView: webView, expectedOrigin: "https://example.invalid")
        defer { service.invalidate() }

        let firstGeneration = service.navigationStarted()
        service.navigationCommitted(generation: firstGeneration)
        XCTAssertTrue(service.isActive(generation: firstGeneration))

        let secondGeneration = service.navigationStarted()
        XCTAssertFalse(service.isActive(generation: firstGeneration))
        XCTAssertFalse(service.isActive(generation: secondGeneration))
        service.navigationCommitted(generation: secondGeneration)
        XCTAssertTrue(service.isActive(generation: secondGeneration))

        service.invalidate()
        XCTAssertFalse(service.isActive(generation: secondGeneration))
    }

    private struct LoadedPage {
        let webView: WKWebView
        let service: DOMSnapshotService
        let navigationObserver: SnapshotNavigationObserver
    }

    private func makePage(
        fixture: URL,
        expectedOrigin: String = "file://",
        maximumNodes: Int = 500,
        maximumBytes: Int = 64_000
    ) async throws -> LoadedPage {
        let configuration = WKWebViewConfiguration()
        configuration.websiteDataStore = .nonPersistent()
        let webView = WKWebView(frame: CGRect(x: 0, y: 0, width: 900, height: 700), configuration: configuration)
        let service = DOMSnapshotService(
            webView: webView,
            expectedOrigin: expectedOrigin,
            maximumNodes: maximumNodes,
            maximumBytes: maximumBytes
        )
        let observer = SnapshotNavigationObserver(service: service)
        webView.navigationDelegate = observer
        let loaded = expectation(description: "Fixture committed")
        observer.onCommit = { loaded.fulfill() }
        webView.loadFileURL(fixture, allowingReadAccessTo: fixture.deletingLastPathComponent())
        await fulfillment(of: [loaded], timeout: 10)
        return LoadedPage(webView: webView, service: service, navigationObserver: observer)
    }

    private func fixtureURL(_ name: String) throws -> URL {
        try XCTUnwrap(DOMSnapshotFixtures.url(named: name))
    }

    private func node(named name: String, in snapshot: DOMSnapshotEnvelope) -> DOMSnapshotNode {
        guard let node = snapshot.nodes.first(where: { $0.name == name }) else {
            XCTFail("Snapshot did not contain node named \(name)")
            return DOMSnapshotNode(role: "", name: "", parentIndex: nil, bounds: nil, reference: nil)
        }
        return node
    }
}

@MainActor
private final class SnapshotNavigationObserver: NSObject, WKNavigationDelegate {
    private let service: DOMSnapshotService
    private var currentGeneration: UUID?
    var onCommit: (() -> Void)?

    init(service: DOMSnapshotService) {
        self.service = service
    }

    func webView(_ webView: WKWebView, didStartProvisionalNavigation navigation: WKNavigation!) {
        currentGeneration = service.navigationStarted()
    }

    func webView(_ webView: WKWebView, didCommit navigation: WKNavigation!) {
        if let currentGeneration {
            service.navigationCommitted(generation: currentGeneration)
        }
        onCommit?()
        onCommit = nil
    }
}
