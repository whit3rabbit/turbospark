import XCTest
import WebKit
@testable import TurboSparkApp

@MainActor
final class BrowserViewportControllerTests: XCTestCase {
    func testModeSwitchUsesFixedDimensionsOrTracksResizedPane() {
        let controller = BrowserViewportController(
            initialPreference: BrowserViewportPreference(width: 1100, height: 720, zoom: 1.25)
        )

        let firstResponsiveLayout = controller.layout(in: CGSize(width: 800, height: 600))
        XCTAssertEqual(controller.mode, .responsive)
        XCTAssertEqual(firstResponsiveLayout.size, CGSize(width: 800, height: 600))
        XCTAssertEqual(firstResponsiveLayout.pageZoom, 1)

        controller.setMode(.fixed)
        let fixedLayout = controller.layout(in: CGSize(width: 800, height: 600))
        XCTAssertEqual(fixedLayout.size, CGSize(width: 1100, height: 720))
        XCTAssertEqual(fixedLayout.pageZoom, 1.25)

        controller.setMode(.responsive)
        let resizedLayout = controller.layout(in: CGSize(width: 960, height: 640))
        XCTAssertEqual(resizedLayout.size, CGSize(width: 960, height: 640))
        XCTAssertEqual(resizedLayout.pageZoom, 1)
    }

    func testChangingViewportRelayoutsTheSamePageWithoutReloading() async throws {
        let controller = BrowserViewportController(
            initialPreference: BrowserViewportPreference(width: 1100, height: 720, zoom: 1.25)
        )
        controller.setMode(.fixed)

        let webView = WKWebView(frame: CGRect(x: 0, y: 0, width: 800, height: 600))
        let recorder = BrowserViewportNavigationRecorder()
        webView.navigationDelegate = recorder
        let loaded = expectation(description: "initial page load completes")
        recorder.onFinish = { loaded.fulfill() }
        webView.loadHTMLString(
            "<html><head><meta name='viewport' content='width=device-width, initial-scale=1'></head><body>viewport fixture</body></html>",
            baseURL: nil
        )
        await fulfillment(of: [loaded], timeout: 5)

        let originalURL = webView.url
        let originalLoadCount = recorder.finishCount
        let initialWidthValue = try await webView.evaluateJavaScript("window.innerWidth")
        let initialCSSWidth = try XCTUnwrap(initialWidthValue as? Int)
        controller.apply(to: webView, containerSize: CGSize(width: 800, height: 600))
        let firstFixedWidthValue = try await webView.evaluateJavaScript("window.innerWidth")
        let firstFixedCSSWidth = try XCTUnwrap(firstFixedWidthValue as? Int)
        controller.updatePreference(width: 900, height: 640, zoom: 1.5)
        controller.apply(to: webView, containerSize: CGSize(width: 800, height: 600))
        let secondFixedWidthValue = try await webView.evaluateJavaScript("window.innerWidth")
        let secondFixedCSSWidth = try XCTUnwrap(secondFixedWidthValue as? Int)

        XCTAssertEqual(webView.frame.size, CGSize(width: 900, height: 640))
        XCTAssertEqual(webView.pageZoom, 1.5)
        XCTAssertNotEqual(firstFixedCSSWidth, initialCSSWidth)
        XCTAssertNotEqual(secondFixedCSSWidth, firstFixedCSSWidth)
        XCTAssertEqual(webView.url, originalURL)
        XCTAssertEqual(recorder.finishCount, originalLoadCount)
    }

    /// Settings > Browser > Default viewport used to be written and never
    /// read. Drives the real Settings edit path into AppModel and opens the
    /// pane's session the way `BrowserPaneView` does.
    func testSettingsDefaultViewportStartsEachSessionAndPopoverEditsStayInIt() {
        let appModel = AppModel()
        let settingsPane = BrowserSettingsPaneViewModel(
            settings: appModel.browserSettings, latest: { appModel.browserSettings }
        ) { appModel.browserSettings = $0 }
        settingsPane.setDefaultViewport(width: 1440, height: 900, zoom: 1.25)
        let chosen = BrowserViewportPreference(width: 1440, height: 900, zoom: 1.25)

        let session = BrowserPaneView.makeViewportController(appModel: appModel)
        XCTAssertEqual(session.preference, chosen)
        session.setMode(.fixed)
        XCTAssertEqual(
            session.layout(in: CGSize(width: 800, height: 600)),
            BrowserViewportLayout(size: CGSize(width: 1440, height: 900), pageZoom: 1.25)
        )

        // A popover edit changes this session only.
        session.updatePreference(width: 1024, height: 768, zoom: 1.5)
        XCTAssertEqual(session.preference, BrowserViewportPreference(width: 1024, height: 768, zoom: 1.5))
        XCTAssertEqual(appModel.browserSettings.defaultViewport, chosen)
        XCTAssertEqual(BrowserPaneView.makeViewportController(appModel: appModel).preference, chosen)
    }
}

@MainActor
private final class BrowserViewportNavigationRecorder: NSObject, WKNavigationDelegate {
    private(set) var finishCount = 0
    var onFinish: (() -> Void)?

    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        finishCount += 1
        onFinish?()
        onFinish = nil
    }
}
