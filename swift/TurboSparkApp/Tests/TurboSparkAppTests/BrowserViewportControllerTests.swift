import XCTest
import WebKit
@testable import TurboSparkApp

@MainActor
final class BrowserViewportControllerTests: XCTestCase {
    func testModeSwitchUsesFixedDimensionsOrTracksResizedPane() {
        var storedPreference = BrowserViewportPreference(width: 1100, height: 720, zoom: 1.25)
        let controller = BrowserViewportController(
            readPreference: { storedPreference },
            writePreference: { storedPreference = $0 }
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
        var storedPreference = BrowserViewportPreference(width: 1100, height: 720, zoom: 1.25)
        let controller = BrowserViewportController(
            readPreference: { storedPreference },
            writePreference: { storedPreference = $0 }
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

    func testUpdatedPreferenceIsRestoredByANewController() {
        var storedPreference = BrowserViewportPreference.standard
        var writeCount = 0
        let firstController = BrowserViewportController(
            readPreference: { storedPreference },
            writePreference: {
                storedPreference = $0
                writeCount += 1
            }
        )

        firstController.updatePreference(width: 1360, height: 900, zoom: 1.2)

        let relaunchedController = BrowserViewportController(
            readPreference: { storedPreference },
            writePreference: { storedPreference = $0 }
        )
        XCTAssertEqual(
            relaunchedController.preference,
            BrowserViewportPreference(width: 1360, height: 900, zoom: 1.2)
        )
        XCTAssertEqual(writeCount, 1)
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
