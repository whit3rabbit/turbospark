import Foundation
import XCTest
@testable import TurboSparkApp

final class BrowserSettingsPersistenceTests: XCTestCase {
    func testBrowserSettingsRoundTripThroughProfileSettings() throws {
        var settings = MacAppSettings()
        settings.browser.enabled = true
        settings.browser.dialogPolicy = .autoDismiss
        settings.browser.defaultViewport = BrowserViewportPreference(width: 1440, height: 900, zoom: 1.25)
        settings.browser.viewportPreference = BrowserViewportPreference(width: 1100, height: 720, zoom: 1.1)
        settings.browser.onboardingDone = true

        let folder = BrowserBookmarkFolder(
            name: "Reading",
            bookmarks: [BrowserBookmark(title: "TurboSpark", url: "https://example.com")]
        )
        let tree = BrowserBookmarkTree(sourceProfileDirectoryLabel: "Profile 1", folders: [folder])
        XCTAssertTrue(settings.browser.replaceBookmarks([tree]))

        let data = try JSONEncoder().encode(settings)
        let decoded = try JSONDecoder().decode(MacAppSettings.self, from: data)

        XCTAssertEqual(decoded.browser, settings.browser)

        let root = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let browser = try XCTUnwrap(root["browser"] as? [String: Any])
        XCTAssertEqual(
            Set(browser.keys),
            Set(["enabled", "dialog_policy", "default_viewport", "onboarding_done", "bookmarks", "viewport_pref"])
        )
    }

    func testOldProfileDecodesWithDefaultsForEveryBrowserSetting() throws {
        let json = #"{"temperature": 0.5}"#
        let decoded = try JSONDecoder().decode(MacAppSettings.self, from: Data(json.utf8))

        XCTAssertEqual(decoded.temperature, 0.5)
        XCTAssertFalse(decoded.browser.enabled)
        XCTAssertEqual(decoded.browser.dialogPolicy, .ask)
        XCTAssertEqual(decoded.browser.defaultViewport, .standard)
        XCTAssertFalse(decoded.browser.onboardingDone)
        XCTAssertTrue(decoded.browser.bookmarks.isEmpty)
        XCTAssertEqual(decoded.browser.viewportPreference, .standard)
    }

    func testMalformedAndUnknownBrowserSettingsFallBackIndependently() throws {
        let json = #"{"browser":{"enabled":"yes","dialog_policy":"manual","default_viewport":{"width":0,"height":800,"zoom":1},"onboarding_done":"done","bookmarks":{"invalid":true},"viewport_pref":"wide"}}"#
        let decoded = try JSONDecoder().decode(MacAppSettings.self, from: Data(json.utf8))

        XCTAssertFalse(decoded.browser.enabled)
        XCTAssertEqual(decoded.browser.dialogPolicy, .ask)
        XCTAssertEqual(decoded.browser.defaultViewport, .standard)
        XCTAssertFalse(decoded.browser.onboardingDone)
        XCTAssertTrue(decoded.browser.bookmarks.isEmpty)
        XCTAssertEqual(decoded.browser.viewportPreference, .standard)
    }

    func testOversizedBookmarkTreesAreRejectedAndOversizedStoredTreesDefaultEmpty() throws {
        var settings = MacAppSettings()
        let existing = BrowserBookmarkTree(
            sourceProfileDirectoryLabel: "Profile 1",
            folders: [BrowserBookmarkFolder(name: "Saved")]
        )
        XCTAssertTrue(settings.browser.replaceBookmarks([existing]))

        let tooLarge = BrowserBookmarkFolder(
            name: String(repeating: "x", count: BrowserSettings.bookmarksSizeLimit + 1)
        )
        let oversizedTree = BrowserBookmarkTree(
            sourceProfileDirectoryLabel: "Profile 1",
            folders: [tooLarge]
        )
        XCTAssertFalse(settings.browser.replaceBookmarks([oversizedTree]))
        XCTAssertEqual(settings.browser.bookmarks, [existing])

        let treeData = try JSONEncoder().encode(oversizedTree)
        let treeObject = try XCTUnwrap(JSONSerialization.jsonObject(with: treeData) as? [String: Any])
        let oversizedProfile = try JSONSerialization.data(withJSONObject: [
            "browser": ["bookmarks": [treeObject]]
        ])
        let decoded = try JSONDecoder().decode(MacAppSettings.self, from: oversizedProfile)

        XCTAssertTrue(decoded.browser.bookmarks.isEmpty)
    }
}
