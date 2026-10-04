import XCTest
@testable import TurboSparkApp

final class BrowserSettingsPaneViewModelTests: XCTestCase {
    @MainActor
    func testFirstEnableRequiresConfirmationAndRecordsOnboardingOnce() {
        var writes: [BrowserSettings] = []
        let viewModel = BrowserSettingsPaneViewModel(settings: BrowserSettings()) {
            writes.append($0)
        }

        XCTAssertEqual(viewModel.requestEnabled(true), .onboardingRequired)
        XCTAssertFalse(viewModel.settings.enabled)
        XCTAssertFalse(viewModel.settings.onboardingDone)
        XCTAssertTrue(writes.isEmpty)

        viewModel.confirmFirstEnable()
        XCTAssertTrue(viewModel.settings.enabled)
        XCTAssertTrue(viewModel.settings.onboardingDone)
        XCTAssertEqual(writes.count, 1)
        XCTAssertEqual(writes.first, viewModel.settings)

        viewModel.confirmFirstEnable()
        XCTAssertEqual(writes.count, 1)
        XCTAssertEqual(viewModel.requestEnabled(false), .changed)
        XCTAssertEqual(viewModel.requestEnabled(true), .changed)
        XCTAssertTrue(viewModel.settings.enabled)
        XCTAssertTrue(viewModel.settings.onboardingDone)
        XCTAssertEqual(writes.count, 3)
    }

    @MainActor
    func testCancellingFirstEnableLeavesFeatureDisabledAndUnrecorded() {
        var writeCount = 0
        let viewModel = BrowserSettingsPaneViewModel(settings: BrowserSettings()) { _ in
            writeCount += 1
        }

        XCTAssertEqual(viewModel.requestEnabled(true), .onboardingRequired)
        viewModel.cancelFirstEnable()
        viewModel.confirmFirstEnable()

        XCTAssertFalse(viewModel.settings.enabled)
        XCTAssertFalse(viewModel.settings.onboardingDone)
        XCTAssertEqual(writeCount, 0)
        XCTAssertEqual(viewModel.requestEnabled(true), .onboardingRequired)
    }

    @MainActor
    func testDialogPolicyAndDefaultViewportRoundTripThroughTheViewModel() {
        var writes: [BrowserSettings] = []
        let viewModel = BrowserSettingsPaneViewModel(settings: BrowserSettings()) {
            writes.append($0)
        }

        viewModel.setDialogPolicy(.autoDismiss)
        viewModel.setDefaultViewport(width: 1440, height: 900, zoom: 1.25)

        XCTAssertEqual(viewModel.settings.dialogPolicy, .autoDismiss)
        XCTAssertEqual(
            viewModel.settings.defaultViewport,
            BrowserViewportPreference(width: 1440, height: 900, zoom: 1.25)
        )
        XCTAssertEqual(writes.last, viewModel.settings)
        XCTAssertEqual(writes.count, 2)
    }

    @MainActor
    func testProjectOriginAddCanonicalizesAndRevocationRemovesTheGrant() {
        var changes: [(AppToolPermission, [String])] = []
        let viewModel = BrowserProjectPermissionsSettingsViewModel(
            permission: .ask,
            originAllowlist: []
        ) { permission, origins in
            changes.append((permission, origins))
        }

        viewModel.originDraft = "HTTPS://Example.COM:443"
        XCTAssertEqual(viewModel.addOrigin(), .added)
        XCTAssertEqual(viewModel.originAllowlist, ["https://example.com"])
        XCTAssertNil(viewModel.feedback)
        XCTAssertEqual(changes.count, 1)

        XCTAssertTrue(viewModel.revokeOrigin("https://example.com"))
        XCTAssertTrue(viewModel.originAllowlist.isEmpty)
        XCTAssertNil(viewModel.feedback)
        XCTAssertEqual(changes.count, 2)
    }

    @MainActor
    func testInvalidAndDuplicateOriginEditsDoNotChangeTheAllowlist() {
        let viewModel = BrowserProjectPermissionsSettingsViewModel(
            permission: .ask,
            originAllowlist: ["https://example.com"]
        )

        viewModel.originDraft = "file:///tmp/page.html"
        XCTAssertEqual(viewModel.addOrigin(), .invalidOrigin)
        XCTAssertEqual(viewModel.feedback, .invalidOrigin)
        XCTAssertEqual(viewModel.originAllowlist, ["https://example.com"])

        viewModel.originDraft = "HTTPS://EXAMPLE.COM:443"
        XCTAssertEqual(viewModel.addOrigin(), .alreadyGranted)
        XCTAssertEqual(viewModel.feedback, .alreadyGranted)
        XCTAssertEqual(viewModel.originAllowlist, ["https://example.com"])
    }

    @MainActor
    func testGrantCapRefusesAnotherOriginWithoutDroppingExistingGrants() {
        let initialOrigins = (0..<BrowserPermissionRuleStore.maximumGrantCount).map {
            "https://site\($0).example"
        }
        let viewModel = BrowserProjectPermissionsSettingsViewModel(
            permission: .ask,
            originAllowlist: initialOrigins
        )

        viewModel.originDraft = "https://new.example"
        XCTAssertEqual(viewModel.addOrigin(), .limitReached)
        XCTAssertEqual(viewModel.feedback, .limitReached)
        XCTAssertEqual(viewModel.originAllowlist, initialOrigins)

        XCTAssertTrue(viewModel.revokeOrigin(initialOrigins[0]))
        viewModel.originDraft = "https://new.example"
        XCTAssertEqual(viewModel.addOrigin(), .added)
        XCTAssertEqual(viewModel.originAllowlist.count, BrowserPermissionRuleStore.maximumGrantCount)
        XCTAssertFalse(viewModel.originAllowlist.contains(initialOrigins[0]))
        XCTAssertTrue(viewModel.originAllowlist.contains("https://new.example"))
    }

    @MainActor
    func testProjectBrowserCategorySelectionIsPersistedWithTheOriginState() {
        var changedPermission: AppToolPermission?
        var changedOrigins: [String]?
        let viewModel = BrowserProjectPermissionsSettingsViewModel(
            permission: .ask,
            originAllowlist: ["https://example.com"]
        ) { permission, origins in
            changedPermission = permission
            changedOrigins = origins
        }

        viewModel.setPermission(.deny)

        XCTAssertEqual(viewModel.permission, .deny)
        XCTAssertEqual(changedPermission, .deny)
        XCTAssertEqual(changedOrigins, ["https://example.com"])
    }
}
