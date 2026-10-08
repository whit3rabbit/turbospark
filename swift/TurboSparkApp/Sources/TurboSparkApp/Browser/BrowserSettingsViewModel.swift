import Foundation
import SwiftUI

enum BrowserFeatureEnableRequest: Equatable {
    case changed
    case unchanged
    case onboardingRequired
}

/// Owns first-enable onboarding and the profile-wide browser preferences.
@MainActor
final class BrowserSettingsPaneViewModel: ObservableObject {
    @Published private(set) var settings: BrowserSettings

    private let onChange: (BrowserSettings) -> Void
    /// The owner's live value. Edits are applied on top of it rather than on
    /// this view model's own copy, which goes stale when something else (the
    /// Chrome bookmark import) changes the settings while the pane is open;
    /// writing the stale copy back used to wipe the imported bookmarks.
    private let latest: (() -> BrowserSettings)?
    private var awaitingFirstEnableConfirmation = false

    init(
        settings: BrowserSettings,
        latest: (() -> BrowserSettings)? = nil,
        onChange: @escaping (BrowserSettings) -> Void = { _ in }
    ) {
        self.settings = settings
        self.latest = latest
        self.onChange = onChange
    }

    @discardableResult
    func requestEnabled(_ enabled: Bool) -> BrowserFeatureEnableRequest {
        if enabled && !settings.onboardingDone {
            awaitingFirstEnableConfirmation = true
            return .onboardingRequired
        }

        awaitingFirstEnableConfirmation = false
        guard settings.enabled != enabled else { return .unchanged }
        update { $0.enabled = enabled }
        return .changed
    }

    func confirmFirstEnable() {
        guard awaitingFirstEnableConfirmation, !settings.onboardingDone else { return }
        awaitingFirstEnableConfirmation = false
        update {
            $0.enabled = true
            $0.onboardingDone = true
        }
    }

    func cancelFirstEnable() {
        awaitingFirstEnableConfirmation = false
    }

    func setDialogPolicy(_ policy: BrowserDialogPolicy) {
        guard settings.dialogPolicy != policy else { return }
        update { $0.dialogPolicy = policy }
    }

    func setDefaultViewport(width: Int, height: Int, zoom: Double) {
        let viewport = BrowserViewportPreference(width: width, height: height, zoom: zoom)
        guard settings.defaultViewport != viewport else { return }
        update { $0.defaultViewport = viewport }
    }

    private func update(_ mutate: (inout BrowserSettings) -> Void) {
        let base = latest?() ?? settings
        var updated = base
        mutate(&updated)
        guard updated != base else { return }
        settings = updated
        onChange(updated)
    }
}

enum BrowserOriginSettingsFeedback: Equatable {
    case alreadyGranted
    case invalidOrigin
    case limitReached

    var localizationKey: String {
        switch self {
        case .alreadyGranted: return "Origin already allowed."
        case .invalidOrigin: return "Enter a valid HTTP(S) origin."
        case .limitReached: return "Project limit reached (256 origins). Remove one before adding another."
        }
    }
}

/// Keeps project browser permission edits on the same canonical grant store as runtime checks.
@MainActor
final class BrowserProjectPermissionsSettingsViewModel: ObservableObject {
    @Published private(set) var permission: AppToolPermission
    @Published private(set) var originAllowlist: [String]
    @Published var originDraft = ""
    @Published private(set) var feedback: BrowserOriginSettingsFeedback?

    private let onChange: (AppToolPermission, [String]) -> Void

    init(
        permission: AppToolPermission,
        originAllowlist: [String],
        onChange: @escaping (AppToolPermission, [String]) -> Void = { _, _ in }
    ) {
        self.permission = permission
        self.originAllowlist = BrowserPermissionRuleStore.normalizedAllowlist(originAllowlist)
        self.onChange = onChange
    }

    func setPermission(_ permission: AppToolPermission) {
        guard self.permission != permission else { return }
        self.permission = permission
        feedback = nil
        onChange(permission, originAllowlist)
    }

    @discardableResult
    func addOrigin() -> BrowserPermissionGrantResult {
        var permissions = AppProjectPermissions(
            browser: permission,
            browserOriginAllowlist: originAllowlist
        )
        let result = BrowserPermissionRuleStore.grant(origin: originDraft, to: &permissions)
        switch result {
        case .added:
            originAllowlist = permissions.browserOriginAllowlist
            originDraft = ""
            feedback = nil
            onChange(permission, originAllowlist)
        case .alreadyGranted:
            feedback = .alreadyGranted
        case .invalidOrigin:
            feedback = .invalidOrigin
        case .limitReached:
            feedback = .limitReached
        }
        return result
    }

    @discardableResult
    func revokeOrigin(_ rawOrigin: String) -> Bool {
        guard let origin = BrowserOrigin(origin: rawOrigin) else {
            feedback = .invalidOrigin
            return false
        }

        var permissions = AppProjectPermissions(
            browser: permission,
            browserOriginAllowlist: originAllowlist
        )
        let revoked = BrowserPermissionRuleStore.revoke(origin: origin, from: &permissions)
        guard revoked else { return false }

        originAllowlist = permissions.browserOriginAllowlist
        feedback = nil
        onChange(permission, originAllowlist)
        return true
    }
}
