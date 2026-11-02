import Foundation
import ComputerUseKit

/// How much of an app the agent may drive once the user grants it.
public enum ComputerUseAppTier: String, Codable, Comparable, Sendable {
    /// Never granted, not even screenshots.
    case refused
    /// Screenshots and snapshots only.
    case read
    /// Read plus single or double left clicks and scrolling. No typing, keys,
    /// or right click (its context menu has Paste).
    case click
    case full

    private var rank: Int {
        switch self {
        case .refused: return 0
        case .read: return 1
        case .click: return 2
        case .full: return 3
        }
    }

    public static func < (lhs: Self, rhs: Self) -> Bool { lhs.rank < rhs.rank }
}

public enum ComputerUseAppPolicy {
    /// TurboSpark and its helper are refused because the approval cards live
    /// in TurboSpark: an agent that can click its own app can approve itself.
    /// System settings and credential stores are refused because one click
    /// there changes security state or reveals secrets.
    static let refusedBundleIDs: Set<String> = [
        "com.whit3rabbit.turbospark",
        "com.whit3rabbit.turbospark.computeruse",
        "com.apple.systempreferences",
        "com.apple.Settings",
        "com.apple.keychainaccess",
        "com.apple.Passwords",
        "com.apple.SecurityAgent",
        "com.apple.ScreenSharing",
        "com.1password.1password",
        "com.agilebits.onepassword7",
        "com.bitwarden.desktop",
        "com.lastpass.LastPass",
        "com.dashlane.dashlanephonefinal",
    ]

    /// Browsers are read-only: the in-app browser tools drive web pages with
    /// origin approvals, which this surface cannot enforce.
    static let readOnlyBundleIDs: Set<String> = [
        "com.apple.Safari",
        "com.google.Chrome",
        "com.google.Chrome.canary",
        "org.mozilla.firefox",
        "com.microsoft.edgemac",
        "company.thebrowser.Browser",
        "com.brave.Browser",
        "com.operasoftware.Opera",
        "com.vivaldi.Vivaldi",
        "org.chromium.Chromium",
    ]

    /// Terminals and editors are click-only: typing there runs code, and the
    /// agent already has its own permission-checked shell and file tools.
    static let clickOnlyBundleIDs: Set<String> = [
        "com.apple.Terminal",
        "com.googlecode.iterm2",
        "com.mitchellh.ghostty",
        "dev.warp.Warp-Stable",
        "net.kovidgoyal.kitty",
        "io.alacritty",
        "com.github.wez.wezterm",
        "com.microsoft.VSCode",
        "com.microsoft.VSCodeInsiders",
        "com.todesktop.230313mzl4w4u92",
        "dev.zed.Zed",
        "com.apple.dt.Xcode",
        "com.sublimetext.4",
    ]

    public static func tier(forBundleID bundleID: String) -> ComputerUseAppTier {
        if refusedBundleIDs.contains(bundleID) || bundleID.hasPrefix("com.whit3rabbit.turbospark") {
            return .refused
        }
        if readOnlyBundleIDs.contains(bundleID) { return .read }
        if clickOnlyBundleIDs.contains(bundleID) || bundleID.hasPrefix("com.jetbrains.") {
            return .click
        }
        return .full
    }

    /// Whether `tier` permits `command`. `listApps` needs no grant and is
    /// not routed here.
    public static func allows(_ command: ComputerUseCommand, tier: ComputerUseAppTier) -> Bool {
        switch command {
        case .listApps:
            return true
        case .snapshot, .screenshot:
            return tier >= .read
        case .openApp, .scroll:
            return tier >= .click
        case .click(_, _, let button, let count):
            if tier == .full { return true }
            return tier == .click && button == .left && (1...2).contains(count)
        case .type, .key:
            return tier == .full
        }
    }

    static func isPlausibleBundleID(_ value: String) -> Bool {
        guard (3...255).contains(value.count), value.contains(".") else { return false }
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: ".-_"))
        return value.unicodeScalars.allSatisfy(allowed.contains)
    }
}
