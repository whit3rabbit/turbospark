import AppKit
import Foundation

extension AppModel {
    /// Whether the sources popover can offer "Open in browser pane": the
    /// pane needs the browser feature on (Settings > Browser).
    var canOpenSourcesInBrowserPane: Bool { browserSettings.enabled }

    /// Opens a source in the browser pane as the USER (OC-09). The click is
    /// direct user input, so this goes through `navigateAsUser`, the same
    /// path as typing an address in the pane, and not through
    /// `authorizeBrowserNavigation`, which gates agent navigation. Only
    /// http and https open (`TurnSource` drops other schemes at collection,
    /// and this checks again because the URL string is data from a tool).
    ///
    /// A closed pane opens on a NEW user tab (its model starts on the
    /// store's active tab). An open pane navigates the tab the user is
    /// looking at, because the pane's own selection does not follow a tab
    /// created behind it.
    @discardableResult
    func openSourceInBrowserPane(_ source: TurnSource) -> Bool {
        guard browserSettings.enabled,
              let url = URL(string: source.url),
              TurnSource.isOpenableScheme(url.scheme)
        else { return false }
        let engine = browserAutomationCoordinator.engine
        do {
            if browserPaneIsOpen, let active = engine.tabStore.activeTabID {
                try engine.navigateAsUser(to: url, in: active)
            } else {
                let tabID = engine.createTab(owner: .user)
                try engine.navigateAsUser(to: url, in: tabID)
                setBrowserPaneOpen(true)
            }
            return true
        } catch {
            showToast(AppStrings.localized("Could not open this source in the browser pane."), style: .warning)
            return false
        }
    }

    /// Copies a source URL. The URL is what the tool returned and nothing
    /// is fetched.
    func copySourceURL(_ source: TurnSource) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(source.url, forType: .string)
        showToast(AppStrings.localized("Copied URL"))
    }
}
