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

    /// Brings the tool card of the call that first produced `source` into
    /// view (OC-09). Opens the turn if it was collapsed, then publishes a
    /// `ToolCardReveal` that the transcript scrolls to and the card (and its
    /// group) expand for. False, with a toast, when the call is not in the
    /// selected chat's transcript any more (compacted away or a stale source).
    @discardableResult
    func revealToolCard(for source: TurnSource) -> Bool {
        guard let origin = source.origin else {
            showToast(AppStrings.localized("That tool call is no longer in the transcript."), style: .warning)
            return false
        }
        return revealToolCard(
            at: origin,
            announcement: AppStrings.localized("Showing the tool call that found this source."))
    }

    /// The same reveal for a file of the end-of-turn file card (component
    /// audit, file-tree): the call that last changed (or read) the file.
    @discardableResult
    func revealToolCard(for entry: TurnFileTree.Entry) -> Bool {
        revealToolCard(
            at: entry.origin,
            announcement: AppStrings.localized("Showing the tool call for this file."))
    }

    private func revealToolCard(at origin: TurnSourceOrigin, announcement: String) -> Bool {
        let messages = selectedTurnMessages
        guard messages.contains(where: { message in
                  message.id == origin.messageID
                      && message.toolCalls.contains { $0.id == origin.callID }
              })
        else {
            showToast(AppStrings.localized("That tool call is no longer in the transcript."), style: .warning)
            return false
        }
        if let anchor = TurnSources.turnAnchor(containing: origin.messageID, in: messages) {
            collapsedTurnAnchors.remove(anchor)
        }
        toolCardReveal = ToolCardReveal(
            messageID: origin.messageID,
            callID: origin.callID,
            token: (toolCardReveal?.token ?? 0) + 1)
        // The popover closes and focus does not move, so say where we went.
        _ = AccessibilityNotification.Announcement.post(.init(announcement))
        return true
    }

    /// Whether `reveal` has not been acted on by a card yet.
    func isToolCardRevealPending(_ reveal: ToolCardReveal) -> Bool {
        reveal.token > toolCardRevealConsumedToken
    }

    /// A card for `callID` takes the pending reveal: true exactly once per
    /// request, and only for the card the request names.
    func consumeToolCardReveal(forCallID callID: UUID) -> Bool {
        guard let reveal = toolCardReveal, reveal.callID == callID,
              isToolCardRevealPending(reveal)
        else { return false }
        toolCardRevealConsumedToken = reveal.token
        return true
    }
}
