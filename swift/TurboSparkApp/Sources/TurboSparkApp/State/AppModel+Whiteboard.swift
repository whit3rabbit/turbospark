import Foundation

// The whiteboard side pane (ZP-10). The flag lives on AppModel
// (`whiteboardPaneIsOpen`); this file owns opening, closing and the export
// handoff to the composer.

extension AppModel {
    /// The board bucket for the current selection: the project's id, or the
    /// shared "chat" bucket outside a project.
    var whiteboardWorkspaceKey: String {
        WhiteboardStore.workspaceKey(projectID: interactionMode == .projects ? selectedProjectID : nil)
    }

    /// Opening the whiteboard closes the browser pane (and the reverse): the
    /// right column is one slot, and a flag left set behind the winner would
    /// reappear when it closed.
    public func setWhiteboardPaneOpen(_ isOpen: Bool) {
        if isOpen {
            guard !activeSection.hidesChatPanes else { return }
            setBrowserPaneOpen(false)
            whiteboardPaneIsOpen = true
        } else {
            whiteboardPaneIsOpen = false
            // Closing is the moment a stroke inside the debounce window would
            // otherwise wait on a timer.
            whiteboardStoreIfCreated?.flush()
        }
    }

    /// Flushes any board write still queued. Called from `shutdown()`.
    func flushWhiteboardsForShutdown() {
        whiteboardStoreIfCreated?.flush()
    }

    /// Attaches a board's PNG export to the FOCUSED chat's composer (the chat
    /// the user is looking at, `selectedChatID`), the same seam an annotated
    /// image uses. The board itself is untouched.
    ///
    /// Images reach the model only through a vision-capable install. The PNG
    /// still attaches (the user may switch models before sending), but the
    /// toast says why a send would be refused instead of claiming success.
    func attachWhiteboardExport(_ png: Data, boardTitle: String) {
        let name = WhiteboardExporter.fileName(forBoardTitle: boardTitle)
        let attached = attachPNGToPrompt(png, fileName: name) { detail in
            AppStrings.localized("Could not attach the whiteboard image: \(detail)")
        }
        guard attached else { return }
        if visionIsActive {
            showToast(AppStrings.localized("Whiteboard image attached to the prompt."), style: .success)
        } else {
            let reason = visionRefusalReason
                ?? AppStrings.localized("the loaded model has no active vision tower")
            showToast(AppStrings.localized("Cannot send this image: \(reason)"), style: .warning)
        }
    }
}
