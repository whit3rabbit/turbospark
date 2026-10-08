import Foundation

extension AppModel {
    /// Recomputes the untrusted project tools for `candidate` (default: the
    /// selected project). Called wherever `detectProjectMcpServers` is: on
    /// project creation and selection. The list is replaced, not appended,
    /// so a tool deleted or already trusted drops out.
    public func refreshPendingProjectTools(for candidate: AppProject? = nil) {
        guard let project = candidate ?? selectedProject,
              let root = project.rootDirectoryURL
        else {
            pendingProjectToolApprovals = []
            return
        }
        pendingProjectToolApprovals = ProjectToolApproval.pending(
            for: root, rejected: rejectedProjectToolIDs)
        // The effective tool list is cached by the catalog and the prompt;
        // nothing to invalidate here because approval only adds tools and
        // `resolveEffectiveTools` re-reads trust on every call.
    }

    /// Explicit per-tool approval. A definition that changed since it was
    /// shown is not trusted; the list is refreshed so the new text is seen.
    public func approveProjectTool(_ review: ProjectToolReview) {
        guard let root = selectedProject?.rootDirectoryURL else { return }
        _ = ProjectToolApproval.approve(review, projectURL: root)
        refreshPendingProjectTools()
    }

    /// Reject leaves the tool untrusted (it never ran) and hides it for the
    /// rest of the session.
    public func rejectProjectTool(_ review: ProjectToolReview) {
        rejectedProjectToolIDs.insert(review.id)
        refreshPendingProjectTools()
    }
}
