import Foundation

struct BrowserPermissionApprovalRequest: Equatable, Identifiable, Sendable {
    let callID: UUID
    let projectID: UUID
    let toolName: String
    let origin: BrowserOrigin

    var id: UUID { callID }

    static func make(
        for call: AppToolCall,
        project: AppProject?,
        currentOrigin: BrowserOrigin?
    ) -> BrowserPermissionApprovalRequest? {
        guard call.category == .browser,
              BrowserToolDefinitions.commandKind(for: call.name.lowercased()) != nil,
              let project
        else {
            return nil
        }

        let origin: BrowserOrigin?
        if call.name.lowercased() == "browser_navigate",
           let rawURL = call.arguments["url"],
           let url = URL(string: rawURL) {
            origin = BrowserOrigin(url: url)
        } else {
            origin = currentOrigin
        }

        guard let origin else { return nil }
        return BrowserPermissionApprovalRequest(
            callID: call.id,
            projectID: project.id,
            toolName: call.name,
            origin: origin
        )
    }
}

enum BrowserPermissionApprovalActionResult: Equatable {
    case approved
    case unavailable
    case grantRefused(BrowserPermissionGrantResult)
}

/// Holds a temporary allow for exactly one tool call, project, and origin.
@MainActor
final class BrowserPermissionApprovalCoordinator {
    private struct Scope: Equatable {
        let projectID: UUID
        let origin: BrowserOrigin
    }

    private var oneTimeApprovals: [UUID: Scope] = [:]

    func authorizeOnce(_ request: BrowserPermissionApprovalRequest) {
        oneTimeApprovals[request.callID] = Scope(
            projectID: request.projectID,
            origin: request.origin
        )
    }

    func allowsOnce(callID: UUID, projectID: UUID, origin: BrowserOrigin) -> Bool {
        oneTimeApprovals[callID] == Scope(projectID: projectID, origin: origin)
    }

    func cancel(callID: UUID) {
        oneTimeApprovals.removeValue(forKey: callID)
    }
}

extension AppModel {
    func browserPermissionContext(
        for call: AppToolCall,
        project: AppProject?
    ) -> BrowserPermissionContext? {
        guard call.category == .browser else { return nil }
        let origin: BrowserOrigin?
        if call.name.lowercased() == "browser_navigate",
           let rawURL = call.arguments["url"],
           let url = URL(string: rawURL) {
            origin = BrowserOrigin(url: url)
        } else {
            // Side-effect free on purpose: this runs while SwiftUI evaluates
            // approval cards, so it must not go through the registry provider,
            // which opens the browser pane and publishes AppModel state.
            origin = browserAutomationCoordinator.runtime(
                for: call,
                project: project,
                availability: browserToolAvailability()
            )?.permissionContext?.origin
        }
        return BrowserPermissionContext(origin: origin)
    }

    func browserPermissionApprovalRequest(for call: AppToolCall) -> BrowserPermissionApprovalRequest? {
        let projectID = pendingToolCallProject?.id
        guard let projectID,
              let project = projects.first(where: { $0.id == projectID }) ?? pendingToolCallProject
        else {
            return nil
        }
        let currentOrigin = browserPermissionContext(for: call, project: project)?.origin
        return BrowserPermissionApprovalRequest.make(
            for: call,
            project: project,
            currentOrigin: currentOrigin
        )
    }

    @discardableResult
    func approvePendingBrowserToolCall(
        id: UUID,
        alwaysAllowOrigin: Bool
    ) -> BrowserPermissionApprovalActionResult {
        guard !generating, !submitting,
              let call = pendingToolCall, call.id == id,
              let projectID = pendingToolCallProject?.id,
              let project = projects.first(where: { $0.id == projectID }) ?? pendingToolCallProject,
              let request = browserPermissionApprovalRequest(for: call)
        else {
            return .unavailable
        }

        if alwaysAllowOrigin {
            var updated = project
            let grantResult = BrowserPermissionRuleStore.grant(
                origin: request.origin.canonicalString,
                to: &updated.permissions
            )
            guard grantResult == .added || grantResult == .alreadyGranted else {
                return .grantRefused(grantResult)
            }
            updateProject(updated)
            pendingToolCallProject = updated
        }

        browserPermissionApprovalCoordinator.authorizeOnce(request)
        approvePendingToolCall(id: id, alwaysAllowSession: false)
        return .approved
    }

    func denyPendingBrowserToolCall(id: UUID) {
        browserPermissionApprovalCoordinator.cancel(callID: id)
        denyPendingToolCall(id: id)
    }
}
