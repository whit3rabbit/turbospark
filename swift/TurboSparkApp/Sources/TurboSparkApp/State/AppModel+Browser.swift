import Foundation

extension AppModel {
    func browserToolAvailability(
        mediaCapability: AppToolMediaCapability? = nil
    ) -> BrowserToolAvailability {
        let currentMediaCapability = mediaCapability ?? {
            guard let vision = session?.info.vision, vision.active else { return .textOnly }
            return .imageBearingToolResults(maximumPixelCount: vision.maxPixels)
        }()
        return BrowserToolAvailability(
            isEnabled: browserSettings.enabled,
            manifest: WebKitBrowserControlPort.manifest,
            mediaCapability: currentMediaCapability
        )
    }

    public func setBrowserPaneOpen(_ isOpen: Bool) {
        if isOpen {
            guard browserSettings.enabled else { return }
            browserPaneIsOpen = true
            browserAutomationCoordinator.openPane()
            return
        }

        let wasOpen = browserPaneIsOpen
        browserPaneIsOpen = false
        guard wasOpen else { return }
        let coordinator = browserAutomationCoordinator
        coordinator.paneWillClose()
        Task {
            // A reopen can precede this queued teardown on MainActor.
            guard !coordinator.isPaneOpen else { return }
            await coordinator.closePane()
        }
    }

    func authorizeBrowserNavigation(
        _ request: BrowserNavigationAuthorizationRequest
    ) async -> BrowserNavigationAuthorizationDecision {
        guard browserSettings.enabled,
              let action = browserAutomationCoordinator.activeAction,
              let project = action.project,
              browserAutomationCoordinator.tabStore.tab(id: request.tabID)?.owner == BrowserTabOwner.agent
        else {
            return .deny
        }

        let arguments = ["url": request.destination.canonicalString]
        let navigationCall = AppToolCall(
            id: action.call.id,
            approvalID: action.call.approvalID,
            name: "browser_navigate",
            arguments: arguments,
            category: .browser,
            riskAssessment: ToolRiskClassifier.assessRisk(
                name: "browser_navigate",
                arguments: arguments
            )
        )
        let actionWasApprovedForDestination = action.permissionContext.currentActionApproved
            && action.permissionContext.origin == request.destination
        let context = BrowserPermissionContext(
            origin: request.destination,
            owner: .agent,
            currentActionApproved: actionWasApprovedForDestination
        )

        switch AppToolPermissionEngine.evaluate(
            call: navigationCall,
            project: project,
            sessionApproved: false,
            globalServers: [],
            browserContext: context
        ) {
        case .allow:
            return .allow
        case .ask, .deny:
            return .deny
        }
    }
}
