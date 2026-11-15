import Foundation

extension AppModel {
    /// Builds the page host with the app's real wiring. A page's tool call
    /// takes the same path as a model-issued deferred MCP call
    /// (`executeDeferredMcpCall`): PreToolUse hooks, deny rules, session
    /// approvals and the permission engine. That path cannot show an approval
    /// card from a page, so an `ask` verdict arrives as a refusal and the
    /// page sees `permission_denied` (fail closed).
    func makePluginUiHost() -> PluginUiHost {
        pluginUiHostIsLive = true
        return PluginUiHost(PluginUiHost.Dependencies(
            optIn: PluginUiOptInStore(),
            plugin: { [weak self] id in
                guard let self else { return nil }
                let projectURL = self.selectedProject?.rootDirectoryURL
                return PluginManager.shared.enabledPlugins(projectURL: projectURL).first { $0.id == id }
            },
            invoker: { [weak self] _, sessionID in
                let project = self?.selectedProject
                let chatID = UUID(uuidString: sessionID)
                return { name, arguments in
                    do {
                        return try await AppToolRegistry.executeDeferredMcpCall(
                            name: name, arguments: arguments, project: project, chatID: chatID)
                    } catch let error as AppToolRegistry.DeferredMcpContinuationStop {
                        throw PageToolError.denied(error.reason)
                    } catch let error as NSError where error.domain == "TurboSparkToolSearch" && [6, 7, 8].contains(error.code) {
                        throw PageToolError.denied(error.localizedDescription)
                    } catch {
                        throw PageToolError.failed(error.localizedDescription)
                    }
                }
            },
            draftSink: { [weak self] text in self?.composeInChat(text) },
            onSidePaneChange: { [weak self] id in self?.openPluginPageID = id }))
    }

    /// Opens a plugin surface in the right column. The right column is one
    /// slot, so the other claimants yield first (`AppRightColumnClaimant`).
    @discardableResult
    func openPluginPage(pluginID: String, surfaceID: String) -> PluginUiOpenRefusal? {
        switch pluginUiHost.open(pluginID: pluginID, surfaceID: surfaceID, sessionID: selectedChatID.uuidString) {
        case .failure(let refusal):
            return refusal
        case .success:
            openArtifactID = nil
            htmlPreview = nil
            previewAttachmentID = nil
            setBrowserPaneOpen(false)
            setWhiteboardPaneOpen(false)
            return nil
        }
    }

    func closePluginPage(_ session: PluginPageSession) {
        pluginUiHost.close(session)
    }
}

extension AppModel {
    func setPluginUiOptIn(_ enabled: Bool, plugin: LoadedPlugin) {
        pluginUiHost.optIn.setEnabled(enabled, pluginID: plugin.id)
        pluginUiHost.reconcile()
    }

    /// The Open button: opens the page or says why it did not.
    func openPluginPageReportingRefusal(pluginID: String, surfaceID: String) {
        guard let refusal = openPluginPage(pluginID: pluginID, surfaceID: surfaceID) else { return }
        switch refusal {
        case .notOptedIn:
            showToast(AppStrings.localized("Turn on UI pages for this plugin first."), style: .warning)
        case .pluginUnavailable:
            showToast(AppStrings.localized("This plugin is disabled or no longer installed."), style: .warning)
        case .noSuchSurface:
            showToast(AppStrings.localized("The plugin no longer declares this page."), style: .warning)
        case .pageLimitReached:
            showToast(AppStrings.localized("Too many plugin pages are open. Close one and try again."), style: .warning)
        }
    }
}
