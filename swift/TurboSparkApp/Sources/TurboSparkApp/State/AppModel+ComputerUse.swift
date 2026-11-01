import Foundation
import ComputerUseKit

extension AppModel {
    /// Computer tools need the setting, a current helper manifest, and a
    /// model with active vision; `ComputerUseToolAvailability` hides all of
    /// them when any one is missing.
    func computerUseToolAvailability(
        mediaCapability: AppToolMediaCapability? = nil
    ) -> ComputerUseToolAvailability {
        let currentMediaCapability = mediaCapability ?? {
            guard let vision = session?.info.vision, vision.active else { return .textOnly }
            return .imageBearingToolResults(maximumPixelCount: vision.maxPixels)
        }()
        return ComputerUseToolAvailability(
            isEnabled: computerUseSettings.enabled,
            manifest: computerUseManifest,
            mediaCapability: currentMediaCapability)
    }

    /// Re-derives availability at execution, so a call parsed under a vision
    /// model is refused if the user has since switched to a text-only one.
    func computerUseToolRuntime(chatID: UUID?) -> ComputerUseToolRuntime? {
        guard let chatID, let port = computerUsePort else { return nil }
        let availability = computerUseToolAvailability()
        guard availability.isOffered else { return nil }
        return ComputerUseToolRuntime(
            availability: availability,
            session: computerUseSession(for: chatID),
            port: port)
    }

    func computerUseSession(for chatID: UUID) -> ComputerUseSession {
        if let existing = computerUseSessions[chatID] { return existing }
        let created = ComputerUseSession()
        computerUseSessions[chatID] = created
        return created
    }

    /// Drops every app grant. Called on model change and by the kill switch.
    public func revokeAllComputerUseGrants() {
        let sessions = Array(computerUseSessions.values)
        computerUseSessions.removeAll()
        Task {
            for session in sessions { await session.revokeAll() }
        }
    }
}

// MARK: - Helper lifecycle

extension AppModel {
    /// Turns the feature on or off. On launches the helper, which reads
    /// grant status without prompting; off shuts it down and drops grants.
    public func setComputerUseEnabled(_ enabled: Bool) {
        guard computerUseSettings.enabled != enabled else { return }
        computerUseSettings.enabled = enabled
        persistSettings()
        if enabled {
            Task { await startComputerUseHelperIfEnabled() }
        } else {
            stopComputerUseHelper()
        }
    }

    func startComputerUseHelperIfEnabled() async {
        guard computerUseSettings.enabled, computerUsePort == nil, computerUseHelperLaunch == nil,
              let helperURL = HelperComputerUsePort.installedHelperURL()
        else { return }
        let launch = Task { @MainActor [weak self] in
            defer { self?.computerUseHelperLaunch = nil }
            do {
                let (port, welcome) = try await HelperComputerUsePort.launch(helperURL: helperURL)
                guard let self, self.computerUseSettings.enabled else {
                    await port.shutdown()
                    return
                }
                self.computerUsePort = port
                self.computerUseManifest = welcome.manifest
                self.computerUseHelperStatus = welcome.status
            } catch {
                self?.computerUseHelperStatus = nil
            }
        }
        computerUseHelperLaunch = launch
        await launch.value
    }

    func stopComputerUseHelper() {
        revokeAllComputerUseGrants()
        let port = computerUsePort as? HelperComputerUsePort
        computerUsePort = nil
        computerUseManifest = nil
        computerUseHelperStatus = nil
        Task { await port?.shutdown() }
    }

    public func refreshComputerUseHelperStatus() async {
        guard let port = computerUsePort as? HelperComputerUsePort else { return }
        computerUseHelperStatus = try? await port.status()
    }

    /// Shows the macOS prompt for one grant. Call only from the user's
    /// Allow click: this is the app's single path to that prompt.
    public func requestComputerUsePermission(_ kind: ComputerUsePermissionKind) async {
        guard let port = computerUsePort as? HelperComputerUsePort else { return }
        computerUseHelperStatus = try? await port.requestPermission(kind)
    }
}
