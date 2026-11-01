import Foundation

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
