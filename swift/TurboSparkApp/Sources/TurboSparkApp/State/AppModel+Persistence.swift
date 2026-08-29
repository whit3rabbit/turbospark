import Foundation
import TurboSpark

extension AppModel {
    func loadSettings() {
        let settings = MacAppSettingsFileStore.load()
        self.maxContextTokens = settings.contextTokens
        self.runtimeOptions.expertCacheSlots = settings.expertCacheSlots
        self.temperature = settings.temperature
        self.topKEnabled = settings.topKEnabled
        self.topK = settings.topK
        self.topPEnabled = settings.topPEnabled
        self.topP = settings.topP
        self.runtimeOptions.prefillEnabled = settings.prefillEnabled
        self.reasoning = GenerateOptions.Reasoning(rawValue: settings.reasoning) ?? .off
        self.maxNewTokens = settings.maxNewTokens
        self.repetitionPenaltyEnabled = settings.repetitionPenaltyEnabled
        self.repetitionPenalty = settings.repetitionPenalty
        self.seedEnabled = settings.seedEnabled
        self.seed = settings.seed
        self.stopSequences = settings.stopSequences
        self.runtimeOptions.powerProfile = AppPowerProfileOption(rawValue: settings.powerProfile) ?? .auto
        self.runtimeOptions.speculation = AppSpeculationOption(rawValue: settings.speculation) ?? .auto
        self.runtimeOptions.speculativeDrafter = AppSpeculativeDrafterOption(rawValue: settings.speculativeDrafter) ?? .auto
        self.runtimeOptions.maxTokensPerSec = settings.maxTokensPerSec
        self.runtimeOptions.steeringPath = settings.steeringPath.isEmpty ? nil : settings.steeringPath
        self.runtimeOptions.steeringMode = AppSteeringModeOption(rawValue: settings.steeringMode) ?? .ablate
        self.runtimeOptions.steeringScale = settings.steeringScale
        self.runtimeOptions.steeringLayers = settings.steeringLayers
        self.runtimeOptions.steeringTarget = settings.steeringTarget
        self.runtimeOptions.steeringGate = settings.steeringGate
        self.steeringPath = self.runtimeOptions.steeringPath
    }

    public func persistSettings() {
        let settings = MacAppSettings(
            contextTokens: maxContextTokens,
            expertCacheSlots: runtimeOptions.expertCacheSlots,
            temperature: temperature,
            topKEnabled: topKEnabled,
            topK: topK,
            topPEnabled: topPEnabled,
            topP: topP,
            prefillEnabled: runtimeOptions.prefillEnabled,
            reasoning: reasoning.rawValue,
            maxNewTokens: maxNewTokens,
            repetitionPenaltyEnabled: repetitionPenaltyEnabled,
            repetitionPenalty: repetitionPenalty,
            seedEnabled: seedEnabled,
            seed: seed,
            stopSequences: stopSequences,
            powerProfile: runtimeOptions.powerProfile.rawValue,
            speculation: runtimeOptions.speculation.rawValue,
            speculativeDrafter: runtimeOptions.speculativeDrafter.rawValue,
            maxTokensPerSec: runtimeOptions.maxTokensPerSec,
            steeringPath: runtimeOptions.steeringPath ?? "",
            steeringMode: runtimeOptions.steeringMode.rawValue,
            steeringScale: runtimeOptions.steeringScale,
            steeringLayers: runtimeOptions.steeringLayers,
            steeringTarget: runtimeOptions.steeringTarget,
            steeringGate: runtimeOptions.steeringGate
        )
        MacAppSettingsFileStore.save(settings)
    }

    func loadChats() {
        let archive = AppChatFileStore.load()
        self.chats = archive.chats
        self.selectedChatID = archive.selectedChatID
    }

    public func persistChats() {
        let archive = AppChatArchive(selectedChatID: selectedChatID, chats: chats)
        AppChatFileStore.save(archive)
    }

    func loadProjects() {
        let archive = AppProjectFileStore.load()
        self.projects = archive.projects
        self.selectedProjectID = archive.selectedProjectID
    }

    public func persistProjects() {
        let archive = AppProjectArchive(selectedProjectID: selectedProjectID, projects: projects)
        AppProjectFileStore.save(archive)
    }
}
