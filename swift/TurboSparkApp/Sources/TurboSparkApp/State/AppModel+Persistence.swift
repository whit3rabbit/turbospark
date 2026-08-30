import Foundation
import TurboSpark

extension AppModel {
    /// Loads persisted generation parameters, execution options, and model paths from disk.
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
        self.runtimeOptions.loadGuard = AppLoadGuardOption(rawValue: settings.loadGuard) ?? .relaxed
        self.runtimeOptions.loadGuardCustomBytes = settings.loadGuardCustomBytes
        self.runtimeOptions.minAutoContextTokens = settings.minAutoContextTokens
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
        self.modelsDirectory = settings.modelsDirectory
        self.enableLMStudioDetection = settings.enableLMStudioDetection
        self.lmStudioDirectory = settings.lmStudioDirectory
        self.customModelDirectories = settings.customModelDirectories
        self.guardrailsMode = AppGuardrailsMode(rawValue: settings.guardrailsMode) ?? .select
        self.modelReasoningDefaults = settings.modelReasoningDefaults
    }

    /// Persists current runtime options, steering parameters, and directory paths to disk.
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
            loadGuard: runtimeOptions.loadGuard.rawValue,
            loadGuardCustomBytes: runtimeOptions.loadGuardCustomBytes,
            minAutoContextTokens: runtimeOptions.minAutoContextTokens,
            speculation: runtimeOptions.speculation.rawValue,
            speculativeDrafter: runtimeOptions.speculativeDrafter.rawValue,
            maxTokensPerSec: runtimeOptions.maxTokensPerSec,
            steeringPath: runtimeOptions.steeringPath ?? "",
            steeringMode: runtimeOptions.steeringMode.rawValue,
            steeringScale: runtimeOptions.steeringScale,
            steeringLayers: runtimeOptions.steeringLayers,
            steeringTarget: runtimeOptions.steeringTarget,
            steeringGate: runtimeOptions.steeringGate,
            modelsDirectory: modelsDirectory,
            enableLMStudioDetection: enableLMStudioDetection,
            lmStudioDirectory: lmStudioDirectory,
            customModelDirectories: customModelDirectories,
            guardrailsMode: guardrailsMode.rawValue,
            modelReasoningDefaults: modelReasoningDefaults
        )
        MacAppSettingsFileStore.save(settings)
    }

    /// Loads persisted chat threads and the active selected conversation ID.
    func loadChats() {
        let archive = AppChatFileStore.load()
        self.chats = archive.chats
        self.selectedChatID = archive.selectedChatID
    }

    /// Persists all conversation threads and active selection to disk.
    public func persistChats() {
        let archive = AppChatArchive(selectedChatID: selectedChatID, chats: chats)
        AppChatFileStore.save(archive)
    }

    /// Loads persisted project configurations and selected project ID.
    func loadProjects() {
        let archive = AppProjectFileStore.load()
        self.projects = archive.projects
        self.selectedProjectID = archive.selectedProjectID
    }

    /// Persists all project workspaces and active project selection.
    public func persistProjects() {
        let archive = AppProjectArchive(selectedProjectID: selectedProjectID, projects: projects)
        AppProjectFileStore.save(archive)
    }

    /// Loads global Model Context Protocol server configurations.
    func loadGlobalMcpServers() {
        let archive = GlobalMcpFileStore.load()
        self.globalMcpServers = archive.servers
    }

    /// Persists global MCP servers to application support directory.
    public func persistGlobalMcpServers() {
        let archive = GlobalMcpArchive(servers: globalMcpServers)
        GlobalMcpFileStore.save(archive)
    }
}
