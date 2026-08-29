import Foundation
import TurboSpark

extension AppModel {
    public func refreshModels() {
        do {
            installed = try TurboSparkCatalog.installed()
            catalog = try TurboSparkCatalog.available()
            telemetry = TurboSparkSession.systemTelemetry
            if selected == nil || !installed.contains(where: { $0.alias == selected?.alias }) {
                selected = installed.first
            }
            if let selected {
                modelPathText = selected.path
            }
        } catch {
            self.error = "\(error)"
        }
    }

    public func selectModel(_ model: InstalledModel) {
        guard !generating, selected?.alias != model.alias else { return }
        selected = model
        modelPathText = model.path
        Task {
            await open(model)
        }
    }

    public func openModelHub() {
        activeSection = .modelHub
    }

    public func openChatWithModel(_ model: InstalledModel) {
        if selected?.alias != model.alias {
            selectModel(model)
        }
        activeSection = .chat
    }

    public func buildOpenOptions() -> OpenOptions {
        var options = OpenOptions()
        if maxContextTokens > 0 {
            options.maxContext = .fixed(UInt32(maxContextTokens))
        }
        if runtimeOptions.expertCacheSlots > 0 {
            options.expertCacheSlots = .fixed(UInt32(runtimeOptions.expertCacheSlots))
        }
        options.powerProfile = runtimeOptions.powerProfile.powerProfile
        options.speculation = runtimeOptions.speculation.speculation
        options.speculativeDrafter = runtimeOptions.speculativeDrafter.drafter
        if runtimeOptions.maxTokensPerSec > 0 {
            options.maxTokensPerSec = runtimeOptions.maxTokensPerSec
        }
        let sPath = (runtimeOptions.steeringPath ?? steeringPath)?.trimmingCharacters(in: .whitespacesAndNewlines)
        if let sPath, !sPath.isEmpty {
            options.steering = sPath
            options.steeringMode = runtimeOptions.steeringMode.steeringMode
            options.steeringScale = runtimeOptions.steeringScale
            let layers = runtimeOptions.steeringLayers.trimmingCharacters(in: .whitespacesAndNewlines)
            if !layers.isEmpty {
                options.steeringLayers = layers
            }
            if runtimeOptions.steeringMode == .clamp {
                options.steeringTarget = runtimeOptions.steeringTarget
            }
            if runtimeOptions.steeringGate > 0 {
                options.steeringGate = runtimeOptions.steeringGate
            }
        }
        return options
    }

    public func open(_ model: InstalledModel) async {
        guard !generating else { return }
        opening = true
        error = nil
        session = nil
        defer { opening = false }

        do {
            let options = buildOpenOptions()
            session = try await TurboSparkSession(modelPath: model.path, options: options)
            selected = model
            modelPathText = model.path
            updateTokenEstimate()
            showToast("Loaded \(model.alias)", style: .success)
        } catch {
            let msg = "Failed to load model: \(error.localizedDescription)"
            self.error = msg
            showToast(msg, style: .error)
        }
    }

    public func loadModel() {
        guard let selected else { return }
        Task {
            await open(selected)
        }
    }

    public func reloadModel() {
        guard let selected else { return }
        Task {
            await open(selected)
        }
    }

    public func unloadModel() {
        guard !generating else { return }
        session = nil
        showToast("Model unloaded", style: .info)
    }

    public func setModelURL(_ url: URL) {
        guard !generating else { return }
        let path = url.standardizedFileURL.path
        modelPathText = path
        session = nil
        opening = true
        defer { opening = false }
        Task {
            do {
                let options = buildOpenOptions()
                session = try await TurboSparkSession(modelPath: path, options: options)
                refreshModels()
                showToast("Opened model at \(url.lastPathComponent)", style: .success)
            } catch {
                let msg = "Failed to open custom model path: \(error.localizedDescription)"
                self.error = msg
                showToast(msg, style: .error)
            }
        }
    }

    public func deleteModel(_ model: InstalledModel) {
        do {
            try TurboSparkCatalog.delete(model.alias)
            if selected?.alias == model.alias {
                session = nil
                selected = nil
            }
            refreshModels()
            showToast("Deleted model '\(model.alias)'", style: .info)
        } catch {
            let msg = "Failed to delete model: \(error.localizedDescription)"
            self.error = msg
            showToast(msg, style: .error)
        }
    }
}

