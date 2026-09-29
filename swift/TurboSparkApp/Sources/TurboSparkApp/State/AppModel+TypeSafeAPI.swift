import Foundation
import OpenKind

extension AppModel {
    var typeSafeBaseURL: URL? {
        guard let port = UInt16(typeSafePort), port > 0 else { return nil }
        return URL(string: "http://127.0.0.1:\(port)")
    }

    public func startTypeSafeServer() async {
        guard typeSafeServer == nil, !typeSafeBusy,
              let url = typeSafeBaseURL else {
            typeSafeError = "Enter a valid loopback port."
            return
        }
        typeSafeBusy = true
        defer { typeSafeBusy = false }
        typeSafeError = nil
        let key = typeSafeAPIKeyInput.trimmingCharacters(in: .whitespacesAndNewlines)
        guard TypeSafeKeychain.saveKey(key) else {
            typeSafeError = "Could not save the TypeSafe API key to Keychain."
            return
        }
        let bundled = Bundle.main.bundleURL.appendingPathComponent("Contents/MacOS/openkindd")
        let override = ProcessInfo.processInfo.environment["OPENKINDD_BINARY"]
        let binary = override ?? bundled.path
        guard FileManager.default.isExecutableFile(atPath: binary) else {
            typeSafeError = "openkindd is missing. Build the app bundle or set OPENKINDD_BINARY for development."
            return
        }
        do {
            let service = try OpenKindServer(
                binary: binary,
                httpAddress: "127.0.0.1:\(url.port ?? 0)",
                models: ["mock"],
                apiKey: key.isEmpty ? nil : key,
                extraArguments: ["--playground", "on"])
            try await service.start()
            typeSafeServer = service
            typeSafeHealth = "Running"
            await refreshTypeSafeModels()
        } catch {
            typeSafeError = error.localizedDescription
            typeSafeHealth = "Stopped"
        }
    }

    public func stopTypeSafeServer() async {
        guard let service = typeSafeServer else { return }
        typeSafeBusy = true
        await service.stop()
        typeSafeServer = nil
        typeSafeModels = []
        typeSafeHealth = "Stopped"
        typeSafeBusy = false
    }

    public func refreshTypeSafeModels() async {
        guard let service = typeSafeServer else { return }
        do {
            let health = try await service.client.health()
            typeSafeHealth = health.data.status
            typeSafeModels = (try await service.client.listLocalModels()).data.models
            typeSafeError = nil
        } catch {
            typeSafeError = error.localizedDescription
            typeSafeHealth = "Error"
        }
    }

    public func setTypeSafeModel(_ name: String, loaded: Bool) async {
        guard let service = typeSafeServer,
              typeSafeModels.contains(where: { $0.name == name && $0.manageable }) else { return }
        typeSafeBusy = true
        defer { typeSafeBusy = false }
        do {
            try await service.client.setLocalModelLoaded(name, loaded: loaded)
            await refreshTypeSafeModels()
        } catch {
            typeSafeError = error.localizedDescription
        }
    }
}
