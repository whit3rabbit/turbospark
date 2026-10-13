import Foundation
import OpenKind
import Security

extension AppModel {
    var typeSafeBaseURL: URL? {
        guard let port = UInt16(typeSafePort), port > 0 else { return nil }
        return URL(string: "http://127.0.0.1:\(port)")
    }

    public func startTypeSafeServer() async {
        guard typeSafeServer == nil, !typeSafeBusy,
              let url = typeSafeBaseURL else {
            typeSafeError = AppStrings.localized("Enter a valid loopback port.")
            return
        }
        typeSafeBusy = true
        defer { typeSafeBusy = false }
        typeSafeError = nil
        let typed = typeSafeAPIKeyInput.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !typed.isEmpty || typeSafeKeyLoadFailed else {
            typeSafeError = AppStrings.localized("TypeSafe API key is required to start the daemon.")
            return
        }
        // Without the opt-in the Keychain is not consulted at all: the key in
        // the field is this session's key and nothing outlives it. With it,
        // the same rule as the server key applies: an unreadable Keychain must
        // not turn an empty field into a delete of the stored key.
        var effectiveKey = typed
        if typeSafeKeyInKeychain {
            let current = TypeSafeKeychain.readKey()
            switch KeychainSecretSync.action(
                current: current, input: typed, loadFailed: typeSafeKeyLoadFailed)
            {
            case .none: break
            case let .save(value):
                let status = TypeSafeKeychain.saveKeyStatus(value)
                guard status == errSecSuccess else {
                    typeSafeError = Self.keychainFailureMessage(
                        "Could not save the TypeSafe API key to Keychain.", status: status)
                    return
                }
                typeSafeKeyLoadFailed = false
            case .delete:
                typeSafeError = AppStrings.localized("TypeSafe API key is required to start the daemon.")
                return
            case let .adopt(value):
                typeSafeAPIKeyInput = value
                typeSafeKeyLoadFailed = false
                effectiveKey = value
            }
            if effectiveKey.isEmpty, case let .error(status) = current {
                typeSafeError = Self.keychainFailureMessage(
                    "TypeSafe API key is required to start the daemon.", status: status)
                return
            }
        }
        let key = effectiveKey
        guard !key.isEmpty else {
            typeSafeError = AppStrings.localized("TypeSafe API key is required to start the daemon.")
            return
        }
        let bundled = Bundle.main.bundleURL.appendingPathComponent("Contents/MacOS/openkindd")
        let override = ProcessInfo.processInfo.environment["OPENKINDD_BINARY"]
        let binary = override ?? bundled.path
        guard FileManager.default.isExecutableFile(atPath: binary) else {
            typeSafeError = AppStrings.localized(
                "openkindd is missing. Build the app bundle or set OPENKINDD_BINARY for development.")
            return
        }
        do {
            let service = try OpenKindServer(
                binary: binary,
                httpAddress: "127.0.0.1:\(url.port ?? 0)",
                models: ["mock"],
                apiKey: key,
                extraArguments: ["--playground", "on"])
            try await service.start()
            typeSafeServer = service
            typeSafeHealth = "Running"
            await refreshTypeSafeModels()
        } catch {
            typeSafeError = Self.typeSafeStartFailureMessage(error, port: url.port)
            typeSafeHealth = "Stopped"
        }
    }

    /// Turns Keychain storage of the TypeSafe key on or off. Turning it on
    /// saves the key in the field now, which is the one moment macOS may ask
    /// for access, and it stays off if that fails. Turning it off deletes the
    /// item, so nothing is left behind that a later build would be asked to
    /// vouch for. The key in the field is kept either way.
    func setTypeSafeKeyInKeychain(_ enabled: Bool) {
        guard enabled != typeSafeKeyInKeychain else { return }
        if enabled {
            let typed = typeSafeAPIKeyInput.trimmingCharacters(in: .whitespacesAndNewlines)
            let value = typed.isEmpty ? TypeSafeKeychain.generateKey() : typed
            let status = TypeSafeKeychain.saveKeyStatus(value)
            guard status == errSecSuccess else {
                typeSafeError = Self.keychainFailureMessage(
                    "Could not save the TypeSafe API key to Keychain.", status: status)
                return
            }
            typeSafeAPIKeyInput = value
            typeSafeKeyLoadFailed = false
        } else {
            TypeSafeKeychain.deleteKey()
            typeSafeKeyLoadFailed = false
        }
        TypeSafeKeychain.setKeychainEnabled(enabled)
        typeSafeKeyInKeychain = enabled
        typeSafeError = nil
    }

    /// A localized failure line followed by the system's own reason, which is
    /// what says "A default keychain could not be found" instead of leaving
    /// the person to guess why saving failed.
    static func keychainFailureMessage(_ key: String, status: OSStatus) -> String {
        AppStrings.localized(String.LocalizationValue(key)) + " " + TypeSafeKeychain.describe(status: status)
    }

    /// `OpenKind.ServerError` bridges to "The operation couldn't be completed.
    /// (OpenKind.ServerError error 1.)", which names nothing. The commonest
    /// cause is a previous openkindd that outlived the app (crash, force-quit)
    /// and still holds the port, so say that.
    static func typeSafeStartFailureMessage(_ error: Error, port: Int?) -> String {
        let description = error.localizedDescription
        guard String(reflecting: type(of: error)).contains("ServerError")
            || description.contains("ServerError")
        else { return description }
        let where_ = port.map { "127.0.0.1:\($0)" } ?? AppStrings.localized("the configured address")
        return AppStrings.localized(
            "openkindd could not start on \(where_). The port may already be in use, possibly by an openkindd left running by a previous session: quit it (Activity Monitor or `pkill openkindd`) or choose another port.")
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
            // A Stop (or restart) that landed while the poll was in flight
            // owns the status now; writing it back would resurrect a dead
            // server's health or error.
            guard typeSafeServer === service, !Task.isCancelled else { return }
            typeSafeHealth = health.data.status
            let models = (try await service.client.listLocalModels()).data.models
            guard typeSafeServer === service, !Task.isCancelled else { return }
            typeSafeModels = models
            typeSafeError = nil
        } catch {
            guard typeSafeServer === service, !Task.isCancelled else { return }
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
