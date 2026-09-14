import Foundation

extension AppHookStore {
    var storageDirectory: URL {
        AppStorageRoot.subdirectory("Hooks")
    }

    var trustedHashesFileURL: URL {
        storageDirectory.appendingPathComponent("trusted_hook_hashes.json")
    }

    var customHooksFileURL: URL {
        storageDirectory.appendingPathComponent("custom_hooks.json")
    }

    var optionsValuesFileURL: URL {
        storageDirectory.appendingPathComponent("hook_options_values.json")
    }

    // MARK: - Persistence IO

    func loadTrustedHashes() {
        if let data = try? Data(contentsOf: trustedHashesFileURL),
           let list = try? JSONDecoder().decode([String].self, from: data) {
            self.trustedHashes = Set(list)
        } else {
            self.trustedHashes = []
        }
    }

    func saveTrustedHashes() {
        let list = Array(trustedHashes).sorted()
        if let data = try? JSONEncoder().encode(list) {
            try? data.write(to: trustedHashesFileURL, options: .atomic)
        }
    }

    func loadOptionValues() {
        if let data = try? Data(contentsOf: optionsValuesFileURL),
           let dict = try? JSONDecoder().decode([String: [String: String]].self, from: data) {
            self.optionValues = dict
        } else {
            self.optionValues = [:]
        }
    }

    func saveOptionValues() {
        var publicValues = optionValues
        for (sourceID, keys) in sensitiveOptionKeys {
            for key in keys {
                publicValues[sourceID]?.removeValue(forKey: key)
            }
            if publicValues[sourceID]?.isEmpty == true {
                publicValues.removeValue(forKey: sourceID)
            }
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        if AppJSONStore.save(
            publicValues, to: optionsValuesFileURL, label: "Hook option values", encoder: encoder)
        {
            try? FileManager.default.setAttributes(
                [.posixPermissions: 0o600], ofItemAtPath: optionsValuesFileURL.path)
        }
    }

    /// Loads Keychain-backed values and removes legacy plaintext secrets from
    /// the preferences file after plugin discovery identifies sensitive keys.
    func synchronizeSensitiveOptionValues() {
        var foundLegacyPlaintext = false
        for (sourceID, keys) in sensitiveOptionKeys {
            for key in keys {
                if let plaintext = optionValues[sourceID]?[key] {
                    optionSecretStore.save(
                        plaintext, sourceID: sourceID, key: key,
                        storageDirectory: storageDirectory)
                    foundLegacyPlaintext = true
                } else if let secret = optionSecretStore.load(
                    sourceID: sourceID, key: key, storageDirectory: storageDirectory)
                {
                    optionValues[sourceID, default: [:]][key] = secret
                }
            }
        }
        if foundLegacyPlaintext { saveOptionValues() }
    }

    /// **A DECODE FAILURE HERE USED TO BE PERMANENT.** The old body was one
    /// `try?` chain returning `[]`, with no report at all -- and `hooks` is
    /// empty until `refresh()` runs, so a mutation before that point wrote an
    /// empty list over the file. `AppJSONStore` quarantines instead.
    func loadCustomHooks() -> [AppHookCommand] {
        AppJSONStore.load([AppHookCommand].self, from: customHooksFileURL, label: "custom hooks")
            ?? []
    }

    func saveCustomHooks() {
        // **NOT REACHED BEFORE `refresh()`.** `hooks` starts empty and this
        // filters it, so saving before discovery has run would truncate the
        // file to `[]` -- the write is skipped rather than made empty.
        guard didRefreshAtLeastOnce else { return }
        let customOnly = hooks.filter { $0.sourceType == .custom }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        AppJSONStore.save(
            customOnly, to: customHooksFileURL, label: "Custom hooks", encoder: encoder)
    }
}
