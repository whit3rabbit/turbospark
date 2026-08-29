import Foundation

extension AppHookStore {
    var storageDirectory: URL {
        let appSupport = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let directory = appSupport.appendingPathComponent("TurboSpark/Hooks", isDirectory: true)
        try? fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
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
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        if let data = try? encoder.encode(optionValues) {
            try? data.write(to: optionsValuesFileURL, options: .atomic)
        }
    }

    func loadCustomHooks() -> [AppHookCommand] {
        guard let data = try? Data(contentsOf: customHooksFileURL),
              let list = try? JSONDecoder().decode([AppHookCommand].self, from: data) else {
            return []
        }
        return list
    }

    func saveCustomHooks() {
        let customOnly = hooks.filter { $0.sourceType == .custom }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        if let data = try? encoder.encode(customOnly) {
            try? data.write(to: customHooksFileURL, options: .atomic)
        }
    }
}
