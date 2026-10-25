import XCTest

/// FluidAudio downloads from Hugging Face `main` and deletes model folders
/// unless told otherwise (docs/FLUIDAUDIO.md). The app keeps every use in
/// `Audio/FluidAudio/`, behind `FluidAudioEngine`, which turns that off. This
/// scan holds the boundary: an import anywhere else, a call to a FluidAudio
/// download, cache or purge API, or an offline switch written outside the
/// engine fails here instead of shipping.
final class FluidAudioIsolationTests: XCTestCase {
    /// Entry points that fetch, purge or re-download. Installs come from the
    /// pinned Rust catalog, and adapters load the files it verified.
    private static let forbiddenAPIs = [
        "ModelHub.download", "ModelHub.loadModels", "ModelHub.fetchFile", "ModelHub.clear",
        "downloadAndLoad", "DownloadUtils", "AssetDownloader", "ResourceDownloader",
        "clearAllCaches", "clearAllModelCaches", "clearCache(", "prepareModels(",
        "loadWithAutoRecovery", "loadFromHuggingFace", "initializeFromHuggingFace",
        "AsrModels.load(", "AsrModels.download",
        // FluidAudio's own loaders skip the signature check that keeps a
        // wrong model from reaching its preconditionFailure.
        "SenseVoiceModels.load(", "ParaformerModels.load(",
    ]
    private static let offlineSwitches = ["ModelHub.offlineMode =", "ModelRegistry.baseURL ="]

    func testFluidAudioStaysBehindTheOfflineEngine() throws {
        let sources = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/TurboSparkApp")
        let files = try XCTUnwrap(FileManager.default.enumerator(at: sources, includingPropertiesForKeys: nil))
            .compactMap { $0 as? URL }.filter { $0.pathExtension == "swift" }
        XCTAssertFalse(files.isEmpty)
        var violations: [String] = []
        var importers = 0
        var switches = 0
        for file in files {
            let name = file.lastPathComponent
            let lines = try String(contentsOf: file, encoding: .utf8).components(separatedBy: "\n")
            let code = lines.map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.hasPrefix("//") }
            let imports = code.contains("import FluidAudio")
            if imports {
                importers += 1
                if !file.path.contains("/Audio/FluidAudio/") {
                    violations.append("\(name) imports FluidAudio outside Audio/FluidAudio/")
                }
                for line in code {
                    for api in Self.forbiddenAPIs where line.contains(api) {
                        violations.append("\(name) calls \(api)")
                    }
                }
            }
            for line in code {
                // Assignments only: the engine also reads both switches back.
                let writes = line.replacingOccurrences(of: "==", with: "")
                for setting in Self.offlineSwitches where writes.contains(setting) {
                    if name == "FluidAudioEngine.swift" { switches += 1 }
                    else { violations.append("\(name) writes \(setting)") }
                }
            }
        }
        XCTAssertEqual(violations, [])
        XCTAssertGreaterThan(importers, 0, "the scan must see the files it guards")
        XCTAssertEqual(switches, Self.offlineSwitches.count, "FluidAudioEngine must set every offline switch")
    }
}
