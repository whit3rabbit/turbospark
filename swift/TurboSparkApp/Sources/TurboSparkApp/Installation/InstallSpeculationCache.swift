import Foundation
import TurboSpark

/// The engine's answer to "would this install speculate under default
/// options", per install path (`TurboSparkCapabilities.speculation`).
///
/// The badge used to guess from the name ("mtp" in it, or a qwen38 alias),
/// which labeled a headless or Q3_K GGUF as MTP-capable. The engine reads the
/// resident index (kilobytes, no weights) with the session open's own rules,
/// so the badge now says what loading the model will do. Cached by the
/// weights file's stamp, since descriptors are resolved while drawing rows.
enum InstallSpeculationCache {
    private struct Stamp: Equatable {
        let modified: Date?
        let size: Int?
    }
    private struct Entry {
        let stamp: Stamp
        let answer: SpeculationPreflight?
    }
    private static let lock = NSLock()
    nonisolated(unsafe) private static var entries: [String: Entry] = [:]
    private static let maxEntries = 256

    /// Nil when the engine cannot answer (no install at `path`, or an
    /// unreadable manifest or index). The caller falls back then.
    static func answer(forInstallPath path: String) -> SpeculationPreflight? {
        let dir = (path as NSString).expandingTildeInPath
        let weights = URL(fileURLWithPath: dir).appendingPathComponent("model_weights.bin")
        guard let attrs = try? FileManager.default.attributesOfItem(atPath: weights.path) else {
            return nil
        }
        let stamp = Stamp(
            modified: attrs[.modificationDate] as? Date,
            size: (attrs[.size] as? NSNumber)?.intValue)
        lock.lock()
        if let hit = entries[dir], hit.stamp == stamp {
            lock.unlock()
            return hit.answer
        }
        lock.unlock()
        let answer = TurboSparkCapabilities.speculation(forInstallAt: dir)
        lock.lock()
        if entries.count >= maxEntries { entries.removeAll() }
        entries[dir] = Entry(stamp: stamp, answer: answer)
        lock.unlock()
        return answer
    }

    /// Test seam.
    static func reset() {
        lock.lock(); entries.removeAll(); lock.unlock()
    }
}

extension ModelFeatureDescriptor {
    /// Which drafter badge an install gets. A DFlash2 name keeps its badge
    /// (`auto` never selects that drafter, so the engine's default-options
    /// answer does not describe it). Otherwise the engine's answer decides
    /// MTP whenever there is one, and the name is consulted only when there
    /// is none (a catalog row not yet installed, or an unreadable install).
    static func speculativeDrafter(
        combinedText: String,
        alias: String,
        engineAnswer: SpeculationPreflight?
    ) -> SpeculativeDrafterType {
        if combinedText.contains("dflash") || combinedText.contains("sloth") {
            return .dynamicSloth
        }
        if let engineAnswer {
            return engineAnswer.available ? .mtp : .none
        }
        return combinedText.contains("mtp") || alias.contains("qwen38") ? .mtp : .none
    }
}
