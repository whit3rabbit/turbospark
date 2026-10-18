import Foundation
import TurboSpark

/// One row of Installed > Audio: a model on disk, a checkpoint folder the
/// user chose, files the engine found but cannot use yet, or a pinned
/// catalog model this Mac can download.
struct AudioInstalledEntry: Identifiable, Equatable {
    /// Rows grouped as the catalog's `AudioInstalledReport` reports them,
    /// plus local folders and the pinned profiles not on disk.
    enum Group: Int, CaseIterable, Comparable {
        case installed, local, needsAdoption, needsAttention, otherAudio, available

        /// The English catalog key of the list's section header.
        var titleKey: String {
            switch self {
            case .installed: return "Installed"
            case .local: return "Local Folders"
            case .needsAdoption: return "Found on Disk"
            case .needsAttention: return "Needs Attention"
            case .otherAudio: return "Not in the Catalog"
            case .available: return "Available to Download"
            }
        }

        static func < (lhs: Group, rhs: Group) -> Bool { lhs.rawValue < rhs.rawValue }
    }

    var group: Group
    var alias: String
    /// The catalog profile, absent for an install the catalog no longer lists.
    var profile: AudioProfile?
    var record: AudioInstalledRecord?
    var legacy: AudioLegacyInstall?
    /// Why an install that needs attention cannot be used.
    var error: String?

    var id: String { "\(group.rawValue):\(alias)" }
    var name: String { profile?.displayName ?? alias }
    var path: String? { record?.path ?? legacy?.record.path }
    var installBytes: UInt64? { record?.installBytes ?? legacy?.record.installBytes }
}

/// Builds the Installed > Audio list. Pure, so tests can drive it with fixtures.
enum AudioInstalledCatalog {
    /// `canRun` is the provider's verdict for this Mac. A pinned profile that
    /// cannot run here is not offered for download, since nothing could use it.
    static func entries(profiles: [AudioProfile], installed: [AudioInstalledRecord], needsAdoption: [AudioLegacyInstall],
                        incompatible: [AudioIncompatibleInstall], otherAudio: [AudioInstalledRecord], localModels: Set<String>,
                        canRun: (AudioProfile) -> Bool) -> [AudioInstalledEntry] {
        let byAlias = Dictionary(profiles.map { ($0.identity.alias, $0) }, uniquingKeysWith: { first, _ in first })
        var entries: [AudioInstalledEntry] = []
        entries += installed.map {
            AudioInstalledEntry(group: .installed, alias: $0.alias, profile: byAlias[$0.alias], record: $0)
        }
        entries += localModels.sorted().map { AudioInstalledEntry(group: .local, alias: $0, profile: byAlias[$0]) }
        entries += needsAdoption.map {
            AudioInstalledEntry(group: .needsAdoption, alias: $0.identity.alias, profile: byAlias[$0.identity.alias],
                                record: $0.record, legacy: $0)
        }
        entries += incompatible.map {
            AudioInstalledEntry(group: .needsAttention, alias: $0.record.alias, profile: byAlias[$0.record.alias],
                                record: $0.record, error: $0.error)
        }
        entries += otherAudio.map { AudioInstalledEntry(group: .otherAudio, alias: $0.alias, record: $0) }
        let onDisk = Set(installed.map(\.alias) + needsAdoption.map(\.identity.alias)
            + incompatible.map(\.record.alias))
        entries += profiles.filter {
            !onDisk.contains($0.identity.alias) && !$0.identity.repository.isEmpty && !$0.identity.revision.isEmpty
                && $0.identity.task.isRunnable && canRun($0)
        }.map { AudioInstalledEntry(group: .available, alias: $0.identity.alias, profile: $0) }
        let taskOrder = Dictionary(AudioTask.allCases.enumerated().map { ($1, $0) }, uniquingKeysWith: { first, _ in first })
        return entries.enumerated().sorted { a, b in
            if a.element.group != b.element.group { return a.element.group < b.element.group }
            let left = a.element.profile.flatMap { taskOrder[$0.identity.task] } ?? Int.max
            let right = b.element.profile.flatMap { taskOrder[$0.identity.task] } ?? Int.max
            if left != right { return left < right }
            let order = a.element.name.localizedStandardCompare(b.element.name)
            return order == .orderedSame ? a.offset < b.offset : order == .orderedAscending
        }.map(\.element)
    }

    static func grouped(_ entries: [AudioInstalledEntry]) -> [(group: AudioInstalledEntry.Group, entries: [AudioInstalledEntry])] {
        AudioInstalledEntry.Group.allCases.compactMap { group in
            let rows = entries.filter { $0.group == group }
            return rows.isEmpty ? nil : (group, rows)
        }
    }
}

extension AudioWorkspaceController {
    var installedAudioEntries: [AudioInstalledEntry] {
        AudioInstalledCatalog.entries(
            profiles: profiles, installed: installedRecords, needsAdoption: needsAdoption, incompatible: incompatibleInstalls,
            otherAudio: otherAudioInstalls, localModels: Set(preferences.localModels.keys),
            canRun: { AudioRuntimeProvider($0).canRun($0) })
    }
}

extension AudioWorkspaceController {
    /// A row's state. Rows the engine reports on disk keep that verdict over
    /// a catalog lookup, which would call an unadopted install missing.
    func entryState(_ entry: AudioInstalledEntry, download: ModelDownload?) -> AudioModelState {
        switch entry.group {
        case .needsAdoption: return .needsAdoption
        case .needsAttention: return .needsAttention
        case .otherAudio: return .installed
        case .installed, .local, .available:
            return entry.profile.map { modelState($0, download: download) } ?? .installed
        }
    }

    /// The Speech-to-Text modes that use `alias` for any job. A mode with
    /// speaker labels off does not use its diarization model.
    func speechToTextModes(using alias: String) -> [SpeechToTextMode] {
        let uses = SpeechToTextModels.uses(of: alias, profiles: profiles, selected: speechToTextSelections, isInstalled: isInstalled)
            .filter { $0.job != .speakers || labelsSpeakers(in: $0.mode) }
        return SpeechToTextMode.allCases.filter { mode in uses.contains { $0.mode == mode } }
    }

    /// The checkpoint folder chosen for `alias`, if it still resolves.
    func localModelURL(_ alias: String) -> URL? {
        guard let bookmark = preferences.localModels[alias] else { return nil }
        var stale = false
        return try? URL(resolvingBookmarkData: bookmark, options: [.withSecurityScope, .withoutUI, .withoutMounting],
                        relativeTo: nil, bookmarkDataIsStale: &stale)
    }
}
