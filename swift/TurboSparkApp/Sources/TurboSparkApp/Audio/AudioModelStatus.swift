import SwiftUI
import TurboSpark

/// Where an audio model stands on this Mac, as the Speech-to-Text models
/// popover and Installed > Audio show it. One spelling for both, so a model
/// never reads Installed in one place and Downloaded in the other.
enum AudioModelState: Equatable {
    case loaded, installed, notInstalled
    /// Fraction done, nil before the size is known.
    case downloading(Double?)
    /// On disk without a receipt; adopting verifies it.
    case needsAdoption
    /// On disk but unusable as it is (`AudioIncompatibleInstall`).
    case needsAttention

    var isOnDisk: Bool { self == .loaded || self == .installed }
}

extension ModelDownload {
    /// Fraction of the bytes received, nil while the total is unknown.
    var fractionDone: Double? {
        guard let total = totalBytes, total > 0 else { return nil }
        return min(1, Double(downloadedBytes) / Double(total))
    }
}

extension AudioWorkspaceController {
    /// `download` is the profile's entry in the app's download queue.
    func modelState(_ profile: AudioProfile, download: ModelDownload?) -> AudioModelState {
        let alias = profile.identity.alias
        // Live text holds its stream only while a take records.
        if isProfileLoaded(profile) || (live.isStreaming && liveTextProfile?.identity.alias == alias) { return .loaded }
        if isInstalled(profile) { return .installed }
        if let download, !download.status.isTerminal { return .downloading(download.fractionDone) }
        if adoptable(alias) != nil { return .needsAdoption }
        if incompatibleInstalls.contains(where: { $0.record.alias == alias }) { return .needsAttention }
        return .notInstalled
    }
}

/// A state as glyph and word, never color alone.
struct AudioModelStateLabel: View {
    let state: AudioModelState

    var body: some View {
        Label { title } icon: { Image(systemName: symbol).accessibilityHidden(true) }
            .themedFont(.tiny, weight: .medium)
            .foregroundStyle(state == .loaded ? AnyShapeStyle(AppThemeColor.appAccent) : AnyShapeStyle(AppThemeColor.appSecondary))
            .lineLimit(1)
    }

    private var symbol: String {
        switch state {
        case .loaded: return "bolt.circle.fill"
        case .installed: return "checkmark.circle"
        case .notInstalled: return "arrow.down.circle"
        case .downloading: return "arrow.down.circle.dotted"
        case .needsAdoption: return "exclamationmark.triangle.fill"
        case .needsAttention: return "xmark.octagon.fill"
        }
    }

    private var title: Text {
        switch state {
        case .loaded: return Text("Loaded", bundle: .module)
        case .installed: return Text("Installed", bundle: .module)
        case .notInstalled: return Text("Not installed", bundle: .module)
        case .downloading(let fraction?):
            return Text("Downloading \(fraction.formatted(.percent.precision(.fractionLength(0))))", bundle: .module)
        case .downloading(nil): return Text("Downloading", bundle: .module)
        case .needsAdoption: return Text("Not verified", bundle: .module)
        case .needsAttention: return Text("Needs attention", bundle: .module)
        }
    }
}

/// "Pinned" and the short revision for a catalog profile, which the engine
/// checks file by file against its sha256. A folder the user chose is
/// "Not pinned". A choice the user made is never called pinned.
struct AudioPinnedBadge: View {
    let profile: AudioProfile
    let isLocalFolder: Bool

    var body: some View {
        if isLocalFolder {
            badge(Text("Not pinned", bundle: .module), symbol: "folder")
                .help(Text("A folder you chose. Its files are not checked.", bundle: .module))
        } else if !profile.identity.revision.isEmpty, !profile.identity.repository.isEmpty {
            let revision = String(profile.identity.revision.prefix(7))
            badge(Text("Pinned", bundle: .module) + Text(verbatim: " " + revision), symbol: "pin.fill")
                .help(Text("Pinned to revision \(profile.identity.revision) in the TurboSpark catalog. Every file is checked against its sha256.", bundle: .module))
        }
    }

    private func badge(_ title: Text, symbol: String) -> some View {
        Label { title } icon: { Image(systemName: symbol).accessibilityHidden(true) }
            .themedFont(.micro, weight: .medium)
            .foregroundStyle(.appSecondary)
            .lineLimit(1)
            .padding(.horizontal, 6).padding(.vertical, 1)
            .overlay { Capsule().stroke(.appBorder, lineWidth: 1) }
            .accessibilityElement(children: .combine)
    }
}

/// Download size and the catalog's resident memory estimate.
struct AudioModelFacts: View {
    let profile: AudioProfile

    var body: some View {
        FlowLayout(spacing: 10, lineSpacing: 2) {
            if profile.downloadBytes > 0 {
                Label { Text(verbatim: MetricFormat.storage(profile.downloadBytes)) } icon: {
                    Image(systemName: "internaldrive").accessibilityHidden(true)
                }
                .help(Text("Download size", bundle: .module))
            }
            if profile.residentMemoryBytes > 0 {
                Label { Text("Up to \(MetricFormat.memory(profile.residentMemoryBytes)), estimate", bundle: .module) } icon: {
                    Image(systemName: "memorychip").accessibilityHidden(true)
                }
                .help(Text("Catalog estimate of the memory it uses loaded, not measured on this Mac.", bundle: .module))
            }
        }
        .themedFont(.tiny)
        .foregroundStyle(.appSecondary)
    }
}
