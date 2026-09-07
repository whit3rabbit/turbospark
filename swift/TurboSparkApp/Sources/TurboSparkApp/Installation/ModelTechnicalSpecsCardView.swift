import AppKit
import SwiftUI
import TurboSpark

/// Card displaying technical specifications and architecture notes for a model.
struct ModelTechnicalSpecsCardView: View {
    let entry: CatalogEntry
    let installedModel: InstalledModel?
    let visuals: ModelFamilyVisuals

    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            technicalSpecificationsCard
            if let notes = entry.notes, !notes.isEmpty {
                notesCard(notes)
            }
        }
    }

    private var technicalSpecificationsCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 6) {
                Image(systemName: visuals.iconSystemName)
                    .foregroundStyle(visuals.accentColor)
                    .help("\(visuals.family) architecture specifications")
                Text("Technical Overview", bundle: .module)
                    .themedFont(.base, weight: .semibold)
            }

            VStack(spacing: 8) {
                specRow(label: "Architecture", value: visuals.architectureType)
                specRow(label: "Format & Quant", value: visuals.formatLabel)
                specRow(label: "Download Size", value: MetricFormat.storage(entry.downloadBytes))
                specRow(label: "Estimated Install", value: MetricFormat.storage(entry.installBytes))
                if let inst = installedModel {
                    HStack(alignment: .firstTextBaseline) {
                        Text("Installed Path", bundle: .module)
                            .themedFont(.small)
                            .foregroundStyle(.secondary)
                        Spacer(minLength: 16)
                        Button {
                            ModelStorageManager.revealInFinder(path: inst.path)
                        } label: {
                            Text(inst.path)
                                .themedCode(.small)
                                .lineLimit(1)
                                .truncationMode(.middle)
                                .foregroundStyle(Color.accentColor)
                        }
                        .buttonStyle(.plain)
                        .help("Reveal in Finder: \(inst.path)")
                        .accessibilityLabel("Installed path: \(inst.path)")
                        .accessibilityHint("Opens Finder to show model directory")
                        .accessibilityAddTraits(.isLink)
                    }
                    specRow(label: "Installed On", value: inst.installedOn)
                }
            }
        }
        .modelCardStyle()
    }

    private func specRow(label: String, value: String) -> some View {
        HStack(alignment: .firstTextBaseline) {
            Text(label)
                .themedFont(.small)
                .foregroundStyle(.secondary)
            Spacer(minLength: 16)
            Text(value)
                .themedCode(.small)
                .lineLimit(1)
                .truncationMode(.middle)
        }
    }

    private func notesCard(_ notes: String) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Benchmark & Architecture Notes", bundle: .module)
                .themedFont(.base, weight: .semibold)

            Text(notes)
                .themedFont(.base)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .modelCardStyle()
    }
}

extension View {
    func modelCardStyle() -> some View {
        self
            .padding(16)
            .background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 12))
            .overlay(RoundedRectangle(cornerRadius: 12).stroke(Color(nsColor: .separatorColor).opacity(0.5), lineWidth: 0.5))
    }
}
