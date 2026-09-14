import AppKit
import SwiftUI
import TurboSpark

/// Header bar for ModelManagerView showing storage stats, rescan, add folder, and hub discovery buttons.
struct ModelManagerHeaderView: View {
    @ObservedObject var model: AppModel

    var body: some View {
        HStack(spacing: 12) {
            HStack(spacing: 8) {
                ZStack {
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .fill(Color.teal.opacity(0.15))
                        .frame(width: 30, height: 30)
                    Image(systemName: "internaldrive.fill")
                        .themedFont(.callout, weight: .semibold)
                        .foregroundStyle(Color.teal)
                        .accessibilityHidden(true)
                }

                VStack(alignment: .leading, spacing: 2) {
                    HStack(spacing: 6) {
                        Text("Installed Models", bundle: .module)
                            .themedFont(.callout, weight: .semibold)
                            .accessibilityAddTraits(.isHeader)

                        Text("LOCAL STORAGE", bundle: .module)
                            .themedFont(.micro, weight: .bold)
                            .padding(.horizontal, 5)
                            .padding(.vertical, 1.5)
                            .background(Color.teal.opacity(0.15), in: RoundedRectangle(cornerRadius: 4))
                            .foregroundStyle(Color.teal)
                    }

                    Text(summaryText)
                        .themedFont(.tiny)
                        .foregroundStyle(.appSecondary)
                }
            }

            Spacer(minLength: 8)

            Button {
                model.refreshModels()
                model.showToast("Refreshed model libraries", style: .info)
            } label: {
                Label { Text("Rescan", bundle: .module) } icon: { Image(systemName: "arrow.clockwise") }
                    .themedFont(.tiny, weight: .medium)
                    .frame(height: 22)
                    .padding(.horizontal, 8)
            }
            .buttonStyle(.plain)
            .background(Color(nsColor: .quaternaryLabelColor).opacity(0.3), in: Capsule())
            .help("Rescan storage directories for models")
            .accessibilityLabel("Rescan storage")

            Button {
                ModelLocationPicker.choose(for: model)
            } label: {
                Label {
                    Text("Add Folder…", bundle: .module)
                } icon: {
                    Image(systemName: "folder.badge.plus")
                }
                    .themedFont(.tiny, weight: .medium)
                    .frame(height: 22)
                    .padding(.horizontal, 8)
            }
            .buttonStyle(.plain)
            .background(Color(nsColor: .quaternaryLabelColor).opacity(0.3), in: Capsule())
            .help("Add an external folder containing .gturbo bundles or .gguf files")
            .accessibilityLabel("Add folder")

            Button {
                model.activeSection = .modelHub
            } label: {
                HStack(spacing: 4) {
                    Image(systemName: "shippingbox.fill")
                        .themedFont(.tiny)
                        .accessibilityHidden(true)
                    Text("Discover Hub ->", bundle: .module)
                        .themedFont(.tiny, weight: .medium)
                }
                .frame(height: 22)
                .padding(.horizontal, 10)
            }
            .buttonStyle(.plain)
            .background(.appAccent.opacity(0.14), in: Capsule())
            .foregroundStyle(.appAccent)
            .help("Browse the curated catalog and download models")
            .accessibilityLabel("Discover models in hub")
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
    }

    private var summaryText: String {
        let count = model.installed.count
        let totalBytes = model.installed.reduce(UInt64(0)) { $0 + $1.installBytes }
        var parts = ["\(count) installed"]
        if totalBytes > 0 {
            parts.append("\(MetricFormat.storage(totalBytes)) on disk")
        }
        if let sel = model.selected, model.session != nil {
            parts.append("Active: \(sel.alias)")
        }
        return parts.joined(separator: " \u{2022} ")
    }
}
