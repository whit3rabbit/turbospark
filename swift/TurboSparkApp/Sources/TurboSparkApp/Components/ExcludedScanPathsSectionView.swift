import SwiftUI

/// The scanned models the user has removed from TurboSpark, and the way back
/// (state#88).
///
/// "Remove from TurboSpark" on a scanned row cannot delete anything -- those
/// bytes were only ever discovered by walking a folder the user pointed the
/// app at. What it really does is exclude the path from future scans, and an
/// exclusion with no list and no undo is indistinguishable from a broken
/// Delete: the row goes away, and nothing anywhere says how to get it back.
struct ExcludedScanPathsSectionView: View {
    @ObservedObject var model: AppModel
    @ObservedObject private var orgStore = ModelOrganizationStore.shared

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("Excluded from scan", systemImage: "eye.slash")
                .font(.headline)

            if orgStore.excludedScanPaths.isEmpty {
                Text(
                    "Models you remove from TurboSpark without deleting their files are listed "
                        + "here, so they can be restored."
                )
                .font(.caption)
                .foregroundStyle(.secondary)
            } else {
                ForEach(orgStore.excludedScanPaths.sorted(), id: \.self) { path in
                    HStack(spacing: 8) {
                        Image(systemName: "eye.slash")
                            .foregroundStyle(.secondary)
                        Text(path)
                            .font(.system(.caption, design: .monospaced))
                            .lineLimit(1)
                            .truncationMode(.middle)
                        Spacer()
                        Button("Restore") {
                            orgStore.restoreToScan(path: path)
                            model.refreshModels()
                        }
                        .controlSize(.small)
                        .help("Include this path in model scans again")
                        .accessibilityHint("Puts this model back in the installed list")
                    }
                    .padding(8)
                    .background(
                        Color(nsColor: .controlBackgroundColor),
                        in: RoundedRectangle(cornerRadius: 8))
                }
            }
        }
        .padding(16)
        .background(Color(nsColor: .windowBackgroundColor), in: RoundedRectangle(cornerRadius: 10))
    }
}
