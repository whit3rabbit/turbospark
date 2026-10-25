import SwiftUI

/// One additional model scan folder in Models and Storage (requirement 3.9).
///
/// **IT DRAWS A `ScanFolderRowModel` AND NOTHING ELSE.** What a sighted user
/// reads and what VoiceOver reads come from the same model, and a table test
/// pins both.
///
/// **A FOLDER THAT CANNOT BE READ SAYS WHY.** A drive that is away shows the
/// status glyph and "Not connected" where the model count would be, the drive to
/// connect under the path, and a Reveal button that is disabled with its reason.
/// A folder that is gone from a drive that is there shows a triangle and "Folder
/// not found". Neither shows "Models: 0", and neither raises an alert, a banner
/// or a toast (requirement 3.4). Color only tints the glyph.
///
/// **A ROW THAT CANNOT BE READ IS ONE ACCESSIBILITY ELEMENT**, in the order path,
/// status, drive. Its buttons are accessibility actions on that element, and only
/// an action that can run is offered. The visible buttons stay real controls for
/// the pointer and for Full Keyboard Access. A connected row keeps its separate
/// controls, as it always had.
struct ScanFolderRowView: View {
    let row: ScanFolderRowModel
    let reveal: () -> Void
    let remove: () -> Void

    var body: some View {
        if row.isOffline {
            content
                .accessibilityElement(children: .ignore)
                .accessibilityLabel(Text(verbatim: row.accessibilityLabel))
                .accessibilityActions {
                    if row.reveal.isEnabled {
                        Button {
                            reveal()
                        } label: {
                            Text("Reveal in Finder", bundle: .module)
                        }
                    }
                    Button {
                        remove()
                    } label: {
                        Text("Remove this folder from scan list", bundle: .module)
                    }
                }
        } else {
            content
        }
    }

    private var content: some View {
        HStack(spacing: 8) {
            Image(systemName: "folder")
                .foregroundStyle(.appSecondary)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 2) {
                pathText
                if let hint = row.hintText {
                    Text(verbatim: hint)
                        .themedFont(.tiny)
                        .foregroundStyle(.appSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            Spacer()
            trailing
            revealButton
            // Quiet, not destructive: this only drops the path from the scan
            // list. No file is touched and Add Folder restores it, so a red
            // trash glyph without a confirmation dialog overstated it
            // (DESIGN.md section 2, Action hierarchy).
            AppIconButton("minus.circle", label: Text("Remove this folder from scan list", bundle: .module)) {
                remove()
            }
            .appAction(.quiet, size: .small)
        }
        .padding(8)
        .background(.appSurface)
        .clipShape(RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous))
    }

    /// Secondary text for a folder that cannot be read, so the status carries
    /// the row's weight.
    @ViewBuilder private var pathText: some View {
        if row.isDimmed {
            SettingsPathText(row.displayPath).foregroundStyle(.appSecondary)
        } else {
            SettingsPathText(row.displayPath)
        }
    }

    /// The model count, or the status where the count would be.
    @ViewBuilder private var trailing: some View {
        if let count = row.countText {
            Text(verbatim: count)
                .themedFont(.tiny)
                .foregroundStyle(.appSecondary)
                .fixedSize()
        } else if let status = row.statusText, let glyph = row.statusGlyph {
            HStack(spacing: 4) {
                glyphImage(glyph)
                Text(verbatim: status)
                    .themedFont(.tiny, weight: .medium)
                    .foregroundStyle(.appText)
            }
            .fixedSize()
        }
    }

    @ViewBuilder private func glyphImage(_ glyph: String) -> some View {
        if glyph == LibraryRowGlyph.warning {
            Image(systemName: glyph).foregroundStyle(.orange).accessibilityHidden(true)
        } else {
            Image(systemName: glyph).foregroundStyle(.appSecondary).accessibilityHidden(true)
        }
    }

    /// Enabled: the shared icon button. Disabled: the same glyph and name with
    /// the REASON as its tooltip (DESIGN.md: a disabled control says why), which
    /// the shared button cannot carry because its tooltip is its name.
    @ViewBuilder private var revealButton: some View {
        if row.reveal.isEnabled {
            AppIconButton("arrow.up.right.square", label: Text("Reveal in Finder", bundle: .module)) {
                reveal()
            }
            .appAction(.quiet, size: .small)
        } else {
            Button {
                reveal()
            } label: {
                Label {
                    Text("Reveal in Finder", bundle: .module)
                } icon: {
                    Image(systemName: "arrow.up.right.square")
                }
                .labelStyle(.iconOnly)
            }
            .appAction(.quiet, size: .small)
            .disabled(true)
            .help(Text(verbatim: row.reveal.disabledReason ?? ""))
        }
    }
}
