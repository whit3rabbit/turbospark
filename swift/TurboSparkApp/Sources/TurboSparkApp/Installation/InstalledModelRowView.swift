import AppKit
import SwiftUI
import TurboSpark

/// A row item in the ModelManager master list representing an installed model.
///
/// Where the model is, and whether it can be opened, come from `rowModel`
/// (requirements 3.1, 3.7 and 5.3): a badge with the library, a glyph AND a word
/// for "Not connected" and "Missing", a faded logo and secondary text for those
/// rows, and the load-time estimate for a model on an external drive. The view
/// draws those strings and nothing else.
struct InstalledModelRowView: View {
    let modelItem: InstalledModel
    let rowModel: InstalledModelRowModel
    let isSelected: Bool
    let isLoaded: Bool
    let isFavorite: Bool
    let tags: [String]
    let onSelect: () -> Void
    /// Multi-select delete (`InstalledModelSelection`). Nil outside selection
    /// mode: the row then opens the detail pane and offers the hover trash.
    var selection: Selection?
    /// Opens the Delete confirmation for this row. Nil hides the trash and the
    /// context menu item (a row that has no files to delete).
    var onRequestDelete: (() -> Void)?

    /// What a row needs in selection mode.
    struct Selection {
        var isChecked: Bool
        /// Why the row cannot be ticked, nil when it can.
        var blockedReason: String?
        var onToggle: () -> Void
    }

    @State private var isHovering = false
    @FocusState private var trashFocused: Bool

    private var descriptor: ModelFeatureDescriptor {
        ModelFeatureDescriptor.resolve(installedModel: modelItem)
    }

    private var visuals: ModelFamilyVisuals {
        ModelFamilyVisuals.resolve(
            alias: modelItem.alias,
            family: modelItem.family,
            name: modelItem.alias
        )
    }

    var body: some View {
        if let selection {
            selectableRow(selection)
        } else {
            openableRow
        }
    }

    /// Selection mode: a native checkbox whose label is the whole row, so a
    /// click anywhere on it toggles, VoiceOver says "checkbox" and the state,
    /// and a blocked row is a disabled checkbox that says why (never color alone).
    private func selectableRow(_ selection: Selection) -> some View {
        Toggle(
            isOn: Binding(get: { selection.isChecked }, set: { _ in selection.onToggle() })
        ) {
            rowContent
                .frame(maxWidth: .infinity, alignment: .leading)
                .contentShape(Rectangle())
        }
        .toggleStyle(.checkbox)
        .disabled(selection.blockedReason != nil)
        .padding(.horizontal, 8)
        .background(
            selection.isChecked ? AnyShapeStyle(.appAccent.opacity(0.14)) : AnyShapeStyle(Color.clear),
            in: RoundedRectangle(cornerRadius: 8)
        )
        .help(selection.blockedReason.map { Text(verbatim: $0) } ?? Text(verbatim: rowModel.accessibilityLabel))
        .accessibilityLabel(rowModel.accessibilityLabel)
        .accessibilityValue(accessibilityDescription)
        .accessibilityHint(selection.blockedReason ?? "")
    }

    /// The row's name, size, state and chips, drawn the same in both modes.
    private var rowContent: some View {
        HStack(alignment: .center, spacing: 10) {
            ModelLogoView(visuals: visuals, size: 36, cornerRadius: 8)
                // Not text, so fading it cannot cost any contrast; the
                // words below stay at full strength.
                .opacity(rowModel.isDimmed ? 0.5 : 1)

            // A row says the name, the size, ONE state and at most two chips.
            // The rest (MoE or Dense, the drafter, tags, the load-time
            // estimate) is in the detail pane, and in the row's spoken value.
            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 4) {
                    Text(verbatim: modelItem.alias)
                        .themedFont(.base, weight: .semibold)
                        // Secondary only for a model that cannot be opened; any
                        // other row keeps the primary text it always had.
                        .foregroundStyle(rowModel.isDimmed ? AnyShapeStyle(.appSecondary) : AnyShapeStyle(.primary))
                        .lineLimit(1)

                    if isFavorite {
                        Image(systemName: "star.fill")
                            .themedFont(.micro)
                            .foregroundStyle(.appWarning)
                    }

                    Spacer(minLength: 2)

                    Text(verbatim: MetricFormat.storage(modelItem.installBytes))
                        .themedFont(.tiny).monospacedDigit()
                        .foregroundStyle(.appSecondary)
                        .fixedSize()
                }

                stateMarker

                FlowLayout(spacing: 4, lineSpacing: 3) {
                    locationOrSourceBadge
                    ModelFeatureBadgeView.quant(descriptor.quantFormat, style: .compact)
                }
            }
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 6)
    }

    /// The normal row: one button that opens the detail pane, plus the trash.
    ///
    /// **THE TRASH IS FINDABLE AND NEVER HOVER-ONLY.** It is drawn while the
    /// pointer is over the row or the trash holds keyboard focus, and it stays in
    /// the accessibility tree and the key-focus chain the whole time (opacity, not
    /// removal), so keyboard and VoiceOver users reach it; the row also exposes
    /// the same action by name, and the context menu offers "Delete...". All three
    /// open the one confirmation in the list.
    private var openableRow: some View {
        ZStack(alignment: .bottomTrailing) {
            openButton
            if let onRequestDelete, rowModel.offersDelete {
                let showTrash = isHovering || trashFocused
                AppIconButton(
                    "trash", label: Text("Delete \(modelItem.alias)", bundle: .module), role: .destructive,
                    action: onRequestDelete
                )
                .appAction(.destructive, size: .small)
                .focused($trashFocused)
                .opacity(showTrash ? 1 : 0)
                // A hidden trash must not take a stray click; keyboard focus and
                // VoiceOver still reach it and make it visible.
                .allowsHitTesting(showTrash)
                .padding(.trailing, 6)
                .padding(.bottom, 4)
            }
        }
        .onHover { isHovering = $0 }
        .contextMenu {
            if let onRequestDelete, rowModel.offersDelete {
                Button(role: .destructive, action: onRequestDelete) {
                    Text("Delete...", bundle: .module)
                }
            }
        }
    }

    private var openButton: some View {
        Button(action: onSelect) {
            rowContent
                .contentShape(RoundedRectangle(cornerRadius: 8))
        }
        .buttonStyle(.plain)
        .background(
            isSelected ? AnyShapeStyle(.appAccent.opacity(0.14)) : AnyShapeStyle(Color.clear),
            in: RoundedRectangle(cornerRadius: 8)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(
                    isSelected ? AnyShapeStyle(.appAccent.opacity(0.4)) : AnyShapeStyle(Color.clear),
                    lineWidth: 1
                )
        )
        .help(Text(verbatim: rowModel.accessibilityLabel))
        .accessibilityElement(children: .combine)
        // One element: name, library, status, size, load-time estimate, in that
        // order (the model builds it), then the state words below as its value.
        .accessibilityLabel(rowModel.accessibilityLabel)
        .accessibilityValue(accessibilityDescription)
        .accessibilityAddTraits(isSelected ? [.isButton, .isSelected] : .isButton)
        .accessibilityAction(named: Text("Delete...", bundle: .module)) {
            if rowModel.offersDelete { onRequestDelete?() }
        }
    }

    /// The one state a row shows. A model that cannot be opened says why, with a
    /// glyph AND a word so it never rests on color, and drawn above the chips so
    /// a narrow list and a long translation still show it. Otherwise the loaded
    /// model says Loaded. A model that is simply there says nothing.
    @ViewBuilder private var stateMarker: some View {
        if let status = rowModel.status, let text = rowModel.statusText {
            HStack(alignment: .firstTextBaseline, spacing: 4) {
                Image(systemName: status.glyph)
                    .themedFont(.tiny)
                    .accessibilityHidden(true)
                Text(verbatim: text)
                    .themedFont(.tiny, weight: .semibold)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .foregroundStyle(.appText)
        } else if isLoaded {
            HStack(spacing: 3) {
                Circle().fill(.appSuccess).frame(width: 5, height: 5)
                Text("Loaded", bundle: .module)
                    .themedFont(.tiny, weight: .bold)
                    .foregroundStyle(.appSuccess)
            }
            .padding(.horizontal, 5)
            .padding(.vertical, 1.5)
            .background(.appSuccessTint, in: Capsule())
        }
    }

    /// A model the engine lists says which library holds it. A model only a scan
    /// found keeps the source badge (TurboSpark, LM Studio, Custom Folder): the
    /// source label of a library model would read "Custom Folder", which it is
    /// not. A long library name is cut at the tail, with the full name on hover.
    @ViewBuilder private var locationOrSourceBadge: some View {
        if let text = rowModel.locationText, let glyph = rowModel.locationGlyph {
            HStack(spacing: 3) {
                Image(systemName: glyph)
                    .themedFont(.micro)
                    .accessibilityHidden(true)
                Text(verbatim: text)
                    .themedFont(.micro, weight: .semibold)
                    .lineLimit(1)
                    .truncationMode(.tail)
            }
            // Short enough that the badge, the size and the logo fit the list
            // column at its narrowest (260 pt): a longer name would push the
            // size off the row's trailing edge.
            .frame(maxWidth: 120, alignment: .leading)
            .padding(.horizontal, 4)
            .padding(.vertical, 1.5)
            .background(.appElevated, in: RoundedRectangle(cornerRadius: 3))
            .foregroundStyle(.appSecondary)
            .help(Text(verbatim: text))
        } else {
            Text(LocalizedStringKey(descriptor.storageSource.shortLabel), bundle: .module)
                .themedFont(.micro, weight: .semibold)
                .lineLimit(1)
                .padding(.horizontal, 4)
                .padding(.vertical, 1.5)
                .background(.appElevated, in: RoundedRectangle(cornerRadius: 3))
                .foregroundStyle(.appSecondary)
        }
    }

    /// Spoken row state: state words from the catalog, data verbatim. The name,
    /// library, status, size and estimate are the label.
    private var accessibilityDescription: Text {
        var parts: [Text] = []
        if isLoaded { parts.append(Text("Loaded", bundle: .module)) }
        if isFavorite { parts.append(Text("Favorite", bundle: .module)) }
        parts.append(Text(LocalizedStringKey(descriptor.routingType.label), bundle: .module))
        if !tags.isEmpty {
            parts.append(Text("Tags:", bundle: .module) + Text(verbatim: " " + tags.joined(separator: ", ")))
        }
        return parts.dropFirst().reduce(parts[0]) { $0 + Text(verbatim: ", ") + $1 }
    }
}
