import SwiftUI
import TurboSpark

/// Dropdown control in the chat composer selecting how tool calls should be approved (Unsloth Studio parity).
struct ToolApprovalDropdown: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel
    @State private var isMenuPresented: Bool = false
    @State private var isHovered: Bool = false

    private var currentMode: AppPermissionMode {
        model.effectivePermissionMode
    }

    var body: some View {
        Button {
            isMenuPresented.toggle()
        } label: {
            HStack(spacing: 5) {
                Image(systemName: currentMode.systemImage)
                    .themedFont(.tiny, weight: .semibold)
                    .foregroundStyle(.appSecondary)

                currentMode.approvalLabelText
                    .font(theme.ui(.small, weight: .medium))
                    .foregroundStyle(.appText)
                    .lineLimit(1)

                Image(systemName: "chevron.down")
                    .themedFont(.micro, weight: .bold)
                    .foregroundStyle(.appSecondary)
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 5)
            .background(
                isHovered ? AnyShapeStyle(.appElevated) : AnyShapeStyle(.appSurface),
                in: RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous)
            )
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .onHover { isHovered = $0 }
        .help(Text("Tool call approval mode: \(currentMode.approvalLabelText)", bundle: .module))
        .accessibilityLabel(Text("Tool call approval: \(currentMode.approvalLabelText)", bundle: .module))
        .popover(
            isPresented: $isMenuPresented,
            arrowEdge: .top
        ) {
            ToolApprovalMenuPopover(
                currentMode: currentMode,
                onSelect: { selected in
                    model.setEffectivePermissionMode(selected)
                    isMenuPresented = false
                }
            )
        }
    }
}

/// Popover menu showing tool approval modes with titles and descriptions matching Unsloth Studio.
struct ToolApprovalMenuPopover: View {
    @Environment(\.appTheme) private var theme
    let currentMode: AppPermissionMode
    let onSelect: (AppPermissionMode) -> Void

    private static let standardModes: [AppPermissionMode] = [
        .ask,
        .auto,
        .permissive,
        .fullAccess,
    ]

    /// The active mode is always listed and checked, even when it is not one
    /// of the standard rows (for example Agent mode), so it can be re-selected.
    static func displayModes(including current: AppPermissionMode) -> [AppPermissionMode] {
        standardModes.contains(current) ? standardModes : [current] + standardModes
    }

    private var displayModes: [AppPermissionMode] {
        Self.displayModes(including: currentMode)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("How should tool calls be approved?", bundle: .module)
                .font(theme.ui(.tiny, weight: .medium))
                .foregroundStyle(.appSecondary)
                .padding(.horizontal, 14)
                .padding(.top, 12)
                .padding(.bottom, 2)

            VStack(spacing: 2) {
                ForEach(displayModes) { mode in
                    ToolApprovalOptionRow(
                        mode: mode,
                        isSelected: mode == currentMode,
                        onSelect: { onSelect(mode) }
                    )
                }
            }
            .padding(.horizontal, 6)
            .padding(.bottom, 8)
        }
        .frame(minWidth: 320, idealWidth: 320)
        .background(.appSurface)
    }
}

private struct ToolApprovalOptionRow: View {
    @Environment(\.appTheme) private var theme
    let mode: AppPermissionMode
    let isSelected: Bool
    let onSelect: () -> Void
    @State private var isHovered: Bool = false

    var body: some View {
        Button(action: onSelect) {
            HStack(alignment: .top, spacing: 10) {
                let glyphSize = (18 * AppMetrics.textScale(theme)).rounded()
                Image(systemName: mode.systemImage)
                    .themedFont(.callout, weight: .semibold)
                    .foregroundStyle(isSelected ? AnyShapeStyle(theme.accent) : AnyShapeStyle(.appSecondary))
                    .frame(width: glyphSize, height: glyphSize)
                    .padding(.top, 1)

                VStack(alignment: .leading, spacing: 2) {
                    mode.approvalLabelText
                        .font(theme.ui(.small, weight: isSelected ? .semibold : .medium))
                        .foregroundStyle(.appText)

                    mode.approvalDescriptionText
                        .font(theme.ui(.tiny))
                        .foregroundStyle(.appSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                        .lineSpacing(1.5)
                }

                Spacer(minLength: 4)

                if isSelected {
                    Image(systemName: "checkmark")
                        .themedFont(.tiny, weight: .bold)
                        .foregroundStyle(.appText)
                        .padding(.top, 2)
                }
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 8)
            .background(
                isHovered
                    ? AnyShapeStyle(.appElevated)
                    : (isSelected ? AnyShapeStyle(theme.accent.opacity(0.12)) : AnyShapeStyle(Color.clear)),
                in: RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous)
            )
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityAddTraits(isSelected ? [.isButton, .isSelected] : .isButton)
        .onHover { isHovered = $0 }
    }
}

extension AppPermissionMode {
    /// `label` in the UI language. Every label and description is a catalog
    /// key, so the composer, the Add menu and the project sheet show the
    /// same translated names for the same consequential choice.
    var approvalLabelText: Text { Text(LocalizedStringKey(label), bundle: .module) }

    /// `descriptionText` in the UI language: what the mode allows.
    var approvalDescriptionText: Text { Text(LocalizedStringKey(descriptionText), bundle: .module) }
}
