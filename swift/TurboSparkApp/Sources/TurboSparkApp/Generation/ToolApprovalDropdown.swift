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
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(Color.secondary)

                Text(currentMode.label)
                    .font(theme.ui(points: 12, weight: .medium))
                    .foregroundStyle(Color.primary.opacity(0.85))
                    .lineLimit(1)

                Image(systemName: "chevron.down")
                    .font(.system(size: 8, weight: .bold))
                    .foregroundStyle(.tertiary)
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 5)
            .background(
                Color.primary.opacity(isHovered ? 0.08 : 0.04),
                in: RoundedRectangle(cornerRadius: 8, style: .continuous)
            )
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .onHover { isHovered = $0 }
        .help("Tool call approval mode: \(currentMode.label)")
        .accessibilityLabel("Tool call approval: \(currentMode.label)")
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

    private let displayModes: [AppPermissionMode] = [
        .ask,
        .auto,
        .permissive,
        .fullAccess,
    ]

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("How should tool calls be approved?")
                .font(theme.ui(points: 11, weight: .medium))
                .foregroundStyle(.tertiary)
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
        .frame(width: 320)
        .background(TurboSparkTheme.surfaceColor)
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
                Image(systemName: mode.systemImage)
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(isSelected ? TurboSparkTheme.accentColor : Color.secondary)
                    .frame(width: 20, height: 20)
                    .padding(.top, 1)

                VStack(alignment: .leading, spacing: 2) {
                    Text(mode.label)
                        .font(theme.ui(points: 12, weight: isSelected ? .semibold : .medium))
                        .foregroundStyle(Color.primary)

                    Text(mode.descriptionText)
                        .font(theme.ui(points: 11))
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                        .lineSpacing(1.5)
                }

                Spacer(minLength: 4)

                if isSelected {
                    Image(systemName: "checkmark")
                        .font(.system(size: 11, weight: .bold))
                        .foregroundStyle(Color.primary)
                        .padding(.top, 2)
                }
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 8)
            .background(
                Color.primary.opacity(isHovered ? 0.08 : (isSelected ? 0.04 : 0)),
                in: RoundedRectangle(cornerRadius: 8, style: .continuous)
            )
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .onHover { isHovered = $0 }
    }
}
