import SwiftUI

// The menu bar panel's building blocks. The panel is a window-style
// MenuBarExtra, so its rows are drawn here to read like menu items while
// staying native Buttons, Menus and a Toggle underneath.

/// The status glyph and word at the top of the panel.
///
/// Only the glyph takes the status color. The word stays in the theme's
/// foreground: system green and orange as text fall under 4.5:1 on the light
/// surfaces, and the glyph plus the word already carry the state without
/// relying on color (DESIGN.md, Color).
@MainActor
struct MenuBarStatusLine: View {
    let phase: MenuBarServerPhase
    let address: String?
    let copyAddress: () -> Void

    var body: some View {
        HStack(spacing: 6) {
            glyph
                .accessibilityHidden(true)
            statusWord
                .themedFont(.small, weight: .semibold)
                .foregroundStyle(.appText)
            Spacer(minLength: 8)
            if let address {
                Button(action: copyAddress) {
                    HStack(spacing: 5) {
                        Text(verbatim: address)
                            .themedCode(.tiny)
                            .lineLimit(1)
                            .truncationMode(.middle)
                        Image(systemName: "doc.on.doc")
                            .themedFont(.micro)
                            .accessibilityHidden(true)
                    }
                    .foregroundStyle(.appSecondary)
                    .padding(.horizontal, 7)
                    .padding(.vertical, 3)
                    .background(
                        RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous)
                            .fill(.appSurface))
                }
                .buttonStyle(.plain)
                .help(Text("Copy the address", bundle: .module))
                .accessibilityLabel(Text("Copy the address", bundle: .module))
                .accessibilityValue(Text(verbatim: address))
            }
        }
    }

    static func color(for level: MenuBarServerPhase.Level) -> Color {
        switch level {
        case .off: return .secondary
        case .ok: return .green
        case .warning: return AppAttention.color
        case .error: return .red
        }
    }

    @ViewBuilder
    private var glyph: some View {
        switch phase {
        case .starting, .working:
            ProgressView().controlSize(.mini)
        default:
            Image(systemName: Self.symbol(for: phase))
                .themedFont(.small)
                .foregroundStyle(Self.color(for: phase.level))
        }
    }

    static func symbol(for phase: MenuBarServerPhase) -> String {
        switch phase {
        case .stopped: return "circle"
        case .running, .daemon: return "checkmark.circle.fill"
        case .idle, .degraded, .starting, .working: return "exclamationmark.triangle.fill"
        case .failed: return "xmark.octagon.fill"
        }
    }

    private var statusWord: Text {
        switch phase {
        case .stopped: return Text("Stopped", bundle: .module)
        case .starting: return Text("Starting TurboSpark Server...", bundle: .module)
        case .working: return Text("Loading...", bundle: .module)
        case .failed: return Text("Failed", bundle: .module)
        case .degraded: return Text("Degraded", bundle: .module)
        case .idle, .running, .daemon: return Text("Running", bundle: .module)
        }
    }
}

/// One resident model: kind glyph, id, kind, and its release button.
@MainActor
struct MenuBarSlotRow: View {
    let slot: MenuBarModelSlot
    let canUnload: Bool
    let unload: () -> Void

    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: slot.kind.systemImage)
                .themedFont(.small)
                .foregroundStyle(slot.kind == .chat ? AnyShapeStyle(.appAccent) : AnyShapeStyle(.appSecondary))
                .frame(width: 26, height: 26)
                .background(
                    RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous)
                        .fill(.appSurface))
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 1) {
                Text(verbatim: slot.name)
                    .themedCode(.small)
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .help(Text(verbatim: slot.name))
                Self.kindLabel(slot.kind)
                    .themedFont(.tiny)
                    .foregroundStyle(.appSecondary)
            }
            Spacer(minLength: 4)
            if let bytes = slot.residentBytes {
                Text(verbatim: MenuBarMemoryReading.format(bytes))
                    .themedCode(.tiny)
                    .foregroundStyle(.appSecondary)
            }
            AppIconButton("eject", label: Text("Eject model", bundle: .module) + Text(verbatim: " \(slot.name)"), action: unload)
                .buttonStyle(.borderless)
                .disabled(!canUnload)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .accessibilityElement(children: .contain)
    }

    static func kindLabel(_ kind: MenuBarModelKind) -> Text {
        switch kind {
        case .chat: return Text("Chat", bundle: .module)
        case .embedding: return Text("Embeddings", bundle: .module)
        case .audio: return Text("Audio", bundle: .module)
        case .image: return Text("Image", bundle: .module)
        case .decision: return Text("TypeSafe", bundle: .module)
        }
    }
}

/// A full-width, menu-like row label: title, optional trailing detail,
/// optional submenu chevron.
struct MenuBarRowLabel: View {
    let title: Text
    var trailing: Text?
    var chevron = false

    var body: some View {
        HStack(spacing: 8) {
            title
                .themedFont(.small)
                .foregroundStyle(.appText)
            Spacer(minLength: 8)
            if let trailing {
                trailing
                    .themedFont(.tiny)
                    .foregroundStyle(.appSecondary)
            }
            if chevron {
                Image(systemName: "chevron.right")
                    .themedFont(.micro, weight: .semibold)
                    .foregroundStyle(.appSecondary)
                    .accessibilityHidden(true)
            }
        }
        .padding(.horizontal, 8)
        .frame(maxWidth: .infinity, minHeight: 26, alignment: .leading)
        .contentShape(Rectangle())
    }
}

/// The menu-item hover fill, drawn behind a row without moving it.
struct MenuBarRowHover: ViewModifier {
    @Environment(\.appTheme) private var theme
    @State private var isHovered = false

    func body(content: Content) -> some View {
        content
            .background(
                RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous)
                    .fill(isHovered ? theme.selection : Color.clear))
            .onHover { isHovered = $0 }
            .tsAnimation(TSMotion.hover, value: isHovered)
    }
}

extension View {
    func menuBarRowHover() -> some View { modifier(MenuBarRowHover()) }
}

/// A plain action row in the panel.
struct MenuBarRowButton: View {
    let title: Text
    var trailing: Text?
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            MenuBarRowLabel(title: title, trailing: trailing)
        }
        .buttonStyle(.plain)
        .menuBarRowHover()
    }
}

/// A row that opens a native submenu.
struct MenuBarSubmenuRow<Items: View>: View {
    let title: Text
    @ViewBuilder let items: () -> Items

    var body: some View {
        Menu {
            items()
        } label: {
            MenuBarRowLabel(title: title, chevron: true)
        }
        .menuStyle(.button)
        .buttonStyle(.plain)
        .menuIndicator(.hidden)
        .menuBarRowHover()
    }
}

/// The hairline between the panel's groups.
struct MenuBarDivider: View {
    var body: some View {
        Divider()
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
    }
}
