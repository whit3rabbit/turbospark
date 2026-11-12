import SwiftUI

/// One option of an `AppPillPicker`.
struct AppPillItem<Value: Hashable>: Identifiable {
    let value: Value
    let label: Text
    var systemImage: String?
    /// Set while this option has live work, such as a recording. Drawn as a
    /// breathing dot and spoken as the option's value, so it is never color alone.
    var liveLabel: Text?
    var id: Value { value }

    init(value: Value, label: Text, systemImage: String? = nil, liveLabel: Text? = nil) {
        self.value = value
        self.label = label
        self.systemImage = systemImage
        self.liveLabel = liveLabel
    }
}

/// Switches one pane between a few modes. Native buttons on a capsule track,
/// with one selection capsule that slides between them, so the move reads as
/// one mark changing place (the rail's pattern). Keep it to two to four short
/// options: it does not shrink, and its header moves it to its own row.
@MainActor
struct AppPillPicker<Value: Hashable>: View {
    @Binding var selection: Value
    let items: [AppPillItem<Value>]
    /// Names the group for VoiceOver and the fallback menu.
    let label: Text

    @Environment(\.appTheme) private var theme
    @Environment(\.layoutDirection) private var layoutDirection
    @Namespace private var namespace
    @FocusState private var focused: Value?
    @State private var hovered: Value?

    init(selection: Binding<Value>, items: [AppPillItem<Value>], label: Text) {
        _selection = selection
        self.items = items
        self.label = label
    }

    // No ViewThatFits fallback to a menu picker: measuring the menu's AppKit
    // popup inside ViewThatFits, under the header's own ViewThatFits, kept the
    // window re-measuring its minimum size every frame (15 to 40% CPU idle,
    // unresponsive at German Extra Large). The header reflows the pills onto
    // their own row instead.
    var body: some View {
        pills
    }

    private var pills: some View {
        HStack(spacing: 2) {
            ForEach(items) { item in segment(item) }
        }
        .padding(3)
        .background(Capsule(style: .continuous).fill(theme.surface))
        .overlay(Capsule(style: .continuous).stroke(theme.border, lineWidth: 1))
        .fixedSize()
        .tsAnimation(TSMotion.select, value: selection)
        .accessibilityElement(children: .contain)
        .accessibilityLabel(label)
        .onKeyPress(.leftArrow) { move(by: -1) }
        .onKeyPress(.rightArrow) { move(by: 1) }
    }

    private func segment(_ item: AppPillItem<Value>) -> some View {
        let isSelected = item.value == selection
        let isFocused = focused == item.value
        return Button { selection = item.value } label: {
            HStack(spacing: 6) {
                if let symbol = item.systemImage {
                    Image(systemName: symbol)
                        .foregroundStyle(isSelected ? theme.accent : theme.secondaryText)
                        .accessibilityHidden(true)
                }
                item.label
                if item.liveLabel != nil {
                    Circle()
                        .fill(theme.accent)
                        .frame(width: 6, height: 6)
                        .tsPulse(active: true)
                        .accessibilityHidden(true)
                }
            }
            // One weight for every state, so selecting never changes a width.
            .themedFont(.small, weight: .medium)
            .lineLimit(1)
            .foregroundStyle(isSelected ? theme.foreground : theme.secondaryText)
            .padding(.horizontal, 12)
            .padding(.vertical, 5)
            .background {
                if isSelected {
                    Capsule(style: .continuous)
                        .fill(theme.selection)
                        .overlay {
                            if theme.isHighContrast {
                                Capsule(style: .continuous).stroke(theme.accent, lineWidth: 1)
                            }
                        }
                        // Only the selected segment hosts the capsule, so
                        // SwiftUI moves its frame from the old one to the new.
                        .matchedGeometryEffect(id: "pillSelection", in: namespace)
                } else if hovered == item.value {
                    Capsule(style: .continuous).fill(theme.foreground.opacity(0.07))
                }
            }
            .overlay {
                // Custom-styled buttons draw no focus ring on macOS.
                if isFocused {
                    Capsule(style: .continuous).stroke(theme.accent, lineWidth: 2)
                }
            }
            .contentShape(Capsule(style: .continuous))
        }
        .buttonStyle(.tsPress)
        .focusEffectDisabled()
        .focused($focused, equals: item.value)
        .help(item.label)
        .onHover { hovering in
            TSMotion.animate(TSMotion.hover) {
                if hovering { hovered = item.value } else if hovered == item.value { hovered = nil }
            }
        }
        .accessibilityLabel(item.label)
        .accessibilityValue(item.liveLabel ?? Text(verbatim: ""))
        .accessibilityAddTraits(isSelected ? [.isButton, .isSelected] : .isButton)
    }

    private func move(by step: Int) -> KeyPress.Result {
        let values = items.map(\.value)
        let next = Self.neighbor(of: selection, in: values, step: step, layoutDirection: layoutDirection)
        guard next != selection else { return .ignored }
        selection = next
        focused = next
        return .handled
    }

    /// The option an arrow key moves to: clamped at the ends, and mirrored
    /// under right-to-left, where the leftmost pill is the last one.
    nonisolated static func neighbor(of value: Value, in values: [Value], step: Int, layoutDirection: LayoutDirection) -> Value {
        guard let index = values.firstIndex(of: value) else { return values.first ?? value }
        let signed = layoutDirection == .rightToLeft ? -step : step
        return values[min(max(index + signed, 0), values.count - 1)]
    }
}
