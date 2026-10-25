import SwiftUI

/// The shared control vocabulary every pane draws its actions, search
/// fields, headers, editors and empty states from.
///
/// The UI audit of 2026-10-11 found each pane picking its own button size,
/// search-field chrome and header layout: Files used a 22pt custom capsule,
/// Settings mixed three native button sizes, and the MCP and Agents headers
/// truncated because a fixed row had no fallback. These are thin wrappers
/// over NATIVE controls on purpose: `Button`, `Menu`, `TextField` and
/// `Toggle` keep their keyboard, focus and VoiceOver semantics, and only the
/// treatment is shared. Colors and type come from the resolved theme
/// (`.appSurface`, `.appBorder`, `.themedFont`) so a custom theme and the
/// text-size setting reach every surface built from these.
enum AppMetrics {
    /// Corner radius of a text field, chip or small control surface.
    static let controlRadius: CGFloat = 7
    /// Corner radius of a card, editor or grouped panel.
    static let cardRadius: CGFloat = 10
    /// Gap between actions in a header row.
    static let actionSpacing: CGFloat = 8

    /// The type role `appAction` and `appMenuAction` give a control's label
    /// at `size`, or nil to inherit. Small and mini follow AppKit's own
    /// control fonts (11 and 9pt against a 13pt regular), which sit closest
    /// to `.small` and `.tiny` on the scale.
    static func labelFontStep(for size: ControlSize) -> AppFontStep? {
        switch size {
        case .mini: return .tiny
        case .small: return .small
        default: return nil
        }
    }
}

/// The four action treatments. Pick by CONSEQUENCE, not by looks: one
/// primary action per surface (the thing the pane exists to do), secondary
/// for the rest, quiet for in-row and repeated actions, destructive for
/// anything that deletes or discards.
enum AppActionRole {
    case primary
    case secondary
    case quiet
    case destructive
}

extension View {
    /// Applies the shared treatment for `role` on top of the native button.
    ///
    /// Destructive keeps the bordered shape and only changes the label
    /// color: pair it with `Button(role: .destructive)` so the semantic
    /// role reaches VoiceOver and confirmation dialogs as well.
    @ViewBuilder
    func appAction(_ role: AppActionRole, size: ControlSize = .regular) -> some View {
        switch role {
        case .primary:
            buttonStyle(.borderedProminent).controlSize(size).appControlLabelFont(size)
        case .secondary:
            buttonStyle(.bordered).controlSize(size).appControlLabelFont(size)
        case .quiet:
            buttonStyle(.borderless).controlSize(size).appControlLabelFont(size)
        case .destructive:
            buttonStyle(.bordered).controlSize(size).appControlLabelFont(size).foregroundStyle(.red)
        }
    }

    /// A pull-down `Menu` drawn like a secondary action, for grouping the
    /// overflow of a crowded header (Import sources, Refresh, Marketplace)
    /// behind one labeled control.
    func appMenuAction(size: ControlSize = .regular) -> some View {
        menuStyle(.button)
            .buttonStyle(.bordered)
            .controlSize(size)
            .appControlLabelFont(size)
            .fixedSize()
    }

    /// The label font a reduced control size needs. A native button keeps
    /// its small bezel height but draws its label in the inherited font, and
    /// every window sets the base UI font, so small row actions drew
    /// base-size text that overflowed the bezel (Choose Model Folder in the
    /// inspector). Regular and large keep inheriting. A font set inside the
    /// label at the call site still wins, being nearer the text.
    @ViewBuilder
    fileprivate func appControlLabelFont(_ size: ControlSize) -> some View {
        if let step = AppMetrics.labelFontStep(for: size) {
            themedFont(step)
        } else {
            self
        }
    }

    /// The shared surface for a multi-line `TextEditor` or code area: the
    /// raised theme surface, the theme hairline and a continuous corner.
    func appEditorSurface(minHeight: CGFloat? = nil) -> some View {
        modifier(AppEditorSurfaceModifier(minHeight: minHeight))
    }
}

private struct AppEditorSurfaceModifier: ViewModifier {
    let minHeight: CGFloat?

    func body(content: Content) -> some View {
        content
            .scrollContentBackground(.hidden)
            .padding(8)
            .frame(minHeight: minHeight)
            .background(.appSurface, in: RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous)
                    .stroke(.appBorder, lineWidth: 1)
            )
    }
}

/// The one search field. `prompt` is a `Text` rather than a key so the call
/// site spells the bundled literal itself, which keeps it visible to the
/// localization source scan (swift/docs/SWIFT_LOCALIZATION.md gates 5 and 7)
/// and doubles as the field's accessible name.
struct AppSearchField: View {
    @Binding var text: String
    let prompt: Text
    var minWidth: CGFloat = 140
    var idealWidth: CGFloat = 220
    var maxWidth: CGFloat = 360
    /// For a field that should take focus (a picker sheet's search) or be
    /// focused from a shortcut. Optional so most call sites ignore it.
    var focus: FocusState<Bool>.Binding?

    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: "magnifyingglass")
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
                .accessibilityHidden(true)
            field
            if !text.isEmpty {
                Button {
                    text = ""
                } label: {
                    Label { Text("Clear search", bundle: .module) } icon: { Image(systemName: "xmark.circle.fill") }
                        .labelStyle(.iconOnly)
                }
                .buttonStyle(.borderless)
                .foregroundStyle(.appSecondary)
                .help(Text("Clear search", bundle: .module))
            }
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 5)
        .background(.appSurface, in: RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous)
                .stroke(.appBorder, lineWidth: 1)
        )
        .frame(minWidth: minWidth, idealWidth: idealWidth, maxWidth: maxWidth)
    }

    @ViewBuilder
    private var field: some View {
        let base = TextField(text: $text, prompt: prompt) { prompt }
            .textFieldStyle(.plain)
            .themedFont(.base)
        if let focus {
            base.focused(focus)
        } else {
            base
        }
    }
}

/// An icon-only action that cannot ship without a name: the label is the
/// VoiceOver name and the tooltip, so neither can drift from the other.
struct AppIconButton: View {
    let systemImage: String
    let label: Text
    var role: ButtonRole?
    let action: () -> Void

    init(_ systemImage: String, label: Text, role: ButtonRole? = nil, action: @escaping () -> Void) {
        self.systemImage = systemImage
        self.label = label
        self.role = role
        self.action = action
    }

    var body: some View {
        Button(role: role, action: action) {
            Label { label } icon: { Image(systemName: systemImage) }
                .labelStyle(.iconOnly)
        }
        .help(label)
    }
}

/// A pane or workspace header that adapts instead of truncating.
///
/// Three layouts, tried in order by `ViewThatFits`: one row (leading content
/// beside the actions), two rows (actions under the leading content), and
/// compact (the leading content over `compactActions`, which should keep
/// the primary action visible and fold the rest into a labeled menu). The
/// last layout always takes the proposal, so long translations and the
/// largest text size still lay out rather than clip.
struct AppAdaptiveHeader<Leading: View, Actions: View, CompactActions: View>: View {
    @ViewBuilder let leading: () -> Leading
    @ViewBuilder let actions: () -> Actions
    @ViewBuilder let compactActions: () -> CompactActions

    init(
        @ViewBuilder leading: @escaping () -> Leading,
        @ViewBuilder actions: @escaping () -> Actions,
        @ViewBuilder compactActions: @escaping () -> CompactActions
    ) {
        self.leading = leading
        self.actions = actions
        self.compactActions = compactActions
    }

    var body: some View {
        ViewThatFits(in: .horizontal) {
            HStack(alignment: .center, spacing: 12) {
                leading()
                Spacer(minLength: 12)
                HStack(spacing: AppMetrics.actionSpacing) { actions() }
            }
            VStack(alignment: .leading, spacing: 10) {
                leading()
                HStack(spacing: AppMetrics.actionSpacing) { actions() }
            }
            VStack(alignment: .leading, spacing: 10) {
                leading()
                HStack(spacing: AppMetrics.actionSpacing) { compactActions() }
            }
        }
    }
}

extension AppAdaptiveHeader where CompactActions == Actions {
    /// A header whose actions already fit a narrow width once stacked.
    init(
        @ViewBuilder leading: @escaping () -> Leading,
        @ViewBuilder actions: @escaping () -> Actions
    ) {
        self.leading = leading
        self.actions = actions
        self.compactActions = actions
    }
}

/// A section heading inside a scrolling pane: semibold title marked as a
/// heading for VoiceOver's rotor, with an optional secondary line.
struct AppSectionHeader: View {
    let title: Text
    var subtitle: Text?

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            title
                .themedFont(.base, weight: .semibold)
                .accessibilityAddTraits(.isHeader)
            if let subtitle {
                subtitle
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

/// The empty state every list and workspace uses: a symbol, a title, an
/// explanation and, when there is one, the action that fills the space.
struct AppEmptyState<Actions: View>: View {
    let systemImage: String
    let title: Text
    var message: Text?
    @ViewBuilder let actions: () -> Actions

    init(
        systemImage: String,
        title: Text,
        message: Text? = nil,
        @ViewBuilder actions: @escaping () -> Actions
    ) {
        self.systemImage = systemImage
        self.title = title
        self.message = message
        self.actions = actions
    }

    var body: some View {
        VStack(spacing: 12) {
            Image(systemName: systemImage)
                .themedFont(.display)
                .foregroundStyle(.appSecondary)
                .accessibilityHidden(true)
            title
                .themedFont(.title3, weight: .semibold)
                .multilineTextAlignment(.center)
                .accessibilityAddTraits(.isHeader)
            if let message {
                message
                    .themedFont(.base)
                    .foregroundStyle(.appSecondary)
                    .multilineTextAlignment(.center)
                    .fixedSize(horizontal: false, vertical: true)
                    .frame(maxWidth: 440)
            }
            HStack(spacing: AppMetrics.actionSpacing) { actions() }
                .padding(.top, 4)
        }
        .padding(32)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

extension AppEmptyState where Actions == EmptyView {
    init(systemImage: String, title: Text, message: Text? = nil) {
        self.init(systemImage: systemImage, title: title, message: message) { EmptyView() }
    }
}

/// The chip every composer-style surface uses (chat composer toggles,
/// Humanize's options): radius 8, a faint fill that deepens on hover, and an
/// accent tint while on. One definition so the chat and Humanize chips stop
/// drifting apart, which the 2026-10-11 audit found they had.
///
/// `label` is the visible title, the tooltip and the VoiceOver name, so an
/// icon-only chip still announces what it does; `isOn` is exposed as the
/// selected trait for toggles.
struct AppChip: View {
    let label: Text
    let systemImage: String
    var isOn = false
    var iconOnly = false
    let action: () -> Void

    @Environment(\.appTheme) private var theme
    @Environment(\.isEnabled) private var isEnabled
    @State private var hovered = false

    init(_ label: Text, systemImage: String, isOn: Bool = false, iconOnly: Bool = false, action: @escaping () -> Void) {
        self.label = label
        self.systemImage = systemImage
        self.isOn = isOn
        self.iconOnly = iconOnly
        self.action = action
    }

    var body: some View {
        Button(action: action) {
            HStack(spacing: 5) {
                Image(systemName: systemImage)
                    .themedFont(.tiny, weight: .medium)
                    .accessibilityHidden(true)
                if !iconOnly {
                    label.themedFont(.small, weight: .medium)
                }
            }
            .lineLimit(1)
            .padding(.horizontal, 8)
            .padding(.vertical, 5)
            .foregroundStyle(isOn ? Color.primary : Color.secondary)
            .background(
                isOn ? theme.accent.opacity(0.14) : Color.primary.opacity(hovered && isEnabled ? 0.08 : 0.04),
                in: RoundedRectangle(cornerRadius: 8, style: .continuous))
            .overlay {
                if isOn {
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .stroke(theme.accent.opacity(0.3), lineWidth: 0.5)
                }
            }
            .opacity(isEnabled ? 1 : 0.45)
            .contentShape(Rectangle())
        }
        .buttonStyle(TSPressScaleStyle(scale: 0.96))
        .onHover { hovered = $0 }
        .tsAnimation(TSMotion.hover, value: isOn)
        .fixedSize()
        .help(label)
        .accessibilityLabel(label)
        .accessibilityAddTraits(isOn ? .isSelected : [])
    }
}
