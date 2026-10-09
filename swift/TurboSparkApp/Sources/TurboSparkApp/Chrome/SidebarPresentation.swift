import SwiftUI

/// Keep both sidebar presentations in the same order without changing shortcuts.
enum SidebarNavigationGroup: CaseIterable, Identifiable {
    case creation, models, utilities

    var id: Self { self }

    var sections: [AppModel.AppNavigationSection] {
        switch self {
        case .creation: return [.chat, .images, .audio]
        case .models: return [.modelManager, .modelHub]
        case .utilities: return [.server, .files]
        }
    }
}

extension ResolvedAppTheme {
    /// Layout follows the same resolved size as text, including the user's
    /// font size, Text Size preference, and accessibility scaling.
    var sidebarScale: CGFloat { max(1, uiFontDescriptor.size / AppFontCatalog.defaultUISize) }
    var sidebarRowHeight: CGFloat { 30 * sidebarScale }
    var sidebarActionSize: CGFloat { 24 * sidebarScale }
}

struct SidebarDivider: View {
    var body: some View {
        Rectangle()
            .fill(.appBorder)
            .frame(height: 0.5)
            .accessibilityHidden(true)
    }
}

/// Shared by primary sidebar actions so platform button defaults cannot
/// make an empty-state action larger than the rows beside it.
struct SidebarActionButtonStyle: ButtonStyle {
    @Environment(\.appTheme) private var theme
    @Environment(\.isEnabled) private var isEnabled

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(theme.ui(.small, weight: .medium))
            .foregroundStyle(.appText)
            .padding(.horizontal, 10)
            .padding(.vertical, 5)
            .frame(maxWidth: .infinity, minHeight: theme.sidebarRowHeight, alignment: .leading)
            .background(configuration.isPressed ? theme.selection : theme.surface,
                        in: RoundedRectangle(cornerRadius: 7))
            .overlay {
                RoundedRectangle(cornerRadius: 7).stroke(.appBorder, lineWidth: 0.5)
            }
            .contentShape(RoundedRectangle(cornerRadius: 7))
            .opacity(isEnabled ? 1 : 0.5)
    }
}

