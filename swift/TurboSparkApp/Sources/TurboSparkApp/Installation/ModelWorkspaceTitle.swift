import SwiftUI

/// The leading block of both model workspace headers (Installed and
/// Discover): a tinted symbol tile, the title with its scope badge, and one
/// summary line.
///
/// One definition so the two workspaces cannot drift apart again; the UI
/// audit of 2026-10-11 found them with different header structure, search
/// placement and action styles. The summary wraps instead of truncating, so
/// a long translation adds a line rather than losing the counts.
struct ModelWorkspaceTitle<Tint: ShapeStyle>: View {
    let systemImage: String
    /// Generic so a theme token (`.appAccent`) and a fixed hue (`.teal`)
    /// both work without resolving the theme here.
    let tint: Tint
    let title: Text
    let badge: Text
    let summary: Text

    var body: some View {
        HStack(spacing: 8) {
            ZStack {
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .fill(tint.opacity(0.15))
                    .frame(width: 30, height: 30)
                Image(systemName: systemImage)
                    .themedFont(.callout, weight: .semibold)
                    .foregroundStyle(tint)
            }
            .accessibilityHidden(true)

            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 6) {
                    title
                        .themedFont(.callout, weight: .semibold)
                        .accessibilityAddTraits(.isHeader)
                    badge
                        .themedFont(.micro, weight: .bold)
                        .padding(.horizontal, 5)
                        .padding(.vertical, 1.5)
                        .background(tint.opacity(0.15), in: RoundedRectangle(cornerRadius: 4))
                        .foregroundStyle(tint)
                }
                summary
                    .themedFont(.tiny)
                    .foregroundStyle(.appSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }
}

/// A filter dropdown shared by the Installed and Discover filter bars: the
/// shared menu treatment, the current value as its title (so an active
/// filter reads without opening it), and the filter's name as tooltip and
/// VoiceOver name.
struct ModelFilterMenu<Content: View>: View {
    let name: Text
    let value: Text
    @ViewBuilder let content: () -> Content

    var body: some View {
        Menu {
            content()
        } label: {
            value.lineLimit(1)
        }
        .appMenuAction(size: .small)
        .help(name)
        .accessibilityLabel(name)
        .accessibilityValue(value)
    }
}
