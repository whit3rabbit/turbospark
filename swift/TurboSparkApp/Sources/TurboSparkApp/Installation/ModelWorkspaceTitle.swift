import SwiftUI

/// The leading block of both model workspace headers (Installed and
/// Discover): a tinted symbol tile, the title with its scope badge, and one
/// summary line.
///
/// One definition so the two workspaces cannot drift apart again; the UI
/// audit of 2026-10-11 found them with different header structure, search
/// placement and action styles. The summary wraps instead of truncating, so
/// a long translation adds a line rather than losing the counts.
///
/// Always the theme accent: Installed used a fixed teal that ignored custom
/// themes and drew its badge text below 4.5:1 in light mode. The wash takes
/// the accent and the text and glyph take `.appAccentText`, which light mode
/// darkens to stay readable.
struct ModelWorkspaceTitle: View {
    let systemImage: String
    let title: Text
    /// A scope tag beside the title. Nil when the title and the summary already
    /// say it (Installed: the summary reads "On disk").
    var badge: Text?
    let summary: Text

    var body: some View {
        HStack(spacing: 8) {
            ZStack {
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .fill(.appAccent.opacity(0.15))
                    .frame(width: 30, height: 30)
                Image(systemName: systemImage)
                    .themedFont(.callout, weight: .semibold)
                    .foregroundStyle(.appAccentText)
            }
            .accessibilityHidden(true)

            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 6) {
                    title
                        .themedFont(.callout, weight: .semibold)
                        .accessibilityAddTraits(.isHeader)
                    if let badge {
                        badge
                            .themedFont(.micro, weight: .bold)
                            .padding(.horizontal, 5)
                            .padding(.vertical, 1.5)
                            .background(.appAccent.opacity(0.15), in: RoundedRectangle(cornerRadius: 4))
                            .foregroundStyle(.appAccentText)
                    }
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
