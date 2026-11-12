import SwiftUI

/// One row of the Model libraries section: icon, name, status, location, count,
/// free space, link, warnings and the actions that apply to it
/// (requirements 1.3, 3.8, 5.1, 5.2 and 5.7).
///
/// **IT DRAWS A `ModelLibraryRowModel` AND NOTHING ELSE.** Every string comes
/// from the model, so what a sighted user reads and what VoiceOver reads cannot
/// drift apart, and a table test pins both.
///
/// **ONE ACCESSIBILITY ELEMENT.** The row is read as a single item in the order
/// name, status, location, count, free space, link, warnings, default. Its
/// actions are accessibility actions on that element, so a VoiceOver user
/// reaches them from the actions rotor rather than by walking six buttons per
/// row. The visible buttons stay real controls for the pointer and for Full
/// Keyboard Access.
///
/// **STATUS IS A GLYPH AND A SENTENCE.** Color only tints the glyph.
///
/// Names and paths are drive-supplied text: the model already escaped and cut
/// them, and the view only draws them (`Text(verbatim:)`, never a key).
struct ModelLibraryRowView: View {
    let row: ModelLibraryRowModel
    /// An install or a move is running: changes are disabled, not hidden.
    let changesBlocked: Bool
    let speedTest: LibrarySpeedTestState?
    let perform: (LibraryRowAction) -> Void
    let cancelSpeedTest: () -> Void
    let dismissSpeedTestFailure: () -> Void

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: row.iconName)
                .themedFont(.title3)
                .foregroundStyle(row.isConnected ? AppThemeColor.appAccent : AppThemeColor.appSecondary)
                .frame(minWidth: 24)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 6) {
                titleLine
                statusLine
                if let location = row.location {
                    SettingsPathText(location, size: .tiny)
                }
                details
                if let warning = row.slowLinkWarning {
                    noteLine(glyph: LibraryRowGlyph.warning, tint: AnyShapeStyle(AppAttention.color), text: warning)
                }
                if let measured = row.measuredSpeedText {
                    Text(verbatim: measured)
                        .themedFont(.small)
                        .foregroundStyle(.appSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                if let explanation = row.explanation {
                    Text(verbatim: explanation)
                        .themedFont(.small)
                        .foregroundStyle(.appSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                speedTestLine
                actionRow
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .appCardSurface(padding: 10)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(Text(verbatim: spokenLabel))
        .accessibilityActions {
            ForEach(spokenActions, id: \.self) { action in
                Button {
                    perform(action)
                } label: {
                    Text(LocalizedStringKey(action.title), bundle: .module)
                }
            }
            if speedTest == .running {
                Button {
                    cancelSpeedTest()
                } label: {
                    Text("Cancel", bundle: .module)
                }
            }
        }
    }

    // MARK: - Lines

    private var titleLine: some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Text(verbatim: row.title)
                .themedFont(.base, weight: .semibold)
                .lineLimit(2)
                .multilineTextAlignment(.leading)
                .help(Text(verbatim: row.title))
            if row.isDefault { defaultBadge }
            Spacer(minLength: 0)
        }
    }

    /// A star and the word, so the badge is not a color. The word is the
    /// existing "Default" key; the sentence is its tooltip and its VoiceOver
    /// text.
    private var defaultBadge: some View {
        HStack(spacing: 4) {
            Image(systemName: LibraryRowGlyph.defaultBadge)
                .accessibilityHidden(true)
            Text("Default", bundle: .module)
        }
        .themedFont(.tiny, weight: .semibold)
        .foregroundStyle(.appText)
        .padding(.horizontal, 7)
        .padding(.vertical, 2)
        .background(.appAccent.opacity(0.18), in: Capsule())
        .overlay(Capsule().stroke(.appBorder, lineWidth: 1))
        .fixedSize()
        .help(Text("Default library for new downloads", bundle: .module))
    }

    private var statusLine: some View {
        HStack(alignment: .firstTextBaseline, spacing: 6) {
            statusGlyph
            Text(verbatim: row.statusText)
                .themedFont(.small, weight: .medium)
                .foregroundStyle(.appText)
        }
        .fixedSize(horizontal: false, vertical: true)
    }

    @ViewBuilder private var statusGlyph: some View {
        if row.isConnected {
            Image(systemName: row.statusGlyph).foregroundStyle(AnyShapeStyle(.appAccent)).accessibilityHidden(true)
        } else {
            Image(systemName: row.statusGlyph).foregroundStyle(.appSecondary).accessibilityHidden(true)
        }
    }

    /// Count, free space and link side by side while they fit, one under the
    /// other when they do not (a narrow pane, a long translation, Extra Large
    /// text).
    private var details: some View {
        ViewThatFits(in: .horizontal) {
            HStack(spacing: 14) { detailItems }
            VStack(alignment: .leading, spacing: 2) { detailItems }
        }
    }

    @ViewBuilder private var detailItems: some View {
        if let count = row.countText { detailText(count) }
        if let free = row.freeSpaceText { detailText(free) }
        if let connection = row.connectionText { detailText(connection) }
    }

    private func detailText(_ text: String) -> some View {
        Text(verbatim: text)
            .themedFont(.small)
            .foregroundStyle(.appSecondary)
            .fixedSize(horizontal: false, vertical: true)
    }

    private func noteLine(glyph: String, tint: AnyShapeStyle, text: String) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 6) {
            Image(systemName: glyph).foregroundStyle(tint).accessibilityHidden(true)
            Text(verbatim: text)
                .themedFont(.small)
                .foregroundStyle(.appText)
        }
        .fixedSize(horizontal: false, vertical: true)
    }

    @ViewBuilder private var speedTestLine: some View {
        switch speedTest {
        case .running?:
            HStack(spacing: 8) {
                AppProgressBar(value: nil, label: Text("Testing speed\u{2026}", bundle: .module), size: .compact)
                    .frame(width: 80)
                // Wraps instead of truncating: a long translation at Extra
                // Large text would otherwise cut the sentence to a word.
                Text("Testing speed\u{2026}", bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
                    .fixedSize(horizontal: false, vertical: true)
                Button {
                    cancelSpeedTest()
                } label: {
                    Text("Cancel", bundle: .module)
                }
                .appAction(.secondary, size: .small)
                .fixedSize()
            }
        case .failed(let sentence)?:
            HStack(alignment: .firstTextBaseline, spacing: 6) {
                noteLine(glyph: LibraryRowGlyph.warning, tint: AnyShapeStyle(AppAttention.color), text: sentence)
                Spacer(minLength: 4)
                AppIconButton("xmark", label: Text("Dismiss", bundle: .module)) {
                    dismissSpeedTestFailure()
                }
                .appAction(.quiet, size: .small)
            }
        case nil:
            EmptyView()
        }
    }

    // MARK: - Actions

    private var visibleActions: [LibraryRowAction] {
        // The speed test button gives way to its own progress and Cancel.
        row.offeredActions.filter { !($0 == .testSpeed && speedTest == .running) }
    }

    private var actionRow: some View {
        FlowLayout(spacing: AppMetrics.actionSpacing, lineSpacing: 6) {
            ForEach(visibleActions, id: \.self) { action in
                actionButton(action)
            }
        }
    }

    @ViewBuilder private func actionButton(_ action: LibraryRowAction) -> some View {
        if let availability = row.availability(of: action, changesBlocked: changesBlocked) {
            let button = Button(role: action.isDestructive ? .destructive : nil) {
                perform(action)
            } label: {
                Label {
                    Text(LocalizedStringKey(action.title), bundle: .module)
                } icon: {
                    Image(systemName: action.systemImage)
                }
            }
            .appAction(action.isDestructive ? .destructive : .secondary, size: .small)
            .disabled(!availability.isEnabled)
            // A disabled control says why.
            if let reason = availability.disabledReason {
                button.help(Text(verbatim: reason))
            } else {
                button
            }
        }
    }

    // MARK: - VoiceOver

    /// The row's label, with a speed test's state appended: the test runs
    /// inside the row and the row is one element.
    private var spokenLabel: String {
        switch speedTest {
        case .running?, nil: return row.accessibilityLabel
        case .failed(let sentence)?: return row.accessibilityLabel + ". " + sentence
        }
    }

    /// Only an action that can run is offered as an accessibility action: a
    /// disabled one would be announced and then do nothing.
    private var spokenActions: [LibraryRowAction] {
        visibleActions.filter { row.availability(of: $0, changesBlocked: changesBlocked)?.isEnabled == true }
    }
}
