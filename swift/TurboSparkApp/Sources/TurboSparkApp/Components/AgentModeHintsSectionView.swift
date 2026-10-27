import SwiftUI

/// Agent Mode section of the Permissions settings pane
/// (`swift/docs/SWIFT_AGENT_MODE.md`): the four natural-language hint lists
/// the classifier embeds in its policy text, plus the mode explainer.
///
/// One hint per line in each editor. This is deliberately NOT a row-list
/// editor: hints are sentences, sentences are written and reordered in a
/// text block, and the lists are capped (`AgentModeHints.normalized`
/// applies the caps at prompt time), so the editing surface stays a plain
/// textarea rather than a CRUD table for values a user pastes wholesale.
@MainActor
struct AgentModeHintsSectionView: View {
    @ObservedObject var model: AppModel

    public var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Label { Text("Agent Mode Classifier Hints", bundle: .module) } icon: { Image(systemName: "brain.head.profile") }
                    .themedFont(.base, weight: .semibold)
                    .accessibilityAddTraits(.isHeader)
                Spacer()
                Text("Used when the tool approval mode is \"Agent (classifier)\"", bundle: .module)
                    .themedFont(.tiny)
                    .foregroundStyle(.appSecondary)
            }
            Text("In Agent mode a local classifier judges each tool call the static rules would have asked about. These lists steer that judgment as plain sentences -- not shell patterns. Allow lines describe work to auto-approve in this workspace. Soft-deny lines describe destructive actions to refuse unless your recent request clearly asked for them. Hard-deny lines describe boundaries the classifier must refuse whatever you asked. Environment lines are facts about this machine it should know.", bundle: .module)
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
                .fixedSize(horizontal: false, vertical: true)

            hintEditor(
                title: "Allow", icon: "checkmark.circle",
                entries: Binding(
                    get: { model.agentModeHints.allow },
                    set: { model.agentModeHints.allow = $0 }),
                placeholder: "Running poetry install and poetry update in this project")
            hintEditor(
                title: "Soft deny", icon: "exclamationmark.arrow.circlepath",
                entries: Binding(
                    get: { model.agentModeHints.softDeny },
                    set: { model.agentModeHints.softDeny = $0 }),
                placeholder: "Running migration scripts that touch the local database")
            hintEditor(
                title: "Hard deny", icon: "hand.raised",
                entries: Binding(
                    get: { model.agentModeHints.hardDeny },
                    set: { model.agentModeHints.hardDeny = $0 }),
                placeholder: "Sending secrets or .env contents to any network endpoint")
            hintEditor(
                title: "Environment", icon: "info.circle",
                entries: Binding(
                    get: { model.agentModeHints.environment },
                    set: { model.agentModeHints.environment = $0 }),
                placeholder: "This is a private monorepo with strict commit signing")
        }
        .padding(14)
        .background(.appPage)
        .clipShape(RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous).stroke(.appBorder, lineWidth: 1))
    }

    private func hintEditor(
        title: LocalizedStringKey, icon: String,
        entries: Binding<[String]>, placeholder: LocalizedStringKey
    ) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 6) {
                Image(systemName: icon)
                    .themedFont(.tiny)
                    .foregroundStyle(.appSecondary)
                Text(title, bundle: .module)
                    .themedFont(.tiny, weight: .semibold)
                    .foregroundStyle(.appSecondary)
                Text("one sentence per line", bundle: .module)
                    .themedFont(.tiny)
                    .foregroundStyle(.appSecondary)
            }
            TextEditor(text: Binding(
                get: { entries.wrappedValue.joined(separator: "\n") },
                set: { newValue in
                    let parsed = newValue
                        .components(separatedBy: .newlines)
                        .map { $0.trimmingCharacters(in: .whitespaces) }
                        .filter { !$0.isEmpty }
                    if parsed != entries.wrappedValue {
                        entries.wrappedValue = parsed
                        model.persistSettingsDebounced()
                    }
                }))
                .themedCode(.small)
                .appEditorSurface(minHeight: 44)
                .accessibilityLabel(Text(title, bundle: .module))
                .overlay(alignment: .topLeading) {
                    // Example sentences are prose, so they are catalog keys
                    // like any other hint text.
                    if entries.wrappedValue.isEmpty {
                        Text(placeholder, bundle: .module)
                            .themedCode(.small)
                            .foregroundStyle(.appSecondary)
                            .padding(.horizontal, 14)
                            .padding(.top, 14)
                            .allowsHitTesting(false)
                            .accessibilityHidden(true)
                    }
                }
        }
    }
}
