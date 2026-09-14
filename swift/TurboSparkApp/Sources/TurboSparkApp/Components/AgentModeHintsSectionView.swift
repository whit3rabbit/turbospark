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
        .clipShape(RoundedRectangle(cornerRadius: 10))
        .overlay(RoundedRectangle(cornerRadius: 10).stroke(.appBorder.opacity(0.4), lineWidth: 1))
    }

    private func hintEditor(
        title: LocalizedStringKey, icon: String,
        entries: Binding<[String]>, placeholder: String
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
                    .foregroundStyle(.tertiary)
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
                .frame(minHeight: 44, maxHeight: 110)
                .scrollContentBackground(.hidden)
                .padding(6)
                .background(.appElevated.opacity(0.5))
                .clipShape(RoundedRectangle(cornerRadius: 6))
                .overlay(
                    RoundedRectangle(cornerRadius: 6)
                        .stroke(Color.secondary.opacity(0.18), lineWidth: 0.5))
                .overlay(alignment: .topLeading) {
                    if entries.wrappedValue.isEmpty {
                        Text(placeholder)
                            .themedCode(.small)
                            .foregroundStyle(.tertiary)
                            .padding(.horizontal, 12)
                            .padding(.top, 12)
                            .allowsHitTesting(false)
                    }
                }
        }
    }
}
