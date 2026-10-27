import AppKit
import SwiftUI

/// Settings section for live repository instructions and saved project guidance.
struct ProjectRulesSectionView: View {
    let rootDirectoryPath: String
    @Binding var rulePreference: AppRulePreference
    @Binding var customInstructions: String
    @Binding var rulesAutoDetectedMessage: String?
    let onAutoDetect: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .center, spacing: 14) {
                TSIdlingAgentsSparkView(size: 64)

                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text("Project Rules & Instructions", bundle: .module)
                            .themedFont(.small, weight: .semibold)
                            .accessibilityAddTraits(.isHeader)
                        Spacer()
                        if !rootDirectoryPath.isEmpty {
                            Button {
                                onAutoDetect()
                            } label: { Text("Check Project Instructions", bundle: .module) }
                            .themedFont(.small)
                            .appAction(.quiet, size: .small)
                            .help(Text("Check the live AGENTS.md, CLAUDE.md, CONTEXT.md, and SOUL.md files for this project", bundle: .module))
                        }
                    }

                    Text("Loads live AGENTS.md or CLAUDE.md instructions from the repository root.", bundle: .module)
                        .themedFont(.small)
                        .foregroundStyle(.appSecondary)
                }
            }

            HStack {
                Text("Conflict Preference", bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
                Spacer()
                Picker(selection: $rulePreference) {
                    ForEach(AppRulePreference.allCases) { pref in
                        // Every preference label is a catalog key.
                        Text(LocalizedStringKey(pref.label), bundle: .module).tag(pref)
                    }
                } label: { Text("Conflict Preference", bundle: .module) }
                .pickerStyle(.menu)
                .frame(minWidth: 140, idealWidth: 180)
                .accessibilityLabel(Text("Rules conflict preference", bundle: .module))
            }

            if let rulesAutoDetectedMessage {
                Text(rulesAutoDetectedMessage)
                    .themedFont(.small)
                    .foregroundStyle(.appAccent)
            }

            TextEditor(text: $customInstructions)
                .themedCode(.base)
                .appEditorSurface(minHeight: 100)
                .accessibilityLabel(Text("Additional project instructions", bundle: .module))
                .accessibilityHint(Text("Guidance stored with this project; repository instruction files are read live", bundle: .module))
        }
    }
}
