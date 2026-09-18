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
                            .buttonStyle(.borderless)
                            .help("Check the live AGENTS.md, CLAUDE.md, CONTEXT.md, and SOUL.md files for this project")
                        }
                    }

                    Text(verbatim: "Loads live AGENTS.md or CLAUDE.md instructions from the repository root.")
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
                        Text(pref.label).tag(pref)
                    }
                } label: { Text("Conflict Preference", bundle: .module) }
                .pickerStyle(.menu)
                .frame(width: 180)
                .accessibilityLabel("Rules conflict preference")
            }

            if let rulesAutoDetectedMessage {
                Text(rulesAutoDetectedMessage)
                    .themedFont(.small)
                    .foregroundStyle(.appAccent)
            }

            TextEditor(text: $customInstructions)
                .themedCode(.base)
                .frame(height: 100)
                .padding(4)
                .background(.appSurface, in: RoundedRectangle(cornerRadius: 8))
                .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.secondary.opacity(0.2), lineWidth: 0.5))
                .accessibilityLabel("Additional project instructions")
                .accessibilityHint("Guidance stored with this project; repository instruction files are read live")
        }
    }
}
