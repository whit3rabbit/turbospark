import AppKit
import SwiftUI

/// Settings section for managing project rules, AGENTS.md / CLAUDE.md auto-detection, and custom instructions.
struct ProjectRulesSectionView: View {
    let rootDirectoryPath: String
    @Binding var rulePreference: AppRulePreference
    @Binding var customInstructions: String
    @Binding var rulesAutoDetectedMessage: String?
    let onAutoDetect: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Project Rules & Instructions", bundle: .module)
                    .themedFont(.small, weight: .semibold)
                    .accessibilityAddTraits(.isHeader)
                Spacer()
                if !rootDirectoryPath.isEmpty {
                    Button("Detect CLAUDE.md / AGENTS.md") {
                        onAutoDetect()
                    }
                    .themedFont(.small)
                    .buttonStyle(.borderless)
                    .help("Detect project rules from AGENTS.md or CLAUDE.md")
                }
            }

            HStack {
                Text("Conflict Preference", bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
                Spacer()
                Picker("Conflict Preference", selection: $rulePreference) {
                    ForEach(AppRulePreference.allCases) { pref in
                        Text(pref.label).tag(pref)
                    }
                }
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
                .accessibilityLabel("Project rules and instructions")
                .accessibilityHint("Free-form text sent to the model as project-specific guidance")
        }
    }
}
