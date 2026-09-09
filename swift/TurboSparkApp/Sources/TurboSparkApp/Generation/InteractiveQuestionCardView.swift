import SwiftUI

/// The tappable question card the transcript shows while the model's turn
/// is parked on an `AskUserQuestion` call (the qwen-code inline question
/// surface). Single-select answers on tap; multi-select toggles a set and
/// answers on Submit; Skip tells the model the user declined, which is an
/// answer it can act on rather than a hang.
@MainActor
struct InteractiveQuestionCardView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel
    let questions: [UserQuestionItem]

    /// Selected labels per question header, for multiSelect sets. A
    /// single-select question answers immediately and never touches this.
    @State private var selections: [String: Set<String>] = [:]

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            ForEach(Array(questions.enumerated()), id: \.offset) { _, question in
                oneQuestion(question)
            }
            skipFooter
        }
        .padding(10)
        .background(TurboSparkTheme.accentColor.opacity(0.06), in: RoundedRectangle(cornerRadius: 8))
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(TurboSparkTheme.accentColor.opacity(0.35), lineWidth: 1)
        )
        .accessibilityElement(children: .contain)
        .accessibilityLabel(accessibilitySummary)
    }

    private var accessibilitySummary: String {
        let count = questions.count
        return "The model asked \(count) question\(count == 1 ? "" : "s")"
    }

    private var skipFooter: some View {
        HStack {
            Spacer()
            Button {
                model.dismissUserQuestions()
            } label: {
                Text("Skip", bundle: .module)
            }
            .buttonStyle(.plain)
            .themedFont(.small)
            .foregroundStyle(.secondary)
            .help("Tell the model you are not answering these")
            .accessibilityLabel("Skip questions")
        }
    }

    @ViewBuilder
    private func oneQuestion(_ question: UserQuestionItem) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 6) {
                Text(question.header)
                    .themedFont(.tiny, weight: .bold)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(TurboSparkTheme.accentColor.opacity(0.15), in: Capsule())
                    .foregroundStyle(TurboSparkTheme.accentColor)
                Text(question.question)
                    .themedFont(.base, weight: .medium)
                    .foregroundStyle(.primary)
            }
            if question.multiSelect {
                VStack(alignment: .leading, spacing: 4) {
                    ForEach(Array(question.options.enumerated()), id: \.offset) { _, option in
                        multiRow(question, option)
                    }
                }
            Button {
                submitMulti(question)
            } label: {
                Text(submitTitle(question))
            }
            .buttonStyle(.plain)
            .themedFont(.small, weight: .semibold)
            .foregroundStyle(TurboSparkTheme.accentColor)
            .disabled(selectedCount(question) == 0)
            .padding(.top, 2)
            .accessibilityLabel("Submit answer for \(question.header)")
            } else {
                VStack(alignment: .leading, spacing: 4) {
                    ForEach(Array(question.options.enumerated()), id: \.offset) { _, option in
                        singleRow(question, option)
                    }
                }
            }
        }
    }

    private func singleRow(_ question: UserQuestionItem, _ option: UserQuestionOption) -> some View {
        Button {
            model.submitUserQuestionAnswers([question.header: option.label])
        } label: {
            optionRow(option, isSelected: false, systemImage: "circle")
        }
        .buttonStyle(.plain)
        .help("Answer with: \(option.label)")
        .accessibilityLabel("\(option.label). \(option.description)")
        .accessibilityAddTraits(.isButton)
    }

    private func multiRow(_ question: UserQuestionItem, _ option: UserQuestionOption) -> some View {
        let isSelected = selections[question.header]?.contains(option.label) ?? false
        return Button {
            toggle(question, option)
        } label: {
            optionRow(option, isSelected: isSelected, systemImage: isSelected ? "checkmark.circle.fill" : "circle")
        }
        .buttonStyle(.plain)
        .accessibilityLabel("\(option.label). \(option.description)")
        .accessibilityAddTraits(isSelected ? [.isButton, .isSelected] : .isButton)
    }

    private func optionRow(_ option: UserQuestionOption, isSelected: Bool, systemImage: String) -> some View {
        HStack(alignment: .top, spacing: 6) {
            Image(systemName: systemImage)
                .themedFont(.tiny)
                .foregroundStyle(isSelected ? TurboSparkTheme.accentColor : Color.secondary)
                .padding(.top, 2)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 1) {
                Text(option.label)
                    .themedFont(.small, weight: .semibold)
                    .foregroundStyle(.primary)
                Text(option.description)
                    .themedFont(.tiny)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(6)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            isSelected ? TurboSparkTheme.accentColor.opacity(0.10) : Color.primary.opacity(0.03),
            in: RoundedRectangle(cornerRadius: 6))
        .contentShape(Rectangle())
    }

    private func toggle(_ question: UserQuestionItem, _ option: UserQuestionOption) {
        var set = selections[question.header] ?? []
        if set.contains(option.label) {
            set.remove(option.label)
        } else {
            set.insert(option.label)
        }
        selections[question.header] = set
    }

    private func selectedCount(_ question: UserQuestionItem) -> Int {
        selections[question.header]?.count ?? 0
    }

    private func submitTitle(_ question: UserQuestionItem) -> String {
        let count = selectedCount(question)
        return count > 0 ? "Submit (\(count))" : "Submit"
    }

    private func submitMulti(_ question: UserQuestionItem) {
        let labels = (selections[question.header] ?? []).sorted()
        guard !labels.isEmpty else { return }
        model.submitUserQuestionAnswers([question.header: labels.joined(separator: ", ")])
    }
}
