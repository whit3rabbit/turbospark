import SwiftUI

/// The tappable question card the transcript shows while the model's turn
/// is parked on an `AskUserQuestion` call (the qwen-code inline question
/// surface). One question: single-select answers on tap, multi-select
/// toggles a set and answers on Submit. Several questions: NOTHING submits
/// until the footer does, because the executor's waiter is all-or-nothing
/// -- an early per-question submit would clear the card with questions
/// 2..N unanswered and the model would see a partial answer. Skip tells the
/// model the user declined, which is an answer it can act on rather than a
/// hang.
@MainActor
struct InteractiveQuestionCardView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel
    let questions: [UserQuestionItem]

    /// Selected labels per question header. Single-select questions record
    /// their (single) pick here too when the card holds several questions;
    /// a lone single-select question answers immediately and never touches
    /// this.
    @State private var selections: [String: Set<String>] = [:]

    /// Whether answers wait for the footer button: any set of more than one
    /// question must submit as ONE map.
    private var defersToFooter: Bool { questions.count > 1 }

    private var allQuestionsAnswered: Bool {
        !questions.isEmpty && questions.allSatisfy { selectedCount($0) > 0 }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            ForEach(Array(questions.enumerated()), id: \.offset) { _, question in
                oneQuestion(question)
            }
            footer
        }
        .padding(10)
        .background(.appAccent.opacity(0.06), in: RoundedRectangle(cornerRadius: 8))
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(.appAccent.opacity(0.35), lineWidth: 1)
        )
        .accessibilityElement(children: .contain)
        .accessibilityLabel(accessibilitySummary)
    }

    private var accessibilitySummary: String {
        let count = questions.count
        return "The model asked \(count) question\(count == 1 ? "" : "s")"
    }

    @ViewBuilder
    private var footer: some View {
        HStack {
            if defersToFooter {
                Button {
                    submitAll()
                } label: {
                    Text("Submit Answers", bundle: .module)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.small)
                .themedFont(.small, weight: .semibold)
                .disabled(!allQuestionsAnswered)
                .help("Answer every question to submit")
                .accessibilityLabel("Submit all answers")
            }
            Spacer()
            Button {
                model.dismissUserQuestions()
            } label: {
                Text("Skip", bundle: .module)
            }
            .buttonStyle(.plain)
            .themedFont(.small)
            .foregroundStyle(.appSecondary)
            .help("Tell the model you are not answering these")
            .accessibilityLabel("Skip questions")
        }
    }

    /// Sends every question's answer in one map. Only reachable when the
    /// footer button is enabled, so the guard is belt.
    private func submitAll() {
        guard allQuestionsAnswered else { return }
        var answers: [String: String] = [:]
        for question in questions {
            answers[question.header] = (selections[question.header] ?? [])
                .sorted()
                .joined(separator: ", ")
        }
        model.submitUserQuestionAnswers(answers)
    }

    @ViewBuilder
    private func oneQuestion(_ question: UserQuestionItem) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 6) {
                Text(question.header)
                    .themedFont(.tiny, weight: .bold)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(.appAccent.opacity(0.15), in: Capsule())
                    .foregroundStyle(.appAccent)
                Text(question.question)
                    .themedFont(.base, weight: .medium)
                    .foregroundStyle(.appText)
            }
            if question.multiSelect {
                VStack(alignment: .leading, spacing: 4) {
                    ForEach(Array(question.options.enumerated()), id: \.offset) { _, option in
                        multiRow(question, option)
                    }
                }
                if !defersToFooter {
                    Button {
                        submitMulti(question)
                    } label: {
                        Text(submitTitle(question))
                    }
                    .buttonStyle(.plain)
                    .themedFont(.small, weight: .semibold)
                    .foregroundStyle(.appAccent)
                    .disabled(selectedCount(question) == 0)
                    .padding(.top, 2)
                    .accessibilityLabel("Submit answer for \(question.header)")
                }
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
        let isSelected = selections[question.header]?.contains(option.label) ?? false
        return Button {
            if defersToFooter {
                // Record the pick and wait for the footer: the submit is
                // all-or-nothing, so an immediate send would strand the
                // other questions unanswered.
                selections[question.header] = [option.label]
            } else {
                model.submitUserQuestionAnswers([question.header: option.label])
            }
        } label: {
            optionRow(option, isSelected: isSelected, systemImage: isSelected ? "circle.fill" : "circle")
        }
        .buttonStyle(.plain)
        .help("Answer with: \(option.label)")
        .accessibilityLabel("\(option.label). \(option.description)")
        .accessibilityAddTraits(isSelected ? [.isButton, .isSelected] : .isButton)
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
                    .foregroundStyle(.appText)
                Text(option.description)
                    .themedFont(.tiny)
                    .foregroundStyle(.appSecondary)
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
