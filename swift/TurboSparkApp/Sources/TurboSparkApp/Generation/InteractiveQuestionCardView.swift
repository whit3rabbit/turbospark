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

    /// Selected labels per question INDEX (headers can repeat). Single-select questions record
    /// their (single) pick here too when the card holds several questions;
    /// a lone single-select question answers immediately and never touches
    /// this.
    @State private var selections: [Int: Set<String>] = [:]

    /// Number keys and Escape act only while the card (or one of its
    /// options) holds focus, so typing a digit in the composer never answers
    /// a question.
    @FocusState private var cardFocused: Bool

    /// The "Other" free text per question index, and which questions show
    /// their field. The text is sent in the same answers map as a label, so
    /// the waiter and the tool result keep their shape.
    @State private var otherText: [Int: String] = [:]
    @State private var otherOpen: Set<Int> = []
    /// The question whose Other field holds focus. While set, number keys
    /// type into the field and Escape closes it instead of skipping.
    @FocusState private var otherFocus: Int?

    /// Key slots per question: its options, then its Other row.
    private var optionCounts: [Int] { questions.map { $0.options.count + 1 } }

    private func answer(_ index: Int) -> String? {
        let question = questions[index]
        return InteractiveQuestionAnswer.compose(
            selected: selections[index] ?? [], freeText: otherText[index],
            options: question.options, multiSelect: question.multiSelect)
    }

    /// Whether answers wait for the footer button: any set of more than one
    /// question must submit as ONE map.
    private var defersToFooter: Bool { questions.count > 1 }

    private var allQuestionsAnswered: Bool {
        // A question with no options can never be answered by tapping, so it
        // must not hold the whole card hostage.
        !questions.isEmpty && questions.enumerated().allSatisfy {
            $0.element.options.isEmpty || answer($0.offset) != nil
        }
    }

    /// The key a question's answer travels under. Headers can repeat (small
    /// local models often reuse one), and a map keyed by a repeated header
    /// would silently drop all but one answer.
    private func answerKey(_ index: Int) -> String {
        let header = questions[index].header
        let duplicated = questions.filter { $0.header == header }.count > 1
        return duplicated ? "\(header) (\(index + 1))" : header
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            ForEach(Array(questions.enumerated()), id: \.offset) { index, question in
                oneQuestion(question, index: index)
            }
            footer
        }
        .padding(10)
        .background(theme.accent.opacity(0.06), in: RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous)
                .stroke(
                    cardFocused ? AnyShapeStyle(theme.focusEdge) : AnyShapeStyle(theme.accent.opacity(0.35)),
                    lineWidth: cardFocused ? 2 : 1)
        )
        // The card draws its own solid focus edge: the system ring would be
        // a rectangle around a rounded card.
        .focusable()
        .focused($cardFocused)
        .focusEffectDisabled()
        .onKeyPress(characters: .decimalDigits, phases: .down) { press in
            // A digit typed into an Other field is text, never a pick.
            guard otherFocus == nil, let key = press.characters.first,
                  let target = InteractiveQuestionKeys.target(for: key, optionCounts: optionCounts)
            else { return .ignored }
            let question = questions[target.question]
            if target.option == question.options.count {
                openOther(target.question)
            } else {
                pick(question, target.question, question.options[target.option])
            }
            return .handled
        }
        // Escape is Skip, the same declined answer the button sends. From
        // an Other field it only goes back to the options.
        .onKeyPress(.escape) {
            if let index = otherFocus {
                closeOther(index)
            } else {
                model.dismissUserQuestions()
            }
            return .handled
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel(Text("Questions from the model: \(questions.count)", bundle: .module))
        .accessibilityHint(Text("Press a number to pick an option. Escape skips.", bundle: .module))
    }

    /// What a tap or a number key does to one option. Keys reuse the tap path
    /// so the answer the model receives is the same either way.
    private func pick(_ question: UserQuestionItem, _ index: Int, _ option: UserQuestionOption) {
        if question.multiSelect {
            toggle(index, option)
        } else if defersToFooter {
            // Record the pick and wait for the footer: the submit is
            // all-or-nothing, so an immediate send would strand the
            // other questions unanswered. A single answer is one or the
            // other, so the pick drops any Other text.
            selections[index] = [option.label]
            otherText[index] = nil
            otherOpen.remove(index)
        } else {
            model.submitUserQuestionAnswers([answerKey(index): option.label])
        }
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
                .appAction(.primary, size: .small)
                .disabled(!allQuestionsAnswered)
                .help(Text("Answer every question to submit", bundle: .module))
                .accessibilityLabel(Text("Submit all answers", bundle: .module))
            }
            Spacer()
            Button {
                model.dismissUserQuestions()
            } label: {
                Text("Skip", bundle: .module)
            }
            .appAction(.quiet, size: .small)
            .help(Text("Tell the model you are not answering these", bundle: .module))
            .accessibilityLabel(Text("Skip questions", bundle: .module))
        }
    }

    /// Sends every question's answer in one map. Only reachable when the
    /// footer button is enabled, so the guard is belt.
    private func submitAll() {
        guard allQuestionsAnswered else { return }
        var answers: [String: String] = [:]
        // A question with no options is included only when Other was typed.
        for index in questions.indices {
            if let value = answer(index) { answers[answerKey(index)] = value }
        }
        model.submitUserQuestionAnswers(answers)
    }

    // MARK: Other

    private func openOther(_ index: Int) {
        otherOpen.insert(index)
        otherFocus = index
    }

    /// Escape from the field: back to the options with the text discarded,
    /// so nothing hidden is sent later. The card keeps focus for keys.
    private func closeOther(_ index: Int) {
        otherText[index] = nil
        otherOpen.remove(index)
        otherFocus = nil
        cardFocused = true
    }

    /// Return in the field. Blank text does nothing. A lone question sends
    /// now (with any multi-select picks); a deferred card keeps the text for
    /// the footer and hands focus back to the card.
    private func commitOther(_ question: UserQuestionItem, _ index: Int) {
        guard InteractiveQuestionAnswer.freeText(otherText[index]) != nil else { return }
        guard defersToFooter else {
            if let value = answer(index) {
                model.submitUserQuestionAnswers([answerKey(index): value])
            }
            return
        }
        if !question.multiSelect { selections[index] = nil }
        otherFocus = nil
        cardFocused = true
    }

    @ViewBuilder
    private func otherRow(_ question: UserQuestionItem, _ index: Int) -> some View {
        let hasText = InteractiveQuestionAnswer.freeText(otherText[index]) != nil
        let selectedImage = question.multiSelect ? "checkmark.circle.fill" : "circle.fill"
        Button {
            openOther(index)
        } label: {
            HStack(alignment: .top, spacing: 6) {
                if let number = keyNumber(index, question.options.count) {
                    Text(verbatim: "\(number)")
                        .themedFont(.tiny, weight: .semibold)
                        .monospacedDigit()
                        .foregroundStyle(.appSecondary)
                        .padding(.top, 1)
                        .accessibilityHidden(true)
                }
                Image(systemName: hasText ? selectedImage : "circle")
                    .themedFont(.tiny)
                    .foregroundStyle(hasText ? AnyShapeStyle(theme.accent) : AnyShapeStyle(.appSecondary))
                    .padding(.top, 2)
                    .accessibilityHidden(true)
                VStack(alignment: .leading, spacing: 1) {
                    Text("Other...", bundle: .module)
                        .themedFont(.small, weight: .semibold)
                        .foregroundStyle(.appText)
                    Text("Answer in your own words", bundle: .module)
                        .themedFont(.tiny)
                        .foregroundStyle(.appSecondary)
                }
            }
            .padding(6)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                hasText ? theme.selection : theme.surface,
                in: RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .help(Text("Answer in your own words", bundle: .module))
        .accessibilityElement(children: .combine)
        .accessibilityAddTraits(hasText ? [.isButton, .isSelected] : .isButton)

        if otherOpen.contains(index) {
            HStack(spacing: 8) {
                TextField(
                    text: Binding(
                        get: { otherText[index] ?? "" },
                        set: { otherText[index] = $0 }),
                    prompt: Text("Your answer", bundle: .module)
                ) {
                    Text("Your answer for \(question.header)", bundle: .module)
                }
                .textFieldStyle(.roundedBorder)
                .focused($otherFocus, equals: index)
                .onAppear { otherFocus = index }
                .onSubmit { commitOther(question, index) }
                .onExitCommand { closeOther(index) }
                .accessibilityHint(Text("Return confirms. Escape returns to the options.", bundle: .module))

                // A lone single-select question answers from here; the
                // multi-select Submit and the footer cover the other cases.
                if !defersToFooter && !question.multiSelect {
                    Button {
                        commitOther(question, index)
                    } label: {
                        Text("Submit", bundle: .module)
                    }
                    .appAction(.quiet, size: .small)
                    .disabled(!hasText)
                    .accessibilityLabel(Text("Submit answer for \(question.header)", bundle: .module))
                }
            }
        }
    }

    @ViewBuilder
    private func oneQuestion(_ question: UserQuestionItem, index: Int) -> some View {
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
                    ForEach(Array(question.options.enumerated()), id: \.offset) { position, option in
                        multiRow(question, index, option, position)
                    }
                    otherRow(question, index)
                }
                if !defersToFooter {
                    Button {
                        submitMulti(question, index)
                    } label: {
                        submitTitle(index)
                    }
                    .appAction(.quiet, size: .small)
                    .disabled(answer(index) == nil)
                    .padding(.top, 2)
                    .accessibilityLabel(Text("Submit answer for \(question.header)", bundle: .module))
                }
            } else {
                VStack(alignment: .leading, spacing: 4) {
                    ForEach(Array(question.options.enumerated()), id: \.offset) { position, option in
                        singleRow(question, index, option, position)
                    }
                    otherRow(question, index)
                }
            }
        }
    }

    private func singleRow(
        _ question: UserQuestionItem, _ index: Int, _ option: UserQuestionOption, _ position: Int
    ) -> some View {
        let isSelected = selections[index]?.contains(option.label) ?? false
        return Button {
            // A click that leaves the card up focuses it, so the next pick
            // can be a number key. One that answers does not: the card goes
            // away and focus would land nowhere instead of the composer.
            if defersToFooter { cardFocused = true }
            pick(question, index, option)
        } label: {
            optionRow(option, number: keyNumber(index, position), isSelected: isSelected,
                      systemImage: isSelected ? "circle.fill" : "circle")
        }
        .buttonStyle(.plain)
        .help(Text("Answer with: \(option.label)", bundle: .module))
        // The model's own words: shown as written.
        .accessibilityLabel(Text(verbatim: "\(option.label). \(option.description)"))
        .accessibilityAddTraits(isSelected ? [.isButton, .isSelected] : .isButton)
    }

    private func multiRow(
        _ question: UserQuestionItem, _ index: Int, _ option: UserQuestionOption, _ position: Int
    ) -> some View {
        let isSelected = selections[index]?.contains(option.label) ?? false
        return Button {
            cardFocused = true  // a toggle keeps the card up
            pick(question, index, option)
        } label: {
            optionRow(option, number: keyNumber(index, position), isSelected: isSelected,
                      systemImage: isSelected ? "checkmark.circle.fill" : "circle")
        }
        .buttonStyle(.plain)
        .accessibilityLabel(Text(verbatim: "\(option.label). \(option.description)"))
        .accessibilityAddTraits(isSelected ? [.isButton, .isSelected] : .isButton)
    }

    /// The number key for an option, by position: labels can repeat.
    private func keyNumber(_ index: Int, _ position: Int) -> Int? {
        InteractiveQuestionKeys.number(question: index, option: position, optionCounts: optionCounts)
    }

    private func optionRow(
        _ option: UserQuestionOption, number: Int?, isSelected: Bool, systemImage: String
    ) -> some View {
        HStack(alignment: .top, spacing: 6) {
            // The key hint: a digit, so it needs no translation. Hidden from
            // VoiceOver, which reads the card's hint instead.
            if let number {
                Text(verbatim: "\(number)")
                    .themedFont(.tiny, weight: .semibold)
                    .monospacedDigit()
                    .foregroundStyle(.appSecondary)
                    .padding(.top, 1)
                    .accessibilityHidden(true)
            }
            Image(systemName: systemImage)
                .themedFont(.tiny)
                .foregroundStyle(isSelected ? AnyShapeStyle(theme.accent) : AnyShapeStyle(.appSecondary))
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
            isSelected ? theme.selection : theme.surface,
            in: RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous))
        .contentShape(Rectangle())
    }

    private func toggle(_ index: Int, _ option: UserQuestionOption) {
        var set = selections[index] ?? []
        if set.contains(option.label) {
            set.remove(option.label)
        } else {
            set.insert(option.label)
        }
        selections[index] = set
    }

    private func selectedCount(_ index: Int) -> Int {
        selections[index]?.count ?? 0
    }

    private func submitTitle(_ index: Int) -> Text {
        let count = selectedCount(index)
        return count > 0 ? Text("Submit (\(count))", bundle: .module) : Text("Submit", bundle: .module)
    }

    /// Picks plus any Other text, in the same comma-joined string as before.
    private func submitMulti(_ question: UserQuestionItem, _ index: Int) {
        guard let value = answer(index) else { return }
        model.submitUserQuestionAnswers([answerKey(index): value])
    }
}
