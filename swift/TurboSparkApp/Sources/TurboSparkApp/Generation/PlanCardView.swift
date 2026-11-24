import SwiftUI

/// The body of an ExitPlanMode tool card: the submitted plan as numbered,
/// expandable steps when `PlanOutlineParser` is sure of them (the markdown
/// block otherwise), and, while `PlanCardActions` says the agent stopped on
/// this plan, Start and Request changes.
///
/// Drawn inside the tool card, which is already the card surface, so this
/// uses the elevated inset fill like the other tool previews instead of a
/// nested card (Card guide: do not nest cards).
///
/// Two button modes. `isAwaitingDecision`: ExitPlanMode is parked on this
/// card, so Start resumes it as approved and Request changes sends the typed
/// text as its result. `isOffered` (post-turn, the gate was not decided, for
/// example a stopped turn or a reloaded chat): Start sends an ordinary reply
/// and Request changes drafts one. Neither mode changes the permission mode
/// or any tool approval.
struct PlanCardView: View {
    @ObservedObject var model: AppModel
    let plan: String
    let isOffered: Bool
    var isAwaitingDecision: Bool = false

    @State private var openStep: Int?
    /// The gate's inline Request changes field is open.
    @State private var isEditingChanges = false
    @State private var changesText = ""
    @FocusState private var changesFieldFocused: Bool

    private var showsActions: Bool { isOffered || isAwaitingDecision }

    private var outline: PlanOutline? { PlanOutlineParser.parse(plan) }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            AppSectionHeader(
                title: showsActions ? Text("Proposed plan", bundle: .module) : Text("Plan", bundle: .module),
                subtitle: outline.map { Text("Steps: \($0.steps.count)", bundle: .module) })

            if let outline {
                if !outline.intro.isEmpty {
                    CollapsibleMessageContentView(text: outline.intro, maxHeight: 120)
                }
                VStack(alignment: .leading, spacing: 2) {
                    ForEach(outline.steps, id: \.number) { step in
                        stepRow(step)
                    }
                }
            } else if !plan.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                CollapsibleMessageContentView(text: plan, maxHeight: 260)
            }

            if isAwaitingDecision {
                gateActions
                    .padding(.top, 2)
            } else if isOffered {
                HStack(spacing: 8) {
                    Button {
                        model.startPlanFromCard()
                    } label: {
                        Text("Start", bundle: .module)
                    }
                    .appAction(.primary, size: .small)
                    .help(Text("Sends a reply asking the agent to carry out this plan", bundle: .module))

                    Button {
                        model.requestPlanChangesFromCard()
                    } label: {
                        Text("Request changes", bundle: .module)
                    }
                    .appAction(.secondary, size: .small)
                    .help(Text("Drafts a reply in the composer so you can describe the changes", bundle: .module))
                }
                .padding(.top, 2)
            }
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.appElevated, in: RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous))
    }

    /// The parked gate's controls. Request changes swaps the two buttons for
    /// a focused field: Return sends, Escape (or Cancel) goes back.
    @ViewBuilder
    private var gateActions: some View {
        if isEditingChanges {
            HStack(spacing: 8) {
                TextField(
                    text: $changesText,
                    prompt: Text("Describe the changes", bundle: .module),
                    axis: .vertical
                ) {
                    Text("Describe the changes", bundle: .module)
                }
                .textFieldStyle(.roundedBorder)
                .lineLimit(1...4)
                .focused($changesFieldFocused)
                .onAppear { changesFieldFocused = true }
                .onSubmit(sendChanges)
                .onExitCommand(perform: cancelChanges)

                Button(action: sendChanges) {
                    Text("Send", bundle: .module)
                }
                .appAction(.primary, size: .small)
                .disabled(changesText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                .help(Text("Sends your changes to the agent, which keeps planning", bundle: .module))

                Button(action: cancelChanges) {
                    Text("Cancel", bundle: .module)
                }
                .appAction(.quiet, size: .small)
            }
        } else {
            HStack(spacing: 8) {
                Button {
                    model.approvePendingPlan()
                } label: {
                    Text("Start", bundle: .module)
                }
                .appAction(.primary, size: .small)
                .help(Text("Approves this plan so the agent carries it out", bundle: .module))

                Button {
                    isEditingChanges = true
                } label: {
                    Text("Request changes", bundle: .module)
                }
                .appAction(.secondary, size: .small)
                .help(Text("Asks the agent to revise this plan", bundle: .module))
            }
        }
    }

    private func sendChanges() {
        guard !changesText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        model.requestChangesToPendingPlan(changesText)
    }

    private func cancelChanges() {
        isEditingChanges = false
        changesText = ""
    }

    @ViewBuilder
    private func stepRow(_ step: PlanCardStep) -> some View {
        let hasDetail = !step.detail.isEmpty
        let isOpen = openStep == step.number
        VStack(alignment: .leading, spacing: 4) {
            Button {
                guard hasDetail else { return }
                TSMotion.animate(TSMotion.select) {
                    openStep = isOpen ? nil : step.number
                }
            } label: {
                HStack(alignment: .firstTextBaseline, spacing: 8) {
                    Text(verbatim: "\(step.number).")
                        .themedFont(.small)
                        .monospacedDigit()
                        .foregroundStyle(.appSecondary)
                        .frame(minWidth: 18, alignment: .trailing)
                    Text(step.title)
                        .themedFont(.base)
                        .foregroundStyle(.appText)
                        .frame(maxWidth: .infinity, alignment: .leading)
                    if hasDetail {
                        Image(systemName: "chevron.right")
                            .themedFont(.tiny)
                            .foregroundStyle(.appSecondary)
                            .rotationEffect(.degrees(isOpen ? 90 : 0))
                            .accessibilityHidden(true)
                    }
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .disabled(!hasDetail)
            .accessibilityElement(children: .combine)
            .accessibilityValue(
                hasDetail
                    ? (isOpen ? Text("Expanded", bundle: .module) : Text("Collapsed", bundle: .module))
                    : Text(verbatim: ""))
            .accessibilityHint(hasDetail ? Text("Shows or hides the details of this step", bundle: .module) : Text(verbatim: ""))

            if isOpen {
                CollapsibleMessageContentView(text: step.detail, maxHeight: 200)
                    .padding(.leading, 26)
                    .transition(.opacity)
            }
        }
        .padding(.vertical, 2)
    }
}
