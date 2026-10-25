import SwiftUI
import TurboSpark

/// The sheet behind "Move to..." (task 6.11, requirements 7.1 to 7.5): where the
/// model is, where it can go, how much room the move needs and the destination
/// has, then progress with Cancel, then what happened.
///
/// **IT DRAWS A `ModelMovePlan` AND A `ModelMoveSheetModel` STAGE AND NOTHING
/// ELSE,** so a table test pins every word and VoiceOver reads the strings a
/// sighted user does. Names and paths come from a drive: the models escaped and
/// cut them, and this only draws them (`Text(verbatim:)`, never a key).
///
/// **IT CANNOT BE DISMISSED WHILE A MOVE RUNS.** Escape, the close gesture and the
/// Close button are all off then (`interactiveDismissDisabled` and no cancel
/// shortcut on any button); Cancel asks the engine to stop and the sheet waits for
/// its answer. Escape is deliberately NOT bound to Cancel: one stray key press must
/// not abandon a copy that has run for twenty minutes.
///
/// **STATUS IS A GLYPH AND A SENTENCE,** never color alone: a refusal is a triangle,
/// a failure an octagon, a cancel an info mark, a success a check.
///
/// Isolated explicitly: its helpers call each other, and only `body` is isolated by
/// the protocol on the macOS 14 SDK.
@MainActor
struct ModelMoveSheet: View {
    @ObservedObject var model: AppModel
    @ObservedObject var sheet: ModelMoveSheetModel
    /// The library state is observed so the free space and the destinations follow a
    /// drive that is plugged or pulled while the sheet is open.
    @ObservedObject private var libraryState: LibraryState
    /// Read so the strings built from `AppStrings` follow the language picker.
    @Environment(\.locale) private var locale

    init(model: AppModel, sheet: ModelMoveSheetModel) {
        self.model = model
        self.sheet = sheet
        _libraryState = ObservedObject(wrappedValue: model.libraryState)
    }

    private var language: AppLanguage {
        _ = locale
        return AppStrings.currentLanguage
    }

    var body: some View {
        let plan = sheet.plan(language: language)
        VStack(alignment: .leading, spacing: 16) {
            header(plan)
            content(plan)
            buttons(plan)
        }
        .padding(24)
        .frame(minWidth: 380, idealWidth: 480, maxWidth: 560)
        .interactiveDismissDisabled(!sheet.canDismiss)
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("model-move-sheet")
    }

    // MARK: - Header

    private func header(_ plan: ModelMovePlan) -> some View {
        let alias = UntrustedText.display(sheet.subject.alias, limit: ModelMoveText.aliasLimit)
        return VStack(alignment: .leading, spacing: 4) {
            Label {
                Text("Move \(alias)", bundle: .module)
                    .themedFont(.title3, weight: .semibold)
                    .fixedSize(horizontal: false, vertical: true)
            } icon: {
                Image(systemName: "arrow.left.arrow.right.circle")
                    .themedFont(.title3)
                    .foregroundStyle(.appAccent)
            }
            .accessibilityAddTraits(.isHeader)
            if let size = plan.requiredText {
                Text(verbatim: size)
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
            }
        }
    }

    // MARK: - Stages

    @ViewBuilder
    private func content(_ plan: ModelMovePlan) -> some View {
        switch sheet.stage {
        case .choosing:
            choosing(plan)
        case .moving:
            moving
        case .finished(let text):
            finished(text)
        case .failed(let text):
            failed(text)
        }
    }

    private func choosing(_ plan: ModelMovePlan) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("TurboSpark copies the model, checks the copy, and only then removes the original.", bundle: .module)
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
                .fixedSize(horizontal: false, vertical: true)
            SummaryRow(label: Text("Location", bundle: .module)) {
                Text(verbatim: plan.sourceName)
            }
            if !plan.choices.isEmpty {
                SummaryRow(label: Text("Move to", bundle: .module)) { destinationMenu(plan) }
            }
            SummaryRow(label: Text("Space required", bundle: .module)) {
                if plan.isMeasuring {
                    HStack(spacing: 6) {
                        ProgressView().controlSize(.small)
                        Text("Measuring...", bundle: .module)
                    }
                    .accessibilityElement(children: .combine)
                } else if let required = plan.requiredText {
                    Text(verbatim: required)
                } else {
                    Text("Unknown", bundle: .module)
                }
            }
            // With nowhere to move to there is no destination whose room to show.
            if !plan.choices.isEmpty {
                SummaryRow(label: Text("Space available", bundle: .module)) {
                    Text(verbatim: plan.availableText)
                }
            }
            if let reason = plan.refusalText {
                MoveNotice(glyph: Self.glyph(for: plan.refusal), tint: Self.tint(for: plan.refusal), text: reason)
            }
        }
    }

    private func destinationMenu(_ plan: ModelMovePlan) -> some View {
        Picker(selection: chosen(plan)) {
            ForEach(plan.choices) { choice in
                Label {
                    Text(verbatim: choice.menuTitle)
                } icon: {
                    Image(systemName: InstallDestinationPicker.glyph(for: choice))
                }
                .tag(choice.id)
            }
        } label: {
            Text("Move to", bundle: .module)
        }
        .pickerStyle(.menu)
        .labelsHidden()
        .fixedSize()
        .themedFont(.small)
        .accessibilityLabel(Text("Move to", bundle: .module))
    }

    private func chosen(_ plan: ModelMovePlan) -> Binding<String> {
        Binding(get: { plan.selectedId ?? "" }, set: { sheet.selection = $0 })
    }

    /// The bar, the phase, the bytes and the reminder, read as one element.
    private var moving: some View {
        let move = sheet.progress
        return VStack(alignment: .leading, spacing: 8) {
            if let fraction = move?.fraction {
                ProgressView(value: fraction)
            } else {
                ProgressView().progressViewStyle(.linear)
            }
            HStack(alignment: .firstTextBaseline) {
                // No move left but the sheet still waiting: the engine has answered and
                // the app is reading the libraries again, which is the finishing step.
                Text(LocalizedStringKey((move?.phase ?? .finishing).title), bundle: .module)
                    .themedFont(.small, weight: .medium)
                    .fixedSize(horizontal: false, vertical: true)
                Spacer(minLength: 8)
                if let move, move.hasProgress, move.total > 0 {
                    Text("\(MetricFormat.storage(move.done)) of \(MetricFormat.storage(move.total))", bundle: .module)
                        .themedFont(.small)
                        .foregroundStyle(.appSecondary)
                        .monospacedDigit()
                }
            }
            Text("Keep the drives connected until the move finishes.", bundle: .module)
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .accessibilityElement(children: .combine)
        .accessibilityValue(Text(verbatim: Self.percent(move?.fraction)))
    }

    private func finished(_ text: MoveSuccessText) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            MoveNotice(
                glyph: "checkmark.circle.fill", tint: AnyShapeStyle(AppThemeColor.appAccent), text: text.headline)
            if let detail = text.detail {
                MoveNotice(glyph: "info.circle", tint: AnyShapeStyle(AppThemeColor.appSecondary), text: detail)
            }
        }
    }

    private func failed(_ text: MoveFailureText) -> some View {
        switch text.kind {
        case .cancelled:
            return MoveNotice(glyph: "info.circle", tint: AnyShapeStyle(AppThemeColor.appSecondary), text: text.text)
        case .failed:
            return MoveNotice(glyph: "xmark.octagon.fill", tint: AnyShapeStyle(Color.red), text: text.text)
        }
    }

    // MARK: - Buttons

    /// The buttons side by side while they fit, stacked when they do not.
    private func buttons(_ plan: ModelMovePlan) -> some View {
        ViewThatFits(in: .horizontal) {
            HStack(spacing: AppMetrics.actionSpacing) {
                Spacer(minLength: 0)
                buttonItems(plan)
            }
            VStack(alignment: .trailing, spacing: 6) { buttonItems(plan) }
        }
        .frame(maxWidth: .infinity, alignment: .trailing)
    }

    @ViewBuilder
    private func buttonItems(_ plan: ModelMovePlan) -> some View {
        switch sheet.stage {
        case .choosing:
            closeButton
            Button {
                sheet.move()
            } label: {
                Text("Move", bundle: .module)
            }
            .appAction(.primary)
            .keyboardShortcut(.defaultAction)
            .disabled(!plan.canMove)
        case .moving:
            // No cancel shortcut here: Escape does nothing while a move runs. Cancel
            // is off once it was pressed, and once the engine has answered.
            let cancelling = sheet.progress.map(\.cancelRequested) ?? true
            // A plain button, not `role: .cancel`: nothing here may answer Escape.
            Button {
                sheet.cancel()
            } label: {
                Text("Cancel", bundle: .module)
            }
            .appAction(.secondary)
            .disabled(cancelling)
        case .finished:
            Button {
                sheet.close()
            } label: {
                Text("Done", bundle: .module)
            }
            .appAction(.primary)
            // One shortcut per button: Return presses Done, and Escape closes the sheet
            // by itself (a finished sheet can be dismissed).
            .keyboardShortcut(.defaultAction)
        case .failed:
            closeButton
            Button {
                sheet.retry()
            } label: {
                Text("Retry", bundle: .module)
            }
            .appAction(.primary)
            .keyboardShortcut(.defaultAction)
        }
    }

    private var closeButton: some View {
        Button(role: .cancel) {
            sheet.close()
        } label: {
            Text("Close", bundle: .module)
        }
        .appAction(.secondary)
        .keyboardShortcut(.cancelAction)
    }

    // MARK: - Glyphs

    /// A shape for each kind of refusal, so the line is never color alone. Nothing
    /// to move to is information; the rest are warnings.
    static func glyph(for refusal: MoveRefusal?) -> String {
        refusal == .noDestination ? "info.circle" : "exclamationmark.triangle.fill"
    }

    static func tint(for refusal: MoveRefusal?) -> AnyShapeStyle {
        refusal == .noDestination ? AnyShapeStyle(AppThemeColor.appSecondary) : AnyShapeStyle(Color.orange)
    }

    /// "40%", for VoiceOver; nothing while the bar is indeterminate.
    static func percent(_ fraction: Double?) -> String {
        guard let fraction else { return "" }
        return fraction.formatted(.percent.precision(.fractionLength(0)))
    }
}

/// A label and its value on one row while they fit, one under the other when they
/// do not (a long library name, a long translation, the largest text size).
private struct SummaryRow<Value: View>: View {
    let label: Text
    @ViewBuilder let value: () -> Value

    var body: some View {
        ViewThatFits(in: .horizontal) {
            HStack(alignment: .firstTextBaseline, spacing: 12) {
                labelText
                Spacer(minLength: 8)
                valueView
            }
            VStack(alignment: .leading, spacing: 2) {
                labelText
                valueView
            }
        }
        .accessibilityElement(children: .combine)
    }

    private var labelText: some View {
        label
            .themedFont(.small, weight: .medium)
            .foregroundStyle(.appSecondary)
            .fixedSize(horizontal: false, vertical: true)
    }

    private var valueView: some View {
        value()
            .themedFont(.small)
            .foregroundStyle(.appText)
            .multilineTextAlignment(.trailing)
            .fixedSize(horizontal: false, vertical: true)
    }
}

/// A glyph and a sentence on a raised surface. The glyph is decoration; the
/// sentence is what VoiceOver reads.
private struct MoveNotice: View {
    let glyph: String
    let tint: AnyShapeStyle
    let text: String

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Image(systemName: glyph)
                .themedFont(.small)
                .foregroundStyle(tint)
                .accessibilityHidden(true)
            Text(verbatim: text)
                .themedFont(.small)
                .foregroundStyle(.appText)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 8)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            .appElevated, in: RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous)
        )
        .overlay(
            RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous)
                .stroke(.appBorder, lineWidth: 1)
        )
        .accessibilityElement(children: .combine)
    }
}

/// Presents the move sheet for the kinds of model its host shows. The Installed
/// Models list hosts text and image models; the Audio picker hosts audio.
///
/// **THE BINDING NEVER LETS A MOVE'S SHEET GO.** A close the system asks for is
/// passed to `AppModel.dismissMoveSheet`, which refuses while a move runs, so no
/// path dismisses the sheet under a running copy.
struct ModelMoveSheetHost: ViewModifier {
    @ObservedObject var model: AppModel
    let kinds: Set<MoveSubject.Kind>

    func body(content: Content) -> some View {
        content.sheet(
            item: Binding(
                get: { model.moveSheet.flatMap { kinds.contains($0.subject.kind) ? $0 : nil } },
                set: { if $0 == nil { _ = model.dismissMoveSheet() } })
        ) { sheet in
            ModelMoveSheet(model: model, sheet: sheet)
        }
    }
}

extension View {
    /// Hosts the move sheet for `kinds` of model.
    func modelMoveSheet(model: AppModel, kinds: Set<MoveSubject.Kind>) -> some View {
        modifier(ModelMoveSheetHost(model: model, kinds: kinds))
    }
}
