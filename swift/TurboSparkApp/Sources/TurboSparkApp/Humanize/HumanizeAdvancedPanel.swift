import SwiftUI

/// Rules that work outside the model, plus the model's own instruction shown
/// read-only. Editing that instruction is deliberately not offered: the
/// checkpoint was trained on those exact bytes and the engine refuses any
/// other prompt.
struct HumanizeAdvancedPanel: View {
    @ObservedObject var controller: HumanizeController
    let onClose: () -> Void

    @Environment(\.appTheme) private var theme
    @State private var showInstruction = false
    @FocusState private var focusedField: Field?

    private enum Field: Hashable { case protected, avoid, replacements }

    var body: some View {
        HumanizeCard {
            Image(systemName: "slider.horizontal.3").font(theme.ui(.small, weight: .semibold))
                .foregroundStyle(.appAccent).accessibilityHidden(true)
            Text("Advanced", bundle: .module).font(theme.ui(.callout, weight: .semibold))
                .accessibilityAddTraits(.isHeader)
            Spacer(minLength: 8)
            HumanizeChip(title: "Close", systemImage: "xmark", iconOnly: true, action: onClose)
        } content: {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    ruleField(
                        "Protected terms",
                        detail: "Each line must appear in the rewrite exactly as written. Missing terms are flagged.",
                        example: "Skills Manager", text: $controller.rules.protectedTerms, field: .protected)
                    ruleField(
                        "Words to avoid",
                        detail: "Flagged and highlighted when they appear in the rewrite.",
                        example: "delve", text: $controller.rules.avoidTerms, field: .avoid)
                    ruleField(
                        "Replacements",
                        detail: "One per line as old -> new. Applied to the rewrite after the model's own checks.",
                        example: "utilize -> use", text: $controller.rules.replacements, field: .replacements)
                    seedSection
                    instructionSection
                }
                .padding(16)
            }
        } footer: {
            Text("Rules apply to the next rewrite.", bundle: .module)
                .font(theme.ui(.tiny)).foregroundStyle(.appSecondary)
            Spacer(minLength: 0)
        }
    }

    private func ruleField(
        _ title: LocalizedStringKey, detail: LocalizedStringKey, example: String,
        text: Binding<String>, field: Field
    ) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title, bundle: .module).font(theme.ui(.small, weight: .semibold))
            Text(detail, bundle: .module).font(theme.ui(.tiny)).foregroundStyle(.appSecondary)
                .fixedSize(horizontal: false, vertical: true)
            ZStack(alignment: .topLeading) {
                if text.wrappedValue.isEmpty {
                    Text(verbatim: example).font(theme.ui(.small)).foregroundStyle(.tertiary)
                        .padding(.horizontal, 9).padding(.vertical, 8)
                        .allowsHitTesting(false)
                }
                TextEditor(text: text)
                    .font(theme.ui(.small))
                    .scrollContentBackground(.hidden)
                    .focused($focusedField, equals: field)
                    .padding(.horizontal, 4).padding(.vertical, 8)
                    .accessibilityLabel(Text(title, bundle: .module))
            }
            .frame(height: 76)
            .background(Color.primary.opacity(0.04), in: RoundedRectangle(cornerRadius: 10, style: .continuous))
            .overlay {
                RoundedRectangle(cornerRadius: 10, style: .continuous).stroke(
                    focusedField == field ? theme.accent.opacity(0.55) : Color.primary.opacity(0.1),
                    lineWidth: focusedField == field ? 1.25 : 0.75)
            }
            .animation(TSMotion.hover, value: focusedField)
        }
    }

    private var seedSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            Toggle(isOn: $controller.rules.fixedSeed) {
                Text("Repeatable output", bundle: .module).font(theme.ui(.small, weight: .semibold))
            }
            .toggleStyle(.switch).controlSize(.mini).tint(theme.accent)
            Text("The same draft and seed give the same rewrite.", bundle: .module)
                .font(theme.ui(.tiny)).foregroundStyle(.appSecondary)
            HStack(spacing: 8) {
                Text("Seed", bundle: .module).font(theme.ui(.small)).foregroundStyle(.appSecondary)
                TextField(value: $controller.rules.seed, format: .number.grouping(.never)) {
                    Text("Seed", bundle: .module)
                }
                .labelsHidden()
                .textFieldStyle(.roundedBorder)
                .font(theme.ui(.small).monospacedDigit())
                .frame(width: 110)
                HumanizeChip(title: "New seed", systemImage: "dice", iconOnly: true) {
                    controller.rules.seed = UInt64.random(in: 1...999_999)
                }
            }
            .disabled(!controller.rules.fixedSeed)
        }
    }

    private var instructionSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Button {
                withAnimation(TSMotion.select) { showInstruction.toggle() }
            } label: {
                HStack(spacing: 6) {
                    Image(systemName: "chevron.right")
                        .font(theme.ui(.micro, weight: .bold))
                        .rotationEffect(.degrees(showInstruction ? 90 : 0))
                    Text("Model instruction", bundle: .module).font(theme.ui(.small, weight: .semibold))
                    Image(systemName: "lock.fill").font(theme.ui(.micro)).foregroundStyle(.appSecondary)
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            if showInstruction {
                Text("The model was trained on this exact text, so it cannot be edited. The rules above work outside the model.", bundle: .module)
                    .font(theme.ui(.tiny)).foregroundStyle(.appSecondary)
                    .fixedSize(horizontal: false, vertical: true)
                Text(verbatim: HumanizeInstruction.text)
                    .font(theme.code(.small))
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(10)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(Color.primary.opacity(0.04), in: RoundedRectangle(cornerRadius: 10, style: .continuous))
                    .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
    }
}
