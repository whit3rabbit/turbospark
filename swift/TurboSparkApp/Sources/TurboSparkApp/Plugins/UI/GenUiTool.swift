import Foundation

/// Schema and executor for `gen_ui_render` (ZP-06). The model passes the card
/// as DATA (`spec`, a JSON string); the executor only validates it and says
/// what it accepted, and the chat draws the card from the same validated spec.
///
/// **Ladder position:** category `.automation` and risk `safe`, because the
/// call has no effect outside the chat: no file, no network, no process, no
/// tool authority, and a follow-up button only fills the composer draft. If a
/// future component could do any of those, this entry has to move up first.
public enum GenUiToolDefinitions {
    public static let render = OpenAITool.function(
        name: "gen_ui_render",
        description: """
            Show a small interactive card in the chat for a question with a visual part: \
            sliders, steppers, toggles, pickers, a live readout, a table, a bar chart, and \
            buttons that draft a follow-up message. The card is drawn by the app from your \
            JSON; it cannot contain HTML or script and cannot call tools or reach the \
            network. A follow_up button only fills the message box with text the person \
            sends themselves. Use it sparingly, once per question, only when adjusting \
            values beats a table in text.
            """,
        parameters: .object(
            properties: [
                "spec": .string(description: """
                    JSON object {"title":"...","components":[...]}, at most 16 KB and 40 components. Components: \
                    {"type":"text","text":"...","style":"heading"|"body"|"caption"} | \
                    {"type":"slider"|"stepper","id":"rate","label":"...","min":0,"max":10,"step":0.5,"value":5,"unit":"%"} | \
                    {"type":"toggle","id":"flag","label":"...","value":true} | \
                    {"type":"picker","id":"mode","label":"...","options":["a","b"],"value":0} | \
                    {"type":"readout","label":"...","bind":"rate","unit":"%"} | \
                    {"type":"table","header":["..."],"rows":[["..."]]} (at most 20 rows, 8 columns) | \
                    {"type":"bars","title":"...","items":[{"label":"...","value":3}]} (at most 24) | \
                    {"type":"follow_up","label":"Ask about this","template":"With rate {rate} and mode {mode}, ..."} \
                    where {id} in a template is replaced by that control's current value. Ids match [a-z][a-z0-9_]{0,31} and are unique. Unknown fields are rejected.
                    """),
            ],
            required: ["spec"]
        )
    )

    public static let all: [OpenAITool] = [render]
}

enum GenUiRenderExecutor {
    /// Validates the spec. A failure names the offending path so the model
    /// can correct it and call again.
    static func execute(arguments: [String: String]) throws -> String {
        guard let raw = arguments["spec"], !raw.isEmpty else {
            throw NSError(domain: "TurboSparkGenUi", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "gen_ui_render needs a 'spec' JSON string."])
        }
        switch GenUiParser.parse(raw) {
        case .success(let spec):
            return "Showed an interactive card \"\(spec.title)\" with \(spec.components.count) components in the chat. "
                + "The person can adjust it; follow-ups arrive as drafts they send themselves."
        case .failure(let error):
            throw NSError(domain: "TurboSparkGenUi", code: 2, userInfo: [
                NSLocalizedDescriptionKey: "The card spec was rejected: \(error.message) Fix it and call gen_ui_render again."])
        }
    }
}
