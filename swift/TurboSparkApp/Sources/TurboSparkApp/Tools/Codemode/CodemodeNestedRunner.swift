import Foundation

/// Executes one `tools.<name>(...)` call made from inside a codemode
/// script. This is the context-bleed guard: a script-issued call MUST take
/// the same path as a model-issued one, so the handler rebuilds the inner
/// MCP call and routes through `executeDeferredMcpCall`, which runs the
/// call's own PreToolUse hooks, session approvals, permission-engine
/// decision, and PostToolUse hooks. A rejected tool stays rejected when it
/// is wrapped in a script.
///
/// The handler only accepts names in the granted descriptor set (deny-rule
/// stripping already applied) and refuses `codemode` itself, so scripts
/// cannot recurse.
enum CodemodeNestedRunner {
    static func callHandler(
        descriptors: [DeferredToolDescriptor],
        project: AppProject?,
        chatID: UUID?
    ) -> CodemodeWorkerSupervisor.CallHandler {
        let byName = Dictionary(
            descriptors.map { ($0.name.lowercased(), $0) },
            uniquingKeysWith: { first, _ in first })
        return { name, argumentsJSON in
            func reject(_ message: String) -> Result<String, CodemodeCallError> {
                .failure(CodemodeCallError(message: message))
            }
            guard name.lowercased() != "codemode" else {
                return reject("codemode cannot invoke itself from inside a script.")
            }
            guard let descriptor = byName[name.lowercased()] else {
                return reject(
                    "'\(name)' is not in the currently granted deferred MCP catalog, so it "
                        + "cannot be called from a script. ALL_TOOLS lists what is available.")
            }
            let arguments: [String: Any]
            if let argumentsJSON, !argumentsJSON.isEmpty {
                do {
                    arguments = try ToolSearchCatalog.parseJSONObject(argumentsJSON)
                } catch {
                    return reject(
                        "the arguments for '\(name)' were not a JSON object: "
                            + error.localizedDescription)
                }
            } else {
                arguments = [:]
            }
            do {
                let output = try await AppToolRegistry.executeDeferredMcpCall(
                    name: descriptor.name,
                    arguments: arguments,
                    project: project,
                    chatID: chatID)
                return .success(output)
            } catch {
                return reject(error.localizedDescription)
            }
        }
    }
}
