import Foundation

/// The versioned authoring surface for workflow source.
///
/// Workflow scripts are parsed by the restricted checker. This declaration is
/// also the contract source for API-reference consumers; it is not a JavaScript
/// runtime and does not evaluate source text.
enum WorkflowFacade {
    static let version = 1
    static let apiReferenceResourceName = "workflow-facade-v1"
    static let entryForm = "async function workflow() { ... }"

    enum Member: String, CaseIterable, Sendable {
        case agent
        case ask
        case parallel
        case join
        case criticLoop
        case phase
        case world
        case command
        case run
        case report
        case artifact
        case args
    }

    static let members = Member.allCases

    enum RunCommandKeyForm: String, Decodable, Sendable {
        case stringLiteral
        case dynamicExpression
    }

    enum RunArgumentForm: String, Decodable, Sendable {
        case namedDynamicValues
        case executableCommandDefinition
    }

    struct RunDispatchShape: Decodable, Sendable {
        let source: String
        let commandKeyForm: RunCommandKeyForm
        let argumentForm: RunArgumentForm
        let accepted: Bool
    }

    enum ContractViolation: Error, Equatable, Sendable {
        case commandKeyMustBeLiteral
        case runAcceptsDynamicValuesOnly
    }

    /// Rejects run-call shapes that can select or replace a pinned executable.
    /// The checker maps parsed call shapes to this contract before authoring is
    /// accepted; parsing itself belongs to WorkflowScriptChecker.
    static func validateRunDispatch(
        commandKeyForm: RunCommandKeyForm,
        argumentForm: RunArgumentForm
    ) throws {
        guard commandKeyForm == .stringLiteral else {
            throw ContractViolation.commandKeyMustBeLiteral
        }
        guard argumentForm == .namedDynamicValues else {
            throw ContractViolation.runAcceptsDynamicValuesOnly
        }
    }

    static func loadAPIReference() throws -> String {
        guard let url = Bundle.module.url(
            forResource: apiReferenceResourceName,
            withExtension: "md")
        else {
            throw APIReferenceError.missingResource(
                bundleURL: Bundle.module.bundleURL.path,
                resourceURL: Bundle.module.resourceURL?.path ?? "")
        }
        return try String(contentsOf: url, encoding: .utf8)
    }

    enum APIReferenceError: Error, Equatable {
        case missingResource(bundleURL: String, resourceURL: String)
    }
}
