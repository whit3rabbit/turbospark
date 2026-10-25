import Foundation


/// UTF-8 byte offset with one-based Unicode-scalar line and column coordinates.
struct WorkflowSourceLocation: Equatable, Sendable {
    let byteOffset: Int
    let line: Int
    let column: Int
}

/// Half-open UTF-8 source range.
struct WorkflowSourceRange: Equatable, Sendable {
    let start: WorkflowSourceLocation
    let end: WorkflowSourceLocation
}

enum WorkflowScriptDiagnosticRule: String, Equatable, Sendable {
    case sourceByteLimitExceeded
    case tokenLimitExceeded
    case astNodeLimitExceeded
    case nestingLimitExceeded
    case invalidEntryForm
    case unexpectedToken
    case unsupportedStatement
    case unsupportedExpression
    case unsupportedLoopForm
    case unsupportedOperator
    case unsupportedCall
    case invalidAwait
    case invalidLiteral
    case unterminatedString
    case invalidEscape
    case unterminatedComment
    case misplacedFacadeCall
    case unsupportedWorldReadOperation
    case facadeCallInsideLoop
    case runDispatchExecutableOverride
    case literalOnlyPosition
    case literalStructureRequired
    case literalRequiredFieldMissing
    case literalUnknownField
    case literalDuplicateField
    case literalValueOutOfRange
    case literalEnumValueUnsupported
    case executablePathMustBeAbsolute
    case shapeMustBeStatic
    case staticAnalysisBudgetExceeded
    case worldOperationMustBeStatic
    case runCommandKeyMustBeLiteral
    case runRequiresNamedDynamicValues
    case duplicateActorName
    case unknownActor
    case duplicateCommandPin
    case unknownCommandPin
    case misplacedPhaseMarker
    case duplicateGraphNodeID
    case unknownGraphDependency
    case graphCycle
    case graphNodeLimitExceeded
    case unknownGraphReference
    case graphNotJoinedOnEveryPath
    case graphJoinedMoreThanOnce
    case graphJoinInsideLoop
    case unserializablePublishedPayload
    case undeclaredWorkflowArgument
    case executableIdentityUnavailable
    case reservedActorName
    case undeclaredIdentifier
    case duplicateBinding
    case reservedBindingName
    case duplicateCommandSlot
}

struct WorkflowScriptDiagnostic: Equatable, Sendable {
    let rule: WorkflowScriptDiagnosticRule
    let message: String
    let location: WorkflowSourceLocation
}

enum WorkflowScriptLiteral: Equatable, Sendable {
    case null
    case boolean(Bool)
    case number(Double)
    case string(String)
}

enum WorkflowBinaryOperator: String, Equatable, Sendable {
    case logicalOr
    case logicalAnd
    case strictEqual
    case strictNotEqual
    case lessThan
    case lessThanOrEqual
    case greaterThan
    case greaterThanOrEqual
}

enum WorkflowCallTarget: String, CaseIterable, Equatable, Sendable {
    case agent
    case ask
    case parallel
    case join
    case criticLoop
    case phase
    case worldRead
    case command
    case run
    case report
    case artifact
    case glob
    case read
    case grep
    case git
}

struct WorkflowCall: Equatable, Sendable {
    let target: WorkflowCallTarget
    let arguments: [WorkflowExpression]
    let sourceRange: WorkflowSourceRange
}

struct WorkflowObjectMember: Equatable, Sendable {
    let name: String
    let nameRange: WorkflowSourceRange
    let value: WorkflowExpression
    let sourceRange: WorkflowSourceRange
}

struct WorkflowExpression: Equatable, Sendable {
    indirect enum Kind: Equatable, Sendable {
        case literal(WorkflowScriptLiteral)
        case identifier(String)
        case member(base: WorkflowExpression, name: String)
        case array([WorkflowExpression])
        case object([WorkflowObjectMember])
        case unaryNot(WorkflowExpression)
        case binary(left: WorkflowExpression, op: WorkflowBinaryOperator, right: WorkflowExpression)
        case call(WorkflowCall)
        case awaited(WorkflowCall)
    }

    let kind: Kind
    let sourceRange: WorkflowSourceRange
    let treeDepth: Int
}

struct WorkflowBlock: Equatable, Sendable {
    let statements: [WorkflowStatement]
    let sourceRange: WorkflowSourceRange
}

struct WorkflowStatement: Equatable, Sendable {
    indirect enum Kind: Equatable, Sendable {
        case declaration(name: String, nameRange: WorkflowSourceRange, value: WorkflowExpression)
        case expression(WorkflowExpression)
        case conditional(
            condition: WorkflowExpression,
            thenBlock: WorkflowBlock,
            elseBlock: WorkflowBlock?)
        case forOf(name: String, nameRange: WorkflowSourceRange, sequence: WorkflowExpression, body: WorkflowBlock)
    }

    let kind: Kind
    let sourceRange: WorkflowSourceRange
}

struct WorkflowScriptAST: Equatable, Sendable {
    static let currentVersion = 1

    let version: Int
    let facadeVersion: Int
    let sourceRange: WorkflowSourceRange
    let body: WorkflowBlock
}

struct WorkflowScriptParseResult: Equatable, Sendable {
    let ast: WorkflowScriptAST?
    let diagnostics: [WorkflowScriptDiagnostic]

    var isValid: Bool {
        ast != nil && diagnostics.isEmpty
    }
}

struct WorkflowCheckedProgram: Equatable, Sendable {
    let ast: WorkflowScriptAST
    let descriptor: WorkflowRunDescriptor
}

struct WorkflowCheckOutcome: Equatable, Sendable {
    let checked: WorkflowCheckedProgram?
    let diagnostics: [WorkflowScriptDiagnostic]

    var isValid: Bool {
        checked != nil && diagnostics.isEmpty
    }
}

struct WorkflowDefinitionCheckOutcome: Equatable, Sendable {
    let ast: WorkflowScriptAST?
    let manifest: WorkflowLaunchManifest?
    let diagnostics: [WorkflowScriptDiagnostic]

    var isValid: Bool {
        ast != nil && manifest != nil && diagnostics.isEmpty
    }
}
