import Foundation

/// Checks the facade v1 fields whose values must be fixed before execution.
/// Dynamic prompts, run values, and report payloads remain in the AST unchanged.
enum WorkflowScriptLiteralValidation {
    static let maximumStaticAnalysisWorkItems = 65_536

    static func validate(_ ast: WorkflowScriptAST) -> [WorkflowScriptDiagnostic] {
        validate(ast, maximumWorkItems: maximumStaticAnalysisWorkItems)
    }

    static func validate(
        _ ast: WorkflowScriptAST,
        maximumWorkItems: Int
    ) -> [WorkflowScriptDiagnostic] {
        var validator = WorkflowScriptLiteralValidator(maximumWorkItems: maximumWorkItems)
        var constants: [String: WorkflowExpression] = [:]
        validator.validate(ast.body, constants: &constants)
        return validator.diagnostics
    }
}

private struct WorkflowScriptLiteralValidator {
    private enum StaticValueStatus {
        case staticLiteral
        case dynamicOrCyclic
        case workLimitExceeded
    }

    var diagnostics: [WorkflowScriptDiagnostic] = []
    private var remainingStaticAnalysisWorkItems: Int
    private var didReportStaticAnalysisBudgetExceeded = false

    init(maximumWorkItems: Int) {
        remainingStaticAnalysisWorkItems = maximumWorkItems
    }

    mutating func validate(_ block: WorkflowBlock, constants: inout [String: WorkflowExpression]) {
        for statement in block.statements {
            switch statement.kind {
            case .declaration(let name, _, let value):
                validateFacadeExpression(value, constants: constants)
                constants[name] = value
            case .expression(let expression):
                validateFacadeExpression(expression, constants: constants)
            case .conditional(_, let thenBlock, let elseBlock):
                var thenConstants = constants
                validate(thenBlock, constants: &thenConstants)
                if let elseBlock {
                    var elseConstants = constants
                    validate(elseBlock, constants: &elseConstants)
                }
            case .forOf(let name, _, _, let body):
                var loopConstants = constants
                loopConstants.removeValue(forKey: name)
                validate(body, constants: &loopConstants)
            }
        }
    }

    private mutating func validateFacadeExpression(
        _ expression: WorkflowExpression,
        constants: [String: WorkflowExpression]
    ) {
        switch expression.kind {
        case .call(let call), .awaited(let call):
            validate(call, constants: constants)
        default:
            break
        }
    }

    private mutating func validate(_ call: WorkflowCall, constants: [String: WorkflowExpression]) {
        switch call.target {
        case .agent:
            _ = requireString(call.arguments[0], field: "agent name")
            _ = requireString(call.arguments[1], field: "agent role")
        case .ask:
            _ = requireString(call.arguments[0], field: "ask actor")
            requireStaticShape(call.arguments[2], field: "ask result shape", constants: constants)
        case .parallel:
            validateParallel(call.arguments[0], constants: constants)
        case .criticLoop:
            validateCriticPolicy(call.arguments[0], constants: constants)
        case .phase:
            _ = requireString(call.arguments[0], field: "phase name")
        case .command:
            validateCommand(call)
        case .run:
            validateRun(call)
        case .worldRead:
            validateWorldRead(call)
        case .join, .report, .artifact, .glob, .read, .grep, .git:
            break
        }
    }

    private mutating func validateParallel(
        _ expression: WorkflowExpression,
        constants: [String: WorkflowExpression]
    ) {
        guard let nodes = requireArray(expression, field: "parallel nodes") else { return }
        for node in nodes {
            guard let fields = requireObject(
                node,
                field: "parallel node",
                required: ["id", "actor", "prompt", "shape", "dependsOn", "maxRetries"],
                allowed: ["id", "actor", "prompt", "shape", "dependsOn", "maxRetries"])
            else { continue }

            if let id = fields["id"] { _ = requireString(id.value, field: "parallel node id") }
            if let actor = fields["actor"] { _ = requireString(actor.value, field: "parallel node actor") }
            if let shape = fields["shape"] {
                requireStaticShape(shape.value, field: "parallel node shape", constants: constants)
            }
            if let dependencies = fields["dependsOn"] {
                validateStringArray(dependencies.value, field: "parallel dependencies")
            }
            if let retries = fields["maxRetries"] {
                _ = requireInteger(
                    retries.value,
                    field: "parallel retry limit",
                    range: 0...3)
            }
        }
    }

    private mutating func validateCriticPolicy(
        _ expression: WorkflowExpression,
        constants: [String: WorkflowExpression]
    ) {
        guard let fields = requireObject(
            expression,
            field: "critic policy",
            required: ["producer", "critic", "verdictField", "feedbackField", "maxIterations"],
            allowed: ["producer", "critic", "verdictField", "feedbackField", "maxIterations"])
        else { return }

        if let producer = fields["producer"],
           let producerFields = requireObject(
               producer.value,
               field: "critic producer",
               required: ["actor", "prompt", "shape"],
               allowed: ["actor", "prompt", "shape"])
        {
            if let actor = producerFields["actor"] {
                _ = requireString(actor.value, field: "critic producer actor")
            }
            if let shape = producerFields["shape"] {
                requireStaticShape(shape.value, field: "critic producer shape", constants: constants)
            }
        }

        if let critic = fields["critic"],
           let criticFields = requireObject(
               critic.value,
               field: "critic reviewer",
               required: ["actor", "prompt"],
               allowed: ["actor", "prompt"]),
           let actor = criticFields["actor"]
        {
            _ = requireString(actor.value, field: "critic actor")
        }

        if let verdict = fields["verdictField"] {
            _ = requireString(verdict.value, field: "critic verdict field")
        }
        if let feedback = fields["feedbackField"] {
            _ = requireString(feedback.value, field: "critic feedback field")
        }
        if let iterations = fields["maxIterations"] {
            _ = requireInteger(
                iterations.value,
                field: "critic iteration limit",
                range: 1...10)
        }
    }

    private mutating func validateCommand(_ call: WorkflowCall) {
        _ = requireString(call.arguments[0], field: "command key")
        guard let fields = requireObject(
            call.arguments[1],
            field: "command definition",
            required: ["executable", "workingDirectory", "argv"],
            allowed: ["executable", "workingDirectory", "argv"])
        else { return }

        if let executable = fields["executable"],
           let path = requireString(executable.value, field: "command executable path"),
           !path.hasPrefix("/")
        {
            append(
                .executablePathMustBeAbsolute,
                "Command executable path must be an absolute literal path.",
                at: executable.value.sourceRange.start)
        }
        if let workingDirectory = fields["workingDirectory"] {
            if let path = requireString(workingDirectory.value, field: "command working directory") {
                // Same rules as WorkflowWorld.validateRelativePath: containment
                // must not depend on a later dispatch path alone.
                let components = path.split(separator: "/", omittingEmptySubsequences: false)
                if path.isEmpty || path.hasPrefix("/") || path.utf8.contains(0)
                    || components.contains(where: { $0.isEmpty || $0 == "." || $0 == ".." })
                {
                    append(
                        .literalValueOutOfRange,
                        "Command working directory must be a non-empty workspace-relative path without '.', '..', or empty components.",
                        at: workingDirectory.value.sourceRange.start)
                }
            }
        }
        if let argv = fields["argv"], let items = requireArray(argv.value, field: "command argv") {
            for item in items {
                switch item.kind {
                case .literal(.string):
                    break
                case .object:
                    validateCommandSlot(item)
                default:
                    append(
                        .literalOnlyPosition,
                        "Command argv items must be fixed string literals or static argument slots.",
                        at: item.sourceRange.start)
                }
            }
        }
    }

    private mutating func validateCommandSlot(_ expression: WorkflowExpression) {
        guard let fields = requireObject(
            expression,
            field: "command argument slot",
            required: ["name", "kind"],
            allowed: ["name", "kind", "values", "maximumBytes"])
        else { return }

        if let name = fields["name"] {
            _ = requireString(name.value, field: "command argument name")
        }
        guard let kindMember = fields["kind"],
              let kind = requireString(kindMember.value, field: "command argument kind")
        else { return }

        switch kind {
        case "workspaceInputPath":
            if let values = fields["values"] {
                append(.literalUnknownField, "workspaceInputPath slots do not accept values.", at: values.nameRange.start)
            }
            if let limit = fields["maximumBytes"] {
                append(.literalUnknownField, "workspaceInputPath slots do not accept maximumBytes.", at: limit.nameRange.start)
            }
        case "allowedValue":
            if let values = fields["values"] {
                validateStringArray(values.value, field: "allowed command values")
            } else {
                append(.literalRequiredFieldMissing, "allowedValue slots require a literal values array.", at: expression.sourceRange.start)
            }
            if let limit = fields["maximumBytes"] {
                append(.literalUnknownField, "allowedValue slots do not accept maximumBytes.", at: limit.nameRange.start)
            }
        case "boundedText":
            if let values = fields["values"] {
                append(.literalUnknownField, "boundedText slots do not accept values.", at: values.nameRange.start)
            }
            if let limit = fields["maximumBytes"] {
                _ = requireInteger(
                    limit.value,
                    field: "boundedText byte limit",
                    range: 1...4_096)
            } else {
                append(.literalRequiredFieldMissing, "boundedText slots require a literal maximumBytes value.", at: expression.sourceRange.start)
            }
        default:
            append(
                .literalEnumValueUnsupported,
                "Command argument kind must be workspaceInputPath, allowedValue, or boundedText.",
                at: kindMember.value.sourceRange.start)
        }
    }

    private mutating func validateRun(_ call: WorkflowCall) {
        let keyIsLiteral: Bool
        if case .literal(.string) = call.arguments[0].kind {
            keyIsLiteral = true
        } else {
            keyIsLiteral = false
            append(
                .runCommandKeyMustBeLiteral,
                "run command key must be a string literal naming a prior command declaration.",
                at: call.arguments[0].sourceRange.start)
        }

        let supportsNamedValues: Bool
        switch call.arguments[1].kind {
        case .object, .identifier, .member:
            supportsNamedValues = true
        default:
            supportsNamedValues = false
            append(
                .runRequiresNamedDynamicValues,
                "run must receive a named dynamic-values object or binding.",
                at: call.arguments[1].sourceRange.start)
        }

        if keyIsLiteral, supportsNamedValues {
            do {
                try WorkflowFacade.validateRunDispatch(
                    commandKeyForm: .stringLiteral,
                    argumentForm: .namedDynamicValues)
            } catch {
                append(
                    .runRequiresNamedDynamicValues,
                    "run accepts named dynamic values only.",
                    at: call.arguments[1].sourceRange.start)
            }
        }
    }

    private mutating func validateWorldRead(_ call: WorkflowCall) {
        guard case .call(let operation) = call.arguments[0].kind else {
            append(
                .worldOperationMustBeStatic,
                "world.read requires a fixed glob, read, grep, or git operation.",
                at: call.arguments[0].sourceRange.start)
            return
        }

        switch operation.target {
        case .glob:
            _ = requireString(operation.arguments[0], field: "glob pattern")
        case .read:
            _ = requireString(operation.arguments[0], field: "world read path")
            // WorkflowWorld refuses reads above its byte cap at run time, so
            // reject them at check time instead of after approval.
            _ = requireInteger(
                operation.arguments[1], field: "world read byte limit",
                range: 1...WorkflowWorldLimits().maximumReadBytes)
        case .grep:
            _ = requireString(operation.arguments[0], field: "grep pattern")
            if operation.arguments.count == 2 {
                let hint = operation.arguments[1]
                if case .literal(.null) = hint.kind {
                    break
                }
                _ = requireString(hint, field: "grep path hint")
            }
        case .git:
            guard let operationName = requireString(operation.arguments[0], field: "git operation") else { return }
            guard ["status", "diff", "log", "changedFiles"].contains(operationName) else {
                append(
                    .literalEnumValueUnsupported,
                    "git operation must be status, diff, log, or changedFiles.",
                    at: operation.arguments[0].sourceRange.start)
                return
            }
        default:
            append(
                .worldOperationMustBeStatic,
                "world.read operation is not a declared read form.",
                at: operation.sourceRange.start)
        }
    }

    private mutating func validateStringArray(_ expression: WorkflowExpression, field: String) {
        guard let values = requireArray(expression, field: field) else { return }
        for value in values {
            _ = requireString(value, field: field)
        }
    }

    private mutating func requireStaticShape(
        _ expression: WorkflowExpression,
        field: String,
        constants: [String: WorkflowExpression]
    ) {
        switch isStaticValue(expression, constants: constants) {
        case .staticLiteral:
            return
        case .dynamicOrCyclic:
            append(
                .shapeMustBeStatic,
                "\(field) must resolve to a statically defined literal value.",
                at: expression.sourceRange.start)
        case .workLimitExceeded:
            if !didReportStaticAnalysisBudgetExceeded {
                didReportStaticAnalysisBudgetExceeded = true
                append(
                    .staticAnalysisBudgetExceeded,
                    "Static literal analysis exceeded its bounded work budget.",
                    at: expression.sourceRange.start)
            }
        }
    }

    private mutating func isStaticValue(
        _ expression: WorkflowExpression,
        constants: [String: WorkflowExpression]
    ) -> StaticValueStatus {
        enum Pending {
            case expression(WorkflowExpression)
            case leaveBinding(String)
        }

        var pending: [Pending] = [.expression(expression)]
        var activeBindings: Set<String> = []
        while let item = pending.popLast() {
            guard remainingStaticAnalysisWorkItems > 0 else { return .workLimitExceeded }
            remainingStaticAnalysisWorkItems -= 1
            switch item {
            case .leaveBinding(let name):
                activeBindings.remove(name)
            case .expression(let value):
                switch value.kind {
                case .literal:
                    break
                case .array(let values):
                    for child in values.reversed() {
                        pending.append(.expression(child))
                    }
                case .object(let members):
                    for member in members.reversed() {
                        pending.append(.expression(member.value))
                    }
                case .identifier(let name):
                    guard !activeBindings.contains(name), let boundValue = constants[name] else {
                        return .dynamicOrCyclic
                    }
                    activeBindings.insert(name)
                    pending.append(.leaveBinding(name))
                    pending.append(.expression(boundValue))
                case .member, .unaryNot, .binary, .call, .awaited:
                    return .dynamicOrCyclic
                }
            }
        }
        return .staticLiteral
    }

    private mutating func requireString(_ expression: WorkflowExpression, field: String) -> String? {
        guard case .literal(.string(let value)) = expression.kind else {
            append(
                .literalOnlyPosition,
                "\(field) must be a string literal.",
                at: expression.sourceRange.start)
            return nil
        }
        return value
    }

    private mutating func requireInteger(
        _ expression: WorkflowExpression,
        field: String,
        range: ClosedRange<Int>
    ) -> Int? {
        guard case .literal(.number(let number)) = expression.kind else {
            append(
                .literalOnlyPosition,
                "\(field) must be an integer literal.",
                at: expression.sourceRange.start)
            return nil
        }
        guard number.isFinite, number.rounded(.towardZero) == number,
              number >= Double(range.lowerBound), number <= Double(range.upperBound)
        else {
            append(
                .literalValueOutOfRange,
                "\(field) must be an integer from \(range.lowerBound) through \(range.upperBound).",
                at: expression.sourceRange.start)
            return nil
        }
        return Int(number)
    }

    private mutating func requireInteger(
        _ expression: WorkflowExpression,
        field: String,
        minimum: Int
    ) -> Int? {
        guard case .literal(.number(let number)) = expression.kind else {
            append(
                .literalOnlyPosition,
                "\(field) must be an integer literal.",
                at: expression.sourceRange.start)
            return nil
        }
        guard number.isFinite, number.rounded(.towardZero) == number,
              number >= Double(minimum), number < Double(Int.max)
        else {
            append(
                .literalValueOutOfRange,
                "\(field) must be an integer greater than or equal to \(minimum).",
                at: expression.sourceRange.start)
            return nil
        }
        return Int(number)
    }

    private mutating func requireArray(_ expression: WorkflowExpression, field: String) -> [WorkflowExpression]? {
        guard case .array(let values) = expression.kind else {
            append(
                .literalStructureRequired,
                "\(field) must use an array literal.",
                at: expression.sourceRange.start)
            return nil
        }
        return values
    }

    private mutating func requireObject(
        _ expression: WorkflowExpression,
        field: String,
        required: [String],
        allowed: [String]
    ) -> [String: WorkflowObjectMember]? {
        guard case .object(let members) = expression.kind else {
            append(
                .literalStructureRequired,
                "\(field) must use an object literal.",
                at: expression.sourceRange.start)
            return nil
        }

        var result: [String: WorkflowObjectMember] = [:]
        for member in members {
            if result[member.name] != nil {
                append(
                    .literalDuplicateField,
                    "\(field) repeats static field '\(member.name)'.",
                    at: member.nameRange.start)
            } else {
                result[member.name] = member
            }
            if !allowed.contains(member.name) {
                append(
                    .literalUnknownField,
                    "\(field) has unsupported static field '\(member.name)'.",
                    at: member.nameRange.start)
            }
        }
        for name in required where result[name] == nil {
            append(
                .literalRequiredFieldMissing,
                "\(field) requires static field '\(name)'.",
                at: expression.sourceRange.start)
        }
        return result
    }

    private mutating func append(
        _ rule: WorkflowScriptDiagnosticRule,
        _ message: String,
        at location: WorkflowSourceLocation
    ) {
        diagnostics.append(WorkflowScriptDiagnostic(rule: rule, message: message, location: location))
    }
}
