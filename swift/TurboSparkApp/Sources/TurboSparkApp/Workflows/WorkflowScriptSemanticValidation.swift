import Foundation

/// Static checks that need the complete parsed body. This pass inspects source
/// structure only; it never executes a branch or resolves runtime values.
enum WorkflowScriptSemanticValidation {
    static func validate(_ ast: WorkflowScriptAST) -> [WorkflowScriptDiagnostic] {
        var validator = WorkflowScriptSemanticValidator()
        return validator.validate(ast)
    }

    static func validateDefinitionArguments(
        _ ast: WorkflowScriptAST,
        declaredNames: Set<String>
    ) -> [WorkflowScriptDiagnostic] {
        var validator = WorkflowScriptSemanticValidator()
        validator.validateDefinitionArguments(ast.body, declaredNames: declaredNames)
        return validator.diagnostics
    }
}

private struct WorkflowScriptSemanticValidator {
    private struct ActorDeclaration {
        let sourceOffset: Int
    }

    private struct CommandDeclaration {
        let sourceOffset: Int
    }

    private struct GraphDependency {
        let name: String
        let location: WorkflowSourceLocation
    }

    private struct GraphNode {
        let id: String
        let dependencies: [GraphDependency]
    }

    private struct GraphDescriptor {
        let bindingName: String
        let identity: Int
        let location: WorkflowSourceLocation
    }

    private struct JoinFlowState: Equatable {
        var graphBindings: [String: Int] = [:]
        var joinCounts: [Int: Int] = [:]
    }

    private(set) var diagnostics: [WorkflowScriptDiagnostic] = []
    private var actorDeclarations: [String: ActorDeclaration] = [:]
    private var commandDeclarations: [String: CommandDeclaration] = [:]
    private var graphDescriptors: [GraphDescriptor] = []
    private var remainingFlowSteps = 100_000
    private var didReportFlowBudget = false
    private var remainingPublishedPayloadSteps = 65_536
    private var didReportPublishedPayloadBudget = false

    mutating func validate(_ ast: WorkflowScriptAST) -> [WorkflowScriptDiagnostic] {
        collectEntryActors(ast.body)
        collectEntryCommands(ast.body)
        validateBindingScopes(ast.body)
        validateFacadePlacement(in: ast.body, isEntryBlock: true)
        validateActorReferences(in: ast.body)
        collectAndValidateGraphs(in: ast.body)

        let endStates = analyzeJoinFlow(in: ast.body, states: [JoinFlowState()], isInsideLoop: false)
        for graph in graphDescriptors {
            for state in endStates {
                guard let count = state.joinCounts[graph.identity] else { continue }
                if count == 0 {
                    append(
                        .graphNotJoinedOnEveryPath,
                        "Graph '\(graph.bindingName)' must be joined exactly once on every path that creates it.",
                        at: graph.location)
                }
            }
        }

        validatePublishedPayloads(in: ast.body)
        return diagnostics
    }

    mutating func validateDefinitionArguments(_ block: WorkflowBlock, declaredNames: Set<String>) {
        validateArgumentReferences(in: block, declaredNames: declaredNames)
    }

    private mutating func collectEntryActors(_ block: WorkflowBlock) {
        for statement in block.statements {
            guard case .expression(let expression) = statement.kind,
                  case .call(let call) = expression.kind,
                  call.target == .agent,
                  let nameExpression = call.arguments.first,
                  case .literal(.string(let name)) = nameExpression.kind
            else { continue }

            if name == "main" {
                append(
                    .reservedActorName,
                    "Actor name 'main' is reserved for the workflow entry lane.",
                    at: nameExpression.sourceRange.start)
            }

            if actorDeclarations[name] != nil {
                append(
                    .duplicateActorName,
                    "Actor name '\(name)' is declared more than once.",
                    at: nameExpression.sourceRange.start)
            } else {
                actorDeclarations[name] = ActorDeclaration(
                    sourceOffset: nameExpression.sourceRange.start.byteOffset)
            }
        }
    }

    private mutating func collectEntryCommands(_ block: WorkflowBlock) {
        for statement in block.statements {
            guard case .expression(let expression) = statement.kind,
                  case .call(let call) = expression.kind,
                  call.target == .command,
                  let keyExpression = call.arguments.first,
                  case .literal(.string(let key)) = keyExpression.kind
            else { continue }

            validateCommandSlots(call)
            if commandDeclarations[key] != nil {
                append(
                    .duplicateCommandPin,
                    "Command pin key '\(key)' is declared more than once.",
                    at: keyExpression.sourceRange.start)
            } else {
                commandDeclarations[key] = CommandDeclaration(
                    sourceOffset: keyExpression.sourceRange.start.byteOffset)
            }
        }
    }

    private mutating func validateFacadePlacement(in block: WorkflowBlock, isEntryBlock: Bool) {
        for statement in block.statements {
            switch statement.kind {
            case .declaration:
                break
            case .expression(let expression):
                guard case .call(let call) = expression.kind else { continue }
                if !isEntryBlock, [.agent, .command].contains(call.target) {
                    append(
                        .misplacedFacadeCall,
                        "Actor and command declarations must be direct statements in the workflow entry block.",
                        at: call.sourceRange.start)
                }
                if !isEntryBlock, call.target == .phase {
                    append(
                        .misplacedPhaseMarker,
                        "Phase markers must be direct statements in the workflow entry block.",
                        at: call.sourceRange.start)
                }
            case .conditional(_, let thenBlock, let elseBlock):
                validateFacadePlacement(in: thenBlock, isEntryBlock: false)
                if let elseBlock {
                    validateFacadePlacement(in: elseBlock, isEntryBlock: false)
                }
            case .forOf(_, _, _, let body):
                validateFacadePlacement(in: body, isEntryBlock: false)
            }
        }
    }

    private mutating func validateActorReferences(in block: WorkflowBlock) {
        for statement in block.statements {
            switch statement.kind {
            case .declaration(_, _, let value):
                validateActorReferences(in: value)
            case .expression(let expression):
                validateActorReferences(in: expression)
            case .conditional(let condition, let thenBlock, let elseBlock):
                validateActorReferences(in: condition)
                validateActorReferences(in: thenBlock)
                if let elseBlock {
                    validateActorReferences(in: elseBlock)
                }
            case .forOf(_, _, let sequence, let body):
                validateActorReferences(in: sequence)
                validateActorReferences(in: body)
            }
        }
    }

    private mutating func validateActorReferences(in expression: WorkflowExpression) {
        switch expression.kind {
        case .call(let call), .awaited(let call):
            switch call.target {
            case .ask:
                if let actor = call.arguments.first {
                    validateActorReference(actor)
                }
            case .run:
                if let commandKey = call.arguments.first {
                    validateCommandReference(commandKey)
                }
            case .criticLoop:
                validateCriticActors(call.arguments[0])
            default:
                break
            }
            for argument in call.arguments {
                validateActorReferences(in: argument)
            }
        case .member(let base, _), .unaryNot(let base):
            validateActorReferences(in: base)
        case .array(let values):
            for value in values { validateActorReferences(in: value) }
        case .object(let members):
            for member in members { validateActorReferences(in: member.value) }
        case .binary(let left, _, let right):
            validateActorReferences(in: left)
            validateActorReferences(in: right)
        case .literal, .identifier:
            break
        }
    }

    private mutating func validateCriticActors(_ expression: WorkflowExpression) {
        guard let producer = objectMember("producer", in: expression),
              let producerActor = objectMember("actor", in: producer),
              let critic = objectMember("critic", in: expression),
              let criticActor = objectMember("actor", in: critic)
        else { return }
        validateActorReference(producerActor)
        validateActorReference(criticActor)
    }

    private mutating func validateActorReference(_ expression: WorkflowExpression) {
        guard case .literal(.string(let name)) = expression.kind else { return }
        guard let declaration = actorDeclarations[name],
              declaration.sourceOffset < expression.sourceRange.start.byteOffset
        else {
            append(
                .unknownActor,
                "Actor '\(name)' must be declared before this use.",
                at: expression.sourceRange.start)
            return
        }
    }

    private mutating func validateCommandReference(_ expression: WorkflowExpression) {
        guard case .literal(.string(let key)) = expression.kind else { return }
        guard let declaration = commandDeclarations[key],
              declaration.sourceOffset < expression.sourceRange.start.byteOffset
        else {
            append(
                .unknownCommandPin,
                "Command pin '\(key)' must be declared before this run call.",
                at: expression.sourceRange.start)
            return
        }
    }

    private mutating func collectAndValidateGraphs(in block: WorkflowBlock) {
        for statement in block.statements {
            switch statement.kind {
            case .declaration(let name, _, let value):
                if case .call(let call) = value.kind, call.target == .parallel {
                    let identity = statement.sourceRange.start.byteOffset
                    let graph = GraphDescriptor(
                        bindingName: name,
                        identity: identity,
                        location: statement.sourceRange.start)
                    validateGraph(call)
                    graphDescriptors.append(graph)
                }
            case .expression:
                break
            case .conditional(_, let thenBlock, let elseBlock):
                collectAndValidateGraphs(in: thenBlock)
                if let elseBlock { collectAndValidateGraphs(in: elseBlock) }
            case .forOf(_, _, _, let body):
                collectAndValidateGraphs(in: body)
            }
        }
    }

    private mutating func validateGraph(_ call: WorkflowCall) {
        guard let argument = call.arguments.first,
              case .array(let expressions) = argument.kind
        else { return }

        if expressions.count > 100 {
            append(
                .graphNodeLimitExceeded,
                "A dependency graph may contain at most 100 nodes.",
                at: argument.sourceRange.start)
        }

        var nodes: [GraphNode] = []
        var firstIDLocations: [String: WorkflowSourceLocation] = [:]
        for expression in expressions {
            guard case .object = expression.kind,
                  let idExpression = objectMember("id", in: expression),
                  let actorExpression = objectMember("actor", in: expression),
                  let dependenciesExpression = objectMember("dependsOn", in: expression),
                  case .literal(.string(let id)) = idExpression.kind,
                  case .literal(.string) = actorExpression.kind,
                  case .array(let dependencyExpressions) = dependenciesExpression.kind
            else { continue }

            if firstIDLocations[id] != nil {
                append(
                    .duplicateGraphNodeID,
                    "Dependency graph node ID '\(id)' is declared more than once.",
                    at: idExpression.sourceRange.start)
            } else {
                firstIDLocations[id] = idExpression.sourceRange.start
            }
            validateActorReference(actorExpression)

            let dependencies = dependencyExpressions.compactMap { dependency -> GraphDependency? in
                guard case .literal(.string(let name)) = dependency.kind else { return nil }
                return GraphDependency(name: name, location: dependency.sourceRange.start)
            }
            nodes.append(GraphNode(
                id: id,
                dependencies: dependencies))
        }

        let knownIDs = Set(nodes.map(\.id))
        for node in nodes {
            for dependency in node.dependencies where !knownIDs.contains(dependency.name) {
                append(
                    .unknownGraphDependency,
                    "Node '\(node.id)' depends on unknown node '\(dependency.name)'.",
                    at: dependency.location)
            }
        }

        let cycleLocations = Self.graphCycleLocations(nodes)
        for location in cycleLocations {
            append(.graphCycle, "Dependency graph contains a cycle.", at: location)
        }
    }

    private static func graphCycleLocations(_ nodes: [GraphNode]) -> [WorkflowSourceLocation] {
        struct Frame {
            let nodeID: String
            var nextDependencyIndex: Int
        }

        var nodesByID: [String: GraphNode] = [:]
        for node in nodes where nodesByID[node.id] == nil {
            nodesByID[node.id] = node
        }

        var colors: [String: Int] = [:]
        var cycleLocations: [WorkflowSourceLocation] = []
        for node in nodes where colors[node.id] == nil {
            colors[node.id] = 1
            var stack = [Frame(nodeID: node.id, nextDependencyIndex: 0)]
            while !stack.isEmpty {
                let frameIndex = stack.count - 1
                let nodeID = stack[frameIndex].nodeID
                guard let current = nodesByID[nodeID],
                      stack[frameIndex].nextDependencyIndex < current.dependencies.count
                else {
                    colors[nodeID] = 2
                    stack.removeLast()
                    continue
                }

                let dependency = current.dependencies[stack[frameIndex].nextDependencyIndex]
                stack[frameIndex].nextDependencyIndex += 1
                guard nodesByID[dependency.name] != nil else { continue }
                if colors[dependency.name] == 1 {
                    cycleLocations.append(dependency.location)
                } else if colors[dependency.name] == nil {
                    colors[dependency.name] = 1
                    stack.append(Frame(nodeID: dependency.name, nextDependencyIndex: 0))
                }
            }
        }
        return cycleLocations
    }

    private mutating func analyzeJoinFlow(
        in block: WorkflowBlock,
        states: [JoinFlowState],
        isInsideLoop: Bool
    ) -> [JoinFlowState] {
        var currentStates = states
        for statement in block.statements {
            guard consumeFlowSteps(currentStates.count, at: statement.sourceRange.start) else { continue }
            switch statement.kind {
            case .declaration(let name, _, let value):
                if case .call(let call) = value.kind, call.target == .parallel {
                    let identity = statement.sourceRange.start.byteOffset
                    for index in currentStates.indices {
                        currentStates[index].graphBindings[name] = identity
                        currentStates[index].joinCounts[identity] = 0
                    }
                } else if case .awaited(let call) = value.kind, call.target == .join {
                    recordJoin(call, states: &currentStates, isInsideLoop: isInsideLoop)
                }
            case .expression:
                break
            case .conditional(_, let thenBlock, let elseBlock):
                let thenStates = analyzeJoinFlow(in: thenBlock, states: currentStates, isInsideLoop: isInsideLoop)
                let elseStates: [JoinFlowState]
                if let elseBlock {
                    elseStates = analyzeJoinFlow(in: elseBlock, states: currentStates, isInsideLoop: isInsideLoop)
                } else {
                    elseStates = currentStates
                }
                currentStates = deduplicated(thenStates + elseStates, at: statement.sourceRange.start)
            case .forOf(_, _, _, let body):
                let bodyStates = analyzeJoinFlow(in: body, states: currentStates, isInsideLoop: true)
                // The loop can execute zero times or repeat. Preserve the zero-iteration
                // path and let join-in-loop diagnostics reject any apparent one-time join.
                currentStates = deduplicated(currentStates + bodyStates, at: statement.sourceRange.start)
            }
        }
        return currentStates
    }

    private mutating func recordJoin(
        _ call: WorkflowCall,
        states: inout [JoinFlowState],
        isInsideLoop: Bool
    ) {
        guard let graphArgument = call.arguments.first,
              case .identifier(let name) = graphArgument.kind
        else {
            append(
                .unknownGraphReference,
                "join must reference a graph binding declared earlier in the workflow.",
                at: call.arguments.first?.sourceRange.start ?? call.sourceRange.start)
            return
        }

        var foundBinding = false
        var missedBinding = false
        for index in states.indices {
            guard let identity = states[index].graphBindings[name] else {
                missedBinding = true
                continue
            }
            foundBinding = true
            if isInsideLoop {
                append(
                    .graphJoinInsideLoop,
                    "A graph join inside a repeatable loop cannot be proven to run exactly once.",
                    at: call.sourceRange.start)
            }
            let count = states[index].joinCounts[identity, default: 0]
            if count >= 1 {
                append(
                    .graphJoinedMoreThanOnce,
                    "Graph '\(name)' is joined more than once on this path.",
                    at: call.sourceRange.start)
            }
            states[index].joinCounts[identity] = min(2, count + 1)
        }
        if !foundBinding || missedBinding {
            append(
                .unknownGraphReference,
                "join must reference a graph binding declared earlier on every path.",
                at: graphArgument.sourceRange.start)
        }
    }

    private mutating func deduplicated(
        _ states: [JoinFlowState],
        at location: WorkflowSourceLocation
    ) -> [JoinFlowState] {
        var unique: [JoinFlowState] = []
        for state in states where !unique.contains(state) {
            if unique.count >= 4_096 {
                reportFlowBudget(at: location)
                break
            }
            unique.append(state)
        }
        return unique
    }

    private mutating func consumeFlowSteps(_ requested: Int, at location: WorkflowSourceLocation) -> Bool {
        guard requested <= remainingFlowSteps else {
            reportFlowBudget(at: location)
            return false
        }
        remainingFlowSteps -= requested
        return true
    }

    private mutating func reportFlowBudget(at location: WorkflowSourceLocation) {
        guard !didReportFlowBudget else { return }
        didReportFlowBudget = true
        append(
            .staticAnalysisBudgetExceeded,
            "Graph path analysis exceeded its bounded work budget.",
            at: location)
    }

    private mutating func validatePublishedPayloads(
        in block: WorkflowBlock,
        constants: [String: WorkflowExpression] = [:]
    ) {
        var visibleConstants = constants
        for statement in block.statements {
            switch statement.kind {
            case .declaration(let name, _, let value):
                validatePublishedPayloads(in: value, constants: visibleConstants)
                visibleConstants[name] = value
            case .expression(let expression):
                validatePublishedPayloads(in: expression, constants: visibleConstants)
            case .conditional(let condition, let thenBlock, let elseBlock):
                validatePublishedPayloads(in: condition, constants: visibleConstants)
                validatePublishedPayloads(in: thenBlock, constants: visibleConstants)
                if let elseBlock { validatePublishedPayloads(in: elseBlock, constants: visibleConstants) }
            case .forOf(let name, _, let sequence, let body):
                validatePublishedPayloads(in: sequence, constants: visibleConstants)
                var loopConstants = visibleConstants
                loopConstants.removeValue(forKey: name)
                validatePublishedPayloads(in: body, constants: loopConstants)
            }
        }
    }

    private mutating func validatePublishedPayloads(
        in expression: WorkflowExpression,
        constants: [String: WorkflowExpression]
    ) {
        switch expression.kind {
        case .call(let call), .awaited(let call):
            if [.report, .artifact].contains(call.target), let payload = call.arguments.first {
                validatePublishedPayload(payload, constants: constants)
            }
            for argument in call.arguments { validatePublishedPayloads(in: argument, constants: constants) }
        case .member(let base, _), .unaryNot(let base):
            validatePublishedPayloads(in: base, constants: constants)
        case .array(let values):
            for value in values { validatePublishedPayloads(in: value, constants: constants) }
        case .object(let members):
            for member in members { validatePublishedPayloads(in: member.value, constants: constants) }
        case .binary(let left, _, let right):
            validatePublishedPayloads(in: left, constants: constants)
            validatePublishedPayloads(in: right, constants: constants)
        case .literal, .identifier:
            break
        }
    }

    private mutating func validatePublishedPayload(
        _ expression: WorkflowExpression,
        constants: [String: WorkflowExpression]
    ) {
        validateDuplicatePayloadFields(in: expression, constants: constants, activeBindings: [])
        guard let value = staticJSONValue(expression, constants: constants, activeBindings: []) else { return }
        do {
            let data = try JSONSerialization.data(withJSONObject: value, options: [.fragmentsAllowed, .sortedKeys])
            if data.count > 65_536 {
                append(
                    .unserializablePublishedPayload,
                    "The minimum serialized form of this published payload exceeds the 65,536-byte limit.",
                    at: expression.sourceRange.start)
            }
        } catch {
            append(
                .unserializablePublishedPayload,
                "Published payload cannot be serialized as JSON.",
                at: expression.sourceRange.start)
        }
    }

    private mutating func validateDuplicatePayloadFields(
        in expression: WorkflowExpression,
        constants: [String: WorkflowExpression],
        activeBindings: Set<String>,
        depth: Int = 0
    ) {
        guard consumePublishedPayloadStep(at: expression.sourceRange.start, depth: depth) else { return }
        switch expression.kind {
        case .object(let members):
            var names: Set<String> = []
            for member in members {
                if !names.insert(member.name).inserted {
                    append(
                        .unserializablePublishedPayload,
                        "Published payload repeats object field '\(member.name)'.",
                        at: member.nameRange.start)
                }
                validateDuplicatePayloadFields(in: member.value, constants: constants, activeBindings: activeBindings, depth: depth + 1)
            }
        case .array(let values):
            for value in values { validateDuplicatePayloadFields(in: value, constants: constants, activeBindings: activeBindings, depth: depth + 1) }
        case .member(let base, _), .unaryNot(let base):
            validateDuplicatePayloadFields(in: base, constants: constants, activeBindings: activeBindings, depth: depth + 1)
        case .binary(let left, _, let right):
            validateDuplicatePayloadFields(in: left, constants: constants, activeBindings: activeBindings, depth: depth + 1)
            validateDuplicatePayloadFields(in: right, constants: constants, activeBindings: activeBindings, depth: depth + 1)
        case .call(let call), .awaited(let call):
            for argument in call.arguments {
                validateDuplicatePayloadFields(in: argument, constants: constants, activeBindings: activeBindings, depth: depth + 1)
            }
        case .identifier(let name):
            guard !activeBindings.contains(name), let value = constants[name] else { return }
            validateDuplicatePayloadFields(
                in: value,
                constants: constants,
                activeBindings: activeBindings.union([name]),
                depth: depth + 1)
        case .literal:
            break
        }
    }

    private mutating func staticJSONValue(
        _ expression: WorkflowExpression,
        constants: [String: WorkflowExpression],
        activeBindings: Set<String>,
        depth: Int = 0
    ) -> Any? {
        guard consumePublishedPayloadStep(at: expression.sourceRange.start, depth: depth) else { return nil }
        switch expression.kind {
        case .literal(.null):
            return NSNull()
        case .literal(.boolean(let value)):
            return value
        case .literal(.number(let value)) where value.isFinite:
            return value
        case .literal(.number):
            return 0
        case .member, .unaryNot, .binary, .call, .awaited:
            // Zero is the shortest JSON representation for an unknown runtime
            // value, so this pass can prove a lower bound without evaluating it.
            return 0
        case .identifier(let name):
            guard !activeBindings.contains(name), let value = constants[name] else { return 0 }
            return staticJSONValue(value, constants: constants, activeBindings: activeBindings.union([name]), depth: depth + 1)
        case .literal(.string(let value)):
            return value
        case .array(let expressions):
            var values: [Any] = []
            for item in expressions {
                guard let value = staticJSONValue(item, constants: constants, activeBindings: activeBindings, depth: depth + 1) else { return nil }
                values.append(value)
            }
            return values
        case .object(let members):
            var values: [String: Any] = [:]
            for member in members {
                guard values[member.name] == nil,
                      let value = staticJSONValue(member.value, constants: constants, activeBindings: activeBindings, depth: depth + 1)
                else { return nil }
                values[member.name] = value
            }
            return values
        }
    }

    private mutating func consumePublishedPayloadStep(at location: WorkflowSourceLocation, depth: Int) -> Bool {
        // Aliases can expand exponentially or add depth beyond the bounded source AST.
        guard !didReportPublishedPayloadBudget,
              remainingPublishedPayloadSteps > 0,
              depth <= WorkflowScriptChecker.Limits.production.maxNesting else {
            if !didReportPublishedPayloadBudget {
                didReportPublishedPayloadBudget = true
                append(
                    .staticAnalysisBudgetExceeded,
                    "Published payload analysis exceeded its bounded work or nesting budget.",
                    at: location)
            }
            return false
        }
        remainingPublishedPayloadSteps -= 1
        return true
    }

    private mutating func validateArgumentReferences(in block: WorkflowBlock, declaredNames: Set<String>) {
        for statement in block.statements {
            switch statement.kind {
            case .declaration(_, _, let value):
                validateArgumentReferences(in: value, declaredNames: declaredNames)
            case .expression(let expression):
                validateArgumentReferences(in: expression, declaredNames: declaredNames)
            case .conditional(let condition, let thenBlock, let elseBlock):
                validateArgumentReferences(in: condition, declaredNames: declaredNames)
                validateArgumentReferences(in: thenBlock, declaredNames: declaredNames)
                if let elseBlock { validateArgumentReferences(in: elseBlock, declaredNames: declaredNames) }
            case .forOf(_, _, let sequence, let body):
                validateArgumentReferences(in: sequence, declaredNames: declaredNames)
                validateArgumentReferences(in: body, declaredNames: declaredNames)
            }
        }
    }

    private mutating func validateArgumentReferences(in expression: WorkflowExpression, declaredNames: Set<String>) {
        switch expression.kind {
        case .member(let base, let name):
            if case .identifier("args") = base.kind, !declaredNames.contains(name) {
                append(
                    .undeclaredWorkflowArgument,
                    "Saved workflow references undeclared argument '\(name)'.",
                    at: memberNameLocation(expression, name: name))
            }
            validateArgumentReferences(in: base, declaredNames: declaredNames)
        case .array(let values):
            for value in values { validateArgumentReferences(in: value, declaredNames: declaredNames) }
        case .object(let members):
            for member in members { validateArgumentReferences(in: member.value, declaredNames: declaredNames) }
        case .unaryNot(let value):
            validateArgumentReferences(in: value, declaredNames: declaredNames)
        case .binary(let left, _, let right):
            validateArgumentReferences(in: left, declaredNames: declaredNames)
            validateArgumentReferences(in: right, declaredNames: declaredNames)
        case .call(let call), .awaited(let call):
            for argument in call.arguments { validateArgumentReferences(in: argument, declaredNames: declaredNames) }
        case .literal, .identifier:
            break
        }
    }

    // MARK: - Binding scopes

    /// Mirrors the interpreter's block-scoped environment so errors it would
    /// raise AFTER approval (and after earlier asks spent model time) are
    /// reported by the checker instead: a use before declaration or typo
    /// (`Unknown workflow identifier`), a `const` repeated in one scope
    /// (including a for-of binding redeclared in its own body), and `args`
    /// as a binding name. A declaration is visible only after its value has
    /// been evaluated, exactly as in the interpreter.
    private mutating func validateBindingScopes(_ block: WorkflowBlock) {
        var scopes: [Set<String>] = [[]]
        validateBindingScopes(block, scopes: &scopes)
    }

    private mutating func validateBindingScopes(_ block: WorkflowBlock, scopes: inout [Set<String>]) {
        for statement in block.statements {
            switch statement.kind {
            case .declaration(let name, let nameRange, let value):
                validateIdentifiers(in: value, scopes: scopes)
                if name == "args" {
                    append(
                        .reservedBindingName,
                        "The frozen args facade cannot be shadowed by a const binding.",
                        at: nameRange.start)
                } else if scopes[scopes.count - 1].contains(name) {
                    append(
                        .duplicateBinding,
                        "Workflow const binding '\(name)' is already declared in this scope.",
                        at: nameRange.start)
                }
                scopes[scopes.count - 1].insert(name)
            case .expression(let expression):
                validateIdentifiers(in: expression, scopes: scopes)
            case .conditional(let condition, let thenBlock, let elseBlock):
                validateIdentifiers(in: condition, scopes: scopes)
                scopes.append([])
                validateBindingScopes(thenBlock, scopes: &scopes)
                scopes.removeLast()
                if let elseBlock {
                    scopes.append([])
                    validateBindingScopes(elseBlock, scopes: &scopes)
                    scopes.removeLast()
                }
            case .forOf(let name, let nameRange, let sequence, let body):
                validateIdentifiers(in: sequence, scopes: scopes)
                if name == "args" {
                    append(
                        .reservedBindingName,
                        "A loop binding cannot shadow the frozen args facade.",
                        at: nameRange.start)
                }
                // The interpreter runs the body in the same environment that
                // holds the loop variable, so redeclaring it there collides.
                scopes.append([name])
                validateBindingScopes(body, scopes: &scopes)
                scopes.removeLast()
            }
        }
    }

    private mutating func validateIdentifiers(in expression: WorkflowExpression, scopes: [Set<String>]) {
        switch expression.kind {
        case .identifier(let name):
            if name != "args", !scopes.contains(where: { $0.contains(name) }) {
                append(
                    .undeclaredIdentifier,
                    "Unknown workflow identifier '\(name)'.",
                    at: expression.sourceRange.start)
            }
        case .member(let base, _):
            validateIdentifiers(in: base, scopes: scopes)
        case .array(let values):
            for value in values { validateIdentifiers(in: value, scopes: scopes) }
        case .object(let members):
            for member in members { validateIdentifiers(in: member.value, scopes: scopes) }
        case .unaryNot(let value):
            validateIdentifiers(in: value, scopes: scopes)
        case .binary(let left, _, let right):
            validateIdentifiers(in: left, scopes: scopes)
            validateIdentifiers(in: right, scopes: scopes)
        case .call(let call), .awaited(let call):
            for argument in call.arguments { validateIdentifiers(in: argument, scopes: scopes) }
        case .literal:
            break
        }
    }

    /// The interpreter rejects a command whose argv declares one dynamic slot
    /// name twice; catch it before approval.
    private mutating func validateCommandSlots(_ call: WorkflowCall) {
        guard call.arguments.count > 1,
              let argv = objectMember("argv", in: call.arguments[1]),
              case .array(let entries) = argv.kind
        else { return }
        var seen = Set<String>()
        for entry in entries {
            guard let nameExpression = objectMember("name", in: entry),
                  case .literal(.string(let slot)) = nameExpression.kind
            else { continue }
            if !seen.insert(slot).inserted {
                append(
                    .duplicateCommandSlot,
                    "Command argument slot '\(slot)' is declared more than once.",
                    at: nameExpression.sourceRange.start)
            }
        }
    }

    private func memberNameLocation(_ expression: WorkflowExpression, name: String) -> WorkflowSourceLocation {
        let byteCount = name.utf8.count
        let scalarCount = name.unicodeScalars.count
        return WorkflowSourceLocation(
            byteOffset: max(expression.sourceRange.start.byteOffset, expression.sourceRange.end.byteOffset - byteCount),
            line: expression.sourceRange.end.line,
            column: max(1, expression.sourceRange.end.column - scalarCount))
    }

    private func objectMember(_ name: String, in expression: WorkflowExpression) -> WorkflowExpression? {
        guard case .object(let members) = expression.kind else { return nil }
        return members.first(where: { $0.name == name })?.value
    }

    private mutating func append(
        _ rule: WorkflowScriptDiagnosticRule,
        _ message: String,
        at location: WorkflowSourceLocation
    ) {
        guard !diagnostics.contains(where: { $0.rule == rule && $0.location == location }) else { return }
        diagnostics.append(WorkflowScriptDiagnostic(rule: rule, message: message, location: location))
    }
}
