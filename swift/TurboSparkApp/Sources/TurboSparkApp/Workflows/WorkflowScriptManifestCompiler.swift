import CryptoKit
import Foundation

struct WorkflowScriptManifestBuildResult: Equatable, Sendable {
    let manifest: WorkflowLaunchManifest?
    let diagnostics: [WorkflowScriptDiagnostic]
}

/// Compiles source-visible declarations into the approval manifest without
/// evaluating any workflow expression.
enum WorkflowScriptManifestCompiler {
    static func build(_ ast: WorkflowScriptAST) -> WorkflowScriptManifestBuildResult {
        var compiler = WorkflowScriptManifestBuilder()
        return compiler.build(ast)
    }
}

private struct WorkflowScriptManifestBuilder {
    private struct SiteCandidate {
        let lane: String
        let sourceOffset: Int
        let insertionOrder: Int
    }

    private(set) var diagnostics: [WorkflowScriptDiagnostic] = []
    private var phases: [String] = []
    private var phaseNames: Set<String> = []
    private var actors: [WorkflowActorSpec] = []
    private var commandPins: [WorkflowCommandPin] = []
    private var siteCandidates: [SiteCandidate] = []
    private var nextInsertionOrder = 0

    mutating func build(_ ast: WorkflowScriptAST) -> WorkflowScriptManifestBuildResult {
        collectEntryDeclarations(ast.body)
        collectSites(in: ast.body, constants: [:])

        guard diagnostics.isEmpty else {
            return WorkflowScriptManifestBuildResult(manifest: nil, diagnostics: diagnostics)
        }

        let siteTable = makeSiteTable()
        return WorkflowScriptManifestBuildResult(
            manifest: WorkflowLaunchManifest(
                phases: phases,
                actors: actors,
                siteTable: siteTable,
                commandPins: commandPins),
            diagnostics: [])
    }

    private mutating func collectEntryDeclarations(_ block: WorkflowBlock) {
        for statement in block.statements {
            guard case .expression(let expression) = statement.kind,
                  case .call(let call) = expression.kind
            else { continue }

            switch call.target {
            case .agent:
                if let name = stringLiteral(at: 0, in: call),
                   let rolePrompt = stringLiteral(at: 1, in: call) {
                    actors.append(WorkflowActorSpec(name: name, rolePrompt: rolePrompt))
                }
            case .phase:
                if let name = stringLiteral(at: 0, in: call), phaseNames.insert(name).inserted {
                    phases.append(name)
                }
            case .command:
                if let pin = makeCommandPin(call) {
                    commandPins.append(pin)
                }
            default:
                break
            }
        }
    }

    private mutating func makeCommandPin(_ call: WorkflowCall) -> WorkflowCommandPin? {
        guard call.arguments.count == 2,
              let commandKey = stringLiteral(at: 0, in: call),
              case .object(let members) = call.arguments[1].kind
        else {
            append(
                .literalStructureRequired,
                "Command pins must have a statically declared key and definition.",
                at: call.sourceRange.start)
            return nil
        }

        let fields = memberMap(members)
        guard let executableMember = fields["executable"],
              case .literal(.string(let executablePath)) = executableMember.value.kind,
              let workingDirectory = fields["workingDirectory"].flatMap({ stringLiteral($0.value) }),
              let argvExpression = fields["argv"]?.value,
              case .array(let argvExpressions) = argvExpression.kind
        else {
            append(
                .literalStructureRequired,
                "Command pin fields must resolve to fixed strings and an argv array.",
                at: call.arguments[1].sourceRange.start)
            return nil
        }

        guard let executable = pinExecutable(
            executablePath,
            location: executableMember.value.sourceRange.start)
        else { return nil }

        var argvTemplate: [WorkflowCommandArgSlot] = []
        for expression in argvExpressions {
            if case .literal(.string(let fixedValue)) = expression.kind {
                argvTemplate.append(.fixed(fixedValue))
            } else if let rule = commandArgumentRule(expression) {
                argvTemplate.append(.dynamic(rule))
            } else {
                append(
                    .literalOnlyPosition,
                    "Command argv items must be fixed strings or declared dynamic argument slots.",
                    at: expression.sourceRange.start)
                return nil
            }
        }

        return WorkflowCommandPin(
            commandKey: commandKey,
            executable: executable,
            argvTemplate: argvTemplate,
            workingDirectory: workingDirectory)
    }

    private mutating func pinExecutable(
        _ path: String,
        location: WorkflowSourceLocation
    ) -> WorkflowExecutableIdentity? {
        guard path.hasPrefix("/") else {
            append(
                .executablePathMustBeAbsolute,
                "Command executable path must be an absolute literal path.",
                at: location)
            return nil
        }

        let canonicalURL = URL(fileURLWithPath: path).resolvingSymlinksInPath().standardizedFileURL
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: canonicalURL.path, isDirectory: &isDirectory),
              !isDirectory.boolValue,
              FileManager.default.isExecutableFile(atPath: canonicalURL.path)
        else {
            append(
                .executableIdentityUnavailable,
                "Command executable must resolve to an existing executable file.",
                at: location)
            return nil
        }

        do {
            let resourceValues = try canonicalURL.resourceValues(forKeys: [.isRegularFileKey])
            guard resourceValues.isRegularFile == true else {
                append(
                    .executableIdentityUnavailable,
                    "Command executable must resolve to an existing executable file.",
                    at: location)
                return nil
            }
            return WorkflowExecutableIdentity(
                canonicalPath: canonicalURL.path,
                sha256: try sha256(fileAt: canonicalURL))
        } catch {
            append(
                .executableIdentityUnavailable,
                "Command executable could not be read to capture its SHA-256 identity.",
                at: location)
            return nil
        }
    }

    private func sha256(fileAt url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }

        var hasher = SHA256()
        while let chunk = try handle.read(upToCount: 64 * 1024), !chunk.isEmpty {
            hasher.update(data: chunk)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    private func commandArgumentRule(_ expression: WorkflowExpression) -> WorkflowCommandArgumentRule? {
        guard case .object(let members) = expression.kind else { return nil }
        let fields = memberMap(members)
        guard let name = fields["name"].flatMap({ stringLiteral($0.value) }),
              let kindName = fields["kind"].flatMap({ stringLiteral($0.value) })
        else { return nil }

        switch kindName {
        case "workspaceInputPath":
            return WorkflowCommandArgumentRule(
                name: name,
                kind: .workspaceInputPath,
                allowedValues: nil,
                maximumBytes: 4_096)
        case "allowedValue":
            guard let valuesExpression = fields["values"]?.value,
                  case .array(let valueExpressions) = valuesExpression.kind
            else { return nil }
            let values = valueExpressions.compactMap(stringLiteral)
            guard values.count == valueExpressions.count else { return nil }
            return WorkflowCommandArgumentRule(
                name: name,
                kind: .allowedValue,
                allowedValues: values,
                maximumBytes: values.map { $0.utf8.count }.max() ?? 0)
        case "boundedText":
            guard let maximumBytes = fields["maximumBytes"].flatMap({ integerLiteral($0.value) }) else {
                return nil
            }
            return WorkflowCommandArgumentRule(
                name: name,
                kind: .boundedText,
                allowedValues: nil,
                maximumBytes: maximumBytes)
        default:
            return nil
        }
    }

    private mutating func collectSites(
        in block: WorkflowBlock,
        constants initialConstants: [String: WorkflowExpression]
    ) {
        var constants = initialConstants
        for statement in block.statements {
            switch statement.kind {
            case .declaration(let name, _, let value):
                collectSites(in: value, constants: constants)
                constants[name] = value
            case .expression(let expression):
                collectSites(in: expression, constants: constants)
            case .conditional(let condition, let thenBlock, let elseBlock):
                collectSites(in: condition, constants: constants)
                collectSites(in: thenBlock, constants: constants)
                if let elseBlock {
                    collectSites(in: elseBlock, constants: constants)
                }
            case .forOf(let name, _, let sequence, let body):
                collectSites(in: sequence, constants: constants)
                var loopConstants = constants
                loopConstants.removeValue(forKey: name)
                collectSites(in: body, constants: loopConstants)
            }
        }
    }

    private mutating func collectSites(
        in expression: WorkflowExpression,
        constants: [String: WorkflowExpression]
    ) {
        switch expression.kind {
        case .call(let call), .awaited(let call):
            collectSites(for: call, constants: constants)
            for argument in call.arguments {
                collectSites(in: argument, constants: constants)
            }
        case .member(let base, _), .unaryNot(let base):
            collectSites(in: base, constants: constants)
        case .array(let values):
            for value in values { collectSites(in: value, constants: constants) }
        case .object(let members):
            for member in members { collectSites(in: member.value, constants: constants) }
        case .binary(let left, _, let right):
            collectSites(in: left, constants: constants)
            collectSites(in: right, constants: constants)
        case .literal, .identifier:
            break
        }
    }

    private mutating func collectSites(for call: WorkflowCall, constants: [String: WorkflowExpression]) {
        switch call.target {
        case .ask:
            if let actor = stringLiteral(at: 0, in: call) {
                addSite(lane: actor, at: call.sourceRange.start)
            }
        case .parallel:
            addSite(lane: "main", at: call.sourceRange.start)
            guard let nodesExpression = call.arguments.first,
                  let resolvedNodes = resolve(nodesExpression, constants: constants),
                  case .array(let nodes) = resolvedNodes.kind
            else { return }
            for node in nodes {
                guard case .object(let members) = node.kind,
                      let actorExpression = memberMap(members)["actor"]?.value,
                      let actor = stringLiteral(actorExpression)
                else { continue }
                addSite(lane: actor, at: actorExpression.sourceRange.start)
            }
        case .criticLoop:
            addSite(lane: "main", at: call.sourceRange.start)
            guard let policyExpression = call.arguments.first,
                  let policy = resolve(policyExpression, constants: constants),
                  case .object(let policyMembers) = policy.kind
            else { return }
            let policyFields = memberMap(policyMembers)
            for laneName in ["producer", "critic"] {
                guard let laneExpression = policyFields[laneName]?.value,
                      let lane = resolve(laneExpression, constants: constants),
                      case .object(let laneMembers) = lane.kind,
                      let actorExpression = memberMap(laneMembers)["actor"]?.value,
                      let actor = stringLiteral(actorExpression)
                else { continue }
                addSite(lane: actor, at: actorExpression.sourceRange.start)
            }
        case .join, .worldRead, .run, .report, .artifact:
            addSite(lane: "main", at: call.sourceRange.start)
        case .agent, .phase, .command, .glob, .read, .grep, .git:
            break
        }
    }

    private func makeSiteTable() -> [WorkflowStaticSite] {
        var nextIndexByLane: [String: Int] = [:]
        return siteCandidates
            .sorted {
                if $0.sourceOffset != $1.sourceOffset {
                    return $0.sourceOffset < $1.sourceOffset
                }
                return $0.insertionOrder < $1.insertionOrder
            }
            .map { candidate in
                let index = nextIndexByLane[candidate.lane, default: 0]
                nextIndexByLane[candidate.lane] = index + 1
                return WorkflowStaticSite(lane: candidate.lane, siteIndex: index)
            }
    }

    private mutating func addSite(lane: String, at location: WorkflowSourceLocation) {
        siteCandidates.append(SiteCandidate(
            lane: lane,
            sourceOffset: location.byteOffset,
            insertionOrder: nextInsertionOrder))
        nextInsertionOrder += 1
    }

    private func resolve(
        _ expression: WorkflowExpression,
        constants: [String: WorkflowExpression]
    ) -> WorkflowExpression? {
        var current = expression
        var visited: Set<String> = []
        while case .identifier(let name) = current.kind {
            guard visited.insert(name).inserted, let value = constants[name] else { return nil }
            current = value
        }
        return current
    }

    private func memberMap(_ members: [WorkflowObjectMember]) -> [String: WorkflowObjectMember] {
        Dictionary(members.map { ($0.name, $0) }, uniquingKeysWith: { _, latest in latest })
    }

    private func stringLiteral(at index: Int, in call: WorkflowCall) -> String? {
        guard call.arguments.indices.contains(index) else { return nil }
        return stringLiteral(call.arguments[index])
    }

    private func stringLiteral(_ expression: WorkflowExpression) -> String? {
        guard case .literal(.string(let value)) = expression.kind else { return nil }
        return value
    }

    private func integerLiteral(_ expression: WorkflowExpression) -> Int? {
        guard case .literal(.number(let value)) = expression.kind,
              value.isFinite,
              value.rounded(.towardZero) == value,
              value >= Double(Int.min),
              value <= Double(Int.max)
        else { return nil }
        return Int(value)
    }

    private mutating func append(
        _ rule: WorkflowScriptDiagnosticRule,
        _ message: String,
        at location: WorkflowSourceLocation
    ) {
        diagnostics.append(WorkflowScriptDiagnostic(rule: rule, message: message, location: location))
    }
}
