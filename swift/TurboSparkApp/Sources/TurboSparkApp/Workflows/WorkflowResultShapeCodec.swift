import CoreFoundation
import Foundation
import TurboSpark

enum WorkflowResultShapeCodec {
  enum Failure: Error, Equatable {
    case invalid(String)

    var message: String {
      switch self {
      case .invalid(let message): message
      }
    }
  }

  static func decode(_ value: WorkflowCanonicalValue) throws -> WorkflowResultShape {
    switch value {
    case .array:
      return WorkflowResultShape(fields: try fields(from: value))
    case .object(let object):
      if case .string("object")? = object["type"], let members = object["fields"] {
        return WorkflowResultShape(fields: try fields(from: members))
      }
      return WorkflowResultShape(fields: try fields(from: value))
    default:
      throw Failure.invalid("The declared result shape must describe an object.")
    }
  }

  static func validate(
    _ value: WorkflowCanonicalValue,
    against shape: WorkflowResultShape,
    maximumIssues: Int = 16
  ) -> [String] {
    guard case .object(let values) = value else {
      return ["The answer must be a JSON object."]
    }

    var issues: [String] = []
    validateFields(shape.fields, in: values, path: "$", issues: &issues, maximumIssues: maximumIssues)
    return issues
  }

  static func canonicalValue(for shape: WorkflowResultShape) -> WorkflowCanonicalValue {
    .array(shape.fields.map { field in
      .object([
        "name": .string(field.name),
        "required": .boolean(field.required),
        "value": canonicalValue(for: field.value),
      ])
    })
  }

  static func parseAnswer(_ text: String) -> WorkflowCanonicalValue? {
    guard let data = text.data(using: .utf8),
          let value = try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed])
    else { return nil }
    return canonicalValue(fromJSON: value)
  }

  static func jsonText(_ value: WorkflowCanonicalValue) -> String? {
    guard JSONSerialization.isValidJSONObject(jsonValue(fromCanonical: value)) else {
      if case .array = value {} else if case .object = value {} else {
        guard let data = try? JSONSerialization.data(
          withJSONObject: jsonValue(fromCanonical: value),
          options: [.fragmentsAllowed, .sortedKeys])
        else { return nil }
        return String(data: data, encoding: .utf8)
      }
      return nil
    }
    guard let data = try? JSONSerialization.data(
      withJSONObject: jsonValue(fromCanonical: value),
      options: [.fragmentsAllowed, .sortedKeys])
    else { return nil }
    return String(data: data, encoding: .utf8)
  }

  static func canonicalValue(fromJSON value: Any) -> WorkflowCanonicalValue? {
    if value is NSNull { return .null }
    if let string = value as? String { return .string(string) }
    if let number = value as? NSNumber {
      if CFGetTypeID(number) == CFBooleanGetTypeID() {
        return .boolean(number.boolValue)
      }
      let double = number.doubleValue
      guard double.isFinite else { return nil }
      if double.rounded(.towardZero) == double, let integer = Int64(number.stringValue) {
        return .integer(integer)
      }
      return .number(double)
    }
    if let values = value as? [Any] {
      var result: [WorkflowCanonicalValue] = []
      result.reserveCapacity(values.count)
      for item in values {
        guard let canonical = canonicalValue(fromJSON: item) else { return nil }
        result.append(canonical)
      }
      return .array(result)
    }
    if let values = value as? [String: Any] {
      var result: [String: WorkflowCanonicalValue] = [:]
      result.reserveCapacity(values.count)
      for (key, item) in values {
        guard let canonical = canonicalValue(fromJSON: item) else { return nil }
        result[key] = canonical
      }
      return .object(result)
    }
    return nil
  }

  static func jsonValue(fromCanonical value: WorkflowCanonicalValue) -> Any {
    switch value {
    case .null:
      NSNull()
    case .boolean(let value):
      NSNumber(value: value)
    case .integer(let value):
      NSNumber(value: value)
    case .number(let value):
      NSNumber(value: value)
    case .string(let value):
      value
    case .array(let values):
      values.map(jsonValue(fromCanonical:))
    case .object(let values):
      values.mapValues(jsonValue(fromCanonical:))
    }
  }

  private static func fields(from value: WorkflowCanonicalValue) throws -> [WorkflowShapeField] {
    switch value {
    case .array(let values):
      let decoded = try values.map { item in
        guard case .object(let fields) = item,
              case .string(let name)? = fields["name"],
              !name.isEmpty,
              case .boolean(let required)? = fields["required"],
              let value = fields["value"]
        else { throw Failure.invalid("Each declared result field needs a name, required flag, and value shape.") }
        guard fields.keys.allSatisfy(["name", "required", "value"].contains) else {
          throw Failure.invalid("Declared result fields contain an unsupported key.")
        }
        return WorkflowShapeField(name: name, required: required, value: try shapeValue(from: value))
      }
      guard Set(decoded.map(\.name)).count == decoded.count else {
        throw Failure.invalid("Declared result field names must be unique.")
      }
      return decoded
    case .object(let values):
      return try values.keys.sorted().map { name in
        guard !name.isEmpty else { throw Failure.invalid("Declared result field names cannot be empty.") }
        guard let value = values[name] else { throw Failure.invalid("A declared result field is missing its shape.") }
        let required: Bool
        let valueShape: WorkflowCanonicalValue
        if case .object(let descriptor) = value, case .boolean(let isRequired)? = descriptor["required"] {
          required = isRequired
          valueShape = .object(descriptor.filter { $0.key != "required" })
        } else {
          required = true
          valueShape = value
        }
        return WorkflowShapeField(name: name, required: required, value: try shapeValue(from: valueShape))
      }
    default:
      throw Failure.invalid("Declared object fields must be an object or field array.")
    }
  }

  private static func shapeValue(from value: WorkflowCanonicalValue) throws -> WorkflowShapeValue {
    switch value {
    case .string("string"):
      return .string(enumValues: nil)
    case .string("integer"):
      return .integer
    case .string("number"):
      return .number
    case .string("boolean"):
      return .boolean
    case .string("null"):
      return .null
    case .object(let descriptor):
      guard case .string(let type)? = descriptor["type"] else {
        return .object(fields: try fields(from: value))
      }
      switch type {
      case "string":
        guard descriptor.keys.allSatisfy(["type", "enum"].contains) else {
          throw Failure.invalid("String shapes accept only type and enum.")
        }
        switch descriptor["enum"] {
        case nil, .null?:
          return .string(enumValues: nil)
        case .array(let values)?:
          var enums: [String] = []
          for value in values {
            guard case .string(let string) = value else {
              throw Failure.invalid("String enum values must all be strings.")
            }
            enums.append(string)
          }
          return .string(enumValues: enums)
        default:
          throw Failure.invalid("String enum must be a static array of strings.")
        }
      case "integer":
        try requireKeys(descriptor, allowed: ["type"])
        return .integer
      case "number":
        try requireKeys(descriptor, allowed: ["type"])
        return .number
      case "boolean":
        try requireKeys(descriptor, allowed: ["type"])
        return .boolean
      case "null":
        try requireKeys(descriptor, allowed: ["type"])
        return .null
      case "array":
        try requireKeys(descriptor, allowed: ["type", "item"])
        guard let item = descriptor["item"] else {
          throw Failure.invalid("Array shapes require an item shape.")
        }
        return .array(item: try shapeValue(from: item))
      case "object":
        try requireKeys(descriptor, allowed: ["type", "fields"])
        guard let declaredFields = descriptor["fields"] else {
          throw Failure.invalid("Object shapes require fields.")
        }
        return .object(fields: try fields(from: declaredFields))
      default:
        throw Failure.invalid("Unsupported result shape type '\(type)'.")
      }
    default:
      throw Failure.invalid("Unsupported result shape value.")
    }
  }

  private static func requireKeys(_ fields: [String: WorkflowCanonicalValue], allowed: [String]) throws {
    guard fields.keys.allSatisfy(allowed.contains) else {
      throw Failure.invalid("Result shape contains an unsupported field.")
    }
  }

  private static func canonicalValue(for shape: WorkflowShapeValue) -> WorkflowCanonicalValue {
    switch shape {
    case .string(let enumValues):
      return .object([
        "enum": enumValues.map { .array($0.map(WorkflowCanonicalValue.string)) } ?? .null,
        "type": .string("string"),
      ])
    case .integer:
      return .string("integer")
    case .number:
      return .string("number")
    case .boolean:
      return .string("boolean")
    case .null:
      return .string("null")
    case .array(let item):
      return .object(["item": canonicalValue(for: item), "type": .string("array")])
    case .object(let fields):
      return .object([
        "fields": canonicalValue(for: WorkflowResultShape(fields: fields)),
        "type": .string("object"),
      ])
    }
  }

  private static func validateFields(
    _ fields: [WorkflowShapeField],
    in values: [String: WorkflowCanonicalValue],
    path: String,
    issues: inout [String],
    maximumIssues: Int
  ) {
    guard issues.count < maximumIssues else { return }
    let declared = Set(fields.map(\.name))
    for extra in values.keys.sorted() where !declared.contains(extra) {
      issues.append("\(path) contains undeclared field '\(extra)'.")
      if issues.count >= maximumIssues { return }
    }
    for field in fields {
      let fieldPath = path == "$" ? field.name : "\(path).\(field.name)"
      guard let value = values[field.name] else {
        if field.required { issues.append("Required field '\(fieldPath)' is missing.") }
        if issues.count >= maximumIssues { return }
        continue
      }
      validate(value, against: field.value, path: fieldPath, issues: &issues, maximumIssues: maximumIssues)
      if issues.count >= maximumIssues { return }
    }
  }

  private static func validate(
    _ value: WorkflowCanonicalValue,
    against shape: WorkflowShapeValue,
    path: String,
    issues: inout [String],
    maximumIssues: Int
  ) {
    guard issues.count < maximumIssues else { return }
    switch (value, shape) {
    case (.string(let actual), .string(let enumValues)):
      if let enumValues, !enumValues.contains(actual) {
        issues.append("\(path) must be one of: \(enumValues.joined(separator: ", ")).")
      }
    case (.integer, .integer), (.integer, .number), (.number, .number):
      break
    case (.number(let number), .integer) where number.isFinite && number.rounded(.towardZero) == number:
      break
    case (.boolean, .boolean), (.null, .null):
      break
    case (.array(let values), .array(let itemShape)):
      for (index, item) in values.enumerated() {
        validate(item, against: itemShape, path: "\(path)[\(index)]", issues: &issues, maximumIssues: maximumIssues)
        if issues.count >= maximumIssues { return }
      }
    case (.object(let values), .object(let fields)):
      validateFields(fields, in: values, path: path, issues: &issues, maximumIssues: maximumIssues)
    default:
      issues.append("\(path) has the wrong JSON type.")
    }
  }
}

enum WorkflowActorTranscriptCodec {
  static func encodedSnapshotData(_ snapshot: WorkflowActorTranscriptSnapshot) throws -> Data {
    try WorkflowCanonicalSerialization.encodedData(.object([
      "payload": snapshot.payload,
      "version": .integer(Int64(snapshot.version)),
    ]))
  }

  static func decode(_ snapshot: WorkflowActorTranscriptSnapshot) throws -> WorkflowActorTranscript {
    guard snapshot.version == WorkflowActorTranscript.currentVersion,
          case .array = snapshot.payload,
          let data = try? JSONSerialization.data(
            withJSONObject: WorkflowResultShapeCodec.jsonValue(fromCanonical: snapshot.payload),
            options: [.fragmentsAllowed, .sortedKeys])
    else { throw Failure.invalid("The prior actor transcript snapshot is invalid or unsupported.") }
    do {
      let messages = try JSONDecoder().decode([ChatMessage].self, from: data)
      return WorkflowActorTranscript(version: snapshot.version, messages: messages)
    } catch {
      throw Failure.invalid("The prior actor transcript snapshot could not be decoded.")
    }
  }

  static func snapshot(from transcript: WorkflowActorTranscript) throws -> WorkflowActorTranscriptSnapshot {
    guard transcript.version == WorkflowActorTranscript.currentVersion else {
      throw Failure.invalid("The actor transcript version is unsupported.")
    }
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    do {
      let data = try encoder.encode(transcript.messages)
      guard let json = try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed]),
            let payload = WorkflowResultShapeCodec.canonicalValue(fromJSON: json),
            case .array = payload
      else { throw Failure.invalid("The actor transcript cannot be represented as canonical JSON.") }
      return WorkflowActorTranscriptSnapshot(payload: payload)
    } catch let failure as Failure {
      throw failure
    } catch {
      throw Failure.invalid("The actor transcript could not be encoded.")
    }
  }

  enum Failure: Error, Equatable {
    case invalid(String)

    var message: String {
      switch self {
      case .invalid(let message): message
      }
    }
  }
}
