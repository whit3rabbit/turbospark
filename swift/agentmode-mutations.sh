#!/usr/bin/env bash
# Mutation checks for Agent mode tests (house rule: every new test is
# mutation-checked before it is believed). Each mutation asserts it APPLIED
# exactly once, runs only the AgentMode filter, and restores the file.
# Expected: the named case reddens; everything else stays green.
set -u
cd "$(dirname "$0")/TurboSparkApp"

run_filter() {
  swift test --filter "AgentModeTests" 2>&1 | grep -E "Test Case.*(failed|passed)|error:" | grep -v "was modified"
}

mutate() {
  local file="$1" expect_fail="$2" desc="$3" old="$4" new="$5"
  cp "$file" /tmp/agentmode-mutation-backup.swift
  python3 - "$file" "$old" "$new" <<'EOF'
import sys
path, old, new = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(path).read()
assert s.count(old) == 1, f"pattern not unique/absent in {path}: {old[:60]!r} count={s.count(old)}"
open(path, "w").write(s.replace(old, new))
EOF
  if [ $? -ne 0 ]; then echo "MUTATION DID NOT APPLY: $desc"; cp /tmp/agentmode-mutation-backup.swift "$file"; return 1; fi
  local out
  out=$(run_filter)
  cp /tmp/agentmode-mutation-backup.swift "$file"
  if echo "$out" | grep -q "failed"; then
    local failed_cases
    failed_cases=$(echo "$out" | grep "Test Case.*failed" | sed "s/.*Test Case 'AgentModeTests\/\(.*\)' failed.*/\1/" | sort -u | tr '\n' ' ')
    echo "OK  [$desc] reddened: $failed_cases"
    if [ -n "$expect_fail" ] && ! echo "$failed_cases" | grep -q "$expect_fail"; then
      echo "WARN: expected '$expect_fail' among them"
    fi
  else
    echo "SURVIVOR [$desc] - mutation applied but nothing reddened"
  fi
}

M=Sources/TurboSparkApp/State/AppModel+AgentMode.swift
C=Sources/TurboSparkApp/Tools/Core/AgentMode/LocalModelToolClassifier.swift
G=Sources/TurboSparkApp/Tools/Core/AgentMode/AgentModeGate.swift
P=Sources/TurboSparkApp/Tools/Core/AgentMode/ToolCallClassifier.swift
R=Sources/TurboSparkApp/Tools/Core/ToolRiskClassifier.swift
E=Sources/TurboSparkApp/Tools/Core/AppToolPermissionEngine.swift

mutate "$M" testAHardGatedAskNeverClassifies "hard-gate check removed" \
'        if assessment.isHardGated {
            return .manualCard
        }
' ''

mutate "$M" testAHighRiskUnrecognizedCommandClassifies "terminal classify flipped to fast-allow" \
'        if !assessment.isHighRisk {
            return .fastAllow
        }
        return .classify' \
'        if !assessment.isHighRisk {
            return .fastAllow
        }
        return .fastAllow'

mutate "$M" testMCPCallsAlwaysClassify "MCP-always-classify removed" \
'        if call.category == .mcp {
            return .classify
        }
' ''

mutate "$C" testAnUnknownVerdictIsNotAVerdict "parse fail-open" \
'        default:
            return nil' \
'        default:
            return .allow'

mutate "$G" testAnAllowVerdictBreaksBothStreaks "recordAllow reset removed" \
'    public func recordAllow(sessionID: String) {
        consecutiveBlocks[sessionID] = 0
        consecutiveUnavailable[sessionID] = 0
    }' \
'    public func recordAllow(sessionID: String) {
    }'

mutate "$G" testThreeConsecutiveBlocksSkipTheClassifier "block threshold 3 to 2" \
'    public static let maxConsecutiveBlocks = 3' \
'    public static let maxConsecutiveBlocks = 2'

mutate "$R" testADenylistCommandIsHardGated "denylist hard mark dropped" \
'            return ToolRiskAssessment(level: .high, category: .terminal, reasons: reasons, hardGated: true)' \
'            return ToolRiskAssessment(level: .high, category: .terminal, reasons: reasons)'

mutate "$E" testRepoImportedServerAsksUnderAgentModeToo "7b agentAuto arm dropped" \
'            if (permissions.mode == .auto || permissions.mode == .agentAuto),' \
'            if permissions.mode == .auto,'

mutate "$P" testAWebProjectionCarriesTheURLAndNotThePrompt "web prompt field leaked into the line" \
'            return "web request: \(target)"' \
'            return "web request: \(target) \(arguments["prompt"] ?? "")"'

echo "mutation checks done"
