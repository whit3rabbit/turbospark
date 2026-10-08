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

# The file under mutation, its backup, and the checksum of the mutated text.
# A signal or an early exit must never leave a security gate removed in the
# source, and a concurrent edit by another session must never be overwritten
# by a stale backup: restore only when the file still holds exactly what this
# script wrote.
CURRENT_FILE=""
CURRENT_BACKUP=""
MUTATED_SUM=""

sum_of() { shasum -a 256 < "$1" | cut -d' ' -f1; }

restore() {
  [ -n "$CURRENT_BACKUP" ] || return 0
  if [ -n "$MUTATED_SUM" ] && [ "$(sum_of "$CURRENT_FILE")" != "$MUTATED_SUM" ]; then
    echo "ABORT: $CURRENT_FILE changed during the mutation run; NOT restoring." >&2
    echo "       Original kept at $CURRENT_BACKUP. Reconcile by hand." >&2
    CURRENT_BACKUP=""
    return 1
  fi
  cp "$CURRENT_BACKUP" "$CURRENT_FILE" && rm -f "$CURRENT_BACKUP"
  CURRENT_BACKUP=""
  MUTATED_SUM=""
}
trap 'restore; exit 130' INT TERM
trap 'restore' EXIT

mutate() {
  local file="$1" expect_fail="$2" desc="$3" old="$4" new="$5"
  # Refuse to touch a file with uncommitted work: the mutation and restore
  # would race whoever is editing it.
  if [ -n "$(git status --porcelain -- "$file" 2>/dev/null)" ]; then
    echo "SKIPPED [$desc]: $file has uncommitted changes" >&2
    return 1
  fi
  CURRENT_FILE="$file"
  CURRENT_BACKUP="$(mktemp -t agentmode-mutation)"
  MUTATED_SUM=""
  cp "$file" "$CURRENT_BACKUP"
  python3 - "$file" "$old" "$new" <<'PYEOF'
import sys
path, old, new = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(path).read()
assert s.count(old) == 1, f"pattern not unique/absent in {path}: {old[:60]!r} count={s.count(old)}"
open(path, "w").write(s.replace(old, new))
PYEOF
  if [ $? -ne 0 ]; then echo "MUTATION DID NOT APPLY: $desc"; restore; return 1; fi
  MUTATED_SUM="$(sum_of "$file")"
  local out
  out=$(run_filter)
  restore || return 1
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
'            return .fastAllow
        }
        return .classify' \
'            return .fastAllow
        }
        return .fastAllow'

mutate "$M" testOnlyReadOnlyCategoriesTakeTheFastPath "fast path widened to every category" \
'        if call.category == .fileRead, assessment.category == .fileRead,
            !assessment.isHighRisk
        {' \
'        if !assessment.isHighRisk {'

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
