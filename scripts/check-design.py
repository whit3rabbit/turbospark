#!/usr/bin/env python3
"""Pre-CI design checks for the macOS app, without building it.

CI does not run the app's `swift test` suite, so the design rules in
`DesignTokenRatchetTests` only fail for someone who runs it by hand. This runs
the same scan in about a second:

  1. The ratchet rules in swift/TurboSparkApp/Tests/TurboSparkAppTests/
     DesignRules.json (the file the Swift test loads). A count above its
     baseline is a new violation. A count below is slack: lower the baseline.
  2. The status hexes in Theme/AppStatusColors.swift match the ones the design
     system publishes (docs/design-system/tools/derive_tokens.py).
  3. `derive_tokens.py --check`: tokens.json and tokens.css match the shipped
     presets.

Usage, from the repository root:
    python3 scripts/check-design.py            # all three, exit 1 on a failure
    python3 scripts/check-design.py --sites    # also list every matching site

Contrast ratios need the app's real surface blending, so they stay in
AppStatusColorTests (`make design-check-full` runs them).
"""
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, 'swift/TurboSparkApp')
SOURCES = os.path.join(APP, 'Sources/TurboSparkApp')
RULES = os.path.join(APP, 'Tests/TurboSparkAppTests/DesignRules.json')
STATUS_SWIFT = os.path.join(SOURCES, 'Theme/AppStatusColors.swift')
DERIVE = os.path.join(ROOT, 'docs/design-system/tools/derive_tokens.py')


def scan_lines():
    for base, _, files in os.walk(SOURCES):
        rel_dir = os.path.relpath(base, SOURCES)
        # Theme/ implements the tokens and is where the literals live.
        if rel_dir == 'Theme' or rel_dir.startswith('Theme' + os.sep):
            continue
        for name in sorted(files):
            if not name.endswith('.swift'):
                continue
            path = os.path.join(base, name)
            rel = os.path.relpath(path, SOURCES)
            with open(path, encoding='utf-8') as handle:
                for number, line in enumerate(handle.read().split('\n'), 1):
                    if not line.strip().startswith('//'):
                        yield rel, number, line


def check_ratchet(show_sites):
    with open(RULES, encoding='utf-8') as handle:
        rules = json.load(handle)['rules']
    lines = list(scan_lines())
    if not lines:
        return ['no Swift sources found under %s' % SOURCES]
    failures = []
    for rule in rules:
        regex = re.compile(rule['pattern'])
        count, sites = 0, []
        for rel, number, text in lines:
            found = len(regex.findall(text))
            if found:
                count += found
                sites.append('%s:%d' % (rel, number))
        status = 'ok' if count == rule['baseline'] else 'FAIL'
        print('  %-4s %-40s %3d (baseline %d)' % (status, rule['name'], count, rule['baseline']))
        if show_sites:
            for site in sites:
                print('         ' + site)
        if count > rule['baseline']:
            failures.append('%s: %d sites, baseline %d. New sites must %s (DESIGN.md section 4). Sites: %s'
                            % (rule['name'], count, rule['baseline'], rule['fix'], ', '.join(sites)))
        elif count < rule['baseline']:
            failures.append('%s: %d sites, baseline %d. Lower the baseline in DesignRules.json to %d so the '
                            'slack cannot hide a new violation.' % (rule['name'], count, rule['baseline'], count))
    return failures


def check_status_hexes():
    with open(STATUS_SWIFT, encoding='utf-8') as handle:
        swift = handle.read()
    found = re.findall(
        r'case \((true|false), (true|false)\): return AppStatusPalette\(success: "(#\w{6})", warning: "(#\w{6})", danger: "(#\w{6})"\)',
        swift)
    if len(found) != 4:
        return ['could not read the four AppStatusPalette cases from %s' % os.path.relpath(STATUS_SWIFT, ROOT)]
    names = {('true', 'false'): 'dark', ('false', 'false'): 'light', ('true', 'true'): 'hc-dark', ('false', 'true'): 'hc-light'}
    published = {}
    with open(DERIVE, encoding='utf-8') as handle:
        source = handle.read()
    for tid in ('dark', 'light', 'hc-dark', 'hc-light'):
        match = re.search(r"'%s': \{'success': '(#\w{6})', 'warning': '(#\w{6})', 'danger': '(#\w{6})'\}" % tid, source)
        if match:
            published[tid] = [value.lower() for value in match.groups()]
    # `dark` and `light` are written through _DARK and _LIGHT.
    for tid, var in (('dark', '_DARK'), ('light', '_LIGHT')):
        match = re.search(r"%s = \{'success': '(#\w{6})', 'warning': '(#\w{6})', 'danger': '(#\w{6})'\}" % var, source)
        if match:
            published[tid] = [value.lower() for value in match.groups()]
    failures = []
    for dark, high, success, warning, danger in found:
        tid = names[(dark, high)]
        if published.get(tid) != [success.lower(), warning.lower(), danger.lower()]:
            failures.append('status hexes for %s: Swift has %s, derive_tokens.py has %s'
                            % (tid, [success, warning, danger], published.get(tid)))
    print('  %-4s %s' % ('FAIL' if failures else 'ok', 'status hexes match the design system'))
    return failures


def check_tokens():
    result = subprocess.run([sys.executable, DERIVE, '--check'], cwd=ROOT, capture_output=True, text=True)
    ok = result.returncode == 0
    print('  %-4s %s' % ('ok' if ok else 'FAIL', 'design-system tokens match the presets'))
    return [] if ok else ['derive_tokens.py --check failed: ' + (result.stdout + result.stderr).strip()]


def main(argv):
    failures = []
    print('design ratchet')
    failures += check_ratchet('--sites' in argv)
    print('design system')
    failures += check_status_hexes()
    failures += check_tokens()
    if failures:
        print('\nDesign check failed:')
        for failure in failures:
            print(' - ' + failure)
        return 1
    print('\nDesign check passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
