#!/usr/bin/env python3
"""Derive the design system's color and shadow tokens from the shipped theme.

TurboSparkApp stores three colors per mode (background, foreground, accent)
plus a contrast value, and derives every surface from them with
NSColor.blended in gamma-encoded sRGB (TurboSparkTheme.swift,
AppThemeColor.swift). This script repeats that math so tokens.json and
tokens.css match what the app draws, and computes the contrast figures the
usage notes quote.

Only the `color` and `shadow` families are generated. Type, spacing, radius
and layout in tokens.json are maintained by hand from the Swift sources listed
in tokens.json `meta.paths`.

Usage, from the repository root:
    python3 docs/design-system/tools/derive_tokens.py           # rewrite tokens.json and tokens.css
    python3 docs/design-system/tools/derive_tokens.py --check   # exit 1 if stale or drifted
    python3 docs/design-system/tools/derive_tokens.py --table   # print the README contrast table
"""
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SYSTEM = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(SYSTEM))
TOKENS = os.path.join(SYSTEM, 'tokens.json')
CSS = os.path.join(SYSTEM, 'tokens.css')
SWIFT_PRESETS = os.path.join(REPO, 'swift/TurboSparkApp/Sources/TurboSparkApp/Theme/AppearanceTypes.swift')
FONT_DIR = '../../swift/TurboSparkApp/Sources/TurboSparkAppResources/Resources/Fonts/'

# id: (name, background, foreground, accent, contrast, isDark, isHighContrast)
# Copied from ThemePreset.presets in AppearanceTypes.swift; --check fails if a
# hex here no longer appears there. The first theme is the CSS default.
THEMES = {
    'dark': ('Spark Blue dark', '#090D16', '#F2F5FB', '#3B9EFF', 68, True, False),
    'light': ('Spark Blue light', '#FFFFFF', '#141A24', '#1D6FE0', 52, False, False),
    'hc-dark': ('High contrast dark', '#000000', '#FFFFFF', '#FFFFFF', 95, True, True),
    'hc-light': ('High contrast light', '#FFFFFF', '#000000', '#000000', 95, False, True),
    'codex-dark': ('Codex dark', '#181818', '#FFFFFF', '#F3F4F6', 60, True, False),
    'codex-light': ('Codex light', '#FFFFFF', '#1A1C1F', '#111827', 45, False, False),
    'emerald-dark': ('Emerald dark', '#151816', '#F0FDF4', '#6ABA71', 65, True, False),
    'emerald-light': ('Emerald light', '#FFFFFF', '#1A1C1F', '#237D32', 50, False, False),
}

# Status colors are an intentional addition: the app uses system
# .green/.orange/.red today. Chosen to hold 4.5:1 on every surface and on
# their own 14% tint (7:1 under high contrast); asserted below.
_DARK = {'success': '#3ddba8', 'warning': '#ffb340', 'danger': '#ff6961'}
_LIGHT = {'success': '#00704f', 'warning': '#8a5a00', 'danger': '#c00013'}
STATUS = {
    'dark': _DARK, 'light': _LIGHT,
    'hc-dark': {'success': '#5ff0c0', 'warning': '#ffc966', 'danger': '#ff9a94'},
    'hc-light': {'success': '#004d3a', 'warning': '#5c3b00', 'danger': '#8f000c'},
    'codex-dark': _DARK, 'codex-light': _LIGHT, 'emerald-dark': _DARK, 'emerald-light': _LIGHT,
}

# Accent as text where the theme's own accent misses 4.5:1 on some surface.
ACCENT_INK = {'light': '#1a64cc'}

WHITE, BLACK = (1.0, 1.0, 1.0), (0.0, 0.0, 0.0)
NEAR_BLACK = '#141417'  # Color.contrastForeground: (0.08, 0.08, 0.09)


def hx(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def tohex(c):
    return '#' + ''.join('%02x' % round(max(0.0, min(1.0, v)) * 255) for v in c)


def blend(a, b, f):
    # NSColor.blended(withFraction:of:): a straight mix of encoded components.
    return tuple(x * (1 - f) + y * f for x, y in zip(a, b))


def over(fg, alpha, bg):
    return blend(bg, fg, alpha)


def lum(c):
    lin = [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in c]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def cr(a, b):
    la, lb = lum(a), lum(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


def floor1(x):
    # Claims read "X or better", so never round up.
    return '%.1f' % (math.floor(x * 10) / 10)


def rgba(c, a):
    r, g, b = (round(v * 255) for v in c)
    return 'rgba(%d, %d, %d, %s)' % (r, g, b, ('%.3f' % a).rstrip('0').rstrip('.'))


def derive():
    out = {}
    for tid, (name, bg, fg, ac, contrast, dark, hc) in THEMES.items():
        B, F, A = hx(bg), hx(fg), hx(ac)
        # Chrome steps toward white (dark) or black (light): TurboSparkTheme.elevate.
        # Content surfaces blend toward the foreground: ResolvedAppTheme.surface.
        mix = WHITE if dark else BLACK
        grounds = {
            'page': B,
            'rail': blend(B, mix, .03 if dark else .02),
            'sidebar': blend(B, mix, .05 if dark else .035),
            'bar': blend(B, mix, .055 if dark else .03),
            'composer': blend(B, mix, .09 if dark else .05),
            'surface': blend(B, F, .055 if dark else .025),
            'elevated': blend(B, F, .09 if dark else .05),
        }
        perceived = .2126 * A[0] + .7152 * A[1] + .0722 * A[2]  # Color.isLight, non-linear on purpose
        t = dict(name=name, B=B, F=F, A=A, dark=dark, hc=hc, g=grounds,
                 sec_a=.96 if hc else (.9 if contrast > 70 else .75),
                 border_a=.75 if hc else .12 + contrast / 500,
                 hair=blend(B, mix, .30 if dark else .18), hair_a=.4 + contrast / 100 * .5,
                 on_accent=hx(NEAR_BLACK) if perceived > .55 else WHITE)
        t['worst'] = lambda col, g=grounds: min(cr(col, y) for y in g.values())
        t['worst_a'] = lambda col, a, g=grounds: min(cr(over(col, a, y), y) for y in g.values())
        t['accent_ink'] = ACCENT_INK.get(tid)
        ink_col = hx(t['accent_ink']) if t['accent_ink'] else A
        assert t['worst'](ink_col) >= 4.5, (tid, 'accent-ink below 4.5:1: add it to ACCENT_INK')
        if not hc:
            t['bs_a'] = next(x / 100 for x in range(20, 100) if t['worst_a'](F, x / 100) >= 3)
        floor = 7 if hc else 4.5
        for key, value in STATUS[tid].items():
            c = hx(value)
            assert t['worst'](c) >= floor and cr(c, over(c, .14, B)) >= floor, (tid, key)
        out[tid] = t
    return out


def color_tokens(T):
    ids = list(THEMES)
    presets = [k for k in ids if not T[k]['hc']]
    darks = [k for k in presets if T[k]['dark']]
    lights = [k for k in presets if not T[k]['dark']]

    def per(fn):
        return {k: fn(T[k]) for k in ids}

    def tok(name, value, usage):
        return {'name': name, 'value': value, 'usage': usage}

    ink_min = min(T[k]['worst'](T[k]['F']) for k in ids)
    sec_d = min(T[k]['worst_a'](T[k]['F'], T[k]['sec_a']) for k in darks)
    sec_l = min(T[k]['worst_a'](T[k]['F'], T[k]['sec_a']) for k in lights)
    bord = [T[k]['worst_a'](T[k]['F'], T[k]['border_a']) for k in presets]
    edge = [T[k]['worst_a'](T[k]['F'], .18 if T[k]['dark'] else .14) for k in presets]
    ring = min(T[k]['worst'](T[k]['A']) for k in ids)
    acc_ink = min(T[k]['worst'](hx(T[k]['accent_ink']) if T[k]['accent_ink'] else T[k]['A']) for k in ids)
    on_acc = min(cr(T[k]['on_accent'], T[k]['A']) for k in ids)
    fe_dark = T['dark']['worst_a'](T['dark']['A'], .55)
    fe_light = T['light']['worst_a'](T['light']['A'], .55)
    status_min = min(min(T[k]['worst'](hx(v)), cr(hx(v), over(hx(v), .14, T[k]['B'])))
                     for k in presets for v in STATUS[k].values())

    colors = [
        tok('page', per(lambda t: tohex(t['B'])),
            'Window canvas and the transcript. The theme background itself (`.appPage`, `ResolvedAppTheme.background`). Spark Blue dark is #090D16, a blue-cast night sky, never neutral gray.'),
        tok('rail', per(lambda t: tohex(t['g']['rail'])),
            'Navigation rail, the 52px icon column at the leading edge. Page stepped 3% toward white (dark) or 2% toward black (light). Drawn at 90% opacity while the sidebar is translucent, solid under Reduce Transparency.'),
        tok('sidebar', per(lambda t: tohex(t['g']['sidebar'])),
            'Sidebar panel beside the rail. Page stepped 5% (dark) or 3.5% (light). Drawn at 94% opacity while translucent, solid under Reduce Transparency.'),
        tok('bar', per(lambda t: tohex(t['g']['bar'])),
            'Top bar and bottom status strip. One step off the page so a band reads as a strip, not a continuation of the canvas.'),
        tok('surface', per(lambda t: tohex(t['g']['surface'])),
            'Cards, search fields, editors, and grouped panels (`.appSurface`). Page blended 5.5% (dark) or 2.5% (light) toward ink.'),
        tok('elevated', per(lambda t: tohex(t['g']['elevated'])),
            'Popovers, menus, and anything raised above a card (`.appElevated`). Page blended 9% (dark) or 5% (light) toward ink.'),
        tok('composer', per(lambda t: tohex(t['g']['composer'])),
            'The prompt composer fill (`ResolvedAppTheme.composerBackground`), and the Humanize and script editor composers. The most raised surface in the window.'),
        tok('night', {k: '#090d16' for k in ids},
            'Intentional addition. The Spark Blue night sky, constant in every theme. Brand ground for the cover, marketing art, and the dock icon only. Never a UI surface (use `page`).'),
        tok('ink', per(lambda t: tohex(t['F'])),
            'Primary text and icons on every surface above (`.appText`). %s:1 or better on all of them in every theme.' % floor1(ink_min)),
        tok('ink-secondary', per(lambda t: rgba(t['F'], t['sec_a'])),
            'Secondary rows, subtitles, metadata, placeholder text (`.appSecondary`). Ink at 75%%, 90%% above contrast 70, 96%% under Increase Contrast. Lowest pair: %s:1 in the dark themes, %s:1 in the light ones, on composer.' % (floor1(sec_d), floor1(sec_l))),
        tok('ink-disabled', per(lambda t: rgba(t['F'], .45)),
            'Disabled labels and chips only (AppChip uses 0.45). Exempt from contrast minimums, so never use it for text a person must read. A disabled control says why in its tooltip.'),
        tok('border', per(lambda t: rgba(t['F'], t['border_a'])),
            'Card, field, and editor strokes (`.appBorder`), 1px. Ink at 0.12 + contrast/500. Decorative separation only: %s:1 to %s:1 across the preset themes, below the 3:1 control-boundary floor. Rises to 75%% under Increase Contrast.' % (floor1(min(bord)), floor1(max(bord)))),
        tok('border-strong', per(lambda t: '{border}' if t['hc'] else rgba(t['F'], t['bs_a'])),
            'Intentional addition. The edge of a control that is identified by its edge alone (an empty field on a matching fill, a checkbox). Ink at 35% to 48% depending on the theme: 3:1 on every surface. Equal to `border` under Increase Contrast.'),
        tok('hairline', per(lambda t: rgba(t['hair'], t['hair_a'])),
            'The 0.5px seams between chrome bands (rail, sidebar, top bar, status strip). A lighter line of the same palette, never a system separator color. Decorative.'),
        tok('accent', per(lambda t: tohex(t['A'])),
            'The one accent: primary action fill, selected rail icon, progress fill, focus edge, live state. Spark Blue #3B9EFF dark, #1D6FE0 light. Codex is near-black and near-white, Emerald is green. As text, Spark Blue light drops to %s:1 on composer, so set text in `accent-ink`.' % floor1(T['light']['worst'](T['light']['A']))),
        tok('accent-ink', per(lambda t: t['accent_ink'] or '{accent}'),
            'Intentional addition. Accent used as text or a small glyph (links, selected rail symbol, "Ready"). %s:1 or better on every surface in every theme. Differs from `accent` only in Spark Blue light.' % floor1(acc_ink)),
        tok('on-accent', per(lambda t: tohex(t['on_accent'])),
            'Label and icon on an accent fill, from `Color.contrastForeground`: near-black #141417 on a bright accent, white on a deep one. %s:1 or better in every theme. White on the Spark Blue dark accent is %s:1: check native .borderedProminent labels against this.' % (floor1(on_acc), floor1(cr(WHITE, T['dark']['A'])))),
        tok('accent-tint', per(lambda t: rgba(t['A'], .14)),
            'Fill behind an on chip and a selected rail item (accent at 14%). Text on it stays `ink`.'),
        tok('accent-edge', per(lambda t: rgba(t['A'], .3)),
            'Hairline stroke around an on chip (accent at 30%, 0.5px). Decorative. The tint and `.isSelected` carry the state.'),
        tok('selection', per(lambda t: rgba(t['A'], (.45 if t['dark'] else .30) if t['hc'] else (.25 if t['dark'] else .12))),
            'Selected text and selected list rows (`ResolvedAppTheme.selection`). Accent at 25% dark, 12% light, 45% and 30% under Increase Contrast.'),
        tok('focus-ring', per(lambda t: '{accent}'),
            'Keyboard focus: a solid 2px accent ring with a 2px gap, 3px under Increase Contrast. %s:1 or better on every surface in every theme. The soft halo (`glow-focus`) is extra, never the only cue.' % floor1(ring)),
        tok('focus-edge', per(lambda t: '{accent}' if t['hc'] else rgba(t['A'], .55)),
            'The composer and editor focus edge as shipped (accent at 55%%, 1.5px, solid 2.5px under Increase Contrast). Spark Blue: %s:1 dark and %s:1 light on composer, under the 3:1 focus floor. Pair it with `focus-ring` for keyboard focus, or raise it.' % (floor1(fe_dark), floor1(fe_light))),
        tok('composer-edge', per(lambda t: '{border}' if t['hc'] else rgba(t['F'], .18 if t['dark'] else .14)),
            'Resting 1px stroke of the composer and focus-glow editors (ink at 18%% dark, 14%% light, `border` under Increase Contrast). Decorative: %s:1 to %s:1.' % (floor1(min(edge)), floor1(max(edge)))),
        tok('hover', per(lambda t: rgba(t['F'], .07)),
            'Hover fill on rail items, sidebar rows, and quiet buttons (ink at 7%). Hover never changes layout.'),
        tok('chip-fill', per(lambda t: rgba(t['F'], .04)), 'Resting fill of an off chip (ink at 4%).'),
        tok('chip-hover', per(lambda t: rgba(t['F'], .08)), 'Hovered off chip (ink at 8%).'),
        tok('track', per(lambda t: rgba(t['F'], .10)),
            'Progress bar track (ink at 10%). Gains a `border` stroke under Increase Contrast.'),
    ]
    labels = [
        ('success', 'Success: done, ready, allowed, resident. Pair with checkmark.circle.fill and a word.'),
        ('warning', 'Warning: needs attention, degraded, partial. Pair with exclamationmark.triangle.fill and a word.'),
        ('danger', 'Danger and destructive: failed, denied, delete. Pair with xmark.octagon.fill and a word. Destructive buttons also carry `role: .destructive`.'),
    ]
    for key, label in labels:
        note = ('Intentional addition, replacing system .green/.orange/.red on themed surfaces. ' + label +
                ' %s:1 or better on every surface and on its own tint (7:1 in the high-contrast themes).' % floor1(status_min))
        if key == 'success':
            note += ' Bluish green, off the red-green axis, so it stays distinct from `danger` with color-vision deficiency.'
        colors.append(tok(key, {k: STATUS[k][key] for k in ids}, note))
    for key in ('success', 'warning', 'danger'):
        colors.append(tok(key + '-tint', {k: rgba(hx(STATUS[k][key]), .14) for k in ids},
                          'Background of a %s status pill or banner (%s at 14%%). Text on it is `%s`, not ink.' % (key, key, key)))
    return colors


def shadow_tokens(T):
    ids = list(THEMES)

    def drop(blur):
        return {k: '0 2px %dpx rgba(0, 0, 0, %s)' % (blur, '0.28' if T[k]['dark'] else '0.08') for k in ids}

    def glow(spec, alpha):
        return {k: 'none' if T[k]['hc'] else spec + rgba(T[k]['A'], alpha) for k in ids}

    return [
        {'name': 'shadow-composer', 'value': drop(12),
         'usage': 'Resting composer lift (SwiftUI radius 6, y 2). The only drop shadow in the system. Cards separate by fill and border.'},
        {'name': 'shadow-composer-focus', 'value': drop(20),
         'usage': 'Composer lift while it holds focus (SwiftUI radius 10).'},
        {'name': 'glow-focus', 'value': glow('0 0 20px ', .35),
         'usage': 'Accent halo outside a focused editor with no native ring (`.appFocusGlow`), cast by an edge behind the surface. None under Increase Contrast, where the ring goes solid.'},
        {'name': 'glow-primary', 'value': glow('0 2px 16px ', .45),
         'usage': 'Halo under the one primary action while it can run (`.appPrimaryGlow`): the composer Send button. Never on a disabled or secondary action.'},
    ]


def build_tokens(current, T):
    tokens = json.loads(json.dumps(current))
    tokens['color']['themes'] = [{'id': k, 'name': THEMES[k][0]} for k in THEMES]
    tokens['color']['tokens'] = color_tokens(T)
    tokens['shadow']['tokens'] = shadow_tokens(T)
    return tokens


def compile_css(tokens):
    ids = [t['id'] for t in tokens['color']['themes']]

    def css_value(v):
        m = re.fullmatch(r'\{(.+)\}', v)
        return 'var(--%s)' % m.group(1) if m else v

    blocks = []
    for i, tid in enumerate(ids):
        lines = []
        for fam in ('color', 'shadow'):
            for tok in tokens[fam]['tokens']:
                v = tok['value'] if isinstance(tok['value'], str) else tok['value'].get(tid, tok['value'][ids[0]])
                lines.append('  --%s: %s;' % (tok['name'], css_value(v)))
        sel = ':root, [data-theme="%s"]' % tid if i == 0 else '[data-theme="%s"]' % tid
        blocks.append('%s {\n%s\n}' % (sel, '\n'.join(lines)))
    root = ['  --%s: %s;' % (t['name'], t['value']) for fam in ('spacing', 'radius', 'layout') for t in tokens[fam]['tokens']]
    root += ['  --font-%s: %s;' % (k, v) for k, v in tokens['type']['families'].items()]
    blocks.append(':root {\n%s\n}' % '\n'.join(root))
    for f in tokens['type']['fonts']:
        blocks.append("@font-face { font-family: '%s'; src: url('%s%s') format('truetype'); font-weight: %s; font-style: %s; }"
                      % (f['family'], FONT_DIR, os.path.basename(f['file']), f['weight'], f.get('style', 'normal')))
    head = '/* TurboSpark design system tokens. Generated by tools/derive_tokens.py from tokens.json. Do not edit. */\n'
    return head + '\n'.join(blocks) + '\n'


def table(T):
    rows = ['| Theme | `ink` | `ink-secondary` | `accent-ink` | `on-accent` | status (worst) | `focus-ring` | `border-strong` |',
            '| --- | --- | --- | --- | --- | --- | --- | --- |']
    order = ['dark', 'light', 'codex-dark', 'codex-light', 'emerald-dark', 'emerald-light', 'hc-dark', 'hc-light']
    for k in order:
        t = T[k]
        ink_acc = hx(t['accent_ink']) if t['accent_ink'] else t['A']
        status = min(min(t['worst'](hx(v)), cr(hx(v), over(hx(v), .14, t['B']))) for v in STATUS[k].values())
        bs = t['worst_a'](t['F'], .75 if t['hc'] else t['bs_a'])
        rows.append('| %s | %s | %s | %s | %s | %s | %s | %s |' % (
            t['name'], floor1(t['worst'](t['F'])), floor1(t['worst_a'](t['F'], t['sec_a'])), floor1(t['worst'](ink_acc)),
            floor1(cr(t['on_accent'], t['A'])), floor1(status), floor1(t['worst'](t['A'])), floor1(bs)))
    return '\n'.join(rows)


def dumps(tokens):
    return json.dumps(tokens, indent=2) + '\n'


def main(argv):
    T = derive()
    current = json.load(open(TOKENS))
    tokens = build_tokens(current, T)
    if '--table' in argv:
        print(table(T))
        return 0
    if '--check' in argv:
        problems = []
        swift = open(SWIFT_PRESETS).read().upper()
        for tid, (_, bg, fg, ac, _c, _d, _h) in THEMES.items():
            for h in (bg, fg, ac):
                if '"%s"' % h.upper() not in swift:
                    problems.append('%s: %s not found in %s' % (tid, h, os.path.relpath(SWIFT_PRESETS, REPO)))
        if dumps(tokens) != open(TOKENS).read():
            problems.append('tokens.json is stale: run tools/derive_tokens.py')
        if not os.path.exists(CSS) or compile_css(tokens) != open(CSS).read():
            problems.append('tokens.css is stale: run tools/derive_tokens.py')
        for p in problems:
            print(p)
        print('ok' if not problems else '%d problem(s)' % len(problems))
        return 1 if problems else 0
    open(TOKENS, 'w').write(dumps(tokens))
    open(CSS, 'w').write(compile_css(tokens))
    print('wrote %s and %s' % (os.path.relpath(TOKENS, REPO), os.path.relpath(CSS, REPO)))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
