---
name: visualize
description: Output contract for agent-written interactive HTML (charts, tables, small dashboards) so the page renders offline in TurboSpark's artifact panel; one self-contained file, no CDN, inline SVG or canvas charts.
when_to_use: Use when the user asks to visualize, chart, plot or present data as an interactive page. Do not use it for plain Markdown reports or for pages that must load remote resources.
---

# Visualize

This file is a contract for the page you write. It guides you and enforces
nothing: the artifact panel is what enforces the offline sandbox. The limits
below that the panel does not check (size, structure) are advisory, and you
are responsible for them.

## How the page is shown

When you write a `.html` or `.htm` file with your file write tool, the app
registers it and opens it in the artifact panel. The panel renders it in a
sandboxed web view that is offline by default: http, https, ws and wss loads
are blocked, no cookies or storage persist, and a file page can read only its
own folder. The first blocked remote load makes the panel show "Network is
off for this preview" with an Allow button. Allow is the user's decision for
that one preview. Never design a page that needs it, and never tell the user
to press it. A page that depends on a blocked resource renders blank or
half-drawn, which is the failure this contract prevents.

The panel has a Source view, and "Save a copy" writes the file wherever the
user chooses. Tell the user the file path you wrote, so they can also open it
in a browser.

## The contract

1. One file. Write a complete document (`<!doctype html>`, `<html>`, `<head>`
   with `<meta charset="utf-8">` and a `<title>`, `<body>`), not a fragment.
   Put CSS in one `<style>` and JavaScript in one `<script>`.
2. No external loads, ever, by default: no CDN scripts or stylesheets, web
   fonts, remote images, `fetch`, `XMLHttpRequest`, `WebSocket`, `<iframe src>`
   to a remote page, `@import`, or `<link href>` to a URL. Use the system font
   stack (`system-ui, sans-serif`). Embed images as inline SVG or `data:` URIs.
3. No charting library. Draw with inline SVG (preferred: crisp, scalable,
   accessible) or `<canvas>` when there are thousands of points. Write the
   few dozen lines of scale and axis code yourself.
4. Keep it under 1 MB (advisory). Embed the data as a JSON literal in the
   script. If the data is larger, aggregate or sample it and say so on the
   page instead of embedding all of it.
5. Do not depend on `localStorage`, `sessionStorage`, cookies, `alert` or
   popups: storage does not persist and may be unavailable.
6. Light and dark: define colors as CSS variables on `:root`, override them in
   `@media (prefers-color-scheme: dark)`, and set an explicit page background.
7. Responsive: no horizontal page scroll; set a viewport `<meta>`; let SVG
   charts scale with `viewBox` and `width: 100%`.
8. Accessible: every chart gets a text title, axis labels with units, a legend
   or direct labels, `role="img"` with an `aria-label` summary, and a data
   table or tooltip so no value is shown by color alone.
9. Be honest about the data: start bar axes at zero, label units, show the
   source file or command in a caption, and show the sample size.

## Chart forms

- Comparison across categories: horizontal bars, sorted by value, value
  labels at the bar end. Many categories: keep the top N and group the rest.
- Trend over time: a line with markers only at the data points; one line per
  series, at most 5; label lines directly at their right end.
- Relationship between two numbers: scatter plot; add hover to show the row.
  Use canvas past about 5,000 points.
- Distribution: histogram with a stated bin width; for a few groups, a strip
  or box summary.
- Part of a whole: a single stacked bar or a sorted bar. Avoid pies beyond
  three slices.
- Many numbers, exact lookup: a sortable table (click a header to sort) with
  right-aligned numeric columns.
- Several related series: small multiples (same axes repeated) rather than
  one crowded chart.

Interaction that fits: hover tooltips, click-to-sort, a series toggle in the
legend, a range filter input. Keep state in JavaScript variables and redraw.

## Procedure

1. Read the real data with your tools. Do not invent numbers; if a value is
   missing, show the gap.
2. Choose the chart form for the question, not for decoration.
3. Write the file next to the data (or where the user says), then state its
   path and the one-line takeaway.
4. If the user reports a blank or broken panel, first look for any external
   reference you wrote (`http`, `//cdn`, `@import`, `fetch(`) and remove it,
   then for a script error in your own code.
