#!/usr/bin/env python3
"""
generate_app_icons.py
Generates Apple macOS compliant app icons (all resolutions, iconset, and .icns)
from a source master image, with clean alpha transparency and proper padding.
"""

import os
import sys
import subprocess
from PIL import Image, ImageFilter
import numpy as np
from collections import deque

def extract_and_generate_icons(source_path: str, output_dirs: list[str]):
    print(f"==> Loading source image: {source_path}")
    img = Image.open(source_path).convert("RGB")
    arr = np.array(img).astype(float)
    h, w, _ = arr.shape
    
    # Estimate background color from outer border perimeter
    bg_sample = np.concatenate([arr[0:5, :], arr[-5:, :], arr[:, 0:5], arr[:, -5:]], axis=None).reshape(-1, 3)
    bg_color = np.median(bg_sample, axis=0)
    print(f"==> Estimated background color: {bg_color.round(1).tolist()}")

    # Measure color distance from background
    dist = np.max(np.abs(arr - bg_color), axis=2)

    # Flood fill from image edges to find the connected exterior region
    visited = np.zeros((h, w), dtype=bool)
    queue = deque()

    for y in range(h):
        for x in [0, w - 1]:
            if dist[y, x] < 5.0 and not visited[y, x]:
                visited[y, x] = True
                queue.append((y, x))
    for x in range(w):
        for y in [0, h - 1]:
            if dist[y, x] < 5.0 and not visited[y, x]:
                visited[y, x] = True
                queue.append((y, x))

    while queue:
        cy, cx = queue.popleft()
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                if dist[ny, nx] < 18.0:
                    visited[ny, nx] = True
                    queue.append((ny, nx))

    # Compute clean alpha channel & un-matte edges
    alpha = np.ones((h, w), dtype=float) * 255.0
    rgb = arr.copy()

    for y in range(h):
        for x in range(w):
            if visited[y, x]:
                if dist[y, x] < 3.5:
                    alpha[y, x] = 0.0
                    rgb[y, x] = [0, 0, 0]
                else:
                    t = (dist[y, x] - 3.5) / (18.0 - 3.5)
                    # Smoothstep curve
                    t_smooth = 3 * t**2 - 2 * t**3
                    alpha[y, x] = t_smooth * 255.0
                    a_norm = t_smooth
                    if a_norm > 0.01:
                        c_unmix = (arr[y, x] - (1.0 - a_norm) * bg_color) / a_norm
                        rgb[y, x] = np.clip(c_unmix, 0, 255)
                    else:
                        rgb[y, x] = [0, 0, 0]

    rgba = np.dstack([rgb, alpha]).astype(np.uint8)
    img_rgba = Image.fromarray(rgba, "RGBA")

    # Center icon bounding box around (512, 284)
    # The icon is 440x440
    crop_box = (512 - 220, 284 - 220, 512 + 220, 284 + 220)
    cropped_icon = img_rgba.crop(crop_box)

    # Standard macOS 1024x1024 master canvas
    # Apple macOS Human Interface Guidelines specify ~824x824 squircle inside 1024x1024 canvas
    master_1024 = Image.new("RGBA", (1024, 1024), (0, 0, 0, 0))
    icon_scaled = cropped_icon.resize((840, 840), Image.Resampling.LANCZOS)
    master_1024.paste(icon_scaled, (92, 92), icon_scaled)

    # Standard macOS iconset specification for iconutil
    iconset_specs = [
        ("icon_16x16.png", 16),
        ("icon_16x16@2x.png", 32),
        ("icon_32x32.png", 32),
        ("icon_32x32@2x.png", 64),
        ("icon_128x128.png", 128),
        ("icon_128x128@2x.png", 256),
        ("icon_256x256.png", 256),
        ("icon_256x256@2x.png", 512),
        ("icon_512x512.png", 512),
        ("icon_512x512@2x.png", 1024),
    ]

    # Additional standalone sizes for generic use (e.g. web, dock, settings, finder)
    standalone_sizes = [16, 32, 48, 64, 128, 256, 512, 1024]

    # Primary assets directory: all standalone sizes, iconset, and icns
    assets_dir = os.path.join(repo_root, "assets", "icons")
    os.makedirs(assets_dir, exist_ok=True)
    print(f"==> Exporting full icon suite to: {assets_dir}")

    # Save 1024 master
    master_1024.save(os.path.join(assets_dir, "icon_1024x1024.png"), "PNG")
    master_1024.save(os.path.join(assets_dir, "AppIcon.png"), "PNG")

    # Save standalone sizes
    for sz in standalone_sizes:
        resized = master_1024.resize((sz, sz), Image.Resampling.LANCZOS)
        resized.save(os.path.join(assets_dir, f"icon_{sz}x{sz}.png"), "PNG")

    # Prepare .iconset directory
    iconset_dir = os.path.join(assets_dir, "AppIcon.iconset")
    os.makedirs(iconset_dir, exist_ok=True)
    for filename, sz in iconset_specs:
        resized = master_1024.resize((sz, sz), Image.Resampling.LANCZOS)
        resized.save(os.path.join(iconset_dir, filename), "PNG")

    # Generate .icns using iconutil
    icns_path = os.path.join(assets_dir, "AppIcon.icns")
    try:
        subprocess.run(["iconutil", "-c", "icns", iconset_dir, "-o", icns_path], check=True)
        print(f"    [+] Created {icns_path} ({os.path.getsize(icns_path):,} bytes)")
    except Exception as e:
        print(f"    [!] Warning: iconutil failed: {e}", file=sys.stderr)

    # Swift Resources: only AppIcon.png and AppIcon.icns to prevent SwiftPM name collisions
    swift_res_dir = os.path.join(repo_root, "swift", "TurboSparkApp", "Sources", "TurboSparkApp", "Resources")
    os.makedirs(swift_res_dir, exist_ok=True)
    print(f"==> Exporting app bundle assets to: {swift_res_dir}")
    master_1024.save(os.path.join(swift_res_dir, "AppIcon.png"), "PNG")
    if os.path.exists(icns_path):
        subprocess.run(["cp", icns_path, os.path.join(swift_res_dir, "AppIcon.icns")], check=True)
        print(f"    [+] Copied AppIcon.icns to Swift resources")

    print("==> All icon assets generated successfully!")

if __name__ == "__main__":
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    src = "/Users/whit3rabbit/.gemini/antigravity-ide/brain/1e79649b-c54a-4e3c-a011-0b3e1b6b0c68/.user_uploaded/media_1788111684876.jpg"
    if len(sys.argv) > 1:
        src = sys.argv[1]

    out_dirs = [
        os.path.join(repo_root, "assets", "icons"),
        os.path.join(repo_root, "swift", "TurboSparkApp", "Sources", "TurboSparkApp", "Resources"),
    ]
    extract_and_generate_icons(src, out_dirs)
