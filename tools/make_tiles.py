#!/usr/bin/env python3
"""Render one evidence tile per evaluated pair, plus a merged contact sheet.

Each tile shows the search image full-frame, the reference inset at its
top-right, every generator parameter for that pair, and the predicted and
true positions pinned to exact pixel coordinates with an 8x magnified
neighbourhood underneath.

    python tools/make_tiles.py \
        --root      <dataset root with pairs.csv>          \
        --pred      predictions.csv                        \
        --truth     ground_truth.csv                       \
        --params    manifest_params.csv   (optional)       \
        --out       <output directory>                     \
        --label     "Phase 3 - CAD2SEM"

No dataset is written to the repository: only the rendered PNGs are kept.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------- palette --

# A light sheet reads better than a dark one: these tiles are printed,
# projected and skimmed at thumbnail size, and the SEM imagery is already
# mid-grey, so dark chrome swallowed the markers.
INK = (22, 27, 36)
DIM = (86, 96, 112)
FAINT = (146, 155, 170)
BG = (255, 255, 255)
PANEL = (255, 255, 255)
LINE = (208, 214, 224)
OVERLAY = (255, 255, 255, 216)    # label plates laid over the micrograph

TRUTH_C = (17, 138, 71)           # ground truth  - green
PRED_C = (16, 96, 202)            # prediction    - blue
ERR_C = (196, 98, 8)              # error vector  - orange
REJECT_C = (196, 38, 46)          # miss / false call

VERDICTS = (                      # (max error px, credit, label, colour)
    (1.0, 1.00, "HIT <= 1 px", (17, 138, 71)),
    (2.0, 0.80, "HIT <= 2 px", (86, 140, 40)),
    (3.0, 0.60, "HIT <= 3 px", (156, 128, 16)),
    (5.0, 0.40, "HIT <= 5 px", (196, 98, 8)),
)

# The panel geometry is fixed so every tile in a sheet aligns exactly.
W = 900
MARGIN = 14
MAIN = 576
COL_X = MARGIN + MAIN + 12
COL_W = W - COL_X - MARGIN
MAIN_Y = 74
ZOOM_Y = MAIN_Y + MAIN + 14
ZOOM = 204
H = ZOOM_Y + ZOOM + 52

FONT_DIRS = (
    "C:/Windows/Fonts", "/usr/share/fonts/truetype/dejavu",
    "/usr/share/fonts/TTF", "/Library/Fonts", "/System/Library/Fonts",
)
FONT_FILES = ("DejaVuSans.ttf", "arial.ttf", "Arial.ttf", "Helvetica.ttc")
MONO_FILES = ("DejaVuSansMono.ttf", "consola.ttf", "cour.ttf", "Menlo.ttc")


def _font(size: int, mono: bool = False):
    for directory in FONT_DIRS:
        for name in (MONO_FILES if mono else FONT_FILES):
            candidate = Path(directory) / name
            if candidate.is_file():
                try:
                    return ImageFont.truetype(str(candidate), size)
                except OSError:
                    continue
    return ImageFont.load_default()


# --------------------------------------------------------------- plumbing --

def read_csv(path: Path, key: str = "pair_id") -> dict:
    if path is None or not Path(path).is_file():
        return {}
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return {row[key]: row for row in csv.DictReader(handle) if row.get(key)}


def read_params(path: Path) -> dict:
    """Generator manifests key on `id` (0, 1, 2 ...) -> pNNN."""
    if path is None or not Path(path).is_file():
        return {}
    out = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            raw = row.get("pair_id") or row.get("id")
            if raw is None:
                continue
            key = raw if str(raw).startswith("p") else f"p{int(raw):03d}"
            out[key] = row
    return out


def load_gray(path: Path) -> np.ndarray | None:
    try:
        with Image.open(path) as image:
            return np.asarray(image.convert("L"), dtype=np.uint8)
    except (OSError, ValueError):
        return None


def number(value, digits=3):
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if f == int(f) and abs(f) < 1e6:
        return str(int(f))
    return f"{f:.{digits}f}".rstrip("0").rstrip(".")


def verdict_of(present: int, found: int, err: float) -> tuple[str, tuple, float]:
    if not present:
        return (("CORRECT REJECT", TRUTH_C, 1.0) if not found
                else ("FALSE POSITIVE", REJECT_C, 0.0))
    if not found:
        return "MISSED (rejected)", REJECT_C, 0.0
    for bound, credit, label, colour in VERDICTS:
        if err <= bound:
            return label, colour, credit
    return f"OFF BY {err:.1f} px", REJECT_C, 0.0


# ---------------------------------------------------------------- drawing --

def marker_truth(draw, x, y, r=13):
    draw.ellipse([x - r - 1, y - r - 1, x + r + 1, y + r + 1], outline=(255, 255, 255), width=1)
    draw.ellipse([x - r, y - r, x + r, y + r], outline=TRUTH_C, width=3)
    draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=TRUTH_C)


def marker_pred(draw, x, y, r=13):
    draw.line([x - r, y, x - 4, y], fill=PRED_C, width=3)
    draw.line([x + 4, y, x + r, y], fill=PRED_C, width=3)
    draw.line([x, y - r, x, y - 4], fill=PRED_C, width=3)
    draw.line([x, y + 4, x, y + r], fill=PRED_C, width=3)


def dashed(draw, p0, p1, colour, dash=6, gap=5, width=1):
    x0, y0 = p0
    x1, y1 = p1
    length = math.hypot(x1 - x0, y1 - y0)
    if length < 1e-6:
        return
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    travelled = 0.0
    while travelled < length:
        end = min(travelled + dash, length)
        draw.line([x0 + ux * travelled, y0 + uy * travelled,
                   x0 + ux * end, y0 + uy * end], fill=colour, width=width)
        travelled = end + gap


def chip(draw, box, text, colour, font):
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=5, fill=(*colour, 28) if len(colour) == 3 else colour,
                           outline=colour, width=1)
    draw.text(((x0 + x1) / 2, (y0 + y1) / 2), text, font=font, fill=colour, anchor="mm")


def zoom_patch(image: np.ndarray, cx: float, cy: float, span: int, out: int) -> Image.Image:
    """Nearest-neighbour magnification so individual pixels stay square."""
    half = span // 2
    x0, y0 = int(round(cx)) - half, int(round(cy)) - half
    canvas = np.zeros((span, span), dtype=np.uint8)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(image.shape[1], x0 + span), min(image.shape[0], y0 + span)
    if sx1 > sx0 and sy1 > sy0:
        canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = image[sy0:sy1, sx0:sx1]
    return Image.fromarray(canvas).resize((out, out), Image.NEAREST), x0, y0


# ------------------------------------------------------------------ tile ---

def render_tile(pair_id, label, search, reference, truth, pred, params) -> Image.Image:
    f_title = _font(21)
    f_head = _font(13)
    f_small = _font(11)
    f_key = _font(11, mono=True)
    f_tick = _font(9, mono=True)

    present = int(float(truth.get("present", 0))) if truth else 0
    gx, gy = float(truth.get("x", 0) or 0), float(truth.get("y", 0) or 0)
    found = int(float(pred.get("found", 0))) if pred else 0
    px, py = float(pred.get("x", 0) or 0), float(pred.get("y", 0) or 0)
    score = float(pred.get("score", 0) or 0)
    ptheta = float(pred.get("theta", 0) or 0)
    pscale = float(pred.get("scale", 0) or 0)
    gtheta = float(truth.get("theta", 0) or 0)
    gscale = float(truth.get("scale", 0) or 0)
    err = math.hypot(px - gx, py - gy) if (present and found) else float("nan")
    text, colour, credit = verdict_of(present, found, err)

    tile = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(tile, "RGBA")

    # ---- header
    draw.text((MARGIN, 18), pair_id, font=f_title, fill=INK)
    draw.text((MARGIN + 86, 24), label, font=f_head, fill=DIM)
    chip(draw, (W - MARGIN - 190, 18, W - MARGIN, 44), text, colour, f_head)
    draw.line([MARGIN, 62, W - MARGIN, 62], fill=LINE, width=1)

    # ---- main search panel
    sh, sw = search.shape
    scale_px = MAIN / max(sw, sh)
    view = Image.fromarray(search).resize((int(sw * scale_px), int(sh * scale_px)), Image.LANCZOS)
    tile.paste(view.convert("RGB"), (MARGIN, MAIN_Y))
    draw.rectangle([MARGIN, MAIN_Y, MARGIN + view.width, MAIN_Y + view.height],
                   outline=LINE, width=1)

    def to_view(x, y):
        return MARGIN + x * scale_px, MAIN_Y + y * scale_px

    # rulers: a coordinate every 100 source px, so any point can be read off
    for t in range(0, sw + 1, 100):
        vx, _ = to_view(t, 0)
        draw.line([vx, MAIN_Y + view.height - 7, vx, MAIN_Y + view.height], fill=(0, 0, 0, 140))
        if t % 200 == 0 and t < sw:
            draw.rectangle([vx, MAIN_Y + view.height - 20, vx + 24, MAIN_Y + view.height - 8],
                           fill=OVERLAY)
            draw.text((vx + 3, MAIN_Y + view.height - 19), str(t), font=f_tick, fill=INK)
    for t in range(0, sh + 1, 100):
        _, vy = to_view(0, t)
        draw.line([MARGIN, vy, MARGIN + 7, vy], fill=(0, 0, 0, 140))
        if t % 200 == 0 and t > 0:
            draw.rectangle([MARGIN + 8, vy - 7, MARGIN + 32, vy + 5], fill=OVERLAY)
            draw.text((MARGIN + 10, vy - 6), str(t), font=f_tick, fill=INK)

    # ---- reference inset, pinned to the top-right of the search frame
    if reference is not None:
        side = 158
        inset = Image.fromarray(reference).resize((side, side), Image.LANCZOS).convert("RGB")
        ix, iy = MARGIN + view.width - side - 10, MAIN_Y + 10
        draw.rectangle([ix - 3, iy - 3, ix + side + 2, iy + side + 16], fill=PANEL)
        tile.paste(inset, (ix, iy))
        draw.rectangle([ix - 1, iy - 1, ix + side, iy + side], outline=(90, 100, 118), width=1)
        draw.text((ix, iy + side + 3), "REFERENCE", font=f_small, fill=DIM)

    # ---- pinpoint markers
    if present:
        vgx, vgy = to_view(gx, gy)
        dashed(draw, (MARGIN, vgy), (MARGIN + view.width, vgy), (*TRUTH_C, 170))
        dashed(draw, (vgx, MAIN_Y), (vgx, MAIN_Y + view.height), (*TRUTH_C, 170))
        marker_truth(draw, vgx, vgy)
    if found:
        vpx, vpy = to_view(px, py)
        dashed(draw, (MARGIN, vpy), (MARGIN + view.width, vpy), (*PRED_C, 170))
        dashed(draw, (vpx, MAIN_Y), (vpx, MAIN_Y + view.height), (*PRED_C, 170))
        marker_pred(draw, vpx, vpy)
    if present and found and err > 4:
        draw.line([*to_view(px, py), *to_view(gx, gy)], fill=ERR_C, width=2)

    # legend sits top-left, clear of both the reference inset and the rulers
    legend_y = MAIN_Y + 10
    entries = ([(marker_truth, "ground truth", TRUTH_C)] if present else []) + \
              ([(marker_pred, "our prediction", PRED_C)] if found else [])
    if entries:
        draw.rectangle([MARGIN + 8, legend_y - 4, MARGIN + 142,
                        legend_y + 13 * len(entries) + 1], fill=OVERLAY)
        for i, (marker, caption, tint) in enumerate(entries):
            marker(draw, MARGIN + 20, legend_y + 5 + 13 * i, 7)
            draw.text((MARGIN + 32, legend_y + 13 * i), caption, font=f_small, fill=tint)

    # ---- right column: the numbers behind the picture
    y = MAIN_Y
    def section(title):
        nonlocal y
        draw.text((COL_X, y), title.upper(), font=f_small, fill=(30, 92, 168))
        y += 14
        draw.line([COL_X, y, W - MARGIN, y], fill=LINE)
        y += 6

    def row(key, value, value_colour=INK):
        nonlocal y
        draw.text((COL_X, y), key, font=f_key, fill=DIM)
        draw.text((W - MARGIN, y), str(value), font=f_key, fill=value_colour, anchor="ra")
        y += 14

    section("result")
    row("found", "yes" if found else "no", PRED_C if found else REJECT_C)
    row("present (truth)", "yes" if present else "no", TRUTH_C)
    row("score", f"{score:.4f}")
    row("localization credit", f"{credit:.2f}" if present else "n/a",
        colour if present else DIM)

    y += 6
    section("position  (search px)")
    row("predicted x", f"{px:.3f}" if found else "-", PRED_C)
    row("predicted y", f"{py:.3f}" if found else "-", PRED_C)
    row("true x", f"{gx:.3f}" if present else "-", TRUTH_C)
    row("true y", f"{gy:.3f}" if present else "-", TRUTH_C)
    if present and found:
        row("dx", f"{px - gx:+.3f}", ERR_C)
        row("dy", f"{py - gy:+.3f}", ERR_C)
        row("error", f"{err:.3f} px", colour)

    y += 6
    section("pose")
    row("theta  pred / true", f"{ptheta:+.2f} / {gtheta:+.2f}" if present else f"{ptheta:+.2f} / -")
    row("scale  pred / true", f"{pscale:.3f} / {gscale:.3f}" if present else f"{pscale:.3f} / -")

    if params:
        y += 6
        section("generator parameters")
        skip = {"id", "pair_id", "reference_path", "search_path", "reference_gds_path",
                "search_gds_path", "reference_sem_path", "reference_preview_path",
                "params_json_path", "gt_x", "gt_y", "gt_box_x", "gt_box_y",
                "gt_box_w", "gt_box_h", "match_found", "present", "x", "y"}
        shown = 0
        limit = (H - 66 - y) // 13
        for key, value in params.items():
            if key in skip or value in ("", None) or shown >= limit:
                continue
            draw.text((COL_X, y), key[:24], font=f_tick, fill=DIM)
            draw.text((W - MARGIN, y), number(value), font=f_tick, fill=INK, anchor="ra")
            y += 13
            shown += 1

    # ---- magnified neighbourhood: the pinpoint, at one screen pixel per source pixel
    span, mag = 58, ZOOM
    if present:
        panels = [(gx, gy, "TRUE SITE  x8"),
                  (px, py, "OUR CALL  x8") if found
                  else (gx, gy, "TRUE SITE - WE MADE NO CALL")]
    elif found:
        panels = [(px, py, "OUR CALL  x8"), (px, py, "NO TRUE SITE EXISTS")]
    else:
        # Nothing to pin: show the middle of the search image rather than an
        # empty black square, so the tile still reads as real imagery.
        mid_x, mid_y = search.shape[1] / 2, search.shape[0] / 2
        panels = [(mid_x, mid_y, "SEARCH CENTRE  x8"),
                  (mid_x + span, mid_y, "NO INSTANCE, NO CALL")]
    for index, (cx, cy, caption) in enumerate(panels):
        zx = MARGIN + index * (mag + 12)
        patch, ox, oy = zoom_patch(search, cx, cy, span, mag)
        tile.paste(patch.convert("RGB"), (zx, ZOOM_Y))
        draw.rectangle([zx, ZOOM_Y, zx + mag, ZOOM_Y + mag], outline=LINE, width=1)
        step = mag / span
        for g in range(0, span + 1, 10):
            draw.line([zx + g * step, ZOOM_Y, zx + g * step, ZOOM_Y + mag], fill=(0, 0, 0, 26))
            draw.line([zx, ZOOM_Y + g * step, zx + mag, ZOOM_Y + g * step], fill=(0, 0, 0, 26))
        if present:
            marker_truth(draw, zx + (gx - ox) * step, ZOOM_Y + (gy - oy) * step, 11)
        if found:
            marker_pred(draw, zx + (px - ox) * step, ZOOM_Y + (py - oy) * step, 11)
        draw.rectangle([zx + 1, ZOOM_Y + 1, zx + mag - 1, ZOOM_Y + 17], fill=OVERLAY)
        draw.text((zx + 5, ZOOM_Y + 3), caption, font=f_small, fill=INK)
        draw.rectangle([zx + 1, ZOOM_Y + mag - 16, zx + mag - 1, ZOOM_Y + mag - 1],
                       fill=OVERLAY)
        draw.text((zx + 5, ZOOM_Y + mag - 14),
                  f"x {ox}..{ox + span}   y {oy}..{oy + span}", font=f_tick, fill=DIM)

    # ---- error read-out beside the magnifiers
    bx = MARGIN + 2 * (mag + 12)
    draw.text((bx, ZOOM_Y + 4), "PINPOINT", font=f_small, fill=(30, 92, 168))
    lines = []
    if present and found:
        lines = [f"true    ({gx:9.3f}, {gy:9.3f})",
                 f"ours    ({px:9.3f}, {py:9.3f})",
                 f"offset  ({px - gx:+9.3f}, {py - gy:+9.3f})",
                 f"|error| {err:9.3f} px",
                 f"        {err * 10.0:9.1f} nm at 10 nm/px"]
    elif present:
        lines = [f"true    ({gx:9.3f}, {gy:9.3f})",
                 "ours     rejected - no call made",
                 f"score   {score:9.4f}"]
    else:
        lines = ["truth    no instance in this search image",
                 f"ours     {'rejected (correct)' if not found else 'CALLED A MATCH'}",
                 f"score   {score:9.4f}"]
    ly = ZOOM_Y + 24
    for line in lines:
        draw.text((bx, ly), line, font=f_key, fill=INK)
        ly += 16
    if present and found:
        gauge_y = ly + 10
        draw.text((bx, gauge_y), "credit ladder  (earned tier highlighted)",
                  font=f_small, fill=DIM)
        earned = next((b for b, _, _, _ in VERDICTS if err <= b), None)
        for i, (bound, tier_credit, _, tier_colour) in enumerate(VERDICTS):
            seg_x = bx + i * 52
            active = bound == earned
            draw.rounded_rectangle([seg_x, gauge_y + 16, seg_x + 46, gauge_y + 36], radius=4,
                                   fill=(*tier_colour, 46) if active else (0, 0, 0, 10),
                                   outline=tier_colour if active else LINE)
            draw.text((seg_x + 23, gauge_y + 22), f"{bound:g}px", font=f_tick,
                      fill=tier_colour if active else FAINT, anchor="mm")
            draw.text((seg_x + 23, gauge_y + 31), f"{tier_credit:.1f}", font=f_tick,
                      fill=tier_colour if active else FAINT, anchor="mm")

    draw.line([MARGIN, H - 32, W - MARGIN, H - 32], fill=LINE)
    draw.text((MARGIN, H - 24),
              "LatticeRank Drift-Sense  -  search image full frame, reference inset top-right, "
              "magnified pinpoint below", font=f_small, fill=FAINT)
    return tile


# ------------------------------------------------------------------ sheet --

def save_light(image: Image.Image, path: Path) -> None:
    """Most of a tile is SEM shot noise, which PNG cannot compress. JPEG at
    high quality with no chroma subsampling keeps the overlay text and the
    coloured markers crisp while holding the whole evidence set to a few
    megabytes -- the repository ships figures, never datasets."""
    image.save(path, quality=84, subsampling=0, optimize=True, progressive=True)


def contact_sheet(tiles: list[Image.Image], columns: int, width: int) -> Image.Image:
    if not tiles:
        raise ValueError("no tiles to merge")
    cell_w = width // columns
    cell_h = int(cell_w * tiles[0].height / tiles[0].width)
    rows = (len(tiles) + columns - 1) // columns
    sheet = Image.new("RGB", (cell_w * columns, cell_h * rows), (226, 230, 238))
    for index, tile in enumerate(tiles):
        r, c = divmod(index, columns)
        sheet.paste(tile.resize((cell_w, cell_h), Image.LANCZOS), (c * cell_w, r * cell_h))
    return sheet


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--pred", required=True, type=Path)
    ap.add_argument("--truth", type=Path)
    ap.add_argument("--params", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--label", default="")
    ap.add_argument("--sheet-columns", type=int, default=5)
    ap.add_argument("--sheet-width", type=int, default=2400)
    ap.add_argument("--sheet-name", default="contact-sheet.jpg")
    args = ap.parse_args()

    pairs_path = args.root / "pairs.csv"
    with pairs_path.open(newline="", encoding="utf-8-sig") as handle:
        pairs = list(csv.DictReader(handle))
    truth = read_csv(args.truth)
    pred = read_csv(args.pred)
    params = read_params(args.params)

    args.out.mkdir(parents=True, exist_ok=True)
    tiles = []
    for row in pairs:
        pair_id = row["pair_id"]
        search = load_gray(args.root / row["search_path"])
        if search is None:
            print(f"{pair_id}: search image unreadable; skipped")
            continue

        ref_raw = (row.get("reference_path")
                   or (params.get(pair_id, {}) or {}).get("reference_preview_path")
                   or row.get("reference_sem_path"))
        reference = load_gray(args.root / ref_raw.replace("\\", "/")) if ref_raw else None

        tile = render_tile(pair_id, args.label, search, reference,
                           truth.get(pair_id, {}), pred.get(pair_id, {}),
                           params.get(pair_id, {}))
        save_light(tile, args.out / f"{pair_id}.jpg")
        tiles.append(tile)
        print(f"{pair_id}: tile written")

    sheet = contact_sheet(tiles, args.sheet_columns, args.sheet_width)
    save_light(sheet, args.out / args.sheet_name)
    print(f"\n{len(tiles)} tiles + {args.sheet_name} in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
