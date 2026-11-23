#!/usr/bin/env python3
"""
Helper for the INT3404 report. Put this next to assign_code.py and campus.tiff, then run:
    python run_report.py
It writes everything into ./report_outputs/ with fixed names and prints the numbers
the report needs (also saved to report_outputs/results.txt).
"""
import os
import numpy as np
from skimage import io, img_as_ubyte

from src import assign_code
from src.assign_code import RawCameraPipeline

# ---- Settings: edit if needed -------------------------------------------------
TIFF = os.path.join("data", "campus.tiff")
BLACK, WHITE = 2048, 15488
WB_MULTS = [1.858398, 1.0, 1.723633, 1.0]
CORRECT_PATTERN = "RGGB"          # the pattern you believe is correct
MANUAL_REGION = (690, 700, 2920, 2930)   # (y0, y1, x0, x1) of a neutral patch; keep y0, x0 EVEN
JPEG_QUALITIES = [95, 90, 75, 50, 25]
OUT = os.path.join("results", "report_outputs")
M_XYZ_CAM = np.array([[6722, -635, -963],
                      [-4287, 12460, 2028],
                      [-908, 2162, 5668]], dtype=np.float64) / 10000.0
# --------------------------------------------------------------------------------

os.makedirs(OUT, exist_ok=True)
log_lines = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    log_lines.append(s)


# Vectorized masks: identical result to the loop version, but ~100x faster on a 20 MP image.
def _fast_masks(h, w, pattern):
    rows = (np.arange(h) % 2)[:, None]
    cols = (np.arange(w) % 2)[None, :]
    idx = rows * 2 + cols
    letters = np.array(list(pattern))
    grid = letters[idx]
    return grid == "R", grid == "G", grid == "B"


RawCameraPipeline._build_bayer_masks = staticmethod(_fast_masks)


def save(name, arr):
    io.imsave(os.path.join(OUT, name), img_as_ubyte(np.clip(arr, 0, 1)), check_contrast=False)


def gamma(lin):
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.power(lin, 1 / 2.4) - 0.055)


def make(pattern, wb="camera", region=None):
    p = RawCameraPipeline(TIFF, BLACK, WHITE, WB_MULTS, pattern)
    p.load_raw_tiff()
    p.linearize()
    p.white_balance(method=wb, manual_region=region)
    p.demosaic_bilinear()
    return p


# 1. Raw stats (printed by load_raw_tiff) ----------------------------------------
log("=== Section 2.2: raw TIFF stats ===")
p0 = RawCameraPipeline(TIFF, BLACK, WHITE, WB_MULTS, CORRECT_PATTERN)
p0.load_raw_tiff()
log("dtype of file on disk:", io.imread(TIFF).dtype)

# 2. Bayer pattern comparison ----------------------------------------------------
log("\n=== Section 4: Bayer patterns (camera WB) ===")
for pat in ["RGGB", "BGGR", "GRBG", "GBRG"]:
    p = make(pat)
    p.color_space_correction(M_XYZ_CAM)
    p.brightness_and_gamma()
    save(f"bayer_{pat}.png", p.rgb_srgb)
    log("saved", f"bayer_{pat}.png")

# 3. White balance comparison ----------------------------------------------------
log("\n=== Section 5: white balance methods (pattern %s) ===" % CORRECT_PATTERN)
for wb in ["camera", "gray_world", "white_world", "manual"]:
    p = RawCameraPipeline(TIFF, BLACK, WHITE, WB_MULTS, CORRECT_PATTERN)
    p.load_raw_tiff.__func__  # no-op, keeps output quiet-ish
    p.raw_image = p0.raw_image
    p.linearize()
    h, w = p.linear_image.shape
    mr, mg, mb = p._build_bayer_masks(h, w, CORRECT_PATTERN)
    img = p.linear_image
    if wb == "gray_world":
        sr, sb = img[mg].mean() / img[mr].mean(), img[mg].mean() / img[mb].mean()
    elif wb == "white_world":
        sr = np.percentile(img[mg], 99) / np.percentile(img[mr], 99)
        sb = np.percentile(img[mg], 99) / np.percentile(img[mb], 99)
    elif wb == "camera":
        sr, sb = WB_MULTS[0] / WB_MULTS[1], WB_MULTS[2] / WB_MULTS[1]
    else:
        y0, y1, x0, x1 = MANUAL_REGION
        patch = img[y0:y1, x0:x1]
        pr, pg, pb = p._build_bayer_masks(y1 - y0, x1 - x0, CORRECT_PATTERN)
        sr, sb = patch[pg].mean() / patch[pr].mean(), patch[pg].mean() / patch[pb].mean()
    log(f"{wb:12s} s_r={sr:.4f}  s_g=1.0000  s_b={sb:.4f}")
    p.white_balance(method=wb, manual_region=MANUAL_REGION if wb == "manual" else None)
    p.demosaic_bilinear()
    p.color_space_correction(M_XYZ_CAM)
    p.brightness_and_gamma()
    save(f"wb_{wb}.png", p.rgb_srgb)
    log("mean R,G,B of final image:", np.round(p.rgb_srgb.reshape(-1, 3).mean(0), 4))

# No white balance reference (demosaic + color correction + gamma only)
p = RawCameraPipeline(TIFF, BLACK, WHITE, WB_MULTS, CORRECT_PATTERN)
p.raw_image = p0.raw_image
p.linearize()
p.demosaic_bilinear()
p.color_space_correction(M_XYZ_CAM)
p.brightness_and_gamma()
save("wb_none.png", p.rgb_srgb)
log("saved wb_none.png")

# 4. Demosaiced channels, color correction before/after, linear vs gamma -----------
log("\n=== Sections 6-8: final pipeline (camera WB, %s) ===" % CORRECT_PATTERN)
p = make(CORRECT_PATTERN)
lin = p.rgb_linear.copy()
for i, c in enumerate("RGB"):
    ch = np.zeros_like(lin)
    ch[..., i] = gamma(np.clip(lin[..., i] / lin.mean() * 0.25, 0, 1))
    save(f"demosaic_{c}.png", ch)

# before color correction (same brightness + gamma so the comparison is fair)
tmp = RawCameraPipeline(TIFF, BLACK, WHITE, WB_MULTS, CORRECT_PATTERN)
tmp.rgb_linear = lin.copy()
tmp.brightness_and_gamma()
save("csc_before.png", tmp.rgb_srgb)

M_srgb_to_xyz = np.array([[0.4124564, 0.3575761, 0.1804375],
                          [0.2126729, 0.7151522, 0.0721750],
                          [0.0193339, 0.1191920, 0.9503041]])
M_raw = np.linalg.inv(M_XYZ_CAM @ M_srgb_to_xyz)
M_norm = M_raw / M_raw.sum(axis=1, keepdims=True)
np.set_printoptions(precision=6, suppress=True)
log("M_cam_to_srgb BEFORE row normalization:\n", M_raw)
log("M_cam_to_srgb AFTER row normalization (what your code uses):\n", M_norm)

p.color_space_correction(M_XYZ_CAM)
save("csc_after_linear_unscaled.png", p.rgb_linear)       # linear image (dark) for section 8
lin_corrected = p.rgb_linear.copy()
p.brightness_and_gamma()
save("csc_after.png", p.rgb_srgb)                          # gamma-encoded
gray = lambda im: 0.2126 * im[..., 0] + 0.7152 * im[..., 1] + 0.0722 * im[..., 2]
log("mean gray, linear (before scaling): %.4f" % gray(lin_corrected).mean())
log("mean gray, linear after scaling to 0.25: %.4f" % gray(np.clip(lin_corrected * 0.25 / gray(lin_corrected).mean(), 0, 1)).mean())
log("mean gray, gamma-encoded: %.4f" % gray(p.rgb_srgb).mean())

# 5. Compression (ALL channels, not blue-only) -----------------------------------
log("\n=== Section 9: compression (full colour image) ===")
final = p.rgb_srgb
rows = []
for q in JPEG_QUALITIES:
    prefix = os.path.join(OUT, f"final_q{q}")
    p.rgb_srgb = final
    p.save_png_jpeg(prefix, quality=q)
    png = os.path.getsize(prefix + ".png") / 1024.0
    jpg = os.path.getsize(prefix + ".jpg") / 1024.0
    rows.append((q, png, jpg, png / jpg))
log("\nquality | PNG KB | JPEG KB | ratio")
for q, a, b, r in rows:
    log(f"{q:7d} | {a:9.2f} | {b:9.2f} | {r:5.2f}")

with open(os.path.join(OUT, "results.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))
print("\nDone. Upload results.txt and the PNGs in report_outputs/ (list in chat).")