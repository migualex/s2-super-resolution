# =====================================================================
# PRODES preview (R=B11, G=B08, B=B04) of the latest s2_super_resolution.py run
# Overview of the whole product plus original 10 m vs super-resolved 2.5 m on the same crop.
#
#   python tools/preview_prodes.py [OUT_DIR]
# or run it as a cell in the same kernel as s2_super_resolution.py (OUT_DIR is reused).
# =====================================================================

# %% PRODES PREVIEW
import glob, os, json, numpy as np, rasterio, matplotlib.pyplot as plt
from rasterio.enums import Resampling
from rasterio.windows import Window, from_bounds
from tqdm.auto import tqdm
import sys

if __name__ == "__main__" and len(sys.argv) > 1 and "ipykernel" not in sys.modules:
    OUT_DIR = sys.argv[1]
OUT_DIR = globals().get("OUT_DIR", "work/cube_resolution")

ZOOM_LR_PX = 256                 # detail crop size, in 10 m pixels
ZOOM_CENTER = None               # (x, y) in 10 m pixels inside the processed area; None = center

_bar = tqdm(total=4, desc="Preview: reading report", unit="step")
_reps = sorted(glob.glob(os.path.join(OUT_DIR, "logs", "*.report.json")), key=os.path.getmtime)
if not _reps:
    _bar.close()
    raise SystemExit(f"No report in {OUT_DIR}/logs: run block 2 first.")
rep = json.load(open(_reps[-1], encoding="utf-8"))
outs, inp = rep.get("outputs", {}), rep.get("input_files", {})
_bar.update(1)


def _stretch(x, lo, hi):
    return np.clip((x.astype("float32") - lo) / max(hi - lo, 1e-6), 0, 1)


_bi = None
if "sr" in outs and os.path.exists(outs["sr"]):
    with rasterio.open(outs["sr"]) as d:
        _desc = list(d.descriptions)
    if all(b in _desc for b in ("B11", "B08", "B04")):
        _bi = [_desc.index(b) + 1 for b in ("B11", "B08", "B04")]
if _bi:
    # PRODES R=B11 G=B08 B=B04 read straight from the SR file; 2-98 % stretch of the original 10 m bands
    _bar.set_description("Preview: PRODES overview")
    lo, hi = [], []
    for b in ("B11", "B08", "B04"):
        with rasterio.open(inp[b]) as o:
            v = o.read(1, out_shape=(max(1, o.height // 8), max(1, o.width // 8)), resampling=Resampling.nearest)
        v = v[v > 0]
        lo.append(float(np.percentile(v, 2)) if v.size else 0.0)
        hi.append(float(np.percentile(v, 98)) if v.size else 10000.0)
    with rasterio.open(outs["sr"]) as d:
        k = max(1, max(d.height, d.width) // 1500)
        over = d.read(_bi, out_shape=(3, d.height // k, d.width // k), resampling=Resampling.average)
        S = 4
        Wlr, Hlr = d.width // S, d.height // S
        ZOOM_LR_PX = int(min(ZOOM_LR_PX, Wlr, Hlr))
        half = ZOOM_LR_PX // 2
        cx, cy = ZOOM_CENTER if ZOOM_CENTER else (Wlr // 2, Hlr // 2)
        cx, cy = min(max(cx, half), Wlr - half), min(max(cy, half), Hlr - half)
        win_sr = Window((cx - half) * S, (cy - half) * S, ZOOM_LR_PX * S, ZOOM_LR_PX * S)
        sr_zoom = d.read(_bi, window=win_sr)
        bounds = rasterio.windows.bounds(win_sr, d.transform)
    _bar.update(1)
    _bar.set_description("Preview: original 10 m crop")
    orig = []
    for b in ("B11", "B08", "B04"):
        with rasterio.open(inp[b]) as d:
            orig.append(d.read(1, window=from_bounds(*bounds, transform=d.transform),
                               out_shape=(ZOOM_LR_PX, ZOOM_LR_PX), resampling=Resampling.nearest))
    orig = np.stack([_stretch(o, lo[i], hi[i]) for i, o in enumerate(orig)], -1)
    _bar.update(1)
    _bar.set_description("Preview: drawing")
    fig, ax = plt.subplots(1, 3, figsize=(20, 7))
    ax[0].imshow(np.stack([_stretch(over[i], lo[i], hi[i]) for i in range(3)], -1))
    ax[0].set_title("SR 2.5 m — overview (R=B11 G=B08 B=B04)")
    r = Window(win_sr.col_off / k, win_sr.row_off / k, win_sr.width / k, win_sr.height / k)
    ax[0].add_patch(plt.Rectangle((r.col_off, r.row_off), r.width, r.height, fill=False, ec="yellow", lw=1.5))
    ax[1].imshow(orig)
    ax[1].set_title(f"Original 10 m — {ZOOM_LR_PX} px crop")
    ax[2].imshow(np.stack([_stretch(sr_zoom[i], lo[i], hi[i]) for i in range(3)], -1))
    ax[2].set_title("Super-resolution 2.5 m — same crop")
    for a_ in ax:
        a_.axis("off")
    plt.tight_layout()
else:
    _bar.update(2)
    _bar.set_description("Preview: no B11 -> showing B08/B04/B03")
    with rasterio.open(outs["sr"]) as d:
        k = max(1, max(d.height, d.width) // 1500)
        desc = list(d.descriptions)
        idx = [desc.index(b) + 1 if b in desc else j for b, j in (("B08", 4), ("B04", 1), ("B03", 2))]
        a = d.read(idx, out_shape=(3, d.height // k, d.width // k), resampling=Resampling.average).astype("float32")
    v = a[:, a[0] > 0]
    lo_, hi_ = np.percentile(v, 2, axis=1), np.percentile(v, 98, axis=1)
    plt.figure(figsize=(10, 10))
    plt.imshow(np.clip((a.transpose(1, 2, 0) - lo_) / (hi_ - lo_ + 1e-6), 0, 1))
    plt.title("SR 2.5 m (B08/B04/B03)")
    plt.axis("off")
_bar.update(1)
_bar.set_description("Preview ready")
_bar.close()
plt.show()
