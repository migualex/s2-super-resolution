# =====================================================================
# Comparison: original 10 m vs LDSR-S2 (fast) vs combined — visible, PRODES and B11 diagnostics
# Requires s2_super_resolution.py runs with the same TILE, DATE and ROI (e.g. "fast" and "combined").
#
#   python tools/compare_modes.py TILE DATE [OUT_DIR]      e.g. 034026 2026-07-12
# or run it as a cell in the same kernel as s2_super_resolution.py (TILE, DATE, OUT_DIR are reused).
# =====================================================================

# %% COMPARISON
# All products found in OUT_DIR are displayed with the stretch of the original 10 m image.
import glob, os, re, json, numpy as np, rasterio, matplotlib.pyplot as plt
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from tqdm.auto import tqdm
import sys

if __name__ == "__main__" and len(sys.argv) > 2 and "ipykernel" not in sys.modules:
    TILE, DATE = sys.argv[1], sys.argv[2]
    if len(sys.argv) > 3:
        OUT_DIR = sys.argv[3]
if "TILE" not in globals() or "DATE" not in globals():
    raise SystemExit("Usage: python tools/compare_modes.py TILE DATE [OUT_DIR]")
OUT_DIR = globals().get("OUT_DIR", "work/cube_resolution")

CMP_ZOOM_PX = 160                # displayed crop, in 10 m pixels
CMP_METRIC_PX = 512              # area used for the numbers, in 10 m pixels
CMP_CENTER = None                # (x, y) in 10 m pixels inside the processed area; None = center
CMP_KEEP = None                  # None = all products found | list of name fragments, e.g. ["combined", "_raw"]

_bar = tqdm(total=5, desc="Comparison: locating products", unit="step")
_cand, TAGS = [], {}
_D8 = DATE.replace("-", "")
for _f in glob.glob(os.path.join(OUT_DIR, f"{TILE}_{_D8}_*.tif")):
    if "partial" in _f:
        continue
    with rasterio.open(_f) as _d:
        _t = _d.tags()
    if (_t.get("MODE") in ("fast", "combined") and _t.get("PRODUCT", "").startswith("SR")
            and (CMP_KEEP is None or any(k in os.path.basename(_f) for k in CMP_KEEP))):
        _cand.append(_f)
        TAGS[_f] = _t
if not _cand:
    _bar.close()
    raise SystemExit(f"No fast/combined product for {TILE} {DATE} in {OUT_DIR}: run block 2 first.")
_grp = TAGS[max(_cand, key=os.path.getmtime)].get("AREA", "")
prods = sorted([f for f in _cand if TAGS[f].get("AREA", "") == _grp],
               key=lambda f: (0 if TAGS[f]["MODE"] == "fast" else 1, os.path.basename(f)))


def _label(f):
    t = TAGS[f]
    sig = float(t.get("DEBLOCK_20M", 0) or 0)
    kind = "LDSR-S2 + guided B11 (fast)" if t["MODE"] == "fast" else "LDSRS2-SEN2SR (combined)"
    return (kind + f", {t.get('DDIM_STEPS', '?')} steps" +
            (f"\n20 m de-blocked, sigma={sig:g}" if sig > 0 else "\n20 m raw BDC")), sig


_inp = {}
for _r in glob.glob(os.path.join(OUT_DIR, "logs", f"{TILE}_{_D8}_*.report.json")):
    _inp.update(json.load(open(_r, encoding="utf-8")).get("input_files", {}))
_need = ("B02", "B03", "B04", "B08", "B11")
_miss = [b for b in _need if b not in _inp or not os.path.exists(_inp[b])]
if _miss:
    _bar.close()
    raise SystemExit(f"Original 10 m bands not found for {_miss}: the reports in {OUT_DIR}/logs do not list them.")
_bar.update(1)
_bar.set_description("Comparison: reading crops")

with rasterio.open(prods[0]) as d:
    _Wlr, _Hlr = d.width // 4, d.height // 4
    M = int(min(CMP_METRIC_PX, _Wlr, _Hlr))
    Z = int(min(CMP_ZOOM_PX, M))
    cx, cy = CMP_CENTER if CMP_CENTER else (_Wlr // 2, _Hlr // 2)
    cx, cy = min(max(cx, M // 2), _Wlr - M // 2), min(max(cy, M // 2), _Hlr - M // 2)
    _win = rasterio.windows.Window((cx - M // 2) * 4, (cy - M // 2) * 4, M * 4, M * 4)
    BOUNDS = rasterio.windows.bounds(_win, d.transform)


def _read_orig(b):
    with rasterio.open(_inp[b]) as d:
        s = d.scales[0] if d.scales and d.scales[0] not in (None, 1.0) else 1e-4
        a = d.read(1, window=from_bounds(*BOUNDS, transform=d.transform), out_shape=(M, M),
                   resampling=Resampling.nearest).astype("float32")
        return np.where(a > 0, a * s, 0.0).astype("float32")


def _read_sr(f, bands):
    with rasterio.open(f) as d:
        desc = list(d.descriptions)
        idx = [desc.index(b) + 1 for b in bands]
        a = d.read(idx, window=from_bounds(*BOUNDS, transform=d.transform), out_shape=(len(idx), M * 4, M * 4),
                   resampling=Resampling.nearest).astype("float32")
        return a * 1e-4


ORIG = {b: _read_orig(b) for b in _need}
SR = {}
for f in prods:
    SR[f] = dict(zip(_need, _read_sr(f, list(_need))))
_bar.update(1)
_bar.set_description("Comparison: computing numbers")


def _blk(a, k=4):
    return a.reshape(a.shape[0] // k, k, a.shape[1] // k, k).mean((1, 3))


def _gauss(a, s):
    if s <= 0:
        return a
    r = int(np.ceil(3 * s))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / s) ** 2)
    k /= k.sum()
    for ax in (0, 1):
        ap = np.pad(a, [(r, r) if i == ax else (0, 0) for i in range(2)], mode="reflect")
        n = a.shape[ax]
        a = sum(k[i] * np.take(ap, np.arange(i, i + n), axis=ax) for i in range(2 * r + 1))
    return a


def _lap_std(a, m):
    c = a[1:-1, 1:-1]
    lap = 4 * c - a[:-2, 1:-1] - a[2:, 1:-1] - a[1:-1, :-2] - a[1:-1, 2:]
    mm = m[1:-1, 1:-1] & m[:-2, 1:-1] & m[2:, 1:-1] & m[1:-1, :-2] & m[1:-1, 2:]
    return float(lap[mm].std())


def _stair(a, m, period):
    """max/min of the mean |gradient| over the `period` phases: ~1 = no periodic grid, >>1 = blocky 20 m pattern"""
    out = []
    for A, Mk in ((a, m), (a.T, m.T)):
        g = np.abs(np.diff(A, axis=1))
        mg = Mk[:, 1:] & Mk[:, :-1]
        per = [g[:, i::period][mg[:, i::period]].mean() for i in range(period)]
        out.append(max(per) / max(min(per), 1e-9))
    return min(max(out), 99.0)


_mo = np.all([ORIG[b] > 0 for b in _need], axis=0)
_ref_det = _lap_std(ORIG["B11"], _mo) / _lap_std(ORIG["B08"], _mo)
rows_ = [dict(name="Original 10 m", dev="-", b11_bias="-", b11_mae="-", det=f"{_ref_det:.2f}",
              st11=f"{_stair(ORIG['B11'], _mo, 2):.2f}", st08=f"{_stair(ORIG['B08'], _mo, 2):.2f}")]
for f in prods:
    lab, sig = _label(f)
    s = SR[f]
    ms = np.all([s[b] > 0 for b in _need], axis=0)
    ml = _blk(ms.astype("float32")) > 0.99
    dev = np.mean([np.abs(_blk(s[b]) - ORIG[b])[ml & _mo].mean() for b in ("B04", "B03", "B02", "B08")]) * 1e4
    ref11 = _gauss(ORIG["B11"], sig)
    dd = (_blk(s["B11"]) - ref11)[ml & _mo]
    rows_.append(dict(name=lab.replace("\n", " | "), dev=f"{dev:.0f}", b11_bias=f"{dd.mean() * 1e4:+.0f}",
                      b11_mae=f"{np.abs(dd).mean() * 1e4:.0f}",
                      det=f"{_lap_std(s['B11'], ms) / _lap_std(s['B08'], ms):.2f}",
                      st11=f"{_stair(s['B11'], ms, 8):.2f}", st08=f"{_stair(s['B08'], ms, 8):.2f}"))
_bar.update(1)

print(f"\nArea compared: {M} x {M} px at 10 m ({M / 100:.1f} x {M / 100:.1f} km) | all values in reflectance x 10,000")
print(f"{'product':72s} {'vis/NIR MAE':>11s} {'B11 bias':>9s} {'B11 MAE':>8s} {'B11/B08 detail':>15s} {'grid B11':>9s} {'grid B08':>9s}")
for r in rows_:
    print(f"{r['name']:72s} {r['dev']:>11s} {r['b11_bias']:>9s} {r['b11_mae']:>8s} {r['det']:>15s} {r['st11']:>9s} {r['st08']:>9s}")
print("vis/NIR MAE : mean |4x4 block mean of the SR band - original band|  (radiometric fidelity, B04 B03 B02 B08)")
print("B11 bias/MAE: same test for B11 against the original B11 (smoothed with the same sigma when de-blocked)")
print("B11/B08 detail: fine-detail energy of B11 relative to B08 (Laplacian std). Real SWIR is smoother than NIR:")
print("                a value well above the original suggests detail injected into B11, not measured.")
print("grid        : ~1.0 = no 20 m block pattern; clearly > 1.3 = visible 2x2 / 8x8 px staircase (capped at 99)")
print("              the original's B11/B08 detail is inflated by the block edges themselves")

if len(prods) > 1:
    print("\nDifference between products (mean |A - B| / mean original reflectance):")
    for f in prods[1:]:
        vis = np.mean([np.abs(SR[prods[0]][b] - SR[f][b]).mean() / ORIG[b][_mo].mean() for b in ("B04", "B03", "B02")])
        nir = np.abs(SR[prods[0]]["B08"] - SR[f]["B08"]).mean() / ORIG["B08"][_mo].mean()
        b11 = np.abs(SR[prods[0]]["B11"] - SR[f]["B11"]).mean() / ORIG["B11"][_mo].mean()
        verdict = ("B11 is what differs" if b11 > 2 * max(vis, nir) else "bands differ by similar amounts")
        _short = lambda g: f"{TAGS[g]['MODE']} (deblock {float(TAGS[g].get('DEBLOCK_20M', 0) or 0):g})"
        print(f"  {_short(prods[0])} vs {_short(f):24s}"
              f" visible {100 * vis:5.1f}% | B08 {100 * nir:5.1f}% | B11 {100 * b11:5.1f}%  -> {verdict}")

_bar.set_description("Comparison: drawing")
o = (M - Z) // 2
zl, zs = (slice(o, o + Z), slice(o, o + Z)), (slice(4 * o, 4 * (o + Z)), slice(4 * o, 4 * (o + Z)))


def _rng(a, lo=2, hi=98):
    v = a[a > 0]
    return (np.percentile(v, lo), np.percentile(v, hi)) if v.size else (0.0, 1.0)


_vlo = min(_rng(ORIG[b])[0] for b in ("B04", "B03", "B02"))
_vhi = max(_rng(ORIG[b])[1] for b in ("B04", "B03", "B02"))
_pr = {b: _rng(ORIG[b]) for b in ("B11", "B08", "B04")}


def _st(a, lo, hi):
    return np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)


def _vis(d, sl):
    return np.dstack([_st(d[b][sl], _vlo, _vhi) for b in ("B04", "B03", "B02")])


def _pro(d, sl):
    return np.dstack([_st(d[b][sl], *_pr[b]) for b in ("B11", "B08", "B04")])


cols = [("Original 10 m\n(BDC values)", ORIG, zl)] + [(_label(f)[0], SR[f], zs) for f in prods]
fig, ax = plt.subplots(3, len(cols), figsize=(4.4 * len(cols), 13), squeeze=False)
for j, (title, dat, sl) in enumerate(cols):
    ax[0, j].imshow(_vis(dat, sl), interpolation="nearest")
    ax[1, j].imshow(_pro(dat, sl), interpolation="nearest")
    ax[2, j].imshow(_st(dat["B11"][sl], *_pr["B11"]), cmap="gray", interpolation="nearest")
    ax[0, j].set_title(title, fontsize=10)
for i, t in enumerate(("Visible  R=B04 G=B03 B=B02", "PRODES  R=B11 G=B08 B=B04", "B11 alone")):
    ax[i, 0].set_ylabel(t, fontsize=11)
for a_ in ax.ravel():
    a_.set_xticks([])
    a_.set_yticks([])
fig.suptitle(f"{TILE} {DATE} — same colour stretch in every panel (taken from the original 10 m)", fontsize=12)
plt.tight_layout()
_bar.update(1)
_bar.set_description("Comparison ready")
_bar.close()
plt.show()
plt.show()
