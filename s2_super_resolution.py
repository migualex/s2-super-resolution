# =====================================================================
# LDSR-S2 super-resolution of Sentinel-2 imagery (BDC S2-16D-2 mosaic)
# =====================================================================

# %% BLOCK 1 — PARAMETERS
TILE = "012014"
DATE = "2026-07-12"

# Working directory for downloaded cubes, models and results (relative to the notebook location)
ROOT    = "rascunho/biomasbr-amazonia/super_resolution"
OUT_DIR = ROOT + "/cube_resolution"            # per-run logs are written to OUT_DIR/logs

# Input source:
#   "bdc16d"  download the BDC S2-16D-2 mosaic to DOWNLOAD_DIR (recommended)
#   "local"   existing files matching LOCAL_PATTERN
#   "16d"     read the BDC mosaic remotely
#   "l2a"     original Sentinel-2 L2A scene
SOURCE        = "bdc16d"
DOWNLOAD_DIR  = ROOT + "/cube_bdc_16d"
LOCAL_PATTERN = ROOT + "/cube/SENTINEL-2_MSI_{tile}_{band}_{date}.tif"
LOCAL_KIND    = "16d"                          # "16d" | "l2a"

# Region of interest. ROI_LAT/ROI_LON = None processes the whole tile (~9,000 windows).
# Otherwise a square of ROI_SIZE_KM is centred on the point (10 km ≈ 100 windows; recommended for "combined").
ROI_LAT     = None                             # e.g. -3.4653
ROI_LON     = None                             # e.g. -62.2159
ROI_SIZE_KM = 10

# Inference
STEPS       = 25                               # DDIM steps. 25 = same quality as 50 (difference = seed noise), 1.6x faster; 15 degrades
UNCERTAINTY = 0                                # 0 = off; N > 3 adds an uncertainty map (N times slower)
PRECISION   = "auto"                           # "auto" | "fp32" | "bf16"
BATCH_SIZE  = 0                                # 0 = auto
OVERLAP     = 16                               # window overlap, 10 m pixels

# Super-resolution mode (all modes write the PRODES composite R=B11 G=B08 B=B04 at 2.5 m):
#   "fast"      LDSR-S2 on B04,B03,B02,B08; B11 sharpened with the LDSR-S2 output   (~1 s/window)
#   "combined"  OpenSR LDSRS2-SEN2SR, 10 bands, B11 from SEN2SR                    (~3 s/window, MIG 2g.24gb)
#   "separate"  LDSR-S2 plus SEN2SR (SEN2SR_DIR) for B11
# B04/B08 (G and B of the PRODES composite) come from the same LDSR-S2 diffusion in "fast" and "combined",
# so the two modes differ only in B11. B11 is a 20 m band: its 2.5 m detail is always borrowed from B04/B03/B02/B08.
MODE         = "fast"
COMBINED_DIR = ROOT + "/model/LDSRS2-SEN2SR"   # downloaded on first use (~1.5 GB)
WITH_B11     = True                            # "separate" only
SEN2SR_DIR   = ROOT + "/model/SEN2SR_main"     # "separate" only; Mamba-free alternative: ROOT + "/model/SEN2SRLite"

# Optional Gaussian sigma (10 m pixels) applied to the 20 m bands before super-resolution (file name gets "_db<sigma>").
# Keep 0 for BDC S2-16D-2: its 20 m bands are already interpolated to 10 m (no 2x2 blocks), so any blur removes real
# detail (sigma 1 removes ~70 % of the contrast at the 20 m scale). Use > 0 only for inputs with 2x2 blocks.
DEBLOCK_20M = 0.0

# Load torch from the conda environment (CUDA 12, required by mamba_ssm). Restart the kernel after changing.
USE_ENV_TORCH = True

MODEL_DIR  = ROOT + "/model/LDSR_S2"
CONFIG     = None                              # config_10m.yaml; None = download
COMPRESS   = "zstd"                            # "zstd" | "deflate" | "lzw" | "none"
OVERWRITE  = False
PROBE_ONLY = False                             # hardware and batch-size check only


# %% BLOCK 2 — RUN
import os, sys, site, importlib, importlib.util, importlib.machinery, subprocess, time, warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")
_T_CELL0 = time.perf_counter()

if globals().get("USE_ENV_TORCH", True):
    _user_sp = site.getusersitepackages()
    _env_sp = [p for p in sys.path if p.startswith(sys.prefix) and p.rstrip("/").endswith("site-packages")]
    _t = sys.modules.get("torch")
    if _t is not None and not str(getattr(_t, "__file__", "")).startswith(sys.prefix):
        raise RuntimeError("torch from ~/.local is already loaded in this kernel. Restart the kernel and run blocks 1 and 2.")
    if _env_sp and _user_sp in sys.path:
        for _p in _env_sp:
            sys.path.remove(_p)
        _i = sys.path.index(_user_sp)
        for _p in reversed(_env_sp):
            sys.path.insert(_i, _p)
        importlib.invalidate_caches()
        print("Using conda env packages before ~/.local:", ", ".join(_env_sp))


def _missing(mod):
    if mod in sys.modules:
        return False
    try:
        return importlib.util.find_spec(mod) is None
    except (ValueError, ImportError):
        return False


def _pip(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])


_REQ = {"tqdm": "tqdm", "ipywidgets": "ipywidgets", "rasterio": "rasterio", "affine": "affine",
        "requests": "requests", "omegaconf": "omegaconf", "opensr_model": "opensr-model", "matplotlib": "matplotlib",
        "einops": "einops", "skimage": "scikit-image", "scipy": "scipy", "pytorch_lightning": "pytorch-lightning",
        "huggingface_hub": "huggingface_hub", "importlib_metadata": "importlib-metadata"}
if globals().get("MODE", "fast") in ("combined", "separate"):
    _REQ.update(mlstac="mlstac", sen2sr="sen2sr")
_installed = [pkg for mod, pkg in _REQ.items() if _missing(mod)]
if "tqdm" in _installed:
    _pip("tqdm")
    _installed.remove("tqdm")
from tqdm.auto import tqdm as _tq
for _pkg in _tq(_installed, desc="Installing dependencies", unit="pkg"):
    _pip(_pkg)

import types as _types
import torch
try:
    import pytorch_lightning  # noqa: F401
except Exception as _e:
    # opensr-model only uses pytorch_lightning for a training class; replace it if a version conflict breaks it
    for _k in list(sys.modules):
        if _k.split(".")[0] in ("pytorch_lightning", "lightning_fabric", "lightning", "torchmetrics", "transformers"):
            del sys.modules[_k]
    _stub = _types.ModuleType("pytorch_lightning")
    _stub.LightningModule = torch.nn.Module
    _stub.__spec__ = importlib.machinery.ModuleSpec("pytorch_lightning", None)
    sys.modules["pytorch_lightning"] = _stub
    print(f"[warning] pytorch_lightning unavailable ({type(_e).__name__}); using a minimal stub (inference unaffected).")

_PKG_ALIASES = {"skimage": "scikit-image", "pytorch_lightning": "pytorch-lightning", "importlib_metadata": "importlib-metadata",
                "yaml": "pyyaml", "PIL": "pillow", "sklearn": "scikit-learn", "cv2": "opencv-python-headless"}
_bar = _tq(total=1, desc="Checking opensr-model", unit="check")
for _ in range(10):
    try:
        import opensr_model  # noqa: F401
        break
    except ModuleNotFoundError as _e:
        _name = (_e.name or "").split(".")[0]
        _pkg = _PKG_ALIASES.get(_name, _name)
        _bar.set_description(f"Installing opensr-model dependency: {_pkg}")
        _installed.append(_pkg)
        _pip(_pkg)
        importlib.invalidate_caches()
_bar.update(1)
_bar.set_description("opensr-model OK")
_bar.close()
if _installed:
    print("Installed:", _installed, "-> restart the kernel if anything fails or progress bars do not render.")

import json
import hashlib
import logging
import argparse
from contextlib import nullcontext, ExitStack
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

log = logging.getLogger("ldsr")


def _nvsmi(args):
    try:
        return subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return ""


def _pre_torch_env():
    listing = _nvsmi(["-L"])
    mig = "MIG" in listing or os.environ.get("CUDA_VISIBLE_DEVICES", "").startswith("MIG-")
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
            "garbage_collection_threshold:0.8,max_split_size_mb:512" if mig else "expandable_segments:True")
    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
    return mig, listing


_ib = tqdm(total=2, desc="Initializing: querying GPU", unit="step", leave=True)
_MIG_HINT, _NVSMI_LIST = _pre_torch_env()
_ib.update(1)
_ib.set_description("Initializing: importing libraries")
import torch.nn.functional as F  # noqa: E402


def _use_matching_proj_db():
    """Point PROJ to the database shipped with this environment's rasterio/GDAL.
    A system-wide PROJ_DATA/PROJ_LIB (e.g. /opt/conda/share/proj) with an older proj.db breaks EPSG lookups."""
    cands = []
    spec = importlib.util.find_spec("rasterio")
    if spec and spec.origin:
        cands.append(Path(spec.origin).parent / "proj_data")
    cands.append(Path(sys.prefix) / "share" / "proj")
    spec = importlib.util.find_spec("pyproj")
    if spec and spec.origin:
        cands.append(Path(spec.origin).parent / "proj_dir" / "share" / "proj")
    for c in cands:
        if (c / "proj.db").is_file():
            os.environ["PROJ_DATA"] = os.environ["PROJ_LIB"] = str(c)
            return str(c)
    return None


_use_matching_proj_db()
import rasterio  # noqa: E402
from rasterio.windows import Window  # noqa: E402
from rasterio.enums import Resampling  # noqa: E402
from affine import Affine  # noqa: E402
_ib.update(1)
_ib.set_description("Initialization done")
_ib.close()
_T_MODULE_LOADED = time.perf_counter()

PATCH = 128
LDSR_BANDS = ["B04", "B03", "B02", "B08"]
SEN2SR_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
SEN2SR_B11_IDX = 8
SEN2SR_URL = "https://huggingface.co/tacofoundation/sen2sr/resolve/main/SEN2SR/main/mlm.json"
SEN2SR_LITE_URL = "https://huggingface.co/tacofoundation/sen2sr/resolve/main/SEN2SRLite/main/mlm.json"
LDSR_SEN2SR_URL = "https://huggingface.co/tacofoundation/RS-SR-LTDF/resolve/main/main/mlm.json"
HF_BASE = "https://huggingface.co/simon-donike/RS-SR-LTDF/resolve/main/"
CONFIG_URLS = [
    "https://raw.githubusercontent.com/ESAOpenSR/opensr-model/main/configs/config_10m.yaml",
    "https://raw.githubusercontent.com/ESAOpenSR/opensr-model/main/opensr_model/configs/config_10m.yaml",
    "https://raw.githubusercontent.com/ESAOpenSR/opensr-model/main/resources/configs/config_10m.yaml",
]
SCL_BAD = (0, 1, 3, 8, 9, 10)
PRECISION_NAMES = {None: "fp32 + TF32", torch.bfloat16: "bf16", torch.float16: "fp16"}


# ---------------------------------------------------------------- stages, timing, logging
def fmt_t(s):
    s = int(round(s))
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


class Stages:
    """Stage progress bar with per-stage and total timing."""

    def __init__(self, plan, init_seconds=None, t0=None):
        self.times, self.order = {}, []
        if init_seconds is not None:
            k = "Initialization (dependencies and imports)"
            self.times[k] = init_seconds
            self.order.append(k)
        self.t0 = t0 if t0 is not None else time.perf_counter()
        self.bar = tqdm(total=len(plan), desc="Stages", unit="stage", leave=True)
        self.cur, self._ts = None, 0.0

    def start(self, name):
        self._end()
        self.cur, self._ts = name, time.perf_counter()
        self.bar.set_description(f"Stage: {name}")
        self.bar.refresh()
        log.info("▶ %s", name)

    def _end(self):
        if self.cur is not None:
            if self.cur not in self.times:
                self.order.append(self.cur)
            self.times[self.cur] = self.times.get(self.cur, 0.0) + time.perf_counter() - self._ts
            if self.bar.n < self.bar.total:
                self.bar.update(1)
            self.cur = None

    def finish(self, ok=True):
        self._end()
        if ok:
            self.bar.n = self.bar.total
        self.bar.set_description("Done" if ok else "Interrupted")
        self.bar.refresh()
        self.bar.close()
        total = time.perf_counter() - self.t0
        log.info("=" * 64)
        log.info("PROCESSING TIME%s", "" if ok else " (interrupted)")
        for k in self.order:
            log.info("  %-46s %s", k, fmt_t(self.times[k]))
        log.info("  %-46s %s", "TOTAL", fmt_t(total))
        log.info("=" * 64)
        return total


class _TqdmHandler(logging.Handler):
    def emit(self, record):
        try:
            tqdm.write(self.format(record), file=sys.stdout)
        except Exception:
            self.handleError(record)


class _BufferHandler(logging.Handler):
    """Keeps the log lines until the name of the output image (and so of its log file) is known."""

    def __init__(self):
        super().__init__(logging.DEBUG)
        self.lines = []

    def emit(self, record):
        self.lines.append(self.format(record))


_LOG_FMT = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S")
_LOGBUF = None


def setup_logging():
    global _LOGBUF
    log.setLevel(logging.DEBUG)
    log.propagate = False
    for old in list(log.handlers):
        log.removeHandler(old)
        old.close()
    sh = _TqdmHandler()
    sh.setLevel(logging.INFO)
    _LOGBUF = _BufferHandler()
    for h in (sh, _LOGBUF):
        h.setFormatter(_LOG_FMT)
        log.addHandler(h)


def attach_log_file(path: Path):
    """Send the log to `path` (appended if the run is resumed): first what was buffered, then everything else."""
    global _LOGBUF
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(path, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(_LOG_FMT)
    fh.stream.write(f"\n===== run started {datetime.now():%Y-%m-%d %H:%M:%S} =====\n")
    if _LOGBUF is not None:
        fh.stream.writelines(line + "\n" for line in _LOGBUF.lines)
        log.removeHandler(_LOGBUF)
        _LOGBUF = None
    log.addHandler(fh)


def dump_unattached_log(path: Path):
    """Run failed before the output name was known: keep what was logged anyway."""
    global _LOGBUF
    if _LOGBUF is not None and _LOGBUF.lines:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"\n===== run started {datetime.now():%Y-%m-%d %H:%M:%S} =====\n")
            f.writelines(line + "\n" for line in _LOGBUF.lines)
    if _LOGBUF is not None:
        log.removeHandler(_LOGBUF)
        _LOGBUF = None


def area_label(args):
    if args.center:
        lat, lon = args.center
        return (f"box{args.box_km:g}km-{abs(lat):.4f}{'N' if lat >= 0 else 'S'}"
                f"-{abs(lon):.4f}{'E' if lon >= 0 else 'W'}")
    return "whole-tile"


def run_stem(args, src_kind, combined, fast, uses_b20):
    """Short file name (no extension): {tile}_{date}_{mode}_{area}, plus a marker only when a setting differs from
    the usual one: _db<sigma> (20 m bands blurred), _s<N> (DDIM steps other than 25), _<source> (not bdc16d).
    Everything else is stored in the GeoTIFF tags, the .log and the .report.json with the same name."""
    mode = "combined" if combined else "bicubic" if args.engine == "bicubic" else "fast" if fast else "separate"
    area = ("tile" if not args.center else
            f"{args.box_km:g}km_{abs(args.center[0]):.4f}{'N' if args.center[0] >= 0 else 'S'}"
            f"_{abs(args.center[1]):.4f}{'E' if args.center[1] >= 0 else 'W'}")
    parts = [args.tile, args.date.replace("-", ""), mode, area]
    if uses_b20 and args.deblock_20m > 0:
        parts.append(f"db{args.deblock_20m:g}")
    if args.engine == "ldsr" and args.steps != 25:
        parts.append(f"s{args.steps}")
    if src_kind != "bdc16d":
        parts.append(src_kind.replace("_", "-"))
    return "_".join(parts)


def run_description(args, src_kind, combined, fast, uses_b20):
    """Full description of the run, written to the log and to the GeoTIFF tag RUN."""
    model = ("LDSRS2-SEN2SR official (10 bands)" if combined else "bicubic" if args.engine == "bicubic" else
             "LDSR-S2 + guided B11" if fast else "LDSR-S2 (B11 from SEN2SR)")
    return (f"{model} | source {src_kind} | area {area_label(args)} | DDIM steps {args.steps} | 20 m bands "
            + (f"de-blocked sigma {args.deblock_20m:g}" if uses_b20 and args.deblock_20m > 0 else "raw BDC values"))


# ---------------------------------------------------------------- hardware
@dataclass
class Hardware:
    cuda: bool = False
    name: str = "CPU"
    is_mig: bool = False
    mig_profile: str = ""
    capability: tuple = (0, 0)
    total_mem: int = 0
    free_mem: int = 0
    sm_count: int = 0
    torch_version: str = ""
    cuda_runtime: str = ""
    cudnn: int = 0
    driver: str = ""
    n_visible: int = 0
    arch_ok: bool = True
    bf16_ok: bool = False

    def gb(self, v=None):
        return (self.total_mem if v is None else v) / 2**30


def detect_hardware() -> Hardware:
    import re
    hw = Hardware(torch_version=torch.__version__, cuda_runtime=str(torch.version.cuda or ""))
    if not torch.cuda.is_available():
        return hw
    p = torch.cuda.get_device_properties(torch.cuda.current_device())
    free, total = torch.cuda.mem_get_info()
    hw.cuda, hw.name, hw.capability = True, p.name, (p.major, p.minor)
    hw.total_mem, hw.free_mem, hw.sm_count = total, free, p.multi_processor_count
    hw.n_visible = torch.cuda.device_count()
    hw.is_mig = ("MIG" in p.name.upper()) or _MIG_HINT
    m = re.search(r"MIG\s+(\d+g\.\d+gb)", p.name + " " + _NVSMI_LIST)
    hw.mig_profile = m.group(1) if m else ""
    try:
        hw.cudnn = int(torch.backends.cudnn.version() or 0)
    except Exception:
        pass
    drv = _nvsmi(["--query-gpu=driver_version", "--format=csv,noheader"]).strip().splitlines()
    hw.driver = drv[0] if drv else ""
    arches = torch.cuda.get_arch_list()
    hw.arch_ok = f"sm_{p.major}{p.minor}" in arches or any(a.startswith("compute_") for a in arches) or not arches
    hw.bf16_ok = p.major >= 8 and torch.cuda.is_bf16_supported()
    return hw


def configure_torch(hw: Hardware):
    if not hw.cuda:
        torch.set_num_threads(max(1, os.cpu_count() or 1))
        return
    torch.backends.cudnn.benchmark = True
    mm = torch.backends.cuda.matmul
    if hasattr(mm, "fp32_precision"):          # torch >= 2.9
        try:
            mm.fp32_precision = "tf32"
            torch.backends.cudnn.conv.fp32_precision = "tf32"
        except Exception as e:
            log.debug("fp32_precision: %s", e)
    else:
        mm.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")


def log_hardware(hw: Hardware):
    log.info("=" * 64)
    if hw.cuda:
        log.info("GPU        : %s%s", hw.name, f"  [MIG {hw.mig_profile}]" if hw.is_mig else "")
        log.info("Memory     : %.1f GiB total / %.1f GiB free | SMs: %d | CC %d.%d",
                 hw.gb(), hw.gb(hw.free_mem), hw.sm_count, *hw.capability)
        log.info("Software   : torch %s | CUDA %s | cuDNN %s | driver %s",
                 hw.torch_version, hw.cuda_runtime, hw.cudnn, hw.driver or "?")
        log.info("torch path : %s", Path(torch.__file__).parent)
        log.info("Precision  : TF32 on | bf16 %s", "available" if hw.bf16_ok else "unavailable")
        if not hw.arch_ok:
            log.warning("This torch build may lack kernels for sm_%d%d", *hw.capability)
        if hw.is_mig:
            log.info("MIG: this process sees only its slice (%d visible device).", hw.n_visible)
    else:
        log.warning("CUDA unavailable: running on CPU (very slow for LDSR-S2).")
    log.info("=" * 64)


# ---------------------------------------------------------------- engines
def gpu_hist_match(sr: torch.Tensor, lr: torch.Tensor) -> torch.Tensor:
    """Per-channel histogram matching SR -> LR on the GPU, ignoring nodata (0)."""
    out = sr.clone()
    B, C = sr.shape[:2]
    for b in range(B):
        for c in range(C):
            s, r = sr[b, c].reshape(-1), lr[b, c].reshape(-1)
            sm, rm = s > 0, r > 0
            ns, nr = int(sm.sum()), int(rm.sum())
            if ns == 0 or nr < 16:
                continue
            sv = s[sm]
            rv = torch.sort(r[rm]).values
            order = torch.argsort(sv)
            q = ((torch.arange(ns, device=s.device, dtype=torch.float32) + 0.5) / ns * nr - 0.5).clamp_(0, nr - 1)
            lo = q.floor().long()
            hi = (lo + 1).clamp_(max=nr - 1)
            f = q - lo.float()
            res = torch.empty_like(sv)
            res[order] = rv[lo] * (1 - f) + rv[hi] * f
            out[b, c].reshape(-1)[sm] = res
    return out


class LDSREngine:
    name = "ldsr"
    out_names = ("sr",)

    def __init__(self, model, device, steps=100, hist="gpu", n_var=0, seed=0, amp=None):
        self.model, self.device = model, device
        self.steps, self.hist, self.n_var, self.seed, self.amp = steps, hist, n_var, seed, amp
        self.scale = 4
        if n_var > 1:
            self.out_names = ("sr", "unc")

    def _forward(self, X, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        use_amp = self.amp is not None and self.device.type == "cuda"
        ctx = torch.autocast("cuda", dtype=self.amp) if use_amp else nullcontext()
        with torch.no_grad(), ctx:
            sr = self.model(X, sampling_steps=self.steps, histogram_matching=(self.hist == "cpu" and not use_amp))
        sr = torch.nan_to_num(sr.float(), nan=0.0).clamp_(min=0.0)
        if self.hist == "gpu" or (self.hist == "cpu" and use_amp):
            sr = gpu_hist_match(sr, X).clamp_(min=0.0)
        return sr

    def predict(self, X):
        if self.n_var <= 1:
            return {"sr": self._forward(X)}
        # own implementation: opensr-model 1.0.3 uncertainty_map() passes an invalid argument to forward()
        first, s, ss = None, 0.0, 0.0
        for k in range(self.n_var):
            v = self._forward(X, seed=self.seed + 1000 + k)
            first = v if first is None else first
            s, ss = s + v, ss + v * v
        var = ((ss - s * s / self.n_var) / (self.n_var - 1)).clamp_(min=0)
        return {"sr": first, "unc": (2.0 * var.sqrt()).mean(dim=1, keepdim=True)}


_GAUSS = {}


def soften_20m(X, idx, sigma):
    """Optional Gaussian smoothing (sigma in 10 m px) of the channels `idx`, nodata (0) excluded by normalised
    convolution. Only useful for inputs whose 20 m bands were replicated in 2x2 blocks (nearest neighbour).
    The BDC S2-16D-2 20 m bands are already smooth (interpolated), so this only removes real detail there."""
    if not sigma or sigma <= 0:
        return X
    idx = list(idx)
    sub = X[:, idx].float()
    m = (sub > 0).float()
    out = torch.where(m > 0, _gauss_blur(sub * m, sigma) / _gauss_blur(m, sigma).clamp_min(1e-6), torch.zeros_like(sub))
    X = X.clone()
    X[:, idx] = out.to(X.dtype)
    return X


B11_SIGMA = 1.0                        # effective resolution of the BDC B11 on the 10 m grid (Gaussian sigma, 10 m px)


def _gauss_blur(t, sigma):
    """Separable Gaussian blur (sigma in pixels), reflect padding; channels blurred independently."""
    key = (round(float(sigma), 3), str(t.device))
    if key not in _GAUSS:
        r = max(1, int(np.ceil(3 * sigma)))
        x = torch.arange(-r, r + 1, dtype=torch.float32, device=t.device)
        k = torch.exp(-0.5 * (x / sigma) ** 2)
        _GAUSS[key] = (k / k.sum(), r)
    k, r = _GAUSS[key]
    c = t.shape[1]
    t = F.pad(t, (r, r, r, r), mode="reflect")
    t = F.conv2d(t, k.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
    return F.conv2d(t, k.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)


def guided_b11(lr4, lr11, sr4, s=4, ridge=1e-2, sigma=B11_SIGMA):
    """B11 at 2.5 m via MTF-matched regression detail injection (pansharpening).
    The BDC B11 is a 20 m band resampled to 10 m: it has no detail finer than ~20 m. So
    1) the B04,B03,B02,B08 regressors are first low-passed to that resolution (sigma), and per window
       B11 ~ c0 + sum(ci * LP(band_i)) is fitted (ridge relative to the Gram diagonal: B02/B03/B04 are collinear);
    2) B11_sr = bicubic(B11) + sum(ci * (SR_i - bicubic(LP(band_i)))): everything finer than 20 m is injected,
       including the 20 -> 10 m scale that a plain 10 m fit leaves empty;
    3) consistency: the SR B11, degraded back to 10 m and to the B11 resolution, matches the input B11.
    Wald-protocol test on BDC data (x4): RMSE -29 % and detail correlation 0.37 -> 0.53 against the previous
    version with DEBLOCK_20M = 1."""
    lr4, lr11, sr4 = lr4.float(), lr11.float(), sr4.float()
    B = lr4.shape[0]
    valid = (lr4 > 0).all(1, keepdim=True) & (lr11 > 0)
    vf = valid.float()
    # normalised convolution so nodata (0) does not leak into the low-pass
    lp4 = _gauss_blur(lr4 * vf, sigma) / _gauss_blur(vf, sigma).clamp_min(1e-6)
    A = torch.cat([torch.ones_like(lr11), lp4], 1)
    coef = torch.zeros(B, 5, device=lr4.device)
    eye = torch.eye(5, device=lr4.device)
    for b in range(B):
        m = valid[b, 0].reshape(-1)
        if int(m.sum()) < 64:
            continue
        Ab = A[b].reshape(5, -1)[:, m].T
        yb = lr11[b].reshape(-1)[m]
        G = Ab.T @ Ab
        coef[b] = torch.linalg.solve(G + eye * (ridge * G.diagonal().mean()), Ab.T @ yb)
    ci = coef[:, 1:, None, None]

    def up(t):
        return F.interpolate(t, scale_factor=s, mode="bicubic", align_corners=False)

    fill = (lr11 * vf).sum((2, 3), keepdim=True) / vf.sum((2, 3), keepdim=True).clamp_min(1)
    lr11f = torch.where(valid, lr11, fill.expand_as(lr11))     # nodata -> window mean (masked out at the end)
    out = up(lr11f) + (ci * (sr4 - up(lp4))).sum(1, keepdim=True)
    corr = lr11f - _gauss_blur(F.avg_pool2d(out, s), sigma)
    out = out + up(corr)
    return out.clamp_(min=0.0) * F.interpolate(vf, scale_factor=s, mode="nearest")


class LDSRGuidedB11Engine:
    """LDSR-S2 (B04,B03,B02,B08) + guided B11. Input order: B04,B03,B02,B08,B11."""
    name = "ldsr+b11"
    scale = 4

    def __init__(self, ldsr, deblock=0.0):
        self.ldsr, self.device, self.amp, self.deblock = ldsr, ldsr.device, None, deblock
        self.out_names = ldsr.out_names

    def predict(self, X):
        x4 = X[:, :4].contiguous()
        x11 = soften_20m(X[:, 4:5], (0,), self.deblock).contiguous()
        out = self.ldsr.predict(x4)
        out["sr"] = torch.cat([out["sr"], guided_b11(x4, x11, out["sr"])], 1)
        return out


class BicubicEngine:
    name = "bicubic"
    out_names = ("sr",)
    scale = 4
    amp = None

    def __init__(self, device):
        self.device = device

    def predict(self, X):
        y = F.interpolate(X, scale_factor=4, mode="bicubic", align_corners=False).clamp_(min=0)
        return {"sr": y * F.interpolate((X > 0).float(), scale_factor=4, mode="nearest")}


def _is_oom(e):
    """On MIG, out-of-memory can surface as 'NVML_SUCCESS == r INTERNAL ASSERT FAILED'."""
    if isinstance(e, torch.cuda.OutOfMemoryError):
        return True
    t = str(e)
    return "out of memory" in t.lower() or "CUDACachingAllocator" in t or "NVML_SUCCESS" in t


def _all_modules(root):
    """All nn.Modules reachable from root, including ones stored outside registered submodules."""
    out, seen = [], set()

    def visit(obj, depth):
        if id(obj) in seen or depth > 8 or torch.is_tensor(obj) or isinstance(obj, (str, bytes, int, float, type)):
            return
        seen.add(id(obj))
        if isinstance(obj, torch.nn.Module):
            out.append(obj)
            ch = list(vars(obj).get("_modules", {}).values()) + [v for k, v in vars(obj).items() if k != "_modules"]
        elif isinstance(obj, (list, tuple, set)):
            ch = list(obj)
        elif isinstance(obj, dict):
            ch = list(obj.values())
        elif hasattr(obj, "__dict__"):
            ch = list(vars(obj).values())
        else:
            return
        for c in ch:
            visit(c, depth + 1)

    visit(root, 0)
    return out


def force_device(root, device):
    """Move every sub-network of an mlstac model to the GPU (including ones .to() cannot reach)
    and fix `device` attributes used for tensor placement. Returns (#params on GPU, #params on CPU)."""
    seen = set()

    def visit(obj, depth):
        if id(obj) in seen or depth > 8 or torch.is_tensor(obj) or isinstance(obj, (str, bytes, int, float, type, _types.ModuleType)):
            return
        seen.add(id(obj))
        if isinstance(obj, torch.nn.Module):
            obj.to(device)
            for m in obj.modules():
                if "device" in vars(m):
                    try:
                        m.device = device
                    except Exception:
                        pass
            children = list(vars(obj).values())
        elif isinstance(obj, (list, tuple, set)):
            children = list(obj)
        elif isinstance(obj, dict):
            children = list(obj.values())
        elif hasattr(obj, "__dict__"):
            if isinstance(vars(obj).get("device"), (str, torch.device)):
                try:
                    obj.device = device
                except Exception:
                    pass
            children = list(vars(obj).values())
        else:
            return
        for c in children:
            visit(c, depth + 1)

    visit(root, 0)
    n_gpu = n_cpu = 0
    for m in _all_modules(root):
        for p in m.parameters(recurse=False):
            if p.device.type == "cuda":
                n_gpu += p.numel()
            else:
                n_cpu += p.numel()
    return n_gpu, n_cpu


def patch_ldsr_classes(model, state):
    """Intercept the SRLatentDiffusion used inside the combined mlstac model to control the DDIM steps
    (state['steps']), bf16 autocast (state['amp']) and GPU histogram matching (state['hist'])."""
    import inspect
    n = 0
    for m in _all_modules(model):
        cls = type(m)
        if cls.__name__ != "SRLatentDiffusion":
            continue
        n += 1
        cls._ldsr_state = state
        if getattr(cls, "_ldsr_patched", False):
            continue
        orig = cls.forward
        sig = inspect.signature(orig)

        def fwd(self, *args, _orig=orig, _sig=sig, **kwargs):
            st_ = type(self)._ldsr_state
            b = _sig.bind(self, *args, **kwargs)
            b.apply_defaults()
            A = b.arguments
            for k in ("sampling_steps", "custom_steps"):
                if st_.get("steps") and k in A:
                    A[k] = st_["steps"]
            want_hm = bool(A.get("histogram_matching", True))
            gpu_hm = st_.get("hist", "gpu") == "gpu" and "histogram_matching" in A
            if gpu_hm:
                A["histogram_matching"] = False
            X = A.get("X", args[0] if args else None)
            amp = st_.get("amp")
            use_amp = amp is not None and X is not None and X.is_cuda
            t0 = time.perf_counter()
            with torch.autocast("cuda", dtype=amp) if use_amp else nullcontext():
                out = _orig(**A)
            if X is not None and X.is_cuda:
                torch.cuda.synchronize()
            st_["t_ldsr"] = st_.get("t_ldsr", 0.0) + time.perf_counter() - t0
            st_["dev_ldsr"] = str(X.device) if X is not None else "?"
            if not torch.is_tensor(out):
                return out
            out = torch.nan_to_num(out.float(), nan=0.0).clamp_(min=0.0)
            if gpu_hm and want_hm and X is not None and X.shape[1] == out.shape[1]:
                out = gpu_hist_match(out, X.float()).clamp_(min=0.0)
            return out

        cls.forward = fwd
        cls._ldsr_patched = True
    return n


def patch_swin_masks(model):
    """Swin2SR (sen2sr) rebuilds the shifted-window attention mask ON THE CPU in every block of every forward
    whenever the input size differs from the training img_size (always the case here: 128 px into the 64 px f2
    model, 512 px into the 128 px f4 model). Caching the mask on the GPU gives identical output and was measured
    at 25 s -> 0.16 s (f2) and 45 s -> 2.1 s (f4) per window on an H100 MIG 2g.24gb slice."""
    n = 0
    for m in _all_modules(model):
        cls = type(m)
        if not hasattr(cls, "calculate_mask") or getattr(cls, "_mask_cached", False):
            continue
        orig = cls.calculate_mask

        def calculate_mask(self, x_size, _orig=orig):
            dev = next(self.parameters()).device
            cache = _SWIN_MASKS.setdefault(id(type(self)), {})
            k = (tuple(int(v) for v in x_size), self.window_size, self.shift_size, str(dev))
            if k not in cache:
                cache[k] = _orig(self, x_size).to(dev)
            return cache[k]

        cls.calculate_mask = calculate_mask
        cls._mask_cached = True
        n += 1
    return n


_SWIN_MASKS = {}


def _lightning_stub_if_broken():
    try:
        import pytorch_lightning  # noqa: F401
        return False
    except Exception as e:
        for k in list(sys.modules):
            if k.split(".")[0] in ("pytorch_lightning", "lightning_fabric", "lightning", "torchmetrics", "transformers"):
                del sys.modules[k]
        stub = _types.ModuleType("pytorch_lightning")
        stub.__spec__ = importlib.machinery.ModuleSpec("pytorch_lightning", None)
        stub.LightningModule = torch.nn.Module
        sys.modules["pytorch_lightning"] = stub
        print(f"[warning] pytorch_lightning unavailable ({type(e).__name__}); using a minimal stub (inference unaffected).")
        return True


class Sen2SREngine:
    """10-band mlstac models: SEN2SR (B11 only) or the combined LDSRS2-SEN2SR (all bands)."""
    name = "sen2sr"
    out_names = ("sr",)
    _no_batch = False
    deblock = 0.0                      # sigma for soften_20m on the 20 m channels (set by main)
    _B20 = (3, 4, 5, 7, 8, 9)          # B05,B06,B07,B8A,B11,B12 in SEN2SR_BANDS order

    def __init__(self, model_dir, device, out_idx=(SEN2SR_B11_IDX,), url=None, label="SEN2SR", ldsr_state=None):
        import mlstac
        _lightning_stub_if_broken()
        model_dir = Path(model_dir)
        bar = tqdm(total=3, desc=f"{label}: checking model", unit="step", leave=True)
        if not (model_dir / "mlm.json").exists():
            bar.set_description(f"{label}: downloading model")
            url = url or (SEN2SR_LITE_URL if "lite" in model_dir.name.lower() else SEN2SR_URL)
            mlstac.download(file=url, output_dir=str(model_dir))
        bar.update(1)
        bar.set_description(f"{label}: loading on GPU")
        try:
            self.model = mlstac.load(str(model_dir)).compiled_model(device=device).to(device)
        except Exception as e:
            bar.close()
            if "mamba" in str(e).lower():
                sys.exit(f"The SEN2SR model in {model_dir} uses Mamba and mamba_ssm failed to load with torch "
                         f"{torch.__version__} (CUDA {torch.version.cuda}). Set USE_ENV_TORCH = True and restart "
                         f"the kernel, or use SEN2SR_DIR = 'model/SEN2SRLite'. Details: {str(e)[:200]}")
            raise
        self.device, self.amp = device, None
        self.out_idx = list(out_idx) if out_idx is not None else None
        bar.update(1)
        if device.type == "cuda":
            n_gpu, n_cpu = force_device(self.model, device)
            log.info("Model parameters: %.1f M on GPU | %.1f M on CPU", n_gpu / 1e6, n_cpu / 1e6)
            if n_cpu:
                log.warning("Some parameters are still on the CPU: that part of the model will be slow.")
        if patch_swin_masks(self.model):
            log.info("Swin2SR attention masks cached on %s (avoids rebuilding them on the CPU every forward).", device)
        self.ldsr_state, self.n_ldsr, self._calls = ldsr_state, 0, 0
        if ldsr_state is not None:
            self.n_ldsr = patch_ldsr_classes(self.model, ldsr_state)
            if self.n_ldsr:
                log.info("Internal LDSR-S2 found: DDIM steps = %s | histogram matching on %s",
                         ldsr_state.get("steps") or "model default", "GPU" if ldsr_state.get("hist") == "gpu" else "CPU")
            else:
                log.warning("Internal LDSR-S2 not found: STEPS and PRECISION have no effect on this model.")
        bar.set_description(f"{label}: initial test")
        steps0 = ldsr_state.get("steps") if ldsr_state else None
        if ldsr_state is not None:
            ldsr_state["steps"] = 2
        with torch.no_grad():
            o = self.model((torch.rand(1, 10, PATCH, PATCH, device=device) * 0.3).clamp_(min=1e-3))
        if ldsr_state is not None:
            ldsr_state["steps"] = steps0
        self.scale = o.shape[-1] // PATCH
        if self.out_idx is None:
            if o.shape[1] != 10:
                raise RuntimeError(f"Unexpected {label} output: {tuple(o.shape)} (expected 10 bands)")
            self.out_idx = list(range(10))
        elif o.shape[1] == 6:          # RSWIR variant (B05,B06,B07,B8A,B11,B12)
            self.out_idx = [4]
        elif o.shape[1] != 10:
            raise RuntimeError(f"Unexpected {label} output: {tuple(o.shape)}")
        bar.update(1)
        bar.set_description(f"{label} ready (x{self.scale}, {o.shape[1]} bands)")
        bar.close()

    def predict(self, X):
        st_ = self.ldsr_state
        t0 = time.perf_counter()
        if st_ is not None:
            st_["t_ldsr"] = 0.0
        out = self._predict(X)
        if st_ is not None and self._calls < 3 and self.device.type == "cuda":
            torch.cuda.synchronize()
            tot, tl = time.perf_counter() - t0, st_.get("t_ldsr", 0.0)
            log.info("Batch time (%d window(s)): total %.1f s | LDSR-S2 %.1f s (%s, %s steps) | SEN2SR and rest %.1f s",
                     X.shape[0], tot, tl, st_.get("dev_ldsr", "?"), st_.get("steps"), tot - tl)
            self._calls += 1
        return out

    def _run_single(self, X):
        return torch.cat([self.model(X[i:i + 1]) for i in range(X.shape[0])], 0)

    def _predict(self, X):
        X = soften_20m(X, self._B20, self.deblock) if X.shape[1] == 10 else X
        with torch.no_grad():
            if self._no_batch or X.shape[0] == 1:
                o = self._run_single(X)
            else:
                try:
                    o = self.model(X)
                except Exception as e:
                    if _is_oom(e):
                        torch.cuda.empty_cache()
                        raise torch.cuda.OutOfMemoryError(str(e)[:200])
                    log.info("Model does not accept batch > 1 (%s); processing one window at a time.", str(e)[:80])
                    self._no_batch = True
                    o = self._run_single(X)
        o = torch.nan_to_num(o.float(), nan=0.0).clamp_(min=0.0)[:, self.out_idx]
        return {"sr": o * F.interpolate((X[:, :1] > 0).float(), scale_factor=self.scale, mode="nearest")}


def _download(urls, dest: Path):
    import requests
    last = None
    for u in ([urls] if isinstance(urls, str) else urls):
        try:
            with requests.get(u, stream=True, timeout=120) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0)) or None
                tmp = dest.with_suffix(dest.suffix + ".dl")
                with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024,
                                                desc=f"Downloading {dest.name}", leave=True) as bar:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                        bar.update(len(chunk))
                tmp.replace(dest)
                return
        except Exception as e:
            last = e
    raise RuntimeError(f"Download failed for {urls}: {last}")


def build_ldsr(args, device):
    bar = tqdm(total=4, desc="LDSR-S2: importing opensr-model", unit="step", leave=True)
    _lightning_stub_if_broken()
    try:
        import opensr_model
        from omegaconf import OmegaConf
    except ImportError as e:
        sys.exit(f"Failed to import opensr-model: {e!r}")
    bar.update(1)
    mdir = Path(args.model_dir)
    mdir.mkdir(parents=True, exist_ok=True)
    cfg_path = Path(args.config) if args.config else mdir / "config_10m.yaml"
    bar.set_description("LDSR-S2: configuration")
    if not cfg_path.exists():
        try:
            _download(CONFIG_URLS, cfg_path)
        except RuntimeError as e:
            sys.exit(f"{e}\nDownload config_10m.yaml from ESAOpenSR/opensr-model and set CONFIG.")
    config = OmegaConf.load(cfg_path)
    ckpt = mdir / str(config.ckpt_version)
    if not ckpt.exists():
        bar.set_description("LDSR-S2: downloading weights")
        _download(HF_BASE + str(config.ckpt_version), ckpt)
    bar.update(1)
    bar.set_description("LDSR-S2: building network")
    model = opensr_model.SRLatentDiffusion(config, device=device)
    bar.update(1)
    bar.set_description("LDSR-S2: loading weights")
    # weights_only=False: torch >= 2.6 rejects Lightning checkpoints by default (official HuggingFace file)
    sd = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    model.model.load_state_dict({k: v for k, v in sd.items() if "loss" not in k}, strict=True)
    model = model.to(device).eval()
    del sd
    bar.update(1)
    bar.set_description("LDSR-S2 ready")
    bar.close()
    try:
        from importlib.metadata import version
        ver = version("opensr-model")
    except Exception:
        ver = "?"
    log.info("opensr-model %s | ckpt %s", ver, config.ckpt_version)
    return LDSREngine(model, device, steps=args.steps, hist=args.hist_match, n_var=args.uncertainty, seed=args.seed)


# ---------------------------------------------------------------- auto-tuning
def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def _batch_of(sample, b, dev):
    return torch.from_numpy(np.repeat(sample[None], b, axis=0)).to(dev)


def _requested_precision(hw, requested):
    """Returns (decided, dtype). decided=False means an automatic test is needed."""
    if not hw.cuda or requested in ("fp32", "tf32"):
        return True, None
    if requested == "bf16":
        return True, torch.bfloat16 if hw.bf16_ok else None
    if requested == "fp16":
        return True, torch.float16
    return (not hw.bf16_ok), None


def _compare_precision(run, set_amp, set_steps, steps0, dev):
    """Runs fp32 vs bf16; adopts bf16 only if the error is negligible and it is clearly faster."""
    bar = tqdm(total=4, desc="Testing precision", unit="test", leave=True)
    try:
        res = {}
        for tag, amp in (("fp32", None), ("bf16", torch.bfloat16)):
            set_amp(amp)
            bar.set_description(f"Testing precision: warming up {tag}")
            set_steps(2)
            run(7)
            set_steps(steps0)
            bar.update(1)
            bar.set_description(f"Testing precision: timing {tag}")
            _sync(dev)
            t0 = time.perf_counter()
            y = run(1234)
            _sync(dev)
            res[tag] = (y, time.perf_counter() - t0)
            bar.update(1)
        mae = (res["fp32"][0] - res["bf16"][0]).abs().mean().item()
        speed = res["fp32"][1] / max(res["bf16"][1], 1e-9)
        ok = mae <= 0.003 and speed >= 1.15
        log.info("Precision: bf16 vs fp32 error = %.5f (refl.) | speed-up x%.2f | 1 window fp32 = %.1f s -> %s",
                 mae, speed, res["fp32"][1], "bf16" if ok else "fp32+TF32")
        return torch.bfloat16 if ok else None
    except Exception as e:
        log.warning("bf16 test failed (%s); using fp32+TF32.", e)
        return None
    finally:
        set_amp(None)
        set_steps(steps0)
        bar.close()
        torch.cuda.empty_cache()


def choose_precision(engine, sample, hw, requested):
    if not isinstance(engine, LDSREngine):
        return None
    decided, dtype = _requested_precision(hw, requested)
    if decided:
        return dtype
    X = _batch_of(sample, 2, engine.device)
    return _compare_precision(lambda seed: engine._forward(X, seed=seed),
                              lambda a: setattr(engine, "amp", a), lambda s: setattr(engine, "steps", s),
                              engine.steps, engine.device)


def choose_precision_combined(eng, sample, hw, requested):
    st_ = eng.ldsr_state
    if st_ is None or not eng.n_ldsr:
        return None
    decided, dtype = _requested_precision(hw, requested)
    if decided:
        return dtype
    X = _batch_of(sample, 1, eng.device)

    def run(seed):
        torch.manual_seed(seed)
        return eng.predict(X)["sr"]

    return _compare_precision(run, lambda a: st_.__setitem__("amp", a), lambda s: st_.__setitem__("steps", s),
                              st_.get("steps"), eng.device)


def tune_batch(engine, sample, hw, max_batch, mem_frac, cands=None, max_single_s=None):
    """Grow the batch while measuring memory and throughput; stop at the memory limit, OOM or < 5 % gain."""
    if not hw.cuda:
        return 1
    dev = engine.device
    saved = getattr(engine, "steps", None), getattr(engine, "n_var", None)
    if isinstance(engine, LDSREngine):
        engine.steps, engine.n_var = min(engine.steps, 4), 0
    best_b, best_thr = 1, 0.0
    cands = [x for x in (cands or (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128)) if x <= max_batch]
    bar = tqdm(cands, desc="Tuning batch size", unit="test", leave=True)
    try:
        for b in bar:
            bar.set_postfix_str(f"testing batch {b}")
            try:
                X = _batch_of(sample, b, dev)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                engine.predict(X)
                _sync(dev)
                t0 = time.perf_counter()
                engine.predict(X)
                _sync(dev)
                dt = time.perf_counter() - t0
                peak = torch.cuda.max_memory_allocated()
            except Exception as e:
                if not _is_oom(e):
                    raise
                torch.cuda.empty_cache()
                log.info("  batch %3d: out of memory", b)
                break
            if getattr(engine, "_no_batch", False):
                best_b = 1
                break
            if max_single_s and b == 1 and dt > max_single_s:
                log.info("  1 window took %.1f s: larger batches are not worth tuning; using batch 1.", dt)
                break
            thr, frac = b / dt, peak / hw.total_mem
            log.debug("  batch %3d: %.1f windows/s (reduced steps) | peak %.1f GiB (%.0f%%)", b, thr, peak / 2**30, 100 * frac)
            bar.set_postfix_str(f"batch {b}: {peak / 2**30:.1f} GiB ({100 * frac:.0f}%)")
            if frac > mem_frac:
                break
            if thr > best_thr * 1.05:
                best_b, best_thr = b, thr
            else:
                break
    finally:
        bar.set_description(f"Batch size: {best_b}")
        bar.close()
        if saved[0] is not None:
            engine.steps, engine.n_var = saved
        torch.cuda.empty_cache()
    return best_b


class BatchRunner:
    """Runs batches, halving the batch size automatically on out-of-memory."""

    def __init__(self, engine, bs):
        self.engine, self.bs = engine, max(1, bs)

    def __call__(self, arr: np.ndarray):
        try:
            X = torch.from_numpy(arr).to(self.engine.device, non_blocking=True)
            return {k: v.float().cpu().numpy() for k, v in self.engine.predict(X).items()}
        except Exception as e:
            if not _is_oom(e):
                raise
            torch.cuda.empty_cache()
            if arr.shape[0] == 1:
                raise RuntimeError("Out of memory with batch=1: lower STEPS/UNCERTAINTY or free the GPU.")
            self.bs = max(1, arr.shape[0] // 2)
            log.warning("Out of memory: batch reduced to %d", self.bs)
            h = arr.shape[0] // 2
            a, b = self(arr[:h]), self(arr[h:])
            return {k: np.concatenate([a[k], b[k]], 0) for k in a}


# ---------------------------------------------------------------- data sources
@dataclass
class Source:
    kind: str
    label: str
    hrefs: dict
    scl: str | None = None
    provenance: str | None = None
    clearob: str | None = None


def _vsi(h):
    return "/vsicurl/" + h if h.startswith("http") else h


def _tok(h, token):
    if token and h.startswith("http") and "access_token" not in h:
        return h + ("&" if "?" in h else "?") + "access_token=" + token
    return h


def _asset_lookup(item, name):
    for k, v in item["assets"].items():
        if k.upper() == name.upper():
            return v["href"]
    return None


def _item_hrefs(item, bands, token):
    hrefs = {}
    for b in bands:
        h = _asset_lookup(item, b)
        if h is None:
            raise RuntimeError(f"Band {b} missing in {item['id']}; assets: {list(item['assets'])}")
        hrefs[b] = _vsi(_tok(h, token))
    return hrefs


def source_local(args, bands):
    paths = {}
    for b in bands:
        p = args.local_pattern.format(tile=args.tile, band=b, date=args.date)
        if not Path(p).exists():
            log.info("Local file missing: %s", p)
            return None
        paths[b] = p
    prov = args.local_pattern.format(tile=args.tile, band="PROVENANCE", date=args.date)
    return Source("local-" + args.local_kind, f"{args.tile} {args.date} (local)", paths,
                  provenance=prov if Path(prov).exists() else None)


def _stac_item_16d(args):
    import requests
    d0 = datetime.strptime(args.date, "%Y-%m-%d")
    r = requests.get(f"{args.stac_url}/collections/{args.collection_16d}/items/S2-16D_V2_{args.tile}_{d0:%Y%m%d}",
                     timeout=60)
    if r.ok:
        return r.json()
    ids = [f"S2-16D_V2_{args.tile}_{(d0 + timedelta(days=k)):%Y%m%d}" for k in range(-args.window_days, args.window_days + 1)]
    r = requests.post(f"{args.stac_url}/search", timeout=60,
                      json={"collections": [args.collection_16d], "ids": ids, "limit": 50})
    r.raise_for_status()
    feats = r.json().get("features", [])
    if not feats:
        return None
    feats.sort(key=lambda f: abs((datetime.strptime(f["id"].split("_")[-1], "%Y%m%d") - d0).days))
    return feats[0]


def source_stac_16d(args, bands, token):
    item = _stac_item_16d(args)
    if item is None:
        return None

    def opt(n):
        h = _asset_lookup(item, n)
        return _vsi(_tok(h, token)) if h else None

    return Source("16d", item["id"], _item_hrefs(item, bands, token), provenance=opt("PROVENANCE"), clearob=opt("CLEAROB"))


def source_stac_l2a(args, bands, token, bbox):
    import requests
    if not bbox:
        raise RuntimeError("L2A fallback needs a lon/lat bbox when no local reference file exists.")
    d0 = datetime.strptime(args.date, "%Y-%m-%d")
    dt = (f"{(d0 - timedelta(days=args.window_days)):%Y-%m-%d}T00:00:00Z/"
          f"{(d0 + timedelta(days=args.window_days)):%Y-%m-%d}T23:59:59Z")
    r = requests.post(f"{args.stac_url}/search", timeout=90,
                      json={"collections": [args.collection_l2a], "bbox": bbox, "datetime": dt, "limit": 100})
    r.raise_for_status()
    feats = r.json().get("features", [])
    feats = [f for f in feats if f["properties"].get("eo:cloud_cover", 0) <= args.max_cloud] or feats
    if not feats:
        return None
    feats.sort(key=lambda f: (f["properties"].get("eo:cloud_cover", 100),
                              abs((datetime.strptime(f["properties"]["datetime"][:10], "%Y-%m-%d") - d0).days)))
    item = feats[0]
    scl = _asset_lookup(item, "SCL")
    log.info("Selected L2A scene: %s (cloud %.1f%%)", item["id"], item["properties"].get("eo:cloud_cover", -1))
    return Source("l2a", item["id"], _item_hrefs(item, bands, token), scl=_vsi(_tok(scl, token)) if scl else None)


def download_16d(args, bands, token):
    """Download the 16-day mosaic COGs from the BDC STAC into DOWNLOAD_DIR (existing files are reused)."""
    item = _stac_item_16d(args)
    if item is None:
        return None
    p = item.get("properties", {})
    log.info("BDC mosaic: %s | period %s to %s | cloud %.1f%%", item["id"], str(p.get("start_datetime", "?"))[:10],
             str(p.get("end_datetime", "?"))[:10], float(p.get("eo:cloud_cover", -1) or -1))
    if not item["id"].endswith(args.date.replace("-", "")):
        log.warning("No mosaic starts on %s; using the closest one: %s", args.date, item["id"])
    ddir = Path(args.download_dir)
    ddir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for b in list(bands) + ["PROVENANCE", "CLEAROB"]:
        h = _asset_lookup(item, b)
        if h is None:
            if b in bands:
                raise RuntimeError(f"Band {b} missing in {item['id']}; assets: {list(item['assets'])}")
            continue
        jobs.append((b, _tok(h, token), ddir / Path(h.split("?")[0]).name))
    paths = {}
    bar = tqdm(jobs, desc=f"Downloading mosaic {item['id']}", unit="band", leave=True)
    for b, url, dest in bar:
        bar.set_postfix_str(b)
        if not (dest.exists() and dest.stat().st_size > 0):
            _download(url, dest)
        paths[b] = str(dest)
    bar.close()
    return Source("bdc16d", item["id"], {b: paths[b] for b in bands},
                  provenance=paths.get("PROVENANCE"), clearob=paths.get("CLEAROB"))


def resolve_source(args, bands):
    token = args.bdc_token or os.environ.get("BDC_ACCESS_TOKEN")
    bbox = args.bbox
    order = {"auto": ["local", "16d", "l2a"], "local": ["local"], "16d": ["16d"], "l2a": ["l2a"],
             "bdc16d": ["bdc16d"]}[args.source]
    names = {"local": "local files", "16d": "STAC S2-16D-2 (remote read)", "l2a": "STAC S2 L2A",
             "bdc16d": "BDC S2-16D-2 mosaic download"}
    bar = tqdm(order, desc="Locating data", unit="source", leave=True)
    try:
        for step in bar:
            bar.set_postfix_str(names[step])
            try:
                if step == "local":
                    s = source_local(args, bands)
                elif step == "bdc16d":
                    s = download_16d(args, bands, token)
                elif step == "16d":
                    s = source_stac_16d(args, bands, token)
                else:
                    if not bbox:
                        ref = args.local_pattern.format(tile=args.tile, band="B02", date=args.date)
                        if Path(ref).exists():
                            from rasterio.warp import transform_bounds
                            with rasterio.open(ref) as ds:
                                bbox = list(transform_bounds(ds.crs, "EPSG:4326", *ds.bounds))
                    s = source_stac_l2a(args, bands, token, bbox)
            except Exception as e:
                log.warning("Source '%s' unavailable: %s", names[step], e)
                continue
            if s is not None:
                bar.set_postfix_str(f"using {names[step]}")
                log.info("Selected source: %s [%s]", s.label, s.kind)
                return s
    finally:
        bar.close()
    sys.exit("No data source available (local, S2-16D-2, L2A).")


def qc_provenance(src: Source, env):
    """Seam density (scene changes between neighbors) of the 16-day mosaic, from the PROVENANCE band."""
    if not src.provenance:
        return None
    try:
        with rasterio.Env(**env), rasterio.open(src.provenance) as ds:
            f = max(1, max(ds.height, ds.width) // 1500)
            a = ds.read(1, out_shape=(ds.height // f, ds.width // f), resampling=Resampling.nearest)
            valid = a != (ds.nodata if ds.nodata is not None else -1)
            dx = (a[:, 1:] != a[:, :-1]) & valid[:, 1:] & valid[:, :-1]
            dy = (a[1:] != a[:-1]) & valid[1:] & valid[:-1]
            dens = (dx.sum() + dy.sum()) / max(1, valid.sum())
            log.info("16D QC: %d scenes in PROVENANCE; seam density = %.3f", len(np.unique(a[valid])), dens)
            if dens > 0.02:
                log.warning("Many scene changes in the mosaic: the model may treat seams as texture.")
            return float(dens)
    except Exception as e:
        log.debug("PROVENANCE QC skipped: %s", e)
        return None


# ---------------------------------------------------------------- band reading
GDAL_ENV = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.TIF,.tiff",
                GDAL_HTTP_MULTIPLEX="YES", GDAL_HTTP_MAX_RETRY="6", GDAL_HTTP_RETRY_DELAY="3",
                VSI_CACHE="TRUE", VSI_CACHE_SIZE=str(256 * 2**20), GDAL_CACHEMAX=1024,
                CPL_VSIL_CURL_CACHE_SIZE=str(256 * 2**20))


class BandStack:
    def __init__(self, hrefs, scale=None, offset=None, roi=None, scl=None):
        self.hrefs = list(hrefs)
        with rasterio.Env(**GDAL_ENV):
            self.ds = [rasterio.open(h) for h in tqdm(self.hrefs, desc="Opening bands", unit="band", leave=True)]
            ref = self.ds[0]
            self.crs, self.src_transform = ref.crs, ref.transform
            self.fullH, self.fullW = ref.height, ref.width
            self.scales, self.offsets, self.nodata = [], [], []
            for d in self.ds:
                s = d.scales[0] if d.scales and d.scales[0] not in (None, 1.0) else None
                o = d.offsets[0] if d.offsets and d.offsets[0] not in (None, 0.0) else 0.0
                self.scales.append(scale if scale is not None else (s or 1e-4))
                self.offsets.append(offset if offset is not None else o)
                self.nodata.append(d.nodata)
            self.ratios = [d.width / ref.width for d in self.ds]
            self.scl = rasterio.open(scl) if scl else None
        x0, y0, x1, y1 = roi if roi else (0, 0, self.fullW, self.fullH)
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(self.fullW, x1), min(self.fullH, y1)
        if x1 - x0 < PATCH or y1 - y0 < PATCH:
            sys.exit(f"Area smaller than {PATCH} px.")
        self.x0, self.y0, self.W, self.H = x0, y0, x1 - x0, y1 - y0
        self.transform = self.src_transform * Affine.translation(x0, y0)

    def read(self, y, h):
        """Strip [y, y+h) x full width -> float32 reflectance (C,h,W); nodata = 0."""
        out = np.zeros((len(self.ds), h, self.W), np.float32)
        ok = np.ones((h, self.W), bool)
        with rasterio.Env(**GDAL_ENV):
            for i, d in enumerate(self.ds):
                r = self.ratios[i]
                win = Window(self.x0 * r, (self.y0 + y) * r, self.W * r, h * r)
                raw = d.read(1, window=win) if r == 1.0 else \
                    d.read(1, window=win, out_shape=(h, self.W), resampling=Resampling.bilinear)
                valid = np.isfinite(raw) & ((raw != self.nodata[i]) if self.nodata[i] is not None else (raw != 0))
                refl = np.clip(np.nan_to_num(raw.astype(np.float32) * self.scales[i] + self.offsets[i]), 0.0, 1.0)
                refl[valid & (refl < 1e-4)] = 1e-4          # 0 is reserved for nodata
                out[i] = refl
                ok &= valid
            if self.scl is not None:
                r = self.scl.width / self.fullW
                s = self.scl.read(1, window=Window(self.x0 * r, (self.y0 + y) * r, self.W * r, h * r),
                                  out_shape=(h, self.W), resampling=Resampling.nearest)
                ok &= ~np.isin(s, SCL_BAD)
        out *= ok[None]
        return out

    def sample_patch(self):
        rng = np.random.default_rng(0)
        strip = None
        for _ in range(12):
            y = int(rng.integers(0, self.H - PATCH + 1))
            x = int(rng.integers(0, self.W - PATCH + 1))
            strip = self.read(y, PATCH)[:, :, x:x + PATCH]
            if (strip[0] > 0).mean() > 0.95:
                break
        return np.ascontiguousarray(strip)


# ---------------------------------------------------------------- strip blending and writing
def grid_positions(n, P, stride):
    pos = list(range(0, n - P + 1, stride))
    if pos[-1] + P < n:
        pos.append(n - P)
    return pos


def _smooth(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


_WIN_CACHE = {}


def window_1d(n_out, border, ramp, first, last):
    key = (n_out, border, ramp, first, last)
    if key not in _WIN_CACHE:
        i = np.arange(n_out, dtype=np.float32) + 0.5
        wl = np.ones(n_out, np.float32) if first else _smooth((i - border) / ramp)
        wr = np.ones(n_out, np.float32) if last else _smooth((n_out - i - border) / ramp)
        _WIN_CACHE[key] = (wl * wr).astype(np.float32)
    return _WIN_CACHE[key]


class RasterSink:
    def __init__(self, path: Path, profile: dict, tags: dict, descriptions, resume: bool):
        self.path, self.profile, self.tags, self.desc = path, profile, tags, descriptions
        self._open("r+" if resume else "w")

    def _open(self, mode):
        with rasterio.Env(GDAL_CACHEMAX=1024):
            if mode == "w":
                self.ds = rasterio.open(self.path, "w", **self.profile)
                self.ds.update_tags(**self.tags)
                for i, d in enumerate(self.desc, 1):
                    self.ds.set_band_description(i, d)
            else:
                self.ds = rasterio.open(self.path, "r+")

    def write(self, arr, row_off):
        self.ds.write(arr, window=Window(0, row_off, arr.shape[2], arr.shape[1]))

    def checkpoint(self):
        self.ds.close()
        self._open("r+")

    def close(self):
        self.ds.close()


class StripBlender:
    """Accumulates weighted SR windows for a band of output rows and flushes finished rows to disk."""

    def __init__(self, C, W_out, scale, block, convert):
        self.convert = convert
        self.buf_h = PATCH * scale + 2 * block
        self.acc = np.zeros((C, self.buf_h, W_out), np.float32)
        self.wgt = np.zeros((self.buf_h, W_out), np.float32)
        self.b0, self.used = 0, 0

    def add(self, sr, y_out, x_out, w2d):
        ry = y_out - self.b0
        if ry < 0:                       # resuming: rows already written
            sr, w2d, ry = sr[:, -ry:], w2d[-ry:], 0
        h, w = sr.shape[1], sr.shape[2]
        if ry + h > self.buf_h:
            raise RuntimeError("strip buffer too small")
        self.acc[:, ry:ry + h, x_out:x_out + w] += sr * w2d
        self.wgt[ry:ry + h, x_out:x_out + w] += w2d
        self.used = max(self.used, ry + h)

    def flush(self, upto, sink: RasterSink):
        n = upto - self.b0
        if n <= 0:
            return
        m = min(n, self.buf_h)
        w = self.wgt[:m]
        sink.write(self.convert(np.where(w > 1e-6, self.acc[:, :m] / np.maximum(w, 1e-6), 0.0)), self.b0)
        rem = max(self.used - n, 0)
        if rem:
            self.acc[:, :rem] = self.acc[:, n:n + rem].copy()
            self.wgt[:rem] = self.wgt[n:n + rem].copy()
        top = max(self.used, rem)
        self.acc[:, rem:top] = 0
        self.wgt[rem:top] = 0
        self.b0 += n
        self.used = rem


def conv_i16(x):
    return np.clip(np.rint(x * 10000.0), 0, 32767).astype(np.int16)


def conv_f32(x):
    return x.astype(np.float32)


def run_pipeline(stack: BandStack, runner: BatchRunner, args, out_paths: dict, band_names: dict,
                 meta: dict, scale: int, tag: str, label: str):
    P, S = PATCH, scale
    stride = P - args.overlap
    border = args.trim * S
    ramp = args.overlap * S - border
    if ramp <= 0:
        sys.exit("OVERLAP must be larger than TRIM + 1")
    block = 512
    H, W = stack.H, stack.W
    rows, cols = grid_positions(H, P, stride), grid_positions(W, P, stride)
    n_total = len(rows) * len(cols)
    outH, outW = H * S, W * S

    key = json.dumps(dict(tag=tag, hrefs=stack.hrefs, roi=[stack.x0, stack.y0, stack.W, stack.H], ov=args.overlap,
                          trim=args.trim, steps=args.steps, nvar=args.uncertainty, S=S, hist=args.hist_match,
                          db=args.deblock_20m, scale=stack.scales, offset=stack.offsets, b11="mtf-v2"), sort_keys=True)
    phash = hashlib.sha1(key.encode()).hexdigest()[:16]

    prog_path = out_paths["sr"].with_suffix(".progress.json")
    partial = {k: p.with_name(p.stem + ".partial.tif") for k, p in out_paths.items()}
    resume_row = 0
    if prog_path.exists() and all(p.exists() for p in partial.values()):
        pr = json.loads(prog_path.read_text())
        if pr.get("hash") == phash:
            resume_row = int(pr["flushed_row"])
            log.info("Resuming: %.1f%% already written", 100 * resume_row / outH)
    if not resume_row:
        for p in partial.values():
            p.unlink(missing_ok=True)

    sinks, blenders = {}, {}
    t_out = stack.transform * Affine.scale(1.0 / S)
    for name in out_paths:
        C = len(band_names[name])
        is_unc = name == "unc"
        prof = dict(driver="GTiff", height=outH, width=outW, count=C, dtype="float32" if is_unc else "int16",
                    crs=stack.crs, transform=t_out, nodata=0, tiled=True, blockxsize=block, blockysize=block,
                    BIGTIFF="YES", compress=args.compress.upper(), NUM_THREADS="ALL_CPUS")
        if args.compress == "zstd":
            prof["ZSTD_LEVEL"] = args.zstd_level
        if args.compress in ("zstd", "deflate", "lzw"):
            prof["predictor"] = 3 if is_unc else 2
        tags = dict(meta, SCALE_FACTOR="1.0" if is_unc else "0.0001",
                    PRODUCT="uncertainty (2*std)" if is_unc else "SR reflectance x10000")
        sinks[name] = RasterSink(partial[name], prof, tags, band_names[name], resume=bool(resume_row))
        blenders[name] = StripBlender(C, outW, S, block, conv_f32 if is_unc else conv_i16)
        blenders[name].b0 = resume_row

    i0 = next((i for i, r in enumerate(rows) if (r + P) * S > resume_row), len(rows)) if resume_row else 0
    log.info("Grid: %d x %d = %d windows | output %d x %d px (x%d) | batch %d",
             len(rows), len(cols), n_total, outH, outW, S, runner.bs)
    done0 = i0 * len(cols)
    t0 = time.perf_counter()
    since_ckpt = 0
    pbar = tqdm(total=n_total, initial=done0, desc=label, unit="window", leave=True, smoothing=0.05, dynamic_ncols=True)
    pool = ThreadPoolExecutor(1)
    fut = pool.submit(stack.read, rows[i0], P) if i0 < len(rows) else None
    try:
        for i in range(i0, len(rows)):
            pbar.set_postfix_str(f"strip {i + 1}/{len(rows)} | reading")
            strip = fut.result()
            if i + 1 < len(rows):
                fut = pool.submit(stack.read, rows[i + 1], P)
            y = rows[i]
            items = [(c, strip[:, :, c:c + P]) for c in cols if (strip[0, :, c:c + P] > 0).mean() >= args.min_valid]
            if len(cols) > len(items):
                pbar.update(len(cols) - len(items))       # windows without valid data
            pbar.set_postfix_str(f"strip {i + 1}/{len(rows)} | batch {runner.bs} | processing")
            wy = window_1d(P * S, border, ramp, y == 0, y + P >= H)
            k = 0
            while k < len(items):
                chunk = items[k:k + runner.bs]
                res = runner(np.ascontiguousarray(np.stack([p for _, p in chunk]), dtype=np.float32))
                for j, (c, _) in enumerate(chunk):
                    w2d = np.outer(wy, window_1d(P * S, border, ramp, c == 0, c + P >= W)).astype(np.float32)
                    for name in blenders:
                        blenders[name].add(res[name][j], y * S, c * S, w2d)
                k += len(chunk)
                pbar.update(len(chunk))
            f = outH if i + 1 >= len(rows) else (rows[i + 1] * S // block) * block
            if f > blenders["sr"].b0:
                pbar.set_postfix_str(f"strip {i + 1}/{len(rows)} | writing")
                for name in blenders:
                    blenders[name].flush(f, sinks[name])
                since_ckpt += 1
                if since_ckpt >= 4 or f >= outH:
                    for s in sinks.values():
                        s.checkpoint()
                    prog_path.write_text(json.dumps(dict(hash=phash, flushed_row=blenders["sr"].b0)))
                    since_ckpt = 0
    finally:
        pool.shutdown(wait=False)
        pbar.close()
        for s in sinks.values():
            try:
                s.close()
            except Exception:
                pass

    for name, p in out_paths.items():
        os.replace(partial[name], p)
    prog_path.unlink(missing_ok=True)
    el = time.perf_counter() - t0
    return dict(windows=n_total, seconds=round(el, 1), windows_per_s=round((n_total - done0) / max(el, 1e-6), 3),
                final_batch=runner.bs)


# ---------------------------------------------------------------- PRODES composite (R=B11, G=B08, B=B04, Int16)
def build_prodes(chans, out_tif: Path, compress="deflate", lo_pct=2, hi_pct=98, stretch_ref=None):
    """chans = [(file, band_index_1based, label)] in R, G, B order. Output grid = channel G.
    Bands keep their reflectance values (Int16, x10000); the 2-98 % stretch is only stored as tags for display.
    stretch_ref = original 10 m files (B11, B08, B04): the stretch is taken from them over the same area,
    so every mode is displayed with the same contrast."""
    with ExitStack() as es:
        ds = [es.enter_context(rasterio.open(p)) for p, _, _ in chans]
        ref = ds[1]
        H, W = ref.height, ref.width
        f = max(1, max(H, W) // 3000)
        small = (H // f, W // f)
        lo, hi = [], []
        for k, (d, (_, bi, _)) in enumerate(zip(ds, chans)):
            if stretch_ref and stretch_ref[k]:
                with rasterio.open(stretch_ref[k]) as o:
                    win = rasterio.windows.from_bounds(*ref.bounds, transform=o.transform)
                    v = o.read(1, window=win, out_shape=(max(1, H // 4 // f), max(1, W // 4 // f)),
                               resampling=Resampling.nearest)
                    sc = o.scales[0] if o.scales and o.scales[0] not in (None, 1.0) else 1e-4
                    v = v[v > 0].astype(np.float32) * sc * 1e4
            else:
                v = d.read(bi, out_shape=small, resampling=Resampling.nearest)
                v = v[v > 0]
            lo.append(np.percentile(v, lo_pct) if v.size else 0)
            hi.append(np.percentile(v, hi_pct) if v.size else 1)
        prof = dict(driver="GTiff", height=H, width=W, count=3, dtype="int16", crs=ref.crs, transform=ref.transform,
                    nodata=0, tiled=True, blockxsize=512, blockysize=512, BIGTIFF="YES",
                    compress=compress.upper(), predictor=2, NUM_THREADS="ALL_CPUS")
        tmp = out_tif.with_name(out_tif.stem + ".partial.tif")
        with rasterio.open(tmp, "w", **prof) as dst:
            for i, (_, _, lab) in enumerate(chans, 1):
                dst.set_band_description(i, lab)
            dst.update_tags(COMPOSITION="R=B11 G=B08 B=B04 (PRODES)",
                            STRETCH_LO=",".join(f"{v:.1f}" for v in lo), STRETCH_HI=",".join(f"{v:.1f}" for v in hi))
            for r0 in tqdm(range(0, H, 1024), desc="PRODES composite", unit="block", leave=True):
                h = min(1024, H - r0)
                win = Window(0, r0, W, h)
                arrs = []
                for d, (_, bi, _) in zip(ds, chans):
                    rr = d.height / H
                    arrs.append(d.read(bi, window=win) if rr == 1.0 else
                                d.read(bi, window=Window(0, r0 * rr, W * rr, h * rr), out_shape=(h, W),
                                       resampling=Resampling.bilinear))
                valid = (arrs[1] > 0) & (arrs[2] > 0)
                out = np.zeros((3, h, W), np.int16)
                for k, ch in enumerate(arrs):
                    out[k] = np.where(valid, np.clip(np.rint(ch), 1, 32767), 0).astype(np.int16)
                dst.write(out, window=win)
        os.replace(tmp, out_tif)


# ---------------------------------------------------------------- main
def parse_args(argv):
    p = argparse.ArgumentParser()
    g = p.add_argument_group("input")
    g.add_argument("--tile", required=True)
    g.add_argument("--date", required=True)
    g.add_argument("--source", choices=["auto", "local", "16d", "l2a", "bdc16d"], default="auto")
    g.add_argument("--local-pattern",
                   default="rascunho/biomasbr-amazonia/super_resolution/cube/SENTINEL-2_MSI_{tile}_{band}_{date}.tif")
    g.add_argument("--local-kind", choices=["16d", "l2a"], default="16d")
    g.add_argument("--download-dir", default="rascunho/biomasbr-amazonia/super_resolution/cube_bdc_16d")
    g.add_argument("--stac-url", default="https://data.inpe.br/bdc/stac/v1")
    g.add_argument("--collection-16d", default="S2-16D-2")
    g.add_argument("--collection-l2a", default="S2_L2A-1")
    g.add_argument("--bdc-token", default=None)
    g.add_argument("--bbox", nargs=4, type=float, default=None)
    g.add_argument("--window-days", type=int, default=8)
    g.add_argument("--max-cloud", type=float, default=30.0)
    g.add_argument("--mask-clouds", action="store_true")
    g.add_argument("--scale", type=float, default=None)
    g.add_argument("--offset", type=float, default=None)
    g.add_argument("--center", nargs=2, type=float, default=None, metavar=("LAT", "LON"))
    g.add_argument("--box-km", type=float, default=10.0)
    g = p.add_argument_group("model")
    g.add_argument("--mode", choices=["fast", "combined", "separate"], default="fast")
    g.add_argument("--combined-dir", default="rascunho/biomasbr-amazonia/super_resolution/model/LDSRS2-SEN2SR")
    g.add_argument("--engine", choices=["ldsr", "bicubic"], default="ldsr")
    g.add_argument("--model-dir", default="rascunho/biomasbr-amazonia/super_resolution/model/LDSR_S2")
    g.add_argument("--config", default=None)
    g.add_argument("--steps", type=int, default=100)
    g.add_argument("--hist-match", choices=["gpu", "cpu", "off"], default="gpu")
    g.add_argument("--uncertainty", type=int, default=0)
    g.add_argument("--deblock-20m", type=float, default=0.0)
    g.add_argument("--seed", type=int, default=42)
    g = p.add_argument_group("performance")
    g.add_argument("--precision", choices=["auto", "fp32", "bf16", "fp16"], default="auto")
    g.add_argument("--batch-size", type=int, default=0)
    g.add_argument("--max-batch", type=int, default=64)
    g.add_argument("--mem-fraction", type=float, default=None)
    g.add_argument("--overlap", type=int, default=16)
    g.add_argument("--trim", type=int, default=2)
    g.add_argument("--min-valid", type=float, default=0.02)
    g = p.add_argument_group("output")
    g.add_argument("--out-dir", default="rascunho/biomasbr-amazonia/super_resolution/cube_resolution")
    g.add_argument("--compress", choices=["zstd", "deflate", "lzw", "none"], default="zstd")
    g.add_argument("--zstd-level", type=int, default=9)
    g.add_argument("--overwrite", action="store_true")
    g.add_argument("--with-b11", action="store_true")
    g.add_argument("--sen2sr-dir", default="rascunho/biomasbr-amazonia/super_resolution/model/SEN2SR_main")
    g.add_argument("--prodes-composite", action="store_true")
    g.add_argument("--probe-only", action="store_true")
    return p.parse_args(argv)


def _size(p: Path):
    try:
        return f"{p.stat().st_size / 2**30:.2f} GiB"
    except OSError:
        return "?"


def cleanup_partials(*dirs):
    """Delete temporary/partial files (from finished or interrupted runs), keeping only final results."""
    removed = []
    for d in {Path(x) for x in dirs if x}:
        if not d.is_dir():
            continue
        for pat in ("*.partial*", "*.progress.json", "*.dl", "*.tmp", "*.aux.xml"):
            for f in d.glob(pat):
                try:
                    f.unlink()
                    removed.append(f.name)
                except OSError as e:
                    log.warning("Could not delete %s: %s", f, e)
    if removed:
        log.info("Temporary files deleted: %s", ", ".join(sorted(removed)))
    return removed


def roi_from_center(href, lat, lon, box_km):
    """Square of box_km x box_km (in the tile's metric CRS) centered on lat/lon -> pixel ROI."""
    from rasterio.warp import transform, transform_bounds
    from rasterio.windows import from_bounds
    with rasterio.Env(**GDAL_ENV), rasterio.open(href) as d:
        xs, ys = transform("EPSG:4326", d.crs, [lon], [lat])
        half = box_km * 500.0
        if d.crs.is_geographic:
            half_x, half_y = half / (111320.0 * np.cos(np.radians(lat))), half / 110540.0
        else:
            half_x = half_y = half
        w = from_bounds(xs[0] - half_x, ys[0] - half_y, xs[0] + half_x, ys[0] + half_y, transform=d.transform)
        x0, y0 = max(0, int(round(w.col_off))), max(0, int(round(w.row_off)))
        x1 = min(d.width, int(round(w.col_off + w.width)))
        y1 = min(d.height, int(round(w.row_off + w.height)))
        full = (int(round(w.width)), int(round(w.height)))
        tb = transform_bounds(d.crs, "EPSG:4326", *d.bounds)
    if x1 - x0 < PATCH or y1 - y0 < PATCH:
        sys.exit(f"ROI ({lat}, {lon}) is outside tile {d.name.split('/')[-1]} or too close to its edge. "
                 f"Tile extent: lon {tb[0]:.4f} to {tb[2]:.4f}, lat {tb[1]:.4f} to {tb[3]:.4f}.")
    if (x1 - x0, y1 - y0) != full:
        log.warning("The %.0f km box was clipped by the tile edge: %d x %d px (%.1f x %.1f km).",
                    box_km, x1 - x0, y1 - y0, (x1 - x0) / 100, (y1 - y0) / 100)
    return [x0, y0, x1, y1]


def main(argv=None, t_start=None):
    args = parse_args(argv)
    if args.compress == "none":
        args.compress = "NONE"
    combined = args.mode == "combined" and args.engine == "ldsr"
    fast = args.mode == "fast"
    if fast or combined:
        args.with_b11 = False
    elif args.prodes_composite:
        args.with_b11 = True
    if combined:
        args.uncertainty = 0
    if args.uncertainty and args.uncertainty <= 3:
        sys.exit("UNCERTAINTY must be 0 or greater than 3")
    out_dir = Path(args.out_dir)
    logs_dir = out_dir / "logs"
    for d in (logs_dir, Path(args.download_dir), Path(args.local_pattern).parent):
        if "{" not in str(d):
            d.mkdir(parents=True, exist_ok=True)
    setup_logging()
    log.info("Results folder: %s", out_dir.resolve())

    sr_stage = ("Super-resolution LDSR-S2 + SEN2SR (10 bands)" if combined else
                "Super-resolution LDSR-S2 + guided B11" if fast else "Super-resolution LDSR-S2")
    plan = ["Detecting GPU/CUDA", "Locating data", "Opening bands",
            "Loading LDSR-S2 + SEN2SR" if combined else "Loading LDSR-S2", "Testing precision", "Tuning batch size"]
    if not args.probe_only:
        plan.append(sr_stage)
        if args.with_b11:
            plan += ["Preparing B11 (SEN2SR)", "Super-resolution B11 (SEN2SR)"]
        if args.prodes_composite:
            plan.append("PRODES composite")
        plan.append("Final report")
    init = (_T_MODULE_LOADED - t_start) if t_start is not None else None
    st = Stages(plan, init_seconds=init, t0=t_start)
    ok, report, outputs = False, {}, []
    try:
        st.start("Detecting GPU/CUDA")
        hw = detect_hardware()
        configure_torch(hw)
        log_hardware(hw)
        device = torch.device("cuda" if hw.cuda else "cpu")
        mem_frac = args.mem_fraction or (0.80 if hw.is_mig else 0.85)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        st.start("Locating data")
        in_bands = SEN2SR_BANDS if combined else (LDSR_BANDS + ["B11"] if fast else LDSR_BANDS)
        src = resolve_source(args, in_bands)
        if src.kind.endswith("16d"):
            qc_provenance(src, GDAL_ENV)

        st.start("Opening bands")
        args.roi = None
        if args.center:
            args.roi = roi_from_center(src.hrefs[in_bands[0]], args.center[0], args.center[1], args.box_km)
            log.info("ROI %s, %.0f km box -> pixels %s (%.1f x %.1f km)", tuple(args.center), args.box_km, args.roi,
                     (args.roi[2] - args.roi[0]) / 100, (args.roi[3] - args.roi[1]) / 100)
        stack = BandStack([src.hrefs[b] for b in in_bands], args.scale, args.offset, args.roi,
                          scl=src.scl if args.mask_clouds else None)
        log.info("Input: %d x %d px%s | CRS %s | scale %s", stack.H, stack.W,
                 f" ({args.box_km:g} km box)" if args.roi else " (whole tile)", stack.crs, stack.scales[0])

        tag = f"{src.kind}_{args.tile}_{args.date}"
        uses_b20 = combined or fast or args.with_b11
        base = run_stem(args, src.kind, combined, fast, uses_b20)
        run_desc = run_description(args, src.kind, combined, fast, uses_b20)
        attach_log_file(logs_dir / f"{base}.log")
        log.info("Output name: %s", base)
        log.info("Run: %s", run_desc)
        out_paths = {"sr": out_dir / f"{base}.tif"}
        if args.uncertainty:
            out_paths["unc"] = out_dir / f"{base}_uncertainty.tif"
        report = dict(args=vars(args), hardware=asdict(hw), source=src.label, kind=src.kind,
                      input_files=dict(src.hrefs), outputs={})
        common_meta = dict(SOURCE=src.label, SOURCE_KIND=src.kind, DDIM_STEPS=str(args.steps), OVERLAP=str(args.overlap),
                           TRIM=str(args.trim), GPU=hw.name, TORCH=torch.__version__,
                           RUN=run_desc, MODE=args.mode if args.engine == "ldsr" else "bicubic", AREA=area_label(args),
                           DEBLOCK_20M=str(args.deblock_20m if uses_b20 else 0),
                           CREATED=datetime.now().isoformat(timespec="seconds"))

        if not args.overwrite and not args.probe_only and all(p.exists() for p in out_paths.values()):
            log.info("Output already exists (%s). Set OVERWRITE = True to redo it.", out_paths["sr"])
            outputs += list(out_paths.values())

        elif combined:
            st.start("Loading LDSR-S2 + SEN2SR")
            lstate = {"steps": args.steps, "amp": None, "hist": "cpu" if args.hist_match == "cpu" else "gpu"}
            eng = Sen2SREngine(args.combined_dir, device, out_idx=None, url=LDSR_SEN2SR_URL, label="LDSR-S2+SEN2SR",
                               ldsr_state=lstate)
            eng.deblock = args.deblock_20m
            log.info("20 m bands: %s", f"Gaussian sigma {args.deblock_20m:g} px (de-blocking)" if args.deblock_20m > 0
                     else "BDC values untouched")
            sample = stack.sample_patch()
            st.start("Testing precision")
            if args.precision == "auto":
                log.info("Combined model: using fp32+TF32 (set PRECISION='bf16' to force bf16).")
            else:
                lstate["amp"] = choose_precision_combined(eng, sample, hw, args.precision)
            prec = PRECISION_NAMES[lstate["amp"]]
            log.info("Precision: %s | DDIM steps: %d", prec, args.steps)
            st.start("Tuning batch size")
            if args.batch_size > 0:
                bs = args.batch_size
            else:
                if eng.n_ldsr:
                    lstate["steps"] = 2          # memory does not depend on the number of steps
                try:
                    bs = tune_batch(eng, sample, hw, args.max_batch, mem_frac, cands=(1, 2, 4, 8),
                                    max_single_s=8.0) if hw.cuda else 1
                finally:
                    lstate["steps"] = args.steps
            log.info("Batch size: %d (memory limit %.0f%% of %.1f GiB)", bs, 100 * mem_frac, hw.gb())
            if args.probe_only:
                ok = True
                return 0
            st.start(sr_stage)
            meta = dict(common_meta, MODEL="LDSRS2-SEN2SR (tacofoundation/RS-SR-LTDF)", PRECISION=prec,
                        BAND_ORDER=",".join(SEN2SR_BANDS))
            report["sr"] = run_pipeline(stack, BatchRunner(eng, bs), args, out_paths, {"sr": SEN2SR_BANDS}, meta,
                                        eng.scale, tag, sr_stage)
            outputs += list(out_paths.values())
            del eng
            if hw.cuda:
                torch.cuda.empty_cache()

        else:
            st.start("Loading LDSR-S2")
            engine = BicubicEngine(device) if args.engine == "bicubic" else build_ldsr(args, device)
            sample = stack.sample_patch()[:4]
            st.start("Testing precision")
            if isinstance(engine, LDSREngine) and hw.cuda:
                engine.amp = choose_precision(engine, sample, hw, args.precision)
            prec = PRECISION_NAMES[engine.amp]
            log.info("Precision: %s", prec)
            st.start("Tuning batch size")
            if args.batch_size > 0:
                bs = args.batch_size
            else:
                bs = tune_batch(engine, sample, hw, args.max_batch, mem_frac) if hw.cuda else 1
                if getattr(engine, "n_var", 0) > 1:
                    bs = max(1, bs // 2)
            log.info("Batch size: %d (memory limit %.0f%% of %.1f GiB)", bs, 100 * mem_frac, hw.gb())
            if args.probe_only:
                ok = True
                return 0
            st.start(sr_stage)
            model_name = "bicubic" if args.engine == "bicubic" else \
                "LDSR-S2 (opensr-model)" + (" + guided B11 (regression detail injection)" if fast else "")
            meta = dict(common_meta, MODEL=model_name, PRECISION=prec, BAND_ORDER=",".join(in_bands))
            run_eng = LDSRGuidedB11Engine(engine, args.deblock_20m) if (fast and isinstance(engine, LDSREngine)) else engine
            if fast:
                log.info("B11 input: %s", f"Gaussian sigma {args.deblock_20m:g} px (de-blocking)" if args.deblock_20m > 0
                         else "BDC values untouched")
            report["ldsr"] = run_pipeline(stack, BatchRunner(run_eng, bs), args, out_paths,
                                          {"sr": in_bands, "unc": ["uncertainty_2std"]}, meta, engine.scale, tag, sr_stage)
            report["precision"] = prec
            outputs += list(out_paths.values())
            del engine
            if hw.cuda:
                torch.cuda.empty_cache()

        comp = out_dir / f"{base}_PRODES.tif"
        p = out_paths["sr"]
        prodes_chans = None
        if args.with_b11:
            b11_path = out_dir / f"{base}_B11.tif"
            st.start("Preparing B11 (SEN2SR)")
            s2 = resolve_source(args, SEN2SR_BANDS)
            report["input_files"].update(s2.hrefs)
            if b11_path.exists() and not args.overwrite:
                log.info("B11 already exists: %s", b11_path)
            else:
                st2 = BandStack([s2.hrefs[b] for b in SEN2SR_BANDS], args.scale, args.offset, args.roi,
                                scl=s2.scl if args.mask_clouds else None)
                eng2 = Sen2SREngine(args.sen2sr_dir, device)
                eng2.deblock = args.deblock_20m
                bs2 = args.batch_size or (tune_batch(eng2, st2.sample_patch(), hw, 32, mem_frac) if hw.cuda else 1)
                st.start("Super-resolution B11 (SEN2SR)")
                report["b11"] = run_pipeline(st2, BatchRunner(eng2, bs2), args, {"sr": b11_path}, {"sr": ["B11"]},
                                             dict(SOURCE=s2.label, MODEL="SEN2SR", BAND="B11", GPU=hw.name),
                                             eng2.scale, "b11_" + tag, "Super-resolution B11 (SEN2SR)")
                del eng2
                if hw.cuda:
                    torch.cuda.empty_cache()
            outputs.append(b11_path)
            report["outputs"]["b11"] = str(b11_path)
            prodes_chans = [(b11_path, 1, "B11 (SEN2SR)"), (p, 4, "B08 (LDSR-S2)"), (p, 1, "B04 (LDSR-S2)")]
        elif fast:
            prodes_chans = [(p, 5, "B11 (guided by LDSR-S2)"), (p, 4, "B08 (LDSR-S2)"), (p, 1, "B04 (LDSR-S2)")]
        elif combined:
            prodes_chans = [(p, 9, "B11 (LDSRS2-SEN2SR)"), (p, 7, "B08 (LDSRS2-SEN2SR)"), (p, 3, "B04 (LDSRS2-SEN2SR)")]

        if args.prodes_composite and prodes_chans and not args.probe_only:
            st.start("PRODES composite")
            if not comp.exists() or args.overwrite:
                build_prodes(prodes_chans, comp, stretch_ref=[src.hrefs.get(b) for b in ("B11", "B08", "B04")])
            outputs.append(comp)
            report["outputs"]["prodes"] = str(comp)

        st.start("Final report")
        ok = True
    finally:
        if ok and not args.probe_only:
            cleanup_partials(out_dir, args.download_dir, args.model_dir, args.combined_dir, args.sen2sr_dir)
        total = st.finish(ok)
        if ok and not args.probe_only and report:
            report["times_s"] = {k: round(v, 1) for k, v in st.times.items()}
            report["total_time"] = fmt_t(total)
            report["outputs"].update({k: str(v) for k, v in out_paths.items()})
            rp = logs_dir / f"{base}.report.json"
            rp.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
            log.info("Files written to %s:", out_dir.resolve())
            for f in outputs:
                log.info("  %s  (%s)", f.name, _size(f))
            log.info("Logs: %s", logs_dir.resolve() / f"{base}.log (+ .report.json)")
        dump_unattached_log(logs_dir / f"{args.tile}_{args.date}_aborted-before-start.log")
    return 0


def _argv_from_params():
    g = globals().get
    mode = g("MODE", "fast")
    R = g("ROOT", "rascunho/biomasbr-amazonia/super_resolution")
    a = ["--tile", g("TILE"), "--date", g("DATE"), "--out-dir", g("OUT_DIR", R + "/cube_resolution"),
         "--source", g("SOURCE", "auto"), "--local-pattern", g("LOCAL_PATTERN", R + "/cube/SENTINEL-2_MSI_{tile}_{band}_{date}.tif"),
         "--local-kind", g("LOCAL_KIND", "16d"), "--download-dir", g("DOWNLOAD_DIR", R + "/cube_bdc_16d"),
         "--steps", str(g("STEPS", 100)), "--precision", g("PRECISION", "auto"),
         "--batch-size", str(g("BATCH_SIZE", 0)), "--overlap", str(g("OVERLAP", 16)),
         "--model-dir", g("MODEL_DIR", R + "/model/LDSR_S2"), "--sen2sr-dir", g("SEN2SR_DIR", R + "/model/SEN2SR_main"),
         "--mode", mode, "--combined-dir", g("COMBINED_DIR", R + "/model/LDSRS2-SEN2SR"), "--compress", g("COMPRESS", "zstd")]
    lat, lon = g("ROI_LAT"), g("ROI_LON")
    if (lat is None) != (lon is None):
        raise SystemExit("Set both ROI_LAT and ROI_LON, or leave both as None for the whole tile.")
    if lat is not None:
        a += ["--center", str(lat), str(lon), "--box-km", str(g("ROI_SIZE_KM", 10))]
    a += ["--deblock-20m", str(g("DEBLOCK_20M", 0.0))]
    if g("UNCERTAINTY"):
        a += ["--uncertainty", str(g("UNCERTAINTY"))]
    if g("CONFIG"):
        a += ["--config", g("CONFIG")]
    if mode in ("fast", "combined") or g("WITH_B11"):
        a += ["--prodes-composite"]
    if mode == "separate" and g("WITH_B11"):
        a += ["--with-b11"]
    if g("OVERWRITE"):
        a += ["--overwrite"]
    if g("PROBE_ONLY"):
        a += ["--probe-only"]
    return a


try:
    main(_argv_from_params(), t_start=_T_CELL0)
except SystemExit as _e:
    if _e.code not in (0, None):
        print("ERROR:", _e.code)
except KeyboardInterrupt:
    print("Interrupted. Run block 2 again to resume where it stopped.")


# %% BLOCK 3 (optional) — PRODES PREVIEW (R=B11, G=B08, B=B04)
import glob, os, json, numpy as np, rasterio, matplotlib.pyplot as plt
from rasterio.enums import Resampling
from rasterio.windows import Window, from_bounds
from tqdm.auto import tqdm

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


if "prodes" in outs and os.path.exists(outs["prodes"]):
    _bar.set_description("Preview: PRODES overview")
    with rasterio.open(outs["prodes"]) as d:
        k = max(1, max(d.height, d.width) // 1500)
        over = d.read([1, 2, 3], out_shape=(3, d.height // k, d.width // k), resampling=Resampling.average)
        t = d.tags()
        lo = [float(v) for v in t.get("STRETCH_LO", "0,0,0").split(",")]
        hi = [float(v) for v in t.get("STRETCH_HI", "10000,10000,10000").split(",")]
        S = 4
        Wlr, Hlr = d.width // S, d.height // S
        ZOOM_LR_PX = int(min(ZOOM_LR_PX, Wlr, Hlr))
        half = ZOOM_LR_PX // 2
        cx, cy = ZOOM_CENTER if ZOOM_CENTER else (Wlr // 2, Hlr // 2)
        cx, cy = min(max(cx, half), Wlr - half), min(max(cy, half), Hlr - half)
        win_sr = Window((cx - half) * S, (cy - half) * S, ZOOM_LR_PX * S, ZOOM_LR_PX * S)
        sr_zoom = d.read([1, 2, 3], window=win_sr)
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

# %% BLOCK 4 (optional) — COMPARISON: original 10 m vs LDSR-S2 (fast) vs combined — visible, PRODES and B11 diagnostics
# Requires block 2 runs with the same TILE, DATE and ROI (e.g. "fast" and "combined").
# All products found in OUT_DIR are displayed with the stretch of the original 10 m image.
import glob, os, re, json, numpy as np, rasterio, matplotlib.pyplot as plt
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from tqdm.auto import tqdm

CMP_ZOOM_PX = 160                # displayed crop, in 10 m pixels
CMP_METRIC_PX = 512              # area used for the numbers, in 10 m pixels
CMP_CENTER = None                # (x, y) in 10 m pixels inside the processed area; None = center
CMP_KEEP = None                  # None = all products found | list of name fragments, e.g. ["combined", "_db1"]

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
