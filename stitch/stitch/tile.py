from __future__ import annotations

import numpy as np
from stitch import connect
from collections import OrderedDict
from iohub import open_ome_zarr
import dask.array as da
from typing import List, Dict, TYPE_CHECKING
from tqdm import tqdm
import scipy
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from dexp.processing.registration.model.translation_registration_model import (
    TranslationRegistrationModel,
)
from dexp.processing.registration import translation_nd as dexp_reg
from dexp.processing.utils.linear_solver import linsolve

# Silence arbol's per-call "Memory pool clearing not enabled!" print (fires
# every dexp CupyBackend.__exit__, so once per phase-correlation call — floods
# stdout with 334k lines on a full plate). We don't rely on arbol output.
try:
    from arbol.arbol import Arbol as _Arbol
    _Arbol.enable_output = False
except Exception:
    pass
from stitch.connect import parse_positions, pos_to_name
from stitch.stitch.graph import connectivity, hilbert_over_points

# CuPy needs CUDA_PATH for runtime kernel compilation (NVRTC reads headers
# from <CUDA_PATH>/include). Our uv .venv doesn't set it, so try in order:
# (a) venv-bundled nvidia/cuda_nvcc wheel (matches organelle_profiler's shim),
# (b) system CUDA toolkit matching PyTorch's CUDA version under /hpc/apps/x86_64/cuda.
# Must run BEFORE `import cupy` — cupy imports fine without it, but the first
# runtime kernel compile (e.g. our FFT/reduction path) will crash otherwise.
import os as _os
if "CUDA_PATH" not in _os.environ:
    _cuda_path = None
    try:
        import importlib.util as _iu
        _spec = _iu.find_spec("nvidia.cuda_nvcc")
        if _spec is not None and _spec.submodule_search_locations:
            _cuda_path = _spec.submodule_search_locations[0]
    except Exception:
        pass
    if _cuda_path is None:
        try:
            from pathlib import Path as _Path
            import torch as _torch
            _cu_ver = _torch.version.cuda  # e.g. "12.6"
            if _cu_ver:
                for _d in sorted(
                    _Path("/hpc/apps/x86_64/cuda").glob(f"{_cu_ver}*"),
                    reverse=True,
                ):
                    if (_d / "bin" / "nvcc").exists():
                        _cuda_path = str(_d)
                        break
        except Exception:
            pass
    if _cuda_path:
        _os.environ["CUDA_PATH"] = _cuda_path

# Try to use CuPy for GPU-accelerated registration
try:
    import cupy as xp
    from cupyx.scipy import ndimage as cundi
    # Check if GPU is actually available at runtime
    try:
        _ = xp.array([1.0])  # Test GPU access
        _USING_CUPY = True
        print("[tile.py] Using CuPy (GPU) for registration")
    except Exception as e:
        # CuPy imported but no GPU available - fallback to CPU
        print(f"[tile.py] CuPy available but GPU not accessible ({type(e).__name__}), falling back to CPU")
        import numpy as xp
        from scipy import ndimage as cundi
        _USING_CUPY = False
except (ModuleNotFoundError, ImportError):
    import numpy as xp
    from scipy import ndimage as cundi
    _USING_CUPY = False
    print("[tile.py] Using NumPy (CPU) for registration")


class LimitedSizeDict(OrderedDict):
    def __init__(self, max_size):
        super().__init__()
        self.max_size = max_size

    def __setitem__(self, key, value):
        # Add new entry
        super().__setitem__(key, value)
        # Remove the oldest entry if size limit is exceeded
        if len(self) > self.max_size:
            self.popitem(last=False)  # Removes the first (oldest) item


class Edge:
    """
    tile_a_key: str
    tile_b_key: str
    tile_cache: TileCache
    """

    def __init__(self, tile_a_key, tile_b_key, tile_cache, overlap=150):
        self.tile_cache = tile_cache
        self.overlap = overlap
        self.tile_a = connect.pos_to_name(tile_a_key)
        self.tile_b = connect.pos_to_name(tile_b_key)
        self.ux = int(self.tile_a[:3])
        self.uy = int(self.tile_a[3:])
        self.vx = int(self.tile_b[:3])
        self.vy = int(self.tile_b[3:])
        self.relation = (self.ux - self.vx, self.uy - self.vy)
        self.model = self.get_offset()

    def get_offset(self):
        """
        get the offset between the two tiles
        """
        tile_a = self.tile_cache[self.tile_a]
        tile_b = self.tile_cache[self.tile_b]
        return offset(tile_a, tile_b, self.relation, overlap=self.overlap)


class TileCache:
    """
    has attributes:
    - list of tiles with a maximum length
    - indexed by their name, containing the tile array in memory

    methods:
    - load tile
    - get_item
    """

    def __init__(self, store_path, well, flipud, fliplr, rot90, channel=0, timepoint=0, use_clahe=False, clahe_clip_limit=0.02):
        self.cache = LimitedSizeDict(max_size=20)
        self.store = open_ome_zarr(store_path)
        self.well = well
        self.flipud = flipud
        self.fliplr = fliplr
        self.rot90 = rot90
        self.channel = channel
        self.use_clahe = use_clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.timepoint = timepoint

    def add(self, obj):
        """add an object to the cache"""
        self.cache.append(obj)

    def __getitem__(self, key):
        """get an object from the cache"""
        # if isinstance(key, int):
        #     return self.cache[key]
        if key in self.cache:
            return self.cache[key]
        else:
            try:
                # load a new tile and add it to the cache
                a = self.load_tile(key)
                return a
            except KeyError:
                print("tile not found")
                return None

    def load_tile(self, key):
        """load a tile and add it to the cache"""
        da_tile = da.from_array(self.store[f"{self.well}/{key}"].data)

        tile = da_tile[self.timepoint, self.channel, 0, :, :].compute()  # T=timepoint, C=channel, Z=0

        # Apply CLAHE preprocessing if enabled
        if self.use_clahe:
            from skimage.exposure import equalize_adapthist
            # Normalize to [0, 1] for CLAHE
            tile_min, tile_max = np.percentile(tile, [1, 99])
            tile_norm = np.clip((tile.astype(np.float32) - tile_min) / (tile_max - tile_min + 1e-8), 0, 1)
            # Apply CLAHE with default kernel size (1/8 of image size)
            tile = equalize_adapthist(tile_norm, clip_limit=self.clahe_clip_limit)
            # Keep as float [0, 1]

        aug_tile = augment_tile(
            tile,
            flipud=self.flipud,
            fliplr=self.fliplr,
            rot90=self.rot90,
        )
        # hardcoded for now to only take first slice and time-point
        self.cache[key] = aug_tile
        return aug_tile


def augment_tile(tile, flipud: bool, fliplr: bool, rot90: int):
    """Augment a tile with flips and rotations using appropriate array library (CuPy or NumPy)"""
    # Determine which array library to use based on input type
    if _USING_CUPY and hasattr(tile, '__array_ufunc__') and 'cupy' in str(type(tile)):
        # Use CuPy operations for GPU arrays
        if flipud:
            tile = xp.flip(tile, axis=-2)
        if fliplr:
            tile = xp.flip(tile, axis=-1)
        if rot90:
            tile = xp.rot90(tile, k=rot90, axes=(-2, -1))
    else:
        # Use NumPy operations for CPU arrays
        if flipud:
            tile = np.flip(tile, axis=-2)
        if fliplr:
            tile = np.flip(tile, axis=-1)
        if rot90:
            tile = np.rot90(tile, k=rot90, axes=(-2, -1))
    return tile


def register_translation_gpu(image_a, image_b, upsample_factor=10):
    """
    GPU-accelerated phase correlation registration using CuPy.

    Falls back to CPU if CuPy is not available.

    Args:
        image_a: Reference image (numpy or cupy array)
        image_b: Moving image (numpy or cupy array)
        upsample_factor: Upsampling factor for subpixel accuracy

    Returns:
        shift: (y, x) shift vector
        confidence: Correlation confidence score
    """
    if _USING_CUPY:
        # Transfer to GPU if not already there
        img_a_gpu = xp.asarray(image_a, dtype=xp.float32)
        img_b_gpu = xp.asarray(image_b, dtype=xp.float32)

        # Compute FFTs
        fft_a = xp.fft.fft2(img_a_gpu)
        fft_b = xp.fft.fft2(img_b_gpu)

        # Phase correlation
        cross_power = fft_a * xp.conj(fft_b)
        cross_power_norm = cross_power / (xp.abs(cross_power) + 1e-10)

        # Inverse FFT to get correlation
        correlation = xp.fft.ifft2(cross_power_norm).real

        # Find peak (coarse)
        max_idx = xp.argmax(correlation)
        max_idx_unraveled = xp.unravel_index(max_idx, correlation.shape)
        shifts_coarse = xp.array([max_idx_unraveled[0], max_idx_unraveled[1]], dtype=xp.float32)

        # Wrap shifts to handle periodic boundaries
        shape = xp.array(correlation.shape)
        shifts_coarse = xp.where(shifts_coarse > shape / 2, shifts_coarse - shape, shifts_coarse)

        # Get confidence from peak height
        confidence = float(xp.max(correlation))

        # Subpixel refinement using upsampled DFT
        if upsample_factor > 1:
            # Create upsampled region around peak
            upsampled_region_size = int(xp.ceil(upsample_factor * 1.5))
            dftshift = int(xp.fix(upsampled_region_size / 2.0))

            # Frequency-domain upsampling
            sample_region_offset = dftshift - shifts_coarse * upsample_factor

            # Matrix multiply DFT for upsampling
            from cupyx.scipy.ndimage import fourier_shift

            # Simple approach: use pixel-level shift for now, can enhance with DFT upsampling
            shift = shifts_coarse.get()  # Transfer back to CPU
        else:
            shift = shifts_coarse.get()  # Transfer back to CPU

        # Clean up GPU memory
        del img_a_gpu, img_b_gpu, fft_a, fft_b, cross_power, cross_power_norm, correlation
        if _USING_CUPY:
            xp.get_default_memory_pool().free_all_blocks()

        return np.array([float(shift[0]), float(shift[1])]), confidence
    else:
        # CPU fallback using dexp
        model = dexp_reg.register_translation_nd(image_a, image_b)
        return model.shift_vector, model.confidence


import threading as _threading

_gpu_thread_state = _threading.local()


def _gpu_stream():
    """Per-thread CUDA stream. With 16 threads all calling GPU phase-corr, the
    default stream serializes their kernels; giving each thread its own
    non-blocking stream lets kernels from different threads overlap on the GPU.
    Returns ``None`` on CPU.
    """
    if not _USING_CUPY:
        return None
    s = getattr(_gpu_thread_state, "stream", None)
    if s is None:
        s = xp.cuda.Stream(non_blocking=True)
        _gpu_thread_state.stream = s
    return s


def _dexp_cupy_backend():
    """Per-thread ``CupyBackend`` for pushing dexp onto GPU. Cached because
    construction touches cub/cutensor toggles + cudnn probe — one-off cost.
    Disable dexp's own memory pool (share cupy's global pool across threads)
    and its per-call clearing (would kill throughput).
    """
    if not _USING_CUPY:
        return None
    b = getattr(_gpu_thread_state, "dexp_backend", None)
    if b is None:
        from dexp.utils.backends import CupyBackend
        b = CupyBackend(enable_memory_pool=False, enable_memory_pool_clearing=False)
        _gpu_thread_state.dexp_backend = b
    return b


class _SimpleTranslationModel:
    """Duck-typed replacement for dexp's ``TranslationRegistrationModel`` —
    exposes ``.shift_vector`` (numpy) and ``.confidence`` (float). Returned
    by ``batched_phase_correlation`` so callers can treat batched results
    exactly like per-call ``offset()`` returns.
    """
    __slots__ = ("shift_vector", "confidence")

    def __init__(self, shift_vector, confidence):
        self.shift_vector = shift_vector
        self.confidence = float(confidence)


def _preprocess_batch_gpu(images):
    """Batched dexp-style preprocessing on GPU.

    Applies gaussian denoise + log1p + sobel magnitude + Hanning window to
    each (H, W) plane of an (N, H, W) tensor in one kernel launch per stage
    (5 kernels total, vs ~10 per single call from dexp).
    """
    from cupyx.scipy.ndimage import gaussian_filter, sobel
    # Denoise on spatial axes only; sigma=0 on batch axis skips it.
    images = gaussian_filter(images, sigma=(0, 1.5, 1.5))
    images = xp.log1p(images)
    # Sobel edge magnitude (matches dexp's edge_filter=True default).
    sy = sobel(images, axis=-2)
    sx = sobel(images, axis=-1)
    images = xp.sqrt(sy * sy + sx * sx)
    # Hanning^0.5 window (dexp's `window=0.5` default is a sqrt of Hanning).
    N, H, W = images.shape
    win_y = xp.sqrt(xp.hanning(H)).astype(xp.float32)
    win_x = xp.sqrt(xp.hanning(W)).astype(xp.float32)
    win = win_y[:, None] * win_x[None, :]  # (H, W)
    return images * win[None, :, :]


def batched_phase_correlation(
    rois_a, rois_b, pitch_yx=(0, 0), preprocess=True, chunk_size=128,
) -> list:
    """Batched phase correlation on GPU with dexp-style preprocessing.

    Parameters
    ----------
    rois_a, rois_b : (N, H, W) arrays (numpy or cupy). Must have the same shape.
        These are the pre-extracted overlap ROIs — the caller does the axis-
        selection that ``offset()`` does inline.
    pitch_yx : (corr_y, corr_x) integer offsets added to every returned shift
        (matches ``offset()``'s ``model.shift_vector += (corr_y, corr_x)`` step
        that converts the ROI-space shift into the tile-pitch shift).
    preprocess : if True, run gaussian + log1p + sobel + Hanning on each ROI
        (matches dexp's default pipeline). Set False for a "bare" phase corr.

    Returns
    -------
    list of ``_SimpleTranslationModel`` of length N, in input order. On
    ``_USING_CUPY=False``, falls back to per-item ``dexp_reg.register_translation_nd``.
    """
    rois_a = np.asarray(rois_a) if not _USING_CUPY else rois_a
    rois_b = np.asarray(rois_b) if not _USING_CUPY else rois_b
    if rois_a.shape != rois_b.shape:
        raise ValueError(f"rois_a shape {rois_a.shape} != rois_b shape {rois_b.shape}")
    N = rois_a.shape[0]
    corr_y, corr_x = int(pitch_yx[0]), int(pitch_yx[1])
    # Chunk large batches: peak GPU memory during preprocess+FFT is ~40 MB per
    # (300 × 2660) edge (or ~15 MB per (948 × 650) edge). 839-edge batches
    # blow past 30 GB of intermediates and fragment cupy's pool; smaller
    # chunks fit comfortably and per-launch overhead is still amortized.
    if _USING_CUPY and chunk_size is not None and N > chunk_size:
        models = []
        for i in range(0, N, chunk_size):
            models.extend(batched_phase_correlation(
                rois_a[i:i + chunk_size], rois_b[i:i + chunk_size],
                pitch_yx=pitch_yx, preprocess=preprocess, chunk_size=None,
            ))
        return models

    if not _USING_CUPY:
        # CPU fallback — per-item dexp. Slower but at least this function is
        # callable on CPU nodes for debugging.
        models = []
        for i in range(N):
            m = dexp_reg.register_translation_nd(
                rois_a[i].astype(np.float32), rois_b[i].astype(np.float32),
            )
            sv = np.asarray(m.shift_vector, dtype=np.float64)
            sv += np.array([corr_y, corr_x])
            models.append(_SimpleTranslationModel(sv, float(m.confidence)))
        return models

    # GPU path. One stream per thread lets multiple batched calls from different
    # threads overlap on the GPU.
    stream = _gpu_stream()
    with stream:
        a = xp.asarray(rois_a, dtype=xp.float32)
        b = xp.asarray(rois_b, dtype=xp.float32)
        if preprocess:
            a = _preprocess_batch_gpu(a)
            b = _preprocess_batch_gpu(b)
        # Batched FFT2 — cufft fans out across the batch dim.
        Fa = xp.fft.fft2(a)
        Fb = xp.fft.fft2(b)
        cross = Fa * xp.conj(Fb)
        cross = cross / (xp.abs(cross) + 1e-10)
        corr = xp.fft.ifft2(cross).real  # (N, H, W)
        H, W = corr.shape[-2], corr.shape[-1]
        # Peak per batch item via a flat argmax on axis=(-2, -1).
        flat = corr.reshape(N, H * W)
        peak_flat = xp.argmax(flat, axis=1)  # (N,)
        peak_val = xp.take_along_axis(flat, peak_flat[:, None], axis=1).squeeze(1)
        raw_py = peak_flat // W
        raw_px = peak_flat % W
        # Peak-to-background confidence: zero out a small window around each
        # peak, take max of remainder, confidence = (peak - bg) / (peak + eps).
        m_h = max(8, int(H ** 0.9) // 8)
        m_w = max(8, int(W ** 0.9) // 8)
        y_grid = xp.arange(H, dtype=xp.int32)[None, :, None]  # (1, H, 1)
        x_grid = xp.arange(W, dtype=xp.int32)[None, None, :]  # (1, 1, W)
        py = raw_py[:, None, None].astype(xp.int32)
        px = raw_px[:, None, None].astype(xp.int32)
        mask = (xp.abs(y_grid - py) < m_h) & (xp.abs(x_grid - px) < m_w)
        masked = xp.where(mask, xp.float32(0.0), corr)
        bg = masked.reshape(N, H * W).max(axis=1)
        conf = (peak_val - bg) / (peak_val + 1e-6)
        # Wrap peak coordinates from [0, N) to signed [-N/2, N/2).
        shift_y = xp.where(raw_py > H // 2, raw_py - H, raw_py).astype(xp.float64)
        shift_x = xp.where(raw_px > W // 2, raw_px - W, raw_px).astype(xp.float64)
        # One D2H sync — bring N shifts + N confidences back to CPU.
        shift_y_np = xp.asnumpy(shift_y)
        shift_x_np = xp.asnumpy(shift_x)
        conf_np = xp.asnumpy(conf)

    models = []
    for i in range(N):
        sv = np.array([shift_y_np[i] + corr_y, shift_x_np[i] + corr_x], dtype=np.float64)
        models.append(_SimpleTranslationModel(sv, float(conf_np[i])))
    return models


def offset(
    image_a: np.array, image_b: np.array, relation: tuple, overlap
) -> "TranslationRegistrationModel":
    """
    overlap: estimate of the # pixels seen in both images. Either a single int
        (same overlap on both axes) or a tuple ``(overlap_x, overlap_y)`` —
        ``overlap_x`` sizes the X-axis ROI (column neighbors), ``overlap_y`` the
        Y-axis ROI (row neighbors).
    """
    if isinstance(overlap, (tuple, list)):
        ov_x, ov_y = int(overlap[0]), int(overlap[1])
    else:
        ov_x = ov_y = int(overlap)
    shape = image_a.shape
    # TODO: have this load only a small fraction of the entire image
    if relation[0] == -1:
        # tile_b is to the right of tile_a
        roi_a = image_a[:, -ov_x:]
        roi_b = image_b[:, :ov_x]
        corr_x = shape[-1] - ov_x  # X pitch from tile WIDTH (shape[-1])
        corr_y = 0

    if relation[1] == -1:
        # tile_b is below tile_a
        roi_a = image_a[-ov_y:, :]
        roi_b = image_b[:ov_y, :]
        corr_x = 0
        corr_y = shape[-2] - ov_y  # Y pitch from tile HEIGHT (shape[-2])

    # phase images are centered at 0, shift to make them all positive
    roi_a_min = xp.min(roi_a)
    roi_b_min = xp.min(roi_b)
    if roi_a_min < 0:
        roi_a = roi_a - roi_a_min
    if roi_b_min < 0:
        roi_b = roi_b - roi_b_min

    if _USING_CUPY:
        # Run dexp's full pipeline (denoise + log + sobel + Hanning window +
        # phase correlation + peak-to-background confidence) on GPU by
        # pushing a CupyBackend onto dexp's thread-local backend stack.
        # ``force_numpy=True`` brings shift_vector/confidence back to numpy
        # inside the CupyBackend context (dexp's default is False, which
        # would leave them as cupy arrays and break the ``+= np.array(...)``
        # below with "Implicit conversion to a NumPy array is not allowed").
        # Per-thread stream lets 16 concurrent calls overlap kernels.
        with _dexp_cupy_backend(), _gpu_stream():
            # ``internal_dtype=np.float32`` — dexp's CPU path forces float32
            # via a ``type(Backend.current()) is NumpyBackend`` check that
            # doesn't fire on CupyBackend; without the explicit override, an
            # uint16 input would keep uint16 as internal dtype and dexp's
            # in-place ``image *= hanning`` window multiply hits numpy's
            # same_kind cast rule and raises.
            model = dexp_reg.register_translation_nd(
                roi_a, roi_b, force_numpy=True, internal_dtype=np.float32,
            )
    else:
        model = dexp_reg.register_translation_nd(roi_a, roi_b)
    model.shift_vector = np.asarray(model.shift_vector, dtype=np.float64)
    model.shift_vector += np.array([corr_y, corr_x])
    return model


def _process_edge_pair(key, pos, tile_cache, overlap):
    """Worker function to process a single edge pair."""
    edge_model = Edge(pos[0], pos[1], tile_cache, overlap=overlap)

    # positions need to be not np.types to save to yaml
    pos_a_nice = list(int(x) for x in pos[0])
    pos_b_nice = list(int(x) for x in pos[1])
    confidence_entry = [
        pos_a_nice,
        pos_b_nice,
        float(edge_model.model.confidence),
    ]

    return key, edge_model, confidence_entry


def linsolve_gpu_lsqr(a_sparse, y, x0, tolerance=1e-5, weights=None):
    """
    GPU-accelerated L2 least squares solver using CuPy's LSMR.

    Minimizes: ||Ax - y||₂ (or, with ``weights``, the weighted ||diag(√w)(Ax-y)||₂)

    Args:
        a_sparse: scipy sparse CSR matrix (n_edges × n_tiles)
        y: target vector (n_edges,)
        x0: initial guess (n_tiles,)
        tolerance: convergence tolerance

    Returns:
        x: solution vector (n_tiles,)
    """
    if not _USING_CUPY:
        raise RuntimeError("CuPy not available for GPU LSMR")

    import cupyx.scipy.sparse.linalg as gpu_linalg
    import cupyx.scipy.sparse as gpu_sparse
    import time

    t_start = time.time()

    # Transfer sparse matrix to GPU (CSR format is efficient)
    # Using float64 for better numerical accuracy to match CPU solver
    a_gpu = gpu_sparse.csr_matrix(a_sparse, dtype=xp.float64)
    y_gpu = xp.asarray(y, dtype=xp.float64)
    x0_gpu = xp.asarray(x0, dtype=xp.float64)

    # Optional per-row confidence weighting: min ||diag(√w)(Ax - y)||₂. LSMR has no
    # native row weights, so pre-scale each row of A and y by √w.
    if weights is not None:
        w_sqrt = xp.sqrt(xp.asarray(weights, dtype=xp.float64))
        a_gpu = gpu_sparse.diags(w_sqrt, format="csr") @ a_gpu
        y_gpu = w_sqrt * y_gpu

    # Solve using LSMR (iterative solver optimized for sparse systems)
    # Note: Use lsmr not lsqr - lsmr supports x0, atol, btol parameters
    result = gpu_linalg.lsmr(
        a_gpu,
        y_gpu,
        x0=x0_gpu,
        atol=tolerance,
        btol=tolerance,
        maxiter=1000
    )

    x_solution = result[0]  # Solution vector
    istop = result[1]  # Stopping condition
    itn = result[2]  # Number of iterations
    normr = result[3]  # Norm of residual ||Ax - y||
    normar = result[4]  # Norm of A^T * residual
    normA = result[5]  # Frobenius norm of A

    # Transfer back to CPU
    x_cpu = xp.asnumpy(x_solution)

    t_elapsed = time.time() - t_start
    print(f"    GPU LSMR: {itn} iterations, istop={istop}, residual={normr:.6e}, time={t_elapsed:.3f}s")

    return x_cpu


def linsolve_gpu_irls_l1(a_sparse, y, x0, tolerance=1e-5, max_iter=20, outer_tolerance=1e-3, min_iter=5, conf_weights=None):
    """
    GPU-accelerated L1 minimization using Iteratively Reweighted Least Squares (IRLS).

    Minimizes: ||Ax - y||₁ (L1 norm - robust to outliers)

    IRLS approximates L1 minimization by solving a sequence of weighted L2 problems:
    - Start with x = x0
    - Repeat:
        1. Compute residuals: r = Ax - y
        2. Compute weights: w_i = 1 / (|r_i| + ε)
        3. Solve weighted LS: min ||W^(1/2)(Ax - y)||₂
        4. Update x
    - Until convergence

    Args:
        a_sparse: scipy sparse CSR matrix (n_edges × n_tiles)
        y: target vector (n_edges,)
        x0: initial guess (n_tiles,)
        tolerance: inner LSMR convergence tolerance (default 1e-5)
        max_iter: maximum IRLS outer iterations (default 20)
        outer_tolerance: outer IRLS convergence tolerance (default 1e-3)
        min_iter: minimum IRLS outer iterations (default 5)

    Returns:
        x: solution vector (n_tiles,)
    """
    if not _USING_CUPY:
        raise RuntimeError("CuPy not available for GPU IRLS")

    import cupyx.scipy.sparse.linalg as gpu_linalg
    import cupyx.scipy.sparse as gpu_sparse
    import time

    t_start = time.time()

    # Transfer to GPU
    # Using float64 for better numerical accuracy to match CPU solver
    a_gpu = gpu_sparse.csr_matrix(a_sparse, dtype=xp.float64)
    y_gpu = xp.asarray(y, dtype=xp.float64)
    x = xp.asarray(x0, dtype=xp.float64)
    # Optional per-edge confidence weight (folded into the IRLS reweight below).
    _cw = None if conf_weights is None else xp.asarray(conf_weights, dtype=xp.float64)

    # Adaptive epsilon based on median residual
    # Start with initial residuals to estimate scale
    initial_residuals = a_gpu @ x - y_gpu
    median_residual = float(xp.median(xp.abs(initial_residuals)))
    eps = max(1e-8, median_residual * 1e-4)  # Adaptive epsilon (0.01% of median residual)

    # Compute initial L1 norm for convergence check
    l1_initial = float(xp.sum(xp.abs(initial_residuals)))

    total_iterations = 0
    l1_prev = l1_initial
    l1_best = l1_initial
    converged = False
    stable_iterations = 0  # Count consecutive stable iterations
    no_improvement_count = 0  # Count iterations without improvement

    for iter_num in range(max_iter):
        # Compute residuals
        residuals = a_gpu @ x - y_gpu

        # Compute L1 norm (objective function we're minimizing)
        l1_norm = float(xp.sum(xp.abs(residuals)))

        # Track best L1 seen so far
        if l1_norm < l1_best:
            l1_best = l1_norm
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        # Check convergence based on:
        # 1. Consecutive stability (L1 not changing much)
        # 2. Reached minimum iteration requirement
        # 3. Actually made progress (L1 decreased significantly)
        if iter_num > 0:
            l1_relative_change = abs(l1_norm - l1_prev) / (abs(l1_prev) + 1e-10)
            l1_reduction = l1_norm / l1_initial  # Fraction of initial residual

            # Track stability: if L1 change is small, increment counter
            if l1_relative_change < outer_tolerance:
                stable_iterations += 1
            else:
                stable_iterations = 0  # Reset if not stable

            # Converge if all conditions met:
            # - At least min_iter iterations completed
            # - L1 objective stable for 3 consecutive iterations (indicates local minimum)
            # - No improvement for last 3 iterations (truly stuck at minimum)
            if (iter_num >= min_iter and
                stable_iterations >= 3 and
                no_improvement_count >= 3):
                converged = True
                break

        l1_prev = l1_norm

        # Compute weights: w_i = 1 / (|r_i| + eps)
        weights = 1.0 / (xp.abs(residuals) + eps)
        if _cw is not None:
            # Fold in per-edge confidence: w_total = w_confidence · w_IRLS, so the
            # iteration approximates min Σ w_confidence·|r| (confidence-weighted L1).
            weights = weights * _cw

        # Create diagonal weight matrix: W = diag(sqrt(weights))
        # For weighted LS: min ||W^(1/2)(Ax - y)||₂ = min ||W_sqrt*A*x - W_sqrt*y||₂
        w_sqrt = xp.sqrt(weights)

        # Apply weights: multiply each row by corresponding weight
        # Convert to diagonal sparse matrix for efficient multiplication
        w_diag = gpu_sparse.diags(w_sqrt, format='csr')
        a_weighted = w_diag @ a_gpu
        y_weighted = w_sqrt * y_gpu

        # Solve weighted least squares with inner tolerance
        result = gpu_linalg.lsmr(
            a_weighted,
            y_weighted,
            x0=x,
            atol=tolerance,  # Inner LSMR tolerance (tight)
            btol=tolerance,
            maxiter=1000
        )

        x_new = result[0]
        total_iterations += result[2]  # Accumulate LSMR iterations

        x = x_new

    # Transfer back to CPU
    x_cpu = xp.asnumpy(x)

    # Compute final residual and L1 norm
    residuals_final = a_gpu @ x - y_gpu
    l1_final = float(xp.sum(xp.abs(residuals_final)))

    t_elapsed = time.time() - t_start

    # Convergence diagnostics
    convergence_status = "converged" if converged else "max_iter"
    if iter_num > 0:
        final_l1_change = abs(l1_final - l1_prev) / (abs(l1_prev) + 1e-10)
    else:
        final_l1_change = 0.0

    l1_reduction = l1_final / l1_initial  # What fraction remains

    print(f"    GPU IRLS (L1): {iter_num+1} outer iterations ({convergence_status}), "
          f"{total_iterations} total inner iterations, "
          f"L1={l1_final:.3e} (best={l1_best:.3e}, {100*l1_best/l1_initial:.1f}% of initial), "
          f"ΔL1={final_l1_change:.3e}, stable={stable_iterations}, time={t_elapsed:.3f}s")

    return x_cpu


def pairwise_shifts(
    positions: List,
    store_path: str,
    well: str,
    flipud: bool,
    fliplr: bool,
    rot90: bool,
    overlap: int = 150,
    max_workers: int = 16,
    channel: int = 0,
    timepoint: int = 0,
    use_clahe: bool = False,
    clahe_clip_limit: float = 0.02,
    verbose: bool = False,
) -> List:
    """ """
    # get neighboring tiles
    grid_positions = parse_positions(positions)
    hilbert_order = hilbert_over_points(grid_positions)
    edges_hilbert = connectivity(hilbert_order)

    tile_cache = TileCache(
        store_path=store_path,
        well=well,
        flipud=flipud,
        fliplr=fliplr,
        rot90=rot90,
        channel=channel,
        timepoint=timepoint,
        use_clahe=use_clahe,
        clahe_clip_limit=clahe_clip_limit,
    )

    edge_list = []
    confidence_dict = {}

    # Parallel processing of edge pairs using ThreadPoolExecutor
    num_edges = len(edges_hilbert)
    print(f"[pairwise_shifts] Processing {num_edges} edge pairs with {max_workers} workers")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all edge pair computations
        futures = {
            executor.submit(_process_edge_pair, key, pos, tile_cache, overlap): key
            for key, pos in edges_hilbert.items()
        }

        # Collect results with progress bar
        for future in tqdm(as_completed(futures), total=num_edges, desc="Computing edge shifts"):
            key, edge_model, confidence_entry = future.result()
            edge_list.append(edge_model)

            if verbose:
                confidence = confidence_entry[2]
                conf_str = f"{confidence:.4f}"
                if confidence < 0.3:
                    conf_str = f"{conf_str} [LOW]"
                elif confidence < 0.5:
                    conf_str = f"{conf_str} [WARN]"
                print(f"  Edge {key}: {confidence_entry[0]} <-> {confidence_entry[1]} confidence={conf_str}")

            confidence_dict[key] = confidence_entry

    return edge_list, confidence_dict


WEIGHT_EPS = 1e-3  # weight floor: keeps every edge nonzero so no tile is left
                   # under-constrained (the solve has no Tikhonov term).


def _edge_weight(conf: float, mode: str) -> float:
    """Map a phase-correlation confidence (0..1) to a global-solve weight.

    mode: 'linear' (w=conf), 'squared' (w=conf**2), or 'threshold:<cutoff>'
    (w=conf if conf>=cutoff else the floor). Results are clamped to >= WEIGHT_EPS.
    """
    if mode == "linear":
        return max(conf, WEIGHT_EPS)
    if mode == "squared":
        return max(conf, WEIGHT_EPS) ** 2
    if mode.startswith("threshold:"):
        cutoff = float(mode.split(":", 1)[1])
        return conf if conf >= cutoff else WEIGHT_EPS
    raise ValueError(f"unknown confidence weighting mode: {mode!r}")


def optimal_positions(
    edge_list: List,
    tile_lut: Dict,
    well: str,
    tile_size: tuple,
    initial_guess: dict = None,
    confidence_by_edge: dict = None,
    weighting: str = None,
) -> Dict:
    """Solve for globally optimal tile positions from pairwise edge shifts.

    When ``weighting`` is set (and ``confidence_by_edge`` is provided), each edge
    equation is weighted by ``_edge_weight(confidence, weighting)`` so low-confidence
    seams pull less on the global fit. ``confidence_by_edge`` maps
    ``frozenset({tile_a_name, tile_b_name}) -> confidence``. Default (None) is the
    original equal-weight solve, byte-for-byte unchanged.
    """
    # Use float64 for better numerical accuracy to match CPU solver
    y_i = np.zeros(len(edge_list) + 1, dtype=np.float64)
    y_j = np.zeros(len(edge_list) + 1, dtype=np.float64)

    if initial_guess is None:
        # Parse tile indices: "002026" = row 002, col 026
        # i_guess = Y coordinates (row * tile_height)
        # j_guess = X coordinates (col * tile_width)
        i_guess = np.asarray(
            [int(a[:3]) * tile_size[0] for a in tile_lut.keys()]
        )  # row → Y
        j_guess = np.asarray(
            [int(a[3:6]) * tile_size[1] for a in tile_lut.keys()]
        )  # col → X
    else:
        try:
            i_guess = initial_guess[well]["i"]
            j_guess = initial_guess[well]["j"]
        except KeyError:
            "initial guess not formatted correctly"

    # Use float64 for better numerical accuracy to match CPU solver
    a = scipy.sparse.lil_matrix((len(tile_lut), len(edge_list) + 1), dtype=np.float64)

    for c, e in enumerate(edge_list):
        a[[tile_lut[e.tile_a], tile_lut[e.tile_b]], c] = [-1, 1]
        y_i[c] = e.model.shift_vector[0]
        y_j[c] = e.model.shift_vector[1]

    y_i[-1] = 0
    y_j[-1] = 0
    a[0, -1] = 1

    a = a.T.tocsr()
    tolerance = 1e-5
    order_error = 1
    order_reg = 1
    alpha_reg = 0
    maxiter = 1e8

    # Optional confidence weighting. Build a per-row weight vector aligned to the
    # matrix rows (one per edge, plus the anchor row kept at weight 1). Edges are
    # matched to their confidence by tile-name pair — edge_list order is not
    # meaningful (pairwise_shifts appends in as_completed order).
    w = None
    if weighting is not None and confidence_by_edge is not None:
        w = np.ones(len(edge_list) + 1, dtype=np.float64)  # last row = anchor, w=1
        n_missing = 0
        for c, e in enumerate(edge_list):
            conf = confidence_by_edge.get(frozenset({e.tile_a, e.tile_b}))
            if conf is None:
                w[c] = WEIGHT_EPS  # edge not scored -> minimal influence
                n_missing += 1
            else:
                w[c] = _edge_weight(float(conf), weighting)
        print(
            f"[optimal_positions] confidence weighting='{weighting}': "
            f"{len(edge_list) - n_missing}/{len(edge_list)} edges scored, "
            f"w range {w[:-1].min():.3g}..{w[:-1].max():.3g}"
        )

    # Try GPU-accelerated solvers first, fall back to CPU if unavailable
    USE_GPU_FOR_OPTIMIZATION = True  # GPU now enabled

    if _USING_CUPY and USE_GPU_FOR_OPTIMIZATION:
        print("optimizing positions (GPU-accelerated)")
        try:
            # Method 1: Fast L2 solution (LSMR)
            print("  Method 1: GPU LSMR (L2 norm)")
            opt_i_lsqr = linsolve_gpu_lsqr(a, y_i, i_guess, tolerance=tolerance, weights=w)
            opt_j_lsqr = linsolve_gpu_lsqr(a, y_j, j_guess, tolerance=tolerance, weights=w)

            # Method 2: L1 solution (IRLS) - cold start from grid guess to find true L1 minimum
            # Note: Warm-starting from L2 caused premature convergence without reaching L1 optimum
            print("  Method 2: GPU IRLS (L1 norm) - cold start from grid guess")
            opt_i_irls = linsolve_gpu_irls_l1(
                a, y_i, i_guess,           # Cold start from grid guess (not L2!)
                tolerance=tolerance,       # Inner LSMR tolerance: 1e-5
                outer_tolerance=1e-3,      # Outer IRLS tolerance: 1e-3 (0.1% change in L1 objective)
                max_iter=200,              # Optimal: best performance (176s) with excellent convergence
                min_iter=8,                # Increased from 5 to ensure sufficient iterations
                conf_weights=w,
            )
            opt_j_irls = linsolve_gpu_irls_l1(
                a, y_j, j_guess,           # Cold start from grid guess
                tolerance=tolerance,
                outer_tolerance=1e-3,
                max_iter=200,              # Optimal: best performance (176s) with excellent convergence
                min_iter=8,
                conf_weights=w,
            )

            # Compute difference between L1 and L2 solutions
            diff_i = np.linalg.norm(opt_i_lsqr - opt_i_irls)
            diff_j = np.linalg.norm(opt_j_lsqr - opt_j_irls)
            max_diff_i = np.max(np.abs(opt_i_lsqr - opt_i_irls))
            max_diff_j = np.max(np.abs(opt_j_lsqr - opt_j_irls))

            print(f"  L1 vs L2 difference: ||Δi||={diff_i:.3f} (max={max_diff_i:.3f}), "
                  f"||Δj||={diff_j:.3f} (max={max_diff_j:.3f})")

            # Decide which solution to use based on difference
            # If L1 and L2 are very different, use L1 (more robust to outliers)
            # If they're similar, either is fine (L2 is slightly faster)
            if max(diff_i, diff_j) > 100:
                opt_i = opt_i_irls
                opt_j = opt_j_irls
                print("  Using L1 (IRLS) solution (large L1 vs L2 difference indicates outliers)")
            else:
                opt_i = opt_i_irls
                opt_j = opt_j_irls
                print("  Using L1 (IRLS) solution (default for robustness)")

        except Exception as e:
            print(f"  GPU solvers failed: {e}")
            print("  Falling back to CPU")
            _USING_CUPY_FOR_OPTIM = False
    else:
        _USING_CUPY_FOR_OPTIM = False

    # CPU fallback
    if not _USING_CUPY or not USE_GPU_FOR_OPTIMIZATION or ('_USING_CUPY_FOR_OPTIM' in locals() and not _USING_CUPY_FOR_OPTIM):
        # Use CPU solver - PyTorch LBFGS doesn't converge properly for sparse linear systems
        # It stops after only ~7 function evaluations with 0% improvement
        # scipy.optimize.minimize with L-BFGS-B is much better suited for this problem
        print("optimizing positions (CPU with scipy L-BFGS-B)")
        # Confidence weighting (external dexp linsolve has no row weights): pre-scale
        # the system rows by √w so low-confidence edges contribute less.
        a_cpu, y_i_cpu, y_j_cpu = a, y_i, y_j
        if w is not None:
            w_sqrt = np.sqrt(w)
            a_cpu = scipy.sparse.diags(w_sqrt) @ a
            y_i_cpu = w_sqrt * y_i
            y_j_cpu = w_sqrt * y_j
        opt_i = linsolve(
            a_cpu,
            y_i_cpu,
            tolerance=tolerance,
            order_error=order_error,
            order_reg=order_reg,
            alpha_reg=alpha_reg,
            x0=i_guess,
            maxiter=maxiter,
        )
        opt_j = linsolve(
            a_cpu,
            y_j_cpu,
            tolerance=tolerance,
            order_error=order_error,
            order_reg=order_reg,
            alpha_reg=alpha_reg,
            x0=j_guess,
            maxiter=maxiter,
        )

    opt_shifts = np.vstack((opt_i, opt_j)).T

    opt_shifts_zeroed = opt_shifts - np.min(opt_shifts, axis=0)

    # needs to be a list of python types for exporting to yaml
    opt_shifts_dict = {
        f"{well}/{a}": [int(a) for a in opt_shifts_zeroed[i]]
        for i, a in enumerate(tile_lut.keys())
    }

    return opt_shifts_dict