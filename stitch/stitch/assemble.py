from iohub.ngff import open_ome_zarr
import os
import math
import json
import zarr
import dask.array as da
from tqdm import tqdm
from stitch.stitch.tile import augment_tile, pairwise_shifts, optimal_positions
from stitch.connect import read_shifts_biahub
from collections import defaultdict, deque
from iohub.ngff import TransformationMeta
from typing import Any, Callable, Dict, Literal, Optional, Tuple, Type, Union
import itertools
from numpy.typing import ArrayLike
from itertools import product
import yaml
from pathlib import Path
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
from joblib import Parallel, delayed

# Optional override of iohub's default zstd codec for v3 outputs.
# STITCH_V3_CODEC=lz4 → faster encode, larger files.
# STITCH_V3_CODEC=zstd (default) → iohub default (clevel=1, bitshuffle).
# STITCH_V3_CODEC=none → no compression, fastest commits, biggest files.
_v3_codec_env = os.environ.get("STITCH_V3_CODEC", "").strip().lower()
if _v3_codec_env in ("lz4", "none"):
    try:
        import iohub.ngff.nodes as _ngff_nodes
        import zarr.codecs as _zc
        _orig_create_compressor_options = _ngff_nodes.Position._create_compressor_options

        def _patched_create_compressor_options(self):
            shuffle = _zc.BloscShuffle.bitshuffle
            if self._zarr_format == 3:
                if _v3_codec_env == "none":
                    return {"compressors": None}
                return {
                    "compressors": _zc.BloscCodec(
                        cname=_v3_codec_env, clevel=1, shuffle=shuffle,
                    )
                }
            # v2 path unchanged
            return _orig_create_compressor_options(self)

        _ngff_nodes.Position._create_compressor_options = _patched_create_compressor_options
        print(f"[v3-codec] Overriding iohub default zstd → {_v3_codec_env}")
    except Exception as e:
        print(f"[v3-codec] WARN: failed to patch iohub codec ({e}); using default zstd")


try:
    from dask.distributed import LocalCluster, Client
    _DASK_DISTRIBUTED_AVAILABLE = True
except ImportError:
    _DASK_DISTRIBUTED_AVAILABLE = False

# Import GPU/CPU abstraction and helper functions from utils
from stitch.stitch.utils import (
    xp,
    cundi,
    _USING_CUPY,
    _EDT_WEIGHT_CACHE,
    _to_numpy,
    _load_to_gpu_async,
    _get_optimal_workers,
    _discover_positions_fast,
    find_contributing_fovs_yx,
    _get_optimal_block_size,
    _resolve_value_dtype,
    _resolve_shift_dtype,
    get_output_shape,
)


# ── Module-level per-process write pipeline ──────────────────────────────────
# These persist across Dask tasks within the same worker process, enabling
# write pipelining: band N's writes run in background while band N+1 loads+GPU.
#
# STITCH_WRITE_WINDOW (default 1) bounds how many bands' writes may be in
# flight per worker. window=1 reproduces the legacy behaviour (drain prior
# before submitting next). window=2+ lets the worker submit and move on,
# only draining when the window fills — burying NFS write latency under
# the next band's compute. Memory cost: each in-flight band holds its
# numpy buffer (~5.6 GB for ISS) until writes complete.
_band_write_pool = None
_band_window: "deque" = deque()  # FIFO of per-band write slots
_blosc_benchmark_done = False


# ── CPU / IO monitoring ──────────────────────────────────────────────────────
from ops_utils.profiling.proc_monitor import start_monitor as _start_cpu_monitor


# Set True while >1 well shares the same GPU (use_thread_wells / use_dask_wells)
# so per-band cleanup hands blocks back to CUDA and other workers can grab them.
# In single-well sequential mode we keep the pool warm — alloc cost drops from
# 300-800ms to ~2-3ms per band (observed in py-spy + STITCH_PROFILE=1).
_PARALLEL_WELLS_ACTIVE = False


def _maybe_free_pool() -> None:
    """Release CuPy memory-pool blocks back to CUDA driver only when needed.

    Defaults: keep blocks (single-well case = fastest re-allocation).
    Auto-flips to "free" when parallel wells are sharing one GPU.
    Override via ``STITCH_FREE_POOL_PER_BAND=1`` (always free) or
    ``STITCH_FREE_POOL_PER_BAND=0`` (never free, even under parallel wells —
    only safe if the GPU has headroom for all wells' working sets).
    """
    override = os.environ.get("STITCH_FREE_POOL_PER_BAND", "")
    if override in ("0", "false", "no"):
        return
    if override in ("1", "true", "yes") or _PARALLEL_WELLS_ACTIVE:
        try:
            xp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass


def _v3_shards_attr(stitched_pos):
    """Return the v3 shards_ratio that the outer stitch() set on this
    position (None if the output is v2). This lets the assembly inner
    functions pass shards_ratio to create_zeros without us having to
    thread a parameter through every layer of the call chain — the outer
    stitch() function tags each position right after it creates them.
    """
    return getattr(stitched_pos, "_v3_shards_ratio", None)


class _TensorStoreWriteAdapter:
    """Adapter that makes a tensorstore-backed array support numpy-style
    ``arr[slices] = value`` assignment.

    zarr-python 3.1's sharding codec has a path in ``encode_partial`` that
    silently loses data on certain unaligned slice writes (we observed
    scattered missing chunks even for nominally aligned writes via
    ``stitched_pos["0"][...] = block``). The convert_v3 conversion script
    sidesteps this by routing all writes through tensorstore, whose
    sharding implementation correctly handles partial-shard
    read-modify-writes.

    This adapter wraps a tensorstore handle and exposes ``__setitem__`` so
    the existing stitcher write call sites
    (``arr_out[(s, s, s, slice(y0,y1), slice(x0,x1))] = norm_cpu``) keep
    working without invasive changes. ``__getattr__`` and ``shape`` /
    ``chunks`` attribute pass-through let the rest of the code interact
    with it like a zarr/iohub array.
    """

    def __init__(self, ts_array, shape, chunks=None, dtype=None,
                 use_transaction: bool = True):
        self._ts = ts_array
        self.shape = tuple(shape)
        self.chunks = tuple(chunks) if chunks is not None else None
        self.dtype = dtype
        # Sharded v3 writes are catastrophically slow without coalescing:
        # each X-block write triggers a full shard read-modify-write
        # (~358ms per write × 2548 writes/well ≈ 970s drain on real bench).
        # We commit every STITCH_TXN_BAND_GROUP bands (default 8) — staging
        # ~16 GB per txn — to bound peak memory. Per-WELL txn (1 commit)
        # OOM'd at 368 GB on real bench; per-BAND txn (52 commits) is
        # memory-safe but ~80s slower from extra shard re-encodes.
        self._use_transaction = use_transaction
        self._txn = None
        self._pending_commits = []  # in-flight async commits from prior groups
        self._txn_lock = threading.Lock()
        # Bands written into the current txn since it was opened.
        self._band_count_in_txn = 0
        # Default 999 = act as per-well (single commit at end of well).
        # Smaller values trade more frequent commits for bounded memory.
        self._band_group_size = max(1, int(os.environ.get("STITCH_TXN_BAND_GROUP", "999")))

    def begin_band(self):
        """Lazy-open a Transaction if one isn't already active. Single
        Transaction batches up to ``_band_group_size`` bands' worth of
        X-block writes before commit, balancing shard-re-encode count
        (fewer = better wall) against staged-memory peak (smaller = no OOM)."""
        if not self._use_transaction:
            return
        if self._txn is not None:
            return
        try:
            import tensorstore as _ts
            with self._txn_lock:
                if self._txn is None:
                    self._txn = _ts.Transaction()
                    self._band_count_in_txn = 0
        except Exception as e:
            print(f"[v3-native] WARN: Transaction unavailable ({e}); falling back to non-coalesced writes")
            self._txn = None

    def end_band_async(self):
        """Mark a band's writes as staged. Commit and reset the txn when
        ``_band_group_size`` bands have accumulated, capping memory."""
        with self._txn_lock:
            self._band_count_in_txn += 1
            if self._band_count_in_txn < self._band_group_size:
                return
            txn = self._txn
            self._txn = None
            self._band_count_in_txn = 0
        if txn is not None:
            self._pending_commits.append(txn.commit_async())

    @property
    def txn_view(self):
        """Returns the tensorstore handle bound to the open transaction (if
        any) so writes accumulate before commit."""
        with self._txn_lock:
            txn = self._txn
        if txn is not None:
            return self._ts.with_transaction(txn)
        return self._ts

    def __setitem__(self, key, value):
        self.txn_view[key].write(value).result()

    def __getitem__(self, key):
        return self.txn_view[key].read().result()

    def commit(self):
        """Final drain: commit any live txn, then wait on all pending
        async commits. Called once at end of well.

        In sequential mode this is THE single per-well commit (~4 shard
        re-encodes). In parallel-wells mode end_band_async has already
        kicked off per-band commits; this just drains them."""
        with self._txn_lock:
            txn = self._txn
            self._txn = None
        if txn is not None:
            # Issue the commit; reuse the same drain loop below.
            self._pending_commits.append(txn.commit_async())
        for fut in self._pending_commits:
            fut.result()
        self._pending_commits = []

    def __getattr__(self, name):
        return getattr(self._ts, name)


def _maybe_wrap_for_tensorstore(arr_obj, stitched_pos=None):
    """If ``arr_obj`` is backed by a v3 sharded zarr array (detected via the
    position's ``_v3_shards_ratio`` tag), return a ``_TensorStoreWriteAdapter``
    wrapping its tensorstore handle. Otherwise return ``arr_obj`` unchanged.

    Inputs:
      - ``arr_obj``: an iohub ImageArray or raw zarr Array
      - ``stitched_pos``: optional iohub Position used to detect v3-sharded mode
    """
    use_ts = False
    if stitched_pos is not None and _v3_shards_attr(stitched_pos) is not None:
        use_ts = True
    if not use_ts:
        return arr_obj
    try:
        import tensorstore as ts
        # Override tensorstore's default thread pools so the commit phase
        # can saturate our SLURM core allocation. Defaults are small (~4-8
        # for data_copy_concurrency) and we observed only ~18 cores in use
        # during commit on a 32-core allocation. Env-var tunable.
        _ts_dc = int(os.environ.get("STITCH_TS_DATA_COPY_CONCURRENCY", "32"))
        _ts_io = int(os.environ.get("STITCH_TS_FILE_IO_CONCURRENCY", "16"))
        # Bound the cache pool. Default tensorstore cache pool is unbounded
        # and stores decoded shards/chunks. For a write-heavy workload like
        # stitch we don't benefit from a big read cache — bounding it lets
        # us fit more workers in the SLURM memory budget. 2 GB default is
        # enough for tensorstore's internal scratch without growing
        # unbounded across the run.
        _ts_cache = int(os.environ.get("STITCH_TS_CACHE_BYTES", str(2 * 1024**3)))
        ts_context = ts.Context({
            "cache_pool": {"total_bytes_limit": _ts_cache},
            "data_copy_concurrency": {"limit": _ts_dc},
            "file_io_concurrency": {"limit": _ts_io},
        })

        # iohub ImageArray exposes .tensorstore(); raw zarr arrays don't.
        if hasattr(arr_obj, "tensorstore"):
            base_ts = arr_obj.tensorstore()
            # Reopen with our context so concurrency limits apply. The spec
            # already encodes the storage location and codec.
            try:
                ts_array = ts.open(base_ts.spec(), context=ts_context).result()
            except Exception:
                ts_array = base_ts  # fall back to iohub's default context
        else:
            # For raw zarr v3 arrays, we need to open them via tensorstore
            # ourselves. The path is reachable via the array's store info.
            store_root = str(arr_obj.store.root) if hasattr(arr_obj.store, "root") else None
            arr_path = arr_obj.path  # e.g. "A/1/0/0"
            if store_root is None:
                return arr_obj  # can't wrap; fall back to direct writes
            full_path = f"{store_root}/{arr_path}" if arr_path else store_root
            ts_array = ts.open({
                "driver": "zarr3",
                "kvstore": {"driver": "file", "path": full_path},
            }, context=ts_context).result()
        return _TensorStoreWriteAdapter(
            ts_array,
            shape=arr_obj.shape,
            chunks=getattr(arr_obj, "chunks", None),
            dtype=getattr(arr_obj, "dtype", None),
        )
    except Exception as e:
        print(f"[v3-native] WARN: tensorstore wrap failed ({e}); using direct zarr writes")
        return arr_obj


def _init_band_write_pool(max_workers=8):
    """Get or create the persistent write thread pool for this worker process."""
    global _band_write_pool
    if _band_write_pool is None:
        _band_write_pool = ThreadPoolExecutor(max_workers=max_workers)
    return _band_write_pool


def _drain_band_slot(slot):
    """Wait for one band's writes; print stats; release buffers."""
    futures = slot["futures"]
    if not futures:
        return
    total_bytes = 0
    block_times = []
    for fut in futures:
        data_bytes, elapsed = fut.result()
        total_bytes += data_bytes
        block_times.append(elapsed)
    wall_time = time.time() - slot["submit_time"]
    n = len(block_times)
    avg_t = sum(block_times) / n if n else 0
    max_t = max(block_times) if block_times else 0
    min_t = min(block_times) if block_times else 0
    throughput = total_bytes / wall_time / 1e6 if wall_time > 0 else 0
    print(f"[Write Stats] {n} blocks, {total_bytes/1e9:.2f}GB, "
          f"wall={wall_time:.1f}s ({throughput:.0f}MB/s), "
          f"per_block: min={min_t:.1f}s avg={avg_t:.1f}s max={max_t:.1f}s")
    slot["futures"].clear()
    slot["data"].clear()


def _maybe_drain_oldest():
    """Drain the oldest band slot if the window is at capacity. Returns the
    wait time. Call BEFORE submitting a new band's writes so the thread
    pool queue ordering matches the legacy `drain-then-submit` semantics
    when STITCH_WRITE_WINDOW=1.
    """
    global _band_window
    max_window = max(1, int(os.environ.get("STITCH_WRITE_WINDOW", "1") or "1"))
    if len(_band_window) < max_window:
        return 0.0
    t0 = time.time()
    _drain_band_slot(_band_window.popleft())
    return time.time() - t0


def _record_band_writes(futures, data_refs):
    """Append a band slot to the window. Caller must have already called
    _maybe_drain_oldest() to make room.
    """
    global _band_window
    _band_window.append({
        "futures": list(futures),
        "data": list(data_refs),  # hold numpy refs until writes complete
        "submit_time": time.time(),
    })


def _push_band_writes(futures, data_refs):
    """Convenience: drain-if-full → record. Equivalent to the legacy
    drain-then-submit pattern at STITCH_WRITE_WINDOW=1.
    """
    t_wait = _maybe_drain_oldest()
    _record_band_writes(futures, data_refs)
    return t_wait


def _drain_all_bands():
    """Drain every queued band slot. Called at well-final or job-final fences."""
    global _band_window
    while _band_window:
        _drain_band_slot(_band_window.popleft())


def _wait_band_writes():
    """Compatibility shim — the previous semantics were `drain everything`,
    which we now provide via _drain_all_bands().
    """
    _drain_all_bands()


def _run_blosc_benchmark(sample_chunk):
    """One-time blosc compression benchmark to separate compression vs I/O costs."""
    global _blosc_benchmark_done
    if _blosc_benchmark_done:
        return
    _blosc_benchmark_done = True
    import numcodecs
    codec = numcodecs.Blosc(cname='lz4', clevel=1, shuffle=numcodecs.Blosc.BITSHUFFLE)
    chunk_bytes = sample_chunk.nbytes
    # Benchmark compression
    t0 = time.time()
    n_iters = 5
    for _ in range(n_iters):
        compressed = codec.encode(sample_chunk)
    compress_time = (time.time() - t0) / n_iters
    ratio = chunk_bytes / len(compressed)
    # Estimate for full band: ~825 chunks, 8 writer threads
    est_total = 825 * compress_time / 8
    print(f"[Blosc Bench] chunk={chunk_bytes/1e6:.1f}MB, compress={compress_time*1000:.1f}ms, "
          f"ratio={ratio:.1f}x, compressed={len(compressed)/1e6:.1f}MB, "
          f"est_825_chunks_8threads={est_total:.1f}s (PID={os.getpid()})")


def _compute_edt_weights(tile_size, blending_exponent, dtype_val):
    """Compute EDT blending weights for a tile of the given size."""
    ty, tx = int(tile_size[0]), int(tile_size[1])
    _mask = xp.zeros((ty, tx), dtype=bool)
    if ty > 2 and tx > 2:
        _mask[1:-1, 1:-1] = True
    _dist = cundi.distance_transform_edt(_mask).astype(xp.float32)
    _dist += 1e-6
    _weights = xp.where(_dist > 0, _dist ** float(blending_exponent), 0.0)
    return xp.asarray(_weights, dtype=dtype_val)


def _process_x_block_gpu(x0, x1, y0, y1, y_tiles, tile_cache, final_shape, dtype_val, use_edt, tile_weights, stream=None, profile=False):
    """
    Process a single X block on GPU, optionally using a specific CUDA stream.

    Returns:
        If profile=False: norm_cpu array
        If profile=True:  (norm_cpu, profile_dict) where profile_dict has timing breakdown
    """
    timings = {} if profile else None

    # Create buffers on the specified stream (or default)
    with xp.cuda.Stream(stream) if stream is not None else xp.cuda.Stream.null:
        if profile:
            xp.cuda.Device().synchronize()
            t0 = time.time()

        numer = xp.zeros(
            (final_shape[0], final_shape[1], final_shape[2], y1 - y0, x1 - x0),
            dtype=dtype_val,
        )
        denom = xp.zeros_like(numer, dtype=dtype_val)

        if profile:
            xp.cuda.Device().synchronize()
            timings['alloc'] = time.time() - t0
            t_accum_start = time.time()
            n_tiles_hit = 0
            t_slice_total = 0.0
            t_gpu_total = 0.0

        for tile_name, t_end, c_end, z_end, ys, ye, xs, xe in y_tiles:
            # X intersection within this block
            ix0 = max(x0, xs)
            ix1 = min(x1, xe)
            if ix0 >= ix1:
                continue
            iy0 = max(y0, ys)
            iy1 = min(y1, ye)

            if profile:
                t_sl = time.time()

            tile_full, t_end, c_end, z_end, ys, ye, xs, xe = tile_cache[tile_name]
            out_sl = (
                slice(0, t_end),
                slice(0, c_end),
                slice(0, z_end),
                slice(iy0 - y0, iy1 - y0),
                slice(ix0 - x0, ix1 - x0),
            )
            tile_sl = (
                slice(0, t_end),
                slice(0, c_end),
                slice(0, z_end),
                slice(iy0 - ys, iy1 - ys),
                slice(ix0 - xs, ix1 - xs),
            )
            # Slice on CPU first (cheap numpy indexing), then transfer to GPU
            block = xp.asarray(tile_full[tile_sl], dtype=dtype_val)

            if profile:
                t_slice_total += time.time() - t_sl
                t_gpu_op = time.time()

            if use_edt:
                wloc = tile_weights[
                    (iy0 - ys) : (iy1 - ys), (ix0 - xs) : (ix1 - xs)
                ]
                wloc = xp.asarray(wloc, dtype=dtype_val)
                nz = block != 0
                wloc = wloc * nz
                numer[out_sl] += block * wloc
                denom[out_sl] += wloc
            else:
                numer[out_sl] += block
                denom[out_sl] += (block != 0).astype(dtype_val)

            if profile:
                xp.cuda.Device().synchronize()
                t_gpu_total += time.time() - t_gpu_op
                n_tiles_hit += 1

        if profile:
            xp.cuda.Device().synchronize()
            timings['accum_total'] = time.time() - t_accum_start
            timings['accum_slice'] = t_slice_total
            timings['accum_gpu'] = t_gpu_total
            timings['accum_python'] = timings['accum_total'] - t_slice_total - t_gpu_total
            timings['n_tiles_hit'] = n_tiles_hit
            t_norm = time.time()

        # Normalize
        norm = xp.nan_to_num(numer / xp.maximum(denom, 1e-12))

        if profile:
            xp.cuda.Device().synchronize()
            timings['normalize'] = time.time() - t_norm
            t_d2h = time.time()

        # Transfer to CPU for writing (synchronous on this stream)
        norm_cpu = _to_numpy(norm)

        if profile:
            timings['d2h'] = time.time() - t_d2h

        # Clean up
        del numer, denom, norm

        if profile:
            return norm_cpu, timings
        return norm_cpu


def _process_y_band_gpu_t_chunked(
    y0, y1, total_x, y_tiles, tile_cache, final_shape,
    dtype_val, use_edt, tile_weights, t_chunk: int,
    profile: bool = False,
):
    """T-chunked variant of _process_y_band_gpu.

    Designed for ISS-style bands where final_shape[0] (cycles) is large.
    Builds the host (T, C, Z, band_h, total_x) result incrementally by
    looping over T-chunks of width ``t_chunk``. Each chunk allocates its
    own (numer_chunk, denom_chunk) on the GPU sized for t_chunk timepoints
    rather than the full T, so the per-worker GPU peak drops by T / t_chunk.

    Always returns a host numpy array. Forgoes the GPU-return / pipelined-D2H
    optimization in the unchunked path because here each chunk's D2H is
    interleaved with the next chunk's compute via the cupy default stream;
    the caller's pipeline still works at the band granularity.

    Average and EDT blending are both safe under T-chunking: per-tile
    accumulation is independent across the leading T axis (no cross-T
    dependencies anywhere in the kernel), so addition order is preserved
    within each chunk and the final stitched output is bit-identical.
    """
    import numpy as _np

    if profile:
        xp.cuda.Device().synchronize()
        t0 = time.time()

    T = int(final_shape[0])
    C = int(final_shape[1])
    Z = int(final_shape[2])
    band_height = y1 - y0

    # numpy.dtype handles both a numpy class (e.g. numpy.float16) and a
    # numpy.dtype instance directly.
    host_dtype = _np.dtype(dtype_val)

    norm_cpu = _np.zeros((T, C, Z, band_height, total_x), dtype=host_dtype)

    n_chunks = (T + t_chunk - 1) // t_chunk
    if profile:
        timings = {"n_chunks": n_chunks, "t_chunk": t_chunk}
        t_chunk_total = 0.0
    else:
        timings = None

    for t0_chunk in range(0, T, t_chunk):
        t1_chunk = min(t0_chunk + t_chunk, T)
        chunk_t = t1_chunk - t0_chunk
        if profile:
            t_chunk_start = time.time()

        numer = xp.zeros(
            (chunk_t, C, Z, band_height, total_x), dtype=dtype_val,
        )
        denom = xp.zeros_like(numer, dtype=dtype_val)

        for tile_name, t_end, c_end, z_end, ys, ye, xs, xe in y_tiles:
            iy0 = max(y0, ys)
            iy1 = min(y1, ye)
            ix0 = max(0, xs)
            ix1 = min(total_x, xe)
            if ix0 >= ix1 or iy0 >= iy1:
                continue
            # Skip tiles that don't overlap this chunk's T range.
            if t_end <= t0_chunk:
                continue
            tt1 = min(t1_chunk, t_end)
            local_chunk_t = tt1 - t0_chunk
            if local_chunk_t <= 0:
                continue

            tile_full, t_end, c_end, z_end, ys, ye, xs, xe = tile_cache[tile_name]
            out_sl = (
                slice(0, local_chunk_t),
                slice(0, c_end),
                slice(0, z_end),
                slice(iy0 - y0, iy1 - y0),
                slice(ix0, ix1),
            )
            tile_sl = (
                slice(t0_chunk, tt1),
                slice(0, c_end),
                slice(0, z_end),
                slice(iy0 - ys, iy1 - ys),
                slice(ix0 - xs, ix1 - xs),
            )
            cpu_slice = tile_full[tile_sl]
            block = xp.asarray(cpu_slice, dtype=dtype_val)

            if use_edt:
                wloc = tile_weights[
                    (iy0 - ys) : (iy1 - ys), (ix0 - xs) : (ix1 - xs)
                ]
                wloc = xp.asarray(wloc, dtype=dtype_val)
                nz = block != 0
                wloc = wloc * nz
                numer[out_sl] += block * wloc
                denom[out_sl] += wloc
            else:
                numer[out_sl] += block
                denom[out_sl] += (block != 0).astype(dtype_val)

        xp.maximum(denom, 1e-12, out=denom)
        xp.divide(numer, denom, out=numer)
        del denom
        xp.get_default_memory_pool().free_all_blocks()
        xp.nan_to_num(numer, copy=False)

        norm_cpu[t0_chunk:t1_chunk, :, :, :, :] = _to_numpy(numer)
        del numer
        xp.get_default_memory_pool().free_all_blocks()

        if profile:
            t_chunk_total += time.time() - t_chunk_start

    if profile:
        xp.cuda.Device().synchronize()
        timings["wall"] = time.time() - t0
        timings["chunk_total"] = t_chunk_total
        return norm_cpu, timings

    return norm_cpu


def _process_y_band_gpu(y0, y1, total_x, y_tiles, tile_cache, final_shape,
                        dtype_val, use_edt, tile_weights, profile=False,
                        return_gpu=False):
    """Process entire Y-band as a single GPU operation.

    Instead of 52 separate X-block operations with individual alloc/normalize/D2H
    cycles, allocates one buffer spanning the full band width and processes all
    tiles in a single pass.

    If return_gpu=True, returns the GPU array without D2H transfer (caller handles it).

    Caller should wrap this in a per-well CUDA stream context to avoid
    default-stream serialization across wells.

    STITCH_T_CHUNK env var: when set to N>0 and N<C, processes the band in
    chunks of N channels at a time, reducing per-worker GPU peak. Forces the
    return-CPU code path (return_gpu is treated as False) since chunked
    accumulation builds a host result incrementally. Designed for ISS where
    final_shape[1] is large (cycles × channels stacked); defaults off so
    track/pheno keep the existing fast path unchanged.
    """
    timings = {} if profile else None

    # Optional T-chunking — see docstring.
    t_chunk_env = int(os.environ.get("STITCH_T_CHUNK", "0") or "0")
    full_t = int(final_shape[0])
    use_t_chunk = t_chunk_env > 0 and t_chunk_env < full_t
    if use_t_chunk:
        return _process_y_band_gpu_t_chunked(
            y0, y1, total_x, y_tiles, tile_cache, final_shape,
            dtype_val, use_edt, tile_weights, t_chunk_env,
            profile=profile,
        )

    if profile:
        xp.cuda.Device().synchronize()
        t0 = time.time()

    band_height = y1 - y0
    numer = xp.zeros(
        (final_shape[0], final_shape[1], final_shape[2], band_height, total_x),
        dtype=dtype_val,
    )
    denom = xp.zeros_like(numer, dtype=dtype_val)

    if profile:
        xp.cuda.Device().synchronize()
        timings['alloc'] = time.time() - t0
        mempool = xp.get_default_memory_pool()
        timings['alloc_gpu_mb'] = mempool.used_bytes() / 1e6
        t_accum_start = time.time()
        n_tiles_hit = 0
        t_slice_cpu_total = 0.0
        t_h2d_total = 0.0
        t_gpu_kernel_total = 0.0

    for tile_name, t_end, c_end, z_end, ys, ye, xs, xe in y_tiles:
        iy0 = max(y0, ys)
        iy1 = min(y1, ye)
        ix0 = max(0, xs)
        ix1 = min(total_x, xe)
        if ix0 >= ix1 or iy0 >= iy1:
            continue

        tile_full, t_end, c_end, z_end, ys, ye, xs, xe = tile_cache[tile_name]
        if profile:
            t_sl = time.time()
        out_sl = (
            slice(0, t_end),
            slice(0, c_end),
            slice(0, z_end),
            slice(iy0 - y0, iy1 - y0),
            slice(ix0, ix1),
        )
        tile_sl = (
            slice(0, t_end),
            slice(0, c_end),
            slice(0, z_end),
            slice(iy0 - ys, iy1 - ys),
            slice(ix0 - xs, ix1 - xs),
        )
        cpu_slice = tile_full[tile_sl]
        if profile:
            t_slice_cpu_total += time.time() - t_sl
            t_h = time.time()

        # Slice on CPU first (cheap numpy indexing), then transfer to GPU
        block = xp.asarray(cpu_slice, dtype=dtype_val)

        if use_edt:
            wloc = tile_weights[
                (iy0 - ys) : (iy1 - ys), (ix0 - xs) : (ix1 - xs)
            ]
            wloc = xp.asarray(wloc, dtype=dtype_val)
            if profile:
                t_h2d_total += time.time() - t_h
                t_k = time.time()
            nz = block != 0
            wloc = wloc * nz
            numer[out_sl] += block * wloc
            denom[out_sl] += wloc
        else:
            if profile:
                t_h2d_total += time.time() - t_h
                t_k = time.time()
            numer[out_sl] += block
            denom[out_sl] += (block != 0).astype(dtype_val)

        if profile:
            t_gpu_kernel_total += time.time() - t_k
            n_tiles_hit += 1

    if profile:
        xp.cuda.Device().synchronize()
        timings['accum'] = time.time() - t_accum_start
        timings['n_tiles_hit'] = n_tiles_hit
        timings['slice_cpu'] = t_slice_cpu_total
        timings['h2d'] = t_h2d_total
        timings['gpu_kernel'] = t_gpu_kernel_total
        timings['num_streams'] = num_streams
        t_norm = time.time()

    # Normalize in-place to minimize GPU memory (avoids ~20GB of temporaries)
    xp.maximum(denom, 1e-12, out=denom)
    xp.divide(numer, denom, out=numer)
    del denom
    # Pool blocks are kept by default for next band's alloc; opt-in via
    # STITCH_FREE_POOL_PER_BAND=1 for the parallel-wells-per-GPU case
    # where cross-worker pool fragmentation matters more than alloc cost.
    _maybe_free_pool()
    xp.nan_to_num(numer, copy=False)
    norm = numer  # just rename, no allocation

    if profile:
        xp.cuda.Device().synchronize()
        timings['normalize'] = time.time() - t_norm
        mempool = xp.get_default_memory_pool()
        timings['post_norm_gpu_mb'] = mempool.used_bytes() / 1e6

    if return_gpu:
        # Return GPU array — caller handles D2H (for pipelined transfer)
        if profile:
            return norm, timings
        return norm

    if profile:
        t_d2h = time.time()

    # Transfer entire band to CPU
    norm_cpu = _to_numpy(norm)

    if profile:
        timings['d2h'] = time.time() - t_d2h

    del norm

    if profile:
        return norm_cpu, timings
    return norm_cpu


def _d2h_and_submit_writes(norm_gpu, transfer_stream, arr_out, final_shape,
                           y0, y1, tx, total_x, _write_executor, _write_futures):
    """Transfer GPU result to CPU on a dedicated stream, then submit zarr writes.

    Runs in a background thread so the main thread can start the next band's
    GPU compute while this D2H transfer is in progress.  The transfer_stream
    ensures the DMA engine handles the copy independently of the compute stream.
    """
    with transfer_stream:
        # _to_numpy handles both cupy arrays (DMA via the active stream)
        # and host numpy arrays (no-op view) — the latter occurs when
        # _process_y_band_gpu took its T-chunked branch which already
        # produced a host accumulator.
        norm_cpu = _to_numpy(norm_gpu)
    del norm_gpu
    # See _maybe_free_pool: pool blocks kept for next band's alloc by default.
    _maybe_free_pool()

    # v3-native: open a per-band transaction so X-block writes coalesce
    # into one re-encode per shard at commit time. Per-band (vs per-well)
    # bounds memory: a full-well txn staged ~110 GB on real bench and
    # OOM'd at 413 GB across 3 parallel wells.
    band_started = False
    if hasattr(arr_out, "begin_band"):
        arr_out.begin_band()
        band_started = True

    # Split into X-block chunks for zarr chunk alignment
    band_write_futures = []
    for x0 in range(0, total_x, tx):
        x1 = min(total_x, x0 + tx)
        block_cpu = norm_cpu[:, :, :, :, x0:x1]
        fut = _write_executor.submit(
            _write_zarr_block, arr_out, final_shape, y0, y1, x0, x1, block_cpu
        )
        band_write_futures.append(fut)
        _write_futures.append(fut)

    # If we opened a band txn, drain its X-block stages before committing.
    # Stages are memory ops (cheap; ~1ms per write per the synth bench),
    # so this drain typically finishes in <100ms even for 52 X-blocks.
    if band_started:
        for fut in band_write_futures:
            fut.result()
        # commit_async returns immediately; the next band can start GPU
        # work while the prior commit re-encodes shards in tensorstore's
        # internal threads. Final drain happens in arr_out.commit().
        arr_out.end_band_async()


def _write_zarr_block(arr_out, final_shape, y0, y1, x0, x1, norm_cpu):
    """Write a single normalized block to the output zarr array.

    Called from a background thread to overlap writes with GPU processing.
    Returns (data_bytes, elapsed) for profiling.
    """
    t0 = time.time()
    data_bytes = norm_cpu.nbytes
    arr_out[
        (
            slice(0, final_shape[0]),
            slice(0, final_shape[1]),
            slice(0, final_shape[2]),
            slice(y0, y1),
            slice(x0, x1),
        )
    ] = norm_cpu
    return data_bytes, time.time() - t0


def _load_band_tiles(y_tiles, store_path, flipud, fliplr, rot90,
                     well_cache=None, well_cache_lock=None):
    """Load all tiles for a Y-band in parallel, returning a tile_cache dict.

    Uses direct zarr array access (bypasses iohub HCS metadata parsing).
    Used for pipelined I/O: loading band N+1 while GPU processes band N.

    When ``well_cache`` is provided, tiles are cached across bands within
    a well — overlap means each tile would otherwise be re-read 2-3× by
    adjacent bands. With a 200-px overlap on 2048-px tiles + 1024-px Y-bands,
    each tile sees roughly 2 bands, so caching halves NFS read volume.
    The cache is keyed by tile_name; entries hold the augmented numpy
    array (~80 MB for production 5ch×2048² float32 tiles).
    """
    # Local view of just the tiles needed for this band — returned to caller.
    band_view = {}

    def _load_single(tile_meta):
        tile_name, t_end, c_end, z_end, ys, ye, xs, xe = tile_meta
        # Cache hit: reuse the already-loaded augmented tile.
        if well_cache is not None:
            cached = well_cache.get(tile_name)
            if cached is not None:
                return tile_name, cached
        tile_full = zarr.open(str(store_path / tile_name / "0"), mode="r")
        tile_cpu = augment_tile(np.asarray(tile_full), flipud, fliplr, rot90)
        entry = (tile_cpu, t_end, c_end, z_end, ys, ye, xs, xe)
        if well_cache is not None and well_cache_lock is not None:
            with well_cache_lock:
                # Double-checked: another thread may have populated while we read.
                existing = well_cache.get(tile_name)
                if existing is None:
                    well_cache[tile_name] = entry
                else:
                    entry = existing
        return tile_name, entry

    # Inner per-band tile-load thread pool. Default 16 is fine for a
    # single-process run, but with N concurrent shard-stripe workers the
    # aggregate NFS fanout (N × outer prefetch × 16) wedges the NFS mount
    # ("D"-state stuck workers, observed in organelle step too). Tunable
    # via STITCH_INNER_LOAD_THREADS — workers default to 4 to keep total
    # concurrent NFS reads ~bounded.
    _inner_max = int(os.environ.get("STITCH_INNER_LOAD_THREADS", "16"))
    with ThreadPoolExecutor(max_workers=min(_inner_max, len(y_tiles))) as loader:
        futures = [loader.submit(_load_single, m) for m in y_tiles]
        for future in as_completed(futures):
            name, data = future.result()
            band_view[name] = data

    return band_view




def assemble(
    shifts: dict,
    tile_size: tuple,
    fov_store_path: str,
    flipud: bool = False,
    fliplr: bool = False,
    rot90: int = 0,
    divide_tile_size: Optional[Tuple[int, int]] = (4000, 4000),
    tcz_policy: Literal["max", "min"] = "min",
    blending_method: Literal["average", "edt"] = "edt",
    blending_exponent: float = 1.0,
    value_precision_bits: Literal[16, 32, 64] = 32,
):
    """Assemble the stitched image give the total shift of all tiles
    - Assume that we have paired image / total shift to be applied
    args:
        - shifts: dict of tile_name: (x_shift, y_shift)
        - tile_size: tuple of the tile size, only x and y dims
        - fov_store_path: path to the zarr file with the tiles
        - blending_method: 'average' (legacy) or 'edt' (distance-transform-based weighting)
        - blending_exponent: exponent applied to EDT distances for weighting
        - value_precision_bits: controls accumulation/output dtype (16→float16, 32→float32, 64→float64)
    """
    print(f"Using blending_method: {blending_method}")

    final_shape_xy = get_output_shape(shifts, tile_size)
    fov_store = open_ome_zarr(fov_store_path)

    # Determine target T, C, Z according to policy across all tiles
    tcz_list = []
    for tname in shifts.keys():
        tcz_list.append(fov_store[tname].data.shape[:3])
    tcz_arr = xp.asarray(tcz_list)
    # Warn if the time dimension varies across tiles, and state how it will be handled
    try:
        unique_T, counts_T = xp.unique(tcz_arr[:, 0], return_counts=True)
        if unique_T.size > 1:
            distribution = {
                int(k): int(v) for k, v in zip(unique_T.tolist(), counts_T.tolist())
            }
            target_T = (
                int(unique_T.max()) if tcz_policy == "max" else int(unique_T.min())
            )
            handling = (
                "zero-pad missing frames (leave zeros beyond each tile's T)"
                if tcz_policy == "max"
                else "truncate extra frames to the minimum T across tiles"
            )
            print(
                "WARNING: Inconsistent number of time frames (T) across tiles. "
                f"Distribution T->count: {distribution}. "
                f"Policy: '{tcz_policy}'. Target T={target_T}; will {handling}."
            )
    except Exception:
        # Best-effort warning; do not fail stitching if unique/count fails for any reason
        pass
    if tcz_policy == "min":
        tcz_target = (
            int(tcz_arr[:, 0].min()),
            int(tcz_arr[:, 1].min()),
            int(tcz_arr[:, 2].min()),
        )
    else:  # "max" (default)
        tcz_target = (
            int(tcz_arr[:, 0].max()),
            int(tcz_arr[:, 1].max()),
            int(tcz_arr[:, 2].max()),
        )

    # Resolve dtypes from requested precisions for 16, 32, 64
    dtype_val = _resolve_value_dtype(value_precision_bits)
    # Start at 32-bit indices and promote as needed
    dtype_idx = _resolve_shift_dtype(shifts, tile_size, base_bits=32)

    print(f"Using dtype for output: {dtype_val}")
    print(f"Using dtype for shifts: {dtype_idx}")

    final_shape = tcz_target + final_shape_xy
    # WARNING: allocating the full canvas can exceed memory for large mosaics.
    # This in-memory path is kept for smaller datasets; a streamed path exists below.
    output_image = xp.zeros(final_shape, dtype=dtype_val)
    # For legacy averaging we accumulate a divisor (counts). For EDT we accumulate a float weight sum.
    use_edt = str(blending_method).lower() == "edt"
    if use_edt:
        weight_sum = xp.zeros(final_shape, dtype=dtype_val)
        # Precompute or reuse a tile-local weight map based on distance to edges
        ty, tx = int(tile_size[0]), int(tile_size[1])
        cache_key = (ty, tx, int(float(blending_exponent) * 1e6))
        tile_weights = _EDT_WEIGHT_CACHE.get(cache_key)
        if tile_weights is None:
            # Boolean mask with interior True, edges False
            if ty > 2 and tx > 2:
                _mask = xp.zeros((ty, tx), dtype=bool)
                _mask[1:-1, 1:-1] = True
                _dist = cundi.distance_transform_edt(_mask).astype(xp.float32)
                _dist += 1e-6
                _weights = xp.where(_dist > 0, _dist ** float(blending_exponent), 0.0)
                tile_weights = xp.asarray(_weights, dtype=dtype_val)
            else:
                tile_weights = xp.ones((ty, tx), dtype=dtype_val)
            _EDT_WEIGHT_CACHE[cache_key] = tile_weights
    else:
        divisor = xp.zeros(final_shape, dtype=xp.uint8)

    for tile_name, shift in tqdm(shifts.items()):

        tile = fov_store[tile_name].data  # 5D array OME (T, C, Z, Y, X)
        tile = augment_tile(xp.asarray(tile), flipud, fliplr, rot90)
        # ignore sub-pixel shifts (which biahub does too by order=0 interpolation)
        # Use a wide integer type to avoid overflow for large mosaics (e.g., 51*2048 > 65535)
        shift_array = xp.asarray(shift, dtype=dtype_idx)
        # Future: add rotation / interpolation by first padding, then placing padded block into
        # final output

        # Compute bounds for T, C, Z respecting target and tile sizes
        t_end = min(tile.shape[0], final_shape[0])
        c_end = min(tile.shape[1], final_shape[1])
        z_end = min(tile.shape[2], final_shape[2])

        ys, ye = shift_array[0], shift_array[0] + tile_size[0]
        xs, xe = shift_array[1], shift_array[1] + tile_size[1]

        tile_block = tile[
            0:t_end, 0:c_end, 0:z_end, : tile_size[0], : tile_size[1]
        ].astype(dtype_val, copy=False)

        if use_edt:
            # Weighted accumulation using precomputed tile-local weights (Y,X)
            # Ignore zero-valued padded pixels
            w = tile_weights
            nz_mask = tile_block != 0
            w_masked = w * nz_mask
            # Expand weights to T,C,Z dims by broadcasting
            output_image[
                0:t_end,
                0:c_end,
                0:z_end,
                ys:ye,
                xs:xe,
            ] += (
                tile_block * w_masked
            )
            weight_sum[
                0:t_end,
                0:c_end,
                0:z_end,
                ys:ye,
                xs:xe,
            ] += w_masked
        else:
            # Legacy simple averaging by counts
            output_image[
                0:t_end,
                0:c_end,
                0:z_end,
                ys:ye,
                xs:xe,
            ] += tile_block

            # Only add divisor where the tile is not zero
            divi_tile = (tile_block != 0).astype(xp.uint8)
            divisor[
                0:t_end,
                0:c_end,
                0:z_end,
                ys:ye,
                xs:xe,
            ] += divi_tile

    stitched = xp.zeros_like(output_image, dtype=dtype_val)

    def _divide(a, b):
        return xp.nan_to_num(a / b)

    if use_edt:
        out = divide_tile(
            output_image,
            weight_sum,
            func=_divide,
            out_array=stitched,
            tile=divide_tile_size,
        )
    else:
        out = divide_tile(
            output_image,
            divisor,
            func=_divide,
            out_array=stitched,
            tile=divide_tile_size,
        )
    del out  # free memory

    return stitched


def assemble_streaming(
    shifts: dict,
    tile_size: tuple,
    fov_store_path: str,
    stitched_pos=None,
    arr_out=None,
    flipud: bool = False,
    fliplr: bool = False,
    rot90: int = 0,
    tcz_policy: Literal["max", "min"] = "min",
    blending_method: Literal["average", "edt"] = "edt",
    blending_exponent: float = 1.0,
    value_precision_bits: Literal[16, 32, 64] = 32,
    chunks_size: Optional[Tuple[int, int, int, int, int]] = None,
    scale: Optional[Tuple[float, float, float, float, float]] = None,
    divide_tile_size: Optional[Tuple[int, int]] = (1024, 1024),
    use_adaptive_blocks: bool = True,
    parallel_x_blocks: bool = True,
    profile: bool = False,
    debug_zero_mask: bool = False,
    per_channel_edt: bool = True,
    parallel_y_bands: bool = False,
    n_workers: Optional[int] = None,
    y_range: Optional[Tuple[int, int]] = None,
):
    """Streamed assembly that avoids saving auxiliary arrays.

    For each output YX block (divide_tile_size), accumulates numerator and denominator in RAM,
    normalizes, and writes directly to the on-disk output image '0'. No auxiliary arrays are saved.

    Args:
        use_adaptive_blocks: If True, automatically determine optimal block size based on GPU memory
        parallel_x_blocks: If True, process X blocks in parallel using CUDA streams (GPU only)
        profile: If True, print detailed timing breakdown per Y-band (adds GPU syncs — slower but accurate)
        debug_zero_mask: If True, print debug info about zero mask statistics
        per_channel_edt: If True, compute EDT weights per-channel based on nonzero mask (data-driven)
        parallel_y_bands: If True, process Y-bands in parallel using joblib (CPU only, ignored for GPU)
        n_workers: Number of workers for parallel Y-band processing. If None, uses _get_optimal_workers()
    """
    # Allow enabling profiling via environment variable (overrides parameter)
    if os.environ.get("STITCH_PROFILE", "").strip() in ("1", "true", "yes"):
        profile = True

    print(f"[assemble.streaming] Using blending_method: {blending_method}")
    print(f"[assemble.streaming] Adaptive blocks: {use_adaptive_blocks}, Parallel X-blocks: {parallel_x_blocks}")
    if profile:
        print(f"[assemble.streaming] PROFILING ENABLED (GPU syncs will slow overall runtime)")

    # Read tile shapes directly from .zarray JSON files (avoids opening full HCS store).
    fov_store_p = Path(fov_store_path)
    tile_shapes = {}
    for tname in shifts.keys():
        with open(fov_store_p / tname / "0" / ".zarray") as f:
            meta = json.load(f)
        tile_shapes[tname] = tuple(meta["shape"])

    # Determine target T,C,Z
    tcz_list = [s[:3] for s in tile_shapes.values()]
    tcz_arr = xp.asarray(tcz_list)
    if tcz_policy == "min":
        tcz_target = (
            int(tcz_arr[:, 0].min()),
            int(tcz_arr[:, 1].min()),
            int(tcz_arr[:, 2].min()),
        )
    else:
        tcz_target = (
            int(tcz_arr[:, 0].max()),
            int(tcz_arr[:, 1].max()),
            int(tcz_arr[:, 2].max()),
        )

    dtype_val = _resolve_value_dtype(value_precision_bits)
    final_shape_xy = get_output_shape(shifts, tile_size)
    final_shape = tcz_target + final_shape_xy

    if any(d == 0 for d in final_shape):
        raise RuntimeError(
            f"[assemble_streaming] Computed output shape has zero dimension: {final_shape}. "
            f"tcz_target={tcz_target}, final_shape_xy={final_shape_xy}, "
            f"num_tiles={len(shifts)}, tcz_policy='{tcz_policy}'"
        )

    # Adaptive block sizing based on GPU memory
    if use_adaptive_blocks:
        divide_tile_size = _get_optimal_block_size(final_shape, tile_size, divide_tile_size)

    if chunks_size is None:
        ty, tx = divide_tile_size if divide_tile_size is not None else (1024, 1024)
        chunks_size = (1, 1, 1, int(ty), int(tx))

    # Create output array on disk only (no auxiliary arrays)
    if arr_out is None:
        try:
            v3_shards_in = _v3_shards_attr(stitched_pos)
            create_kwargs = dict(
                name="0",
                shape=final_shape,
                chunks=chunks_size,
                dtype=dtype_val,
                transform=(
                    [TransformationMeta(type="scale", scale=scale)]
                    if scale is not None
                    else None
                ),
            )
            # When v3-sharded, set a chunk-aligned shards_ratio. Writes
            # go through a tensorstore adapter (see _maybe_wrap_for_tensorstore
            # below) — convert_v3 does the same and gets correct partial-shard
            # writes; raw zarr-python's sharding codec silently drops chunks.
            if v3_shards_in is not None:
                # Original ratio assumed chunks=(1,1,1,512,512); rescale spatial
                # dims so that chunks × shards_ratio still gives ~1 GB shards.
                wb_y, wb_x = chunks_size[3], chunks_size[4]
                ratio_scale_y = max(1, wb_y // 512)
                ratio_scale_x = max(1, wb_x // 512)
                v3_shards = (
                    v3_shards_in[0], v3_shards_in[1], v3_shards_in[2],
                    max(1, v3_shards_in[3] // ratio_scale_y),
                    max(1, v3_shards_in[4] // ratio_scale_x),
                )
                create_kwargs["shards_ratio"] = v3_shards
            stitched_pos.create_zeros(**create_kwargs)
        except Exception:
            pass

    use_edt = str(blending_method).lower() == "edt"

    # Precompute EDT weights for one tile
    tile_weights = _compute_edt_weights(tile_size, blending_exponent, dtype_val) if use_edt else None

    if arr_out is None:
        arr_out = stitched_pos["0"]
        # If this is a v3-sharded output, route writes through tensorstore
        # for correct partial-shard handling (zarr-python 3.1's sharding
        # codec drops chunks on certain unaligned writes; tensorstore's
        # implementation is what convert_v3 uses and is known correct).
        arr_out = _maybe_wrap_for_tensorstore(arr_out, stitched_pos=stitched_pos)
    dtype_idx = _resolve_shift_dtype(shifts, tile_size, base_bits=32)

    # Precompute tile metadata for fast intersection checks (using cached shapes)
    tile_meta = []
    for tile_name, shift in shifts.items():
        ts = tile_shapes[tile_name]
        shift_array = xp.asarray(shift, dtype=dtype_idx)
        t_end = min(ts[0], final_shape[0])
        c_end = min(ts[1], final_shape[1])
        z_end = min(ts[2], final_shape[2])
        ys, ye = int(shift_array[0]), int(shift_array[0] + tile_size[0])
        xs, xe = int(shift_array[1]), int(shift_array[1] + tile_size[1])
        tile_meta.append((tile_name, t_end, c_end, z_end, ys, ye, xs, xe))

    # Blockwise accumulation over YX
    ty, tx = divide_tile_size if divide_tile_size is not None else (1024, 1024)
    total_y, total_x = final_shape[-2], final_shape[-1]

    # Pre-identify all Y-bands and their tiles for batch loading.
    # When y_range=(ys_lo, ys_hi) is set (shard-stripe parallelism), only
    # bands fully inside the requested Y range are emitted. Band starts
    # remain on chunk-aligned multiples of ty so stripe boundaries that
    # are also chunk-aligned (e.g. shard cell heights = 14*1024) cleanly
    # partition the work.
    y_bands = []
    if y_range is not None:
        y_lo, y_hi = y_range
        y_lo = max(0, int(y_lo))
        y_hi = min(int(total_y), int(y_hi))
    else:
        y_lo, y_hi = 0, total_y
    # Snap loop start to a multiple of ty so chunk-aligned stripes pick up
    # only the bands fully inside their range.
    start = (y_lo // ty) * ty
    for y0 in range(start, y_hi, ty):
        if y0 < y_lo:
            continue
        y1 = min(y_hi, y0 + ty)
        y_tiles = [
            (nm, t_end, c_end, z_end, ys, ye, xs, xe)
            for (nm, t_end, c_end, z_end, ys, ye, xs, xe) in tile_meta
            if not (ye <= y0 or ys >= y1)
        ]
        y_bands.append((y0, y1, y_tiles))
    if not y_bands:
        print(f"[assemble.streaming] No bands in y_range={y_range}; nothing to do")
        return arr_out

    # Pipeline: prefetch multiple Y-bands ahead so tiles are ready when GPU needs them.
    # On 32-CPU production nodes NFS-bound tile reads benefit from more concurrency.
    # Tuned via STITCH_PIPELINE_WORKERS / STITCH_PREFETCH_DEPTH (defaults: 8, 5).
    _pl_workers = int(os.environ.get("STITCH_PIPELINE_WORKERS", "8"))
    _pipeline_executor = ThreadPoolExecutor(max_workers=_pl_workers)
    _prefetch_depth = int(os.environ.get("STITCH_PREFETCH_DEPTH", "5"))

    # Background writer: submit zarr writes to a thread pool so GPU can
    # continue with the next Y-band while chunks are flushed to disk.
    # Zarr chunks are independent files — parallel writes are safe.
    _write_executor = ThreadPoolExecutor(max_workers=14)
    _write_futures = []

    # D2H pipelining: transfer GPU results to CPU on a dedicated CUDA stream
    # in a background thread, so GPU can start the next band's compute immediately.
    _d2h_executor = ThreadPoolExecutor(max_workers=1)
    _transfer_stream = xp.cuda.Stream(non_blocking=True) if _USING_CUPY else None
    _prev_d2h_future = None

    # Per-well tile cache: with overlap (e.g. 200 px on 2048 px tiles +
    # 1024 px Y-bands) each tile is needed by ~2 adjacent bands. Default
    # off — empirical runs showed wash-or-regression vs no cache, possibly
    # because the kernel NFS client cache already deduplicates re-reads.
    # Set STITCH_TILE_CACHE=1 to enable.
    _use_tile_cache = os.environ.get("STITCH_TILE_CACHE", "0") in ("1", "true", "yes")
    _well_tile_cache = {} if _use_tile_cache else None
    _well_tile_cache_lock = threading.Lock() if _use_tile_cache else None

    _load_futures = deque()
    for i in range(min(_prefetch_depth, len(y_bands))):
        _load_futures.append(_pipeline_executor.submit(
            _load_band_tiles, y_bands[i][2], fov_store_p, flipud, fliplr, rot90,
            _well_tile_cache, _well_tile_cache_lock,
        ))

    for band_idx, (y0, y1, y_tiles) in enumerate(tqdm(y_bands, desc="Stitching Y")):
        t_band_start = time.time()

        # Wait for pre-loaded tiles (pipelined: loaded during previous bands' processing)
        t_load_start = time.time()
        tile_cache = _load_futures.popleft().result()
        t_load_elapsed = time.time() - t_load_start

        # Keep prefetch pipeline full
        next_prefetch = band_idx + _prefetch_depth
        if next_prefetch < len(y_bands):
            _load_futures.append(_pipeline_executor.submit(
                _load_band_tiles, y_bands[next_prefetch][2], fov_store_p, flipud, fliplr, rot90,
                _well_tile_cache, _well_tile_cache_lock,
            ))

        print(f"  [Y-band {band_idx+1}/{len(y_bands)}] Loaded {len(y_tiles)} tiles (waited {t_load_elapsed:.2f}s)")

        # X-block list for zarr chunk alignment (used by both GPU and CPU paths)
        x_blocks = [(x0, min(total_x, x0 + tx)) for x0 in range(0, total_x, tx)]

        # Full Y-band GPU processing with pipelined D2H
        t_process_start = time.time()
        if _USING_CUPY:
            ret = _process_y_band_gpu(
                y0, y1, total_x, y_tiles, tile_cache, final_shape,
                dtype_val, use_edt, tile_weights, profile=profile,
                return_gpu=True,
            )
            if profile:
                norm_gpu, band_timings = ret
            else:
                norm_gpu = ret

            t_process_elapsed = time.time() - t_process_start

            # Wait for previous band's D2H + writes to complete before submitting new one
            if _prev_d2h_future is not None:
                _prev_d2h_future.result()

            # Submit D2H + write for current band in background thread.
            # GPU is free to process next band while DMA engine handles the transfer.
            _prev_d2h_future = _d2h_executor.submit(
                _d2h_and_submit_writes, norm_gpu, _transfer_stream,
                arr_out, final_shape, y0, y1, tx, total_x,
                _write_executor, _write_futures,
            )

            # See _maybe_free_pool: pool blocks kept for next band's alloc
            # by default (saves 300-800ms/band of CUDA driver round-trips).
            _maybe_free_pool()

            if profile:
                # Break accum down into its CPU-slice / H2D / GPU-kernel components
                # so we can see which dominates and which is parallelisable.
                _sl = band_timings.get('slice_cpu', 0) * 1000
                _h2d = band_timings.get('h2d', 0) * 1000
                _gk = band_timings.get('gpu_kernel', 0) * 1000
                print(f"  [Y-band {band_idx+1}/{len(y_bands)}] PROFILE: "
                      f"alloc={band_timings['alloc']*1000:.1f}ms  "
                      f"accum={band_timings['accum']*1000:.1f}ms ({band_timings['n_tiles_hit']} tiles "
                      f"= slice_cpu {_sl:.0f}ms + h2d {_h2d:.0f}ms + gpu_kernel {_gk:.0f}ms)  "
                      f"norm={band_timings['normalize']*1000:.1f}ms  "
                      f"total={t_process_elapsed:.2f}s (D2H pipelined)")
            else:
                print(f"  [Y-band {band_idx+1}/{len(y_bands)}] GPU processing: {t_process_elapsed:.2f}s (D2H pipelined)")

        else:
            # Sequential CPU processing with per-channel EDT support
            for x0, x1 in x_blocks:
                numer = xp.zeros(
                    (final_shape[0], final_shape[1], final_shape[2], y1 - y0, x1 - x0),
                    dtype=dtype_val,
                )
                denom = xp.zeros_like(numer, dtype=dtype_val)

                for tile_name, t_end, c_end, z_end, ys, ye, xs, xe in y_tiles:
                    ix0 = max(x0, xs)
                    ix1 = min(x1, xe)
                    if ix0 >= ix1:
                        continue
                    iy0 = max(y0, ys)
                    iy1 = min(y1, ye)
                    tile_full, t_end, c_end, z_end, ys, ye, xs, xe = tile_cache[tile_name]
                    tile_sl = (
                        slice(0, t_end),
                        slice(0, c_end),
                        slice(0, z_end),
                        slice(iy0 - ys, iy1 - ys),
                        slice(ix0 - xs, ix1 - xs),
                    )
                    block = tile_full[tile_sl].astype(dtype_val, copy=False)

                    if use_edt and per_channel_edt:
                        for c_idx in range(c_end):
                            ch_out_sl = (
                                slice(0, t_end),
                                slice(c_idx, c_idx + 1),
                                slice(0, z_end),
                                slice(iy0 - y0, iy1 - y0),
                                slice(ix0 - x0, ix1 - x0),
                            )
                            block_ch = block[:, c_idx : c_idx + 1, :, :, :]
                            nz2d = xp.any(block[:, c_idx, :, :, :] != 0, axis=(0, 1))

                            if debug_zero_mask:
                                total_px = nz2d.size
                                nonzero_px = int(xp.count_nonzero(nz2d))
                                zero_px = int(total_px - nonzero_px)
                                print(f"    [debug] tile={tile_name} ch={c_idx}: {nonzero_px}/{total_px} nonzero ({zero_px} zeros)")

                            if not xp.any(nz2d):
                                continue

                            if int(xp.count_nonzero(nz2d)) == nz2d.size:
                                wloc2d = tile_weights[iy0 - ys : iy1 - ys, ix0 - xs : ix1 - xs]
                            else:
                                _dist = cundi.distance_transform_edt(nz2d).astype(xp.float32)
                                _dist += 1e-6
                                wloc2d = xp.where(_dist > 0, _dist ** float(blending_exponent), 0.0)
                                wloc2d = xp.asarray(wloc2d, dtype=dtype_val)

                            wloc5 = wloc2d[None, None, None, :, :]
                            numer[ch_out_sl] += (block_ch.astype(xp.float32) * wloc5).astype(dtype_val)
                            denom[ch_out_sl] += wloc5

                    elif use_edt:
                        out_sl = (
                            slice(0, t_end),
                            slice(0, c_end),
                            slice(0, z_end),
                            slice(iy0 - y0, iy1 - y0),
                            slice(ix0 - x0, ix1 - x0),
                        )
                        wloc = tile_weights[
                            (iy0 - ys) : (iy1 - ys), (ix0 - xs) : (ix1 - xs)
                        ]
                        wloc = xp.asarray(wloc, dtype=dtype_val)
                        nz = block != 0
                        wloc = wloc * nz
                        numer[out_sl] += block * wloc
                        denom[out_sl] += wloc
                    else:
                        out_sl = (
                            slice(0, t_end),
                            slice(0, c_end),
                            slice(0, z_end),
                            slice(iy0 - y0, iy1 - y0),
                            slice(ix0 - x0, ix1 - x0),
                        )
                        numer[out_sl] += block
                        denom[out_sl] += (block != 0).astype(dtype_val)

                norm = xp.nan_to_num(numer / xp.maximum(denom, 1e-12))
                norm_cpu = _to_numpy(norm)
                arr_out[
                    (
                        slice(0, final_shape[0]),
                        slice(0, final_shape[1]),
                        slice(0, final_shape[2]),
                        slice(y0, y1),
                        slice(x0, x1),
                    )
                ] = norm_cpu
                del numer, denom, norm, norm_cpu

            t_process_elapsed = time.time() - t_process_start
            print(f"  [Y-band {band_idx+1}/{len(y_bands)}] Processing: {t_process_elapsed:.2f}s")

        # Free cache for this Y band
        t_cleanup = time.time()
        tile_cache.clear()
        t_cleanup_elapsed = time.time() - t_cleanup

        t_band_elapsed = time.time() - t_band_start
        if profile:
            print(f"  [Y-band {band_idx+1}/{len(y_bands)}] cleanup={t_cleanup_elapsed:.2f}s  "
                  f"band_total={t_band_elapsed:.2f}s ({len(x_blocks)} X-blocks)")
        else:
            print(f"  [Y-band {band_idx+1}/{len(y_bands)}] Total: {t_band_elapsed:.2f}s ({len(x_blocks)} X-blocks)")

    _pipeline_executor.shutdown(wait=False)

    # Wait for last band's D2H + write submission to complete
    if _prev_d2h_future is not None:
        _prev_d2h_future.result()
    _d2h_executor.shutdown(wait=True)

    # Drain all background zarr writes and shut down the write executor
    t_drain_start = time.time()
    n_failed = 0
    for fut in _write_futures:
        try:
            fut.result()
        except Exception as e:
            n_failed += 1
            print(f"  WARNING: background zarr write failed: {e}")
    _write_executor.shutdown(wait=True)
    t_drain_elapsed = time.time() - t_drain_start
    print(f"  [Write drain] Waited {t_drain_elapsed:.2f}s for {len(_write_futures)} background writes"
          f"{f' ({n_failed} failed)' if n_failed else ''}")

    # v3-native: tensorstore transaction commit. All X-block writes have
    # staged into the per-position transaction; commit now coalesces them
    # per-shard (one re-encode per shard, instead of one per X-block).
    if hasattr(arr_out, "commit"):
        t_commit_start = time.time()
        arr_out.commit()
        print(f"  [Txn commit] {time.time() - t_commit_start:.2f}s "
              f"(coalesced {len(_write_futures)} writes per shard)")

    return arr_out


def _process_well_subprocess_entry(args):
    """Module-level entry point for ``parallel_mode='wells_processes'``.

    Each well runs in its own spawn-spawned Python interpreter — independent
    CUDA context, independent tensorstore handle, independent tile prefetch
    pool. CPU-side work and tensorstore commits truly parallel across wells
    (no GIL). GPU-side work is multiplexed by the CUDA driver across the
    sibling processes' contexts (effectively time-sliced unless MPS is
    enabled).

    Args is a tuple so it pickles cleanly across the spawn boundary.
    """
    import os as _os
    (well_id, shifts, output_store_path, input_store_path, tile_shape,
     flipud, fliplr, rot90, kwargs, blending_method, chunks_size, scale,
     env_vars) = args

    # Propagate the parent's STITCH_* / CUDA_* env so codec/concurrency
    # overrides are honoured in the subprocess.
    for k, v in env_vars.items():
        _os.environ[k] = v

    # Re-import in subprocess (triggers iohub codec patch on stitch import)
    from iohub.ngff import open_ome_zarr as _open
    from stitch.stitch.assemble import assemble_streaming as _assemble

    try:
        # Bypass iohub entirely in the worker. iohub's r+ mode fails to
        # populate ``_channel_names`` on Plate, Row, Well, Position
        # objects (the private attribute the inner navigation code
        # references), and patching each level is fragile. Instead open
        # the position's array directly via raw zarr and pass a tiny
        # stub to assemble_streaming that mimics the only iohub Position
        # attributes the streaming path actually uses (__getitem__,
        # create_zeros, _v3_shards_ratio).
        import zarr as _zarr
        v3_shards = kwargs.get("v3_shards_ratio")

        class _PositionStub:
            """Minimal stand-in for an iohub Position. Backed by a raw
            zarr group at ``A/{well_id}/0``. The output array (``"0"``)
            is created on demand by assemble_streaming; if it already
            exists (parent pre-created), create_zeros catches the error
            and the streaming path falls through to ``self["0"]``."""
            def __init__(self, group_path):
                self._group_path = group_path
                self._group = _zarr.open(group_path, mode="r+")
                self._v3_shards_ratio = v3_shards

            def __getitem__(self, key):
                arr_path = f"{self._group_path}/{key}"
                return _zarr.open(arr_path, mode="r+")

            def create_zeros(self, name, shape, dtype, chunks=None,
                              shards_ratio=None, transform=None,
                              check_shape=True):
                # Match iohub's create_zeros signature. Use raw zarr
                # to materialise the v3 array. Skip if it already exists.
                arr_path = f"{self._group_path}/{name}"
                try:
                    if shards_ratio is not None:
                        shards = tuple(c * r for c, r in zip(chunks, shards_ratio))
                        return _zarr.create_array(
                            store=arr_path, shape=shape, dtype=dtype,
                            chunks=chunks, shards=shards,
                            zarr_format=3, overwrite=False,
                        )
                    return _zarr.create_array(
                        store=arr_path, shape=shape, dtype=dtype,
                        chunks=chunks, zarr_format=3, overwrite=False,
                    )
                except Exception:
                    # Already exists — assemble_streaming's outer try/except
                    # handles this case.
                    pass

        group_path = f"{output_store_path}/A/{well_id}/0"
        try:
            stitched_pos = _PositionStub(group_path)

            _assemble(
                shifts=shifts,
                tile_size=tile_shape[-2:],
                fov_store_path=input_store_path,
                stitched_pos=stitched_pos,
                flipud=flipud, fliplr=fliplr, rot90=rot90,
                tcz_policy=kwargs.get("tcz_policy", "min"),
                blending_method=blending_method,
                blending_exponent=kwargs.get("blending_exponent", 1.0),
                value_precision_bits=kwargs.get("value_precision_bits", 32),
                chunks_size=chunks_size,
                scale=scale,
                divide_tile_size=kwargs.get("target_chunks_yx", (1024, 1024)),
                profile=kwargs.get("profile", False),
            )
            return well_id, True
        except Exception:
            raise
    except Exception as e:
        import traceback
        print(f"[Process Wells] Worker for well {well_id} failed: {e}")
        traceback.print_exc()
        return well_id, False


def _process_stripe_subprocess_entry(args):
    """Module-level entry for ``parallel_mode='shard_stripes'``.

    Each task processes a horizontal Y-stripe of one well. Multiple
    stripes can run concurrently on separate Python processes, so:
      - Tile reads are partitioned (each worker only loads tiles
        overlapping its stripe → ~1/N NFS read traffic per worker).
      - Output writes are partitioned by Y range → workers never write
        to the same shard rows → independent tensorstore commits.
      - Memory per worker scales with stripe size, not well size.

    Args is a tuple of pickleable values. The output array is assumed
    to already be created by the parent (so workers don't race on
    create_zeros / plate metadata).
    """
    import os as _os
    (well_id, y_start, y_end, shifts, output_store_path, input_store_path,
     tile_shape, flipud, fliplr, rot90, kwargs, blending_method,
     chunks_size, scale, env_vars) = args
    for k, v in env_vars.items():
        _os.environ[k] = v
    # NOTE: previously this line forced STITCH_FREE_POOL_PER_BAND=1 across
    # workers under the rationale that "siblings need freed VRAM" — but
    # measured: pool-free per band costs 200-800ms of alloc per band per
    # worker, compounding to 50-150s of regression across a full real-bench
    # run. With shard_stripes on H100/H200, GPU VRAM is plentiful relative
    # to per-worker working set (~5 GB × N workers ≪ device VRAM), so the
    # warm-pool optimisation should remain on. Operators can opt back into
    # eager freeing via STITCH_FREE_POOL_PER_BAND=1 if running on a much
    # tighter GPU.
    # NFS fanout limit: N workers × outer prefetch × inner per-band loaders
    # easily reaches 500+ concurrent reads, which wedges the NFS mount
    # (workers stick in "D"-state, same failure mode as organelle Pass 1).
    # Cap per-worker NFS concurrency so total stays within ~64-128.
    # Caller can override by setting these env vars themselves.
    _os.environ.setdefault("STITCH_PIPELINE_WORKERS", "2")
    _os.environ.setdefault("STITCH_PREFETCH_DEPTH", "2")
    _os.environ.setdefault("STITCH_INNER_LOAD_THREADS", "4")
    # Auto-size tensorstore thread pools to match the per-worker share of
    # the SLURM core allocation. Without this, each worker opens a 32-thread
    # data_copy + 16-thread file_io pool — N workers all running commit
    # simultaneously oversubscribes the cgroup (4×48=192 CPU-bound threads
    # on 32 cores → context-switch + cache thrash). Total budget ≈ allocated
    # cores; per-worker = cores / N. Operators can still override the env.
    try:
        n_w = int(_os.environ.get("STITCH_STRIPE_WORKERS", "4"))
        n_cpus = int(_os.environ.get(
            "SLURM_CPUS_PER_TASK",
            _os.environ.get("OMP_NUM_THREADS_TOTAL", "32"),
        ))
        per_worker = max(2, n_cpus // max(1, n_w))
        _os.environ.setdefault("STITCH_TS_DATA_COPY_CONCURRENCY", str(per_worker))
        _os.environ.setdefault(
            "STITCH_TS_FILE_IO_CONCURRENCY", str(max(2, per_worker // 2))
        )
    except Exception:
        pass
    import zarr as _zarr
    from stitch.stitch.assemble import assemble_streaming as _assemble

    v3_shards = kwargs.get("v3_shards_ratio")

    class _StripePositionStub:
        """Same interface as iohub Position but reads + writes through
        raw zarr (workers don't go through iohub). The output array is
        opened existing — parent already created it."""
        def __init__(self, group_path):
            self._group_path = group_path
            self._v3_shards_ratio = v3_shards
        def __getitem__(self, key):
            return _zarr.open(f"{self._group_path}/{key}", mode="r+")
        def create_zeros(self, *a, **kw):
            # Output array already exists — parent created it. No-op.
            pass

    try:
        group_path = f"{output_store_path}/A/{well_id}/0"
        stitched_pos = _StripePositionStub(group_path)

        _assemble(
            shifts=shifts,
            tile_size=tile_shape[-2:],
            fov_store_path=input_store_path,
            stitched_pos=stitched_pos,
            flipud=flipud, fliplr=fliplr, rot90=rot90,
            tcz_policy=kwargs.get("tcz_policy", "min"),
            blending_method=blending_method,
            blending_exponent=kwargs.get("blending_exponent", 1.0),
            value_precision_bits=kwargs.get("value_precision_bits", 32),
            chunks_size=chunks_size,
            scale=scale,
            divide_tile_size=kwargs.get("target_chunks_yx", (1024, 1024)),
            profile=kwargs.get("profile", False),
            y_range=(y_start, y_end),
        )
        return (well_id, y_start, y_end, True)
    except Exception as e:
        import traceback
        print(f"[Stripe] Worker for well {well_id} stripe [{y_start}:{y_end}] failed: {e}")
        traceback.print_exc()
        return (well_id, y_start, y_end, False)


def _process_single_well(well_id, shifts, output_store, input_store_path, tile_shape,
                        flipud, fliplr, rot90, kwargs, blending_method, chunks_size, scale,
                        well_lock, parallel_y_bands=False, n_workers=None):
    """
    Process a single well for parallel stitching.
    Thread-safe helper function to stitch one well.

    Args:
        parallel_y_bands: If True, process Y-bands in parallel (CPU mode)
        n_workers: Number of workers for parallel Y-band processing
    """
    try:
        print(f"[Well Processing] Starting well {well_id}")

        # Thread-safe creation of stitched position
        with well_lock:
            stitched_pos = output_store.create_position("A", well_id, "0")
            # Propagate v3 shards_ratio (if any) so downstream create_zeros calls
            # write v3-formatted arrays. Tagged on the position because the inner
            # functions don't know about it otherwise.
            v3_shards = kwargs.get("v3_shards_ratio")
            if v3_shards is not None:
                stitched_pos._v3_shards_ratio = v3_shards

        # Process well (this is where GPU computation happens)
        assemble_streaming(
            shifts=shifts,
            tile_size=tile_shape[-2:],
            fov_store_path=input_store_path,
            stitched_pos=stitched_pos,
            flipud=flipud,
            fliplr=fliplr,
            rot90=rot90,
            tcz_policy=kwargs.get("tcz_policy", "min"),
            blending_method=blending_method,
            blending_exponent=kwargs.get("blending_exponent", 1.0),
            value_precision_bits=kwargs.get("value_precision_bits", 32),
            chunks_size=chunks_size,
            scale=scale,
            divide_tile_size=kwargs.get("target_chunks_yx", (1024, 1024)),
            profile=kwargs.get("profile", False),
            debug_zero_mask=kwargs.get("debug_zero_mask", False),
            per_channel_edt=kwargs.get("per_channel_edt", True),
            parallel_y_bands=parallel_y_bands,
            n_workers=n_workers,
        )

        del stitched_pos  # free reference
        print(f"[Well Processing] Completed well {well_id}")
        return well_id, True

    except Exception as e:
        print(f"[Well Processing] Failed well {well_id}: {e}")
        return well_id, False


def _stitch_band_dask_worker(well_id, band_idx, y0, y1, y_tiles,
                              output_store_path, input_store_path,
                              final_shape, tile_size, tx,
                              flipud, fliplr, rot90,
                              blending_method, blending_exponent,
                              value_precision_bits, cuda_path="",
                              is_last_band=False):
    """
    Process one Y-band for one well. Self-contained Dask work unit.

    Each worker runs in its own process with its own GIL and CUDA context.
    Uses a persistent write pool to pipeline writes: band N's writes run in
    background while band N+1 does load+GPU, cutting idle time significantly.
    """
    import numcodecs
    numcodecs.blosc.set_nthreads(4)

    # Ensure CUDA is findable for CuPy JIT kernel compilation
    if cuda_path and 'CUDA_PATH' not in os.environ:
        os.environ['CUDA_PATH'] = cuda_path

    try:
        t_start = time.time()

        print(f"[Band Worker] Starting well={well_id} band={band_idx+1} "
              f"y=[{y0}:{y1}] tiles={len(y_tiles)} (PID={os.getpid()})")

        # Open output zarr array directly (instant, no HCS parsing)
        arr_out = zarr.open(os.path.join(output_store_path, "A", well_id, "0", "0"), mode="r+")

        # Compute EDT weights on this worker's GPU
        dtype_val = _resolve_value_dtype(value_precision_bits)
        use_edt = str(blending_method).lower() == "edt"
        tile_weights = _compute_edt_weights(tile_size, blending_exponent, dtype_val) if use_edt else None

        # Load tiles for this band
        t_load = time.time()
        fov_store_p = Path(input_store_path)
        tile_cache = _load_band_tiles(y_tiles, fov_store_p, flipud, fliplr, rot90)
        t_load_elapsed = time.time() - t_load

        # GPU process this Y-band
        t_gpu = time.time()
        total_x = final_shape[-1]
        norm_gpu = _process_y_band_gpu(
            y0, y1, total_x, y_tiles, tile_cache, final_shape,
            dtype_val, use_edt, tile_weights, return_gpu=True,
        )
        t_gpu_elapsed = time.time() - t_gpu

        # D2H transfer (no-op when norm_gpu is already a host numpy array,
        # which happens under the T-chunked path)
        t_d2h = time.time()
        norm_cpu = _to_numpy(norm_gpu)
        del norm_gpu, tile_cache
        xp.get_default_memory_pool().free_all_blocks()
        t_d2h_elapsed = time.time() - t_d2h

        # Drain oldest slot first if window is full — keeps the thread-
        # pool queue ordering identical to the legacy code at window=1
        # (drain-prior-then-submit).
        t_wait_elapsed = _maybe_drain_oldest()

        # One-time blosc compression benchmark (first band only)
        _run_blosc_benchmark(
            np.ascontiguousarray(norm_cpu[0, 0, 0, :, :tx])
        )

        # Submit writes to persistent pool (non-blocking).
        pool = _init_band_write_pool()
        new_futures = []
        n_blocks = 0
        for x0 in range(0, total_x, tx):
            x1 = min(total_x, x0 + tx)
            new_futures.append(pool.submit(
                _write_zarr_block, arr_out, final_shape, y0, y1, x0, x1,
                norm_cpu[:, :, :, :, x0:x1]
            ))
            n_blocks += 1

        # Record this band in the window (no drain — already done above).
        _record_band_writes(new_futures, [norm_cpu])
        print(f"[Band Worker] Submitted {n_blocks} write blocks, "
              f"{norm_cpu.nbytes/1e9:.2f}GB ({norm_cpu.dtype})")

        # For the last band of a well, drain everything still in flight
        if is_last_band:
            t_flush = time.time()
            _drain_all_bands()
            t_flush_elapsed = time.time() - t_flush
            print(f"[Band Worker] Final flush well={well_id}: {t_flush_elapsed:.1f}s")

        t_total = time.time() - t_start
        print(f"[Band Worker] Done well={well_id} band={band_idx+1} "
              f"load={t_load_elapsed:.1f}s gpu={t_gpu_elapsed:.1f}s "
              f"d2h={t_d2h_elapsed:.1f}s wait_prev={t_wait_elapsed:.1f}s total={t_total:.1f}s")
        return well_id, band_idx, True

    except Exception as e:
        import traceback
        print(f"[Band Worker] Failed well={well_id} band={band_idx+1}: {e}")
        traceback.print_exc()
        return well_id, band_idx, False


# ---------------------------------------------------------------------------
# Prefetch-enabled worker loop: processes a chunk of bands with tile
# prefetching so that band N+1's tiles load while band N is on GPU.
# ---------------------------------------------------------------------------

# Module-level prefetch executor (one per Dask worker process, lazy-init)
_prefetch_executor = None


def _read_depth() -> int:
    """STITCH_READ_DEPTH bounds how many bands' tile loads run concurrently
    with GPU compute. depth=1 reproduces the legacy single-prefetch
    behaviour (one band ahead). depth>1 keeps multiple band-loads in
    flight, paying off only when NFS read bandwidth has slack."""
    try:
        return max(1, int(os.environ.get("STITCH_READ_DEPTH", "1") or "1"))
    except ValueError:
        return 1


def _init_prefetch_executor():
    global _prefetch_executor
    if _prefetch_executor is None:
        # Pool size matches read depth — each prefetch task runs a
        # joblib-or-thread-pool _load_band_tiles internally, so a single
        # thread is enough to dispatch and wait on it.
        _prefetch_executor = ThreadPoolExecutor(
            max_workers=max(1, _read_depth()), thread_name_prefix="prefetch"
        )
    return _prefetch_executor


def _stitch_bands_loop_worker(
    band_list, output_store_path, input_store_path,
    tile_size, tx, flipud, fliplr, rot90,
    blending_method, blending_exponent,
    value_precision_bits, cuda_path="",
):
    """Process multiple bands in sequence with tile prefetching.

    Each Dask worker is assigned a chunk of bands. While the GPU processes
    band N, a background thread prefetches tiles for band N+1, hiding load
    latency behind GPU computation (which releases the GIL).
    """
    from datetime import datetime

    import numcodecs
    numcodecs.blosc.set_nthreads(4)

    if cuda_path and 'CUDA_PATH' not in os.environ:
        os.environ['CUDA_PATH'] = cuda_path

    profile = os.environ.get("STITCH_PROFILE", "").strip() in ("1", "true", "yes")

    fov_store_p = Path(input_store_path)
    dtype_val = _resolve_value_dtype(value_precision_bits)
    use_edt = str(blending_method).lower() == "edt"
    tile_weights = (
        _compute_edt_weights(tile_size, blending_exponent, dtype_val)
        if use_edt else None
    )

    prefetch_pool = _init_prefetch_executor()
    read_depth = _read_depth()
    # Deque of in-flight prefetches: each entry is (target_band_index, future).
    # FIFO — index 0 is the next band's load.
    prefetch_queue: "deque" = deque()
    results = []

    # Prime the queue by submitting the first `read_depth` bands ahead.
    # Band 0 is loaded fresh below (cold path), so prime starts at band 1.
    for ahead in range(1, min(read_depth, len(band_list))):
        ahead_y_tiles = band_list[ahead][4]
        prefetch_queue.append((ahead, prefetch_pool.submit(
            _load_band_tiles, ahead_y_tiles, fov_store_p, flipud, fliplr, rot90,
        )))

    for i, (well_id, band_idx, y0, y1, y_tiles, final_shape, is_last) in enumerate(band_list):
        gpu_timings = None
        for attempt in range(2):
            try:
                t_start = time.time()
                ts = datetime.now().strftime("%H:%M:%S.%f")[:12]
                retry_tag = f" [retry {attempt}]" if attempt > 0 else ""
                print(
                    f"[{ts}] [Band Worker]{retry_tag} Starting well={well_id} band={band_idx+1} "
                    f"y=[{y0}:{y1}] tiles={len(y_tiles)} (PID={os.getpid()})"
                )

                arr_out = zarr.open(
                    os.path.join(output_store_path, "A", well_id, "0", "0"),
                    mode="r+",
                )

                # ---- Load tiles: pop matching prefetch or blocking load ----
                # On retry the prefetch for THIS band may have been cancelled;
                # fall back to a cold load. The deque can also hold prefetches
                # for bands i+1 ... i+depth-1 — leave those alone.
                t_load = time.time()
                tile_cache = None
                if prefetch_queue and prefetch_queue[0][0] == i:
                    _, fut = prefetch_queue.popleft()
                    tile_cache = fut.result()
                    t_load_elapsed = time.time() - t_load
                    load_src = "prefetch"
                else:
                    tile_cache = _load_band_tiles(
                        y_tiles, fov_store_p, flipud, fliplr, rot90,
                    )
                    t_load_elapsed = time.time() - t_load
                    load_src = "cold"

                # ---- Top up the prefetch queue ----
                # After consuming i, queue's frontmost prefetch is for i+1
                # (if any). Submit a new prefetch for i+read_depth so the
                # queue stays read_depth-1 ahead of the consumer (since the
                # current band's tile_cache is also held alive).
                next_to_submit = i + read_depth
                if next_to_submit < len(band_list):
                    nts_y_tiles = band_list[next_to_submit][4]
                    prefetch_queue.append((
                        next_to_submit,
                        prefetch_pool.submit(
                            _load_band_tiles,
                            nts_y_tiles, fov_store_p, flipud, fliplr, rot90,
                        ),
                    ))

                # ---- GPU process this Y-band ----
                t_gpu = time.time()
                total_x = final_shape[-1]
                gpu_result = _process_y_band_gpu(
                    y0, y1, total_x, y_tiles, tile_cache, final_shape,
                    dtype_val, use_edt, tile_weights, return_gpu=True,
                    profile=profile,
                )
                if profile:
                    norm_gpu, gpu_timings = gpu_result
                else:
                    norm_gpu = gpu_result
                t_gpu_elapsed = time.time() - t_gpu

                # ---- D2H transfer ----
                # _to_numpy is a no-op when norm_gpu is already a host
                # numpy array (T-chunked path); otherwise issues the
                # standard cupy DMA copy.
                t_d2h = time.time()
                norm_cpu = _to_numpy(norm_gpu)
                del norm_gpu, tile_cache
                xp.get_default_memory_pool().free_all_blocks()
                t_d2h_elapsed = time.time() - t_d2h

                # ---- Drain oldest slot if window full (legacy ordering) ----
                t_wait_elapsed = _maybe_drain_oldest()

                # One-time blosc benchmark
                _run_blosc_benchmark(
                    np.ascontiguousarray(norm_cpu[0, 0, 0, :, :tx])
                )

                # ---- Build write futures for THIS band ----
                pool = _init_band_write_pool()
                new_futures = []
                n_blocks = 0
                for x0 in range(0, total_x, tx):
                    x1 = min(total_x, x0 + tx)
                    new_futures.append(pool.submit(
                        _write_zarr_block, arr_out, final_shape, y0, y1, x0, x1,
                        norm_cpu[:, :, :, :, x0:x1],
                    ))
                    n_blocks += 1

                # ---- Record in window (no drain — already done above) ----
                _record_band_writes(new_futures, [norm_cpu])
                print(
                    f"[Band Worker] Submitted {n_blocks} write blocks, "
                    f"{norm_cpu.nbytes/1e9:.2f}GB ({norm_cpu.dtype})"
                )

                # Flush on last band of a well
                if is_last:
                    t_flush = time.time()
                    _drain_all_bands()
                    t_flush_elapsed = time.time() - t_flush
                    print(
                        f"[Band Worker] Final flush well={well_id}: "
                        f"{t_flush_elapsed:.1f}s"
                    )

                t_total = time.time() - t_start
                ts = datetime.now().strftime("%H:%M:%S.%f")[:12]
                print(
                    f"[{ts}] [Band Worker] Done well={well_id} band={band_idx+1} "
                    f"load={t_load_elapsed:.1f}s({load_src}) "
                    f"gpu={t_gpu_elapsed:.1f}s d2h={t_d2h_elapsed:.1f}s "
                    f"wait_prev={t_wait_elapsed:.1f}s total={t_total:.1f}s"
                )
                if profile and gpu_timings:
                    gt = gpu_timings
                    print(
                        f"  [GPU detail] alloc={gt.get('alloc',0):.2f}s "
                        f"accum={gt.get('accum',0):.2f}s "
                        f"({gt.get('n_tiles_hit',0)} tiles: "
                        f"slice_cpu={gt.get('slice_cpu',0):.2f}s "
                        f"h2d={gt.get('h2d',0):.2f}s "
                        f"gpu_kernel={gt.get('gpu_kernel',0):.2f}s) "
                        f"norm={gt.get('normalize',0):.2f}s "
                        f"gpu_mem: alloc={gt.get('alloc_gpu_mb',0):.0f}MB "
                        f"post_norm={gt.get('post_norm_gpu_mb',0):.0f}MB"
                    )
                results.append((well_id, band_idx, True))
                break  # success — no retry needed

            except Exception as e:
                import traceback
                ts = datetime.now().strftime("%H:%M:%S.%f")[:12]
                print(
                    f"[{ts}] [Band Worker] Failed (attempt {attempt + 1}/2) "
                    f"well={well_id} band={band_idx+1}: {e}"
                )
                traceback.print_exc()
                # Note: with STITCH_READ_DEPTH>=1 the prefetch queue holds
                # futures for FUTURE bands (i+1 ... i+depth), not the
                # currently-failing band. Leave them in flight — they're
                # still useful for the next iteration. The retry path
                # cold-loads its own tiles via the `else` branch above.
                # Flush GPU memory pools to recover from OOM before retrying.
                try:
                    xp.get_default_memory_pool().free_all_blocks()
                    xp.get_default_pinned_memory_pool().free_all_blocks()
                except Exception:
                    pass
                if attempt == 1:
                    results.append((well_id, band_idx, False))

    # Final flush: ensure all pending writes complete before returning
    _wait_band_writes()

    return results


def stitch(
    config_path: str,
    input_store_path: str,
    output_store_path: str,
    flipud: bool = False,
    fliplr: bool = False,
    rot90: int = 0,
    blending_method: Literal["average", "edt"] = "edt",
    parallel_mode: Literal["auto", "wells", "wells_threads", "wells_processes", "shard_stripes", "y_bands", "sequential"] = "auto",
    **kwargs,
):
    """Mimic of biahub stitch function

    Args:
        config_path: Path to the YAML config file with shift estimates
        input_store_path: Path to the input OME-Zarr store
        output_store_path: Path for the output stitched OME-Zarr store
        flipud: Flip tiles vertically
        fliplr: Flip tiles horizontally
        rot90: Number of 90-degree rotations to apply
        blending_method: 'average' or 'edt' for blending overlapping tiles
        parallel_mode: Parallelization strategy:
            - "auto" (default): GPU + Dask uses multiprocess wells, GPU without Dask uses threaded wells, CPU uses parallel Y-bands
            - "wells": Force parallel well processing (Dask multiprocessing if available, else threads)
            - "wells_threads": Force parallel well processing via ThreadPoolExecutor (legacy)
            - "wells_processes": Force multiprocess wells (spawn)
            - "shard_stripes": Force shard-stripe multiprocess (recommended for v3-native)
            - "y_bands": Force parallel Y-band processing (joblib)
            - "sequential": No parallelization
        **kwargs: Additional arguments passed to assemble_streaming()

    STITCH_PARALLEL_MODE env var, if set to one of the above values, overrides
    the parameter — used by production callers that don't take parallel_mode
    in their own signature (e.g. estimate_and_stitch).
    """
    # Env-var override so production callers (estimate_and_stitch) can pick
    # shard_stripes without a signature change.
    _pm_env = os.environ.get("STITCH_PARALLEL_MODE", "").strip().lower()
    if _pm_env in ("auto", "wells", "wells_threads", "wells_processes",
                   "shard_stripes", "y_bands", "sequential"):
        parallel_mode = _pm_env

    # get the shifts and split into a list of lists per well
    all_shifts = read_shifts_biahub(config_path)

    def get_group(key):
        return key.split("/")[1]

    grouped_shifts = defaultdict(dict)
    for key, value in all_shifts.items():
        group = get_group(key)
        grouped_shifts[group][key] = value

    # Read metadata from a single position via direct JSON access.
    # open_ome_zarr() parses the full HCS hierarchy (all 7035 positions) which
    # takes minutes; reading one position's JSON files is instant.
    input_store_p = Path(input_store_path)
    first_pos_key = next(iter(all_shifts.keys()))  # e.g. "A/1/002026"
    with open(input_store_p / first_pos_key / ".zattrs") as f:
        pos_attrs = json.load(f)
    channel_names = [c["label"] for c in pos_attrs["omero"]["channels"]]
    scale_transforms = pos_attrs.get("multiscales", [{}])[0].get("datasets", [{}])[0].get("coordinateTransformations", [])
    scale = tuple(scale_transforms[0]["scale"]) if scale_transforms and scale_transforms[0].get("type") == "scale" else None
    with open(input_store_p / first_pos_key / "0" / ".zarray") as f:
        arr_meta = json.load(f)
    tile_shape = tuple(arr_meta["shape"])
    chunks_size = tuple(arr_meta["chunks"])
    # Optional override for output chunking to improve viewer (dask/napari) responsiveness
    # Accept either full 5D "target_chunks" or YX-only via "target_chunks_yx"
    target_chunks = kwargs.get("target_chunks")
    target_chunks_yx = kwargs.get("target_chunks_yx")
    if target_chunks is not None:
        try:
            chunks_size = tuple(int(x) for x in target_chunks)
        except Exception:
            pass
    elif target_chunks_yx is not None:
        try:
            ty, tx = int(target_chunks_yx[0]), int(target_chunks_yx[1])
            # Use singleton chunks for T,C,Z for snappier per-channel/plane reads
            chunks_size = (1, 1, 1, ty, tx)
        except Exception:
            pass

    # initialize output zarr store
    # zarr_version controls the on-disk format: "0.4" for OME-Zarr v0.4 (zarr v2),
    # "0.5" for OME-Zarr v0.5 (zarr v3 with sharding). Default keeps v2 behavior;
    # callers opt in to v3-native by passing zarr_version="0.5". When v3 is on we
    # also compute the channel-aware shard ratio (matches the standalone
    # convert_v3 script's `calculate_channel_based_shards`) so a single-pass
    # stitcher produces stores byte-equivalent to "v2 stitch + v3 conversion".
    zarr_version = kwargs.get("zarr_version", "0.4")
    if zarr_version not in ("0.4", "0.5"):
        raise ValueError(f"zarr_version must be '0.4' or '0.5', got {zarr_version!r}")
    use_v3_native = zarr_version == "0.5"

    if use_v3_native:
        # Channel-aware sharding to match convert_v3's recipe: one shard ≈ 1 GB,
        # group all channels per shard, square spatial sharding for the rest.
        # Same formula as ops_process/.../convert_v3.calculate_channel_based_shards.
        n_ch = len(channel_names)
        target_chunks_per_shard = 4096
        spatial_shard_ratio = max(1, int((target_chunks_per_shard / n_ch) ** 0.5))
        v3_native_shards_ratio = (1, n_ch, 1, spatial_shard_ratio, spatial_shard_ratio)
        # Make the shards_ratio visible to all inner functions via kwargs —
        # _process_single_well and the dask-bands path both read this key.
        kwargs["v3_shards_ratio"] = v3_native_shards_ratio
        print(f"[v3-native] channel-aware shards_ratio={v3_native_shards_ratio} "
              f"(channels={n_ch}, target_chunks_per_shard={target_chunks_per_shard})")
    else:
        v3_native_shards_ratio = None

    output_store = open_ome_zarr(
        output_store_path, layout="hcs", mode="w-",
        channel_names=channel_names,
        version=zarr_version,
    )
    print(f"output store created (zarr_version={zarr_version})")

    # When v3-native, mirror initialize_v3_store's channels_metadata zattrs so
    # downstream tools (zarr_inspector_data, napari) see the same plate-level
    # metadata they'd see after run_v3_conversion. Best-effort — failures here
    # are non-fatal because the conversion script's metadata helper isn't
    # always importable from the stitch submodule.
    if use_v3_native:
        try:
            from ops_analysis.processes.convert_v3 import build_channels_metadata
            cm = build_channels_metadata(channel_names, experiment=kwargs.get("experiment"))
            if cm:
                root = zarr.open(str(output_store_path), mode="r+")
                attrs = dict(root.attrs)
                attrs["channels_metadata"] = cm
                root.attrs.update(attrs)
                print(f"[v3-native] wrote channels_metadata for {len(cm)} channels to plate root")
        except Exception as e:
            print(f"[v3-native] WARN: could not write channels_metadata ({e})")

    # Determine parallelization strategy based on parallel_mode and hardware
    use_dask_wells = False
    use_thread_wells = False
    use_process_wells = False
    use_stripe_workers = False
    use_parallel_y_bands = False
    n_workers = None

    if parallel_mode == "auto":
        if _USING_CUPY and _DASK_DISTRIBUTED_AVAILABLE:
            use_dask_wells = True
            print("[stitch] Auto mode: GPU + Dask detected, using Dask multiprocess wells strategy")
        elif _USING_CUPY:
            use_thread_wells = True
            print("[stitch] Auto mode: GPU detected (no Dask), using threaded wells strategy")
        else:
            use_parallel_y_bands = True
            n_workers = _get_optimal_workers(use_gpu=False, verbose=True)
            print(f"[stitch] Auto mode: CPU detected, using parallel Y-bands strategy ({n_workers} workers)")
    elif parallel_mode == "wells":
        if _DASK_DISTRIBUTED_AVAILABLE:
            use_dask_wells = True
            print("[stitch] Forced Dask multiprocess wells strategy")
        else:
            use_thread_wells = True
            print("[stitch] Forced threaded wells strategy (Dask unavailable)")
    elif parallel_mode == "wells_threads":
        use_thread_wells = True
        print("[stitch] Forced threaded wells strategy (legacy)")
    elif parallel_mode == "wells_processes":
        use_process_wells = True
        print("[stitch] Forced multiprocess wells strategy (spawn, true GIL bypass)")
    elif parallel_mode == "shard_stripes":
        use_stripe_workers = True
        print("[stitch] Forced shard-stripe multiprocess strategy")
    elif parallel_mode == "y_bands":
        use_parallel_y_bands = True
        n_workers = _get_optimal_workers(use_gpu=False, verbose=True)
        print(f"[stitch] Forced parallel Y-bands strategy ({n_workers} workers)")
    else:  # "sequential"
        print("[stitch] Sequential processing (no parallelization)")

    well_ids = list(grouped_shifts.keys())
    num_wells = len(well_ids)

    if use_dask_wells:
        # DASK BAND-LEVEL PARALLEL PROCESSING
        # Break work into (well, Y-band) pairs for fine-grained parallelism.
        # Workers naturally desynchronize — some load tiles while others use GPU.
        t_phase1 = time.time()
        print(f"[Dask Bands] Processing {num_wells} wells with band-level Dask parallelism")
        print(f"[Dask Bands] Wells: {well_ids}")

        # Resolve divide_tile_size for Y-band computation and write chunking
        # Band size and output chunks MUST match to avoid concurrent writes to the same chunk.
        #
        # Three env knobs to tune metadata vs storage layout:
        #   STITCH_TILE_YX=4096      — bumps Y/X chunk and band size (default 2048).
        #   STITCH_CHUNK_TC=1        — collapse leading T,C dims into ONE chunk
        #                              (default 0 = chunk=(1,1,1,...)). For ISS
        #                              this drops file count 50× (per (T,C,Z) the
        #                              chunk now holds all 50 leading positions).
        #   STITCH_SHARD_RATIO_YX=N  — zarr v3 sharding with N×N chunks per shard
        #                              (default 0 = stay v2, no shards).
        divide_tile_yx = kwargs.get("target_chunks_yx", (2048, 2048))
        tile_yx_env = int(os.environ.get("STITCH_TILE_YX", "0") or "0")
        if tile_yx_env > 0:
            ty_band = tx_write = tile_yx_env
        else:
            ty_band, tx_write = int(divide_tile_yx[0]), int(divide_tile_yx[1])

        chunk_tc = int(os.environ.get("STITCH_CHUNK_TC", "0") or "0") == 1
        # Two ways to enable v3 sharding in the Dask-bands path:
        #  - kwargs["v3_shards_ratio"]: outer stitch() set this when zarr_version="0.5"
        #  - STITCH_SHARD_RATIO_YX env var: legacy escape hatch for v2-plate experiments
        v3_shards_ratio_kw = kwargs.get("v3_shards_ratio")
        shard_ratio_yx = int(os.environ.get("STITCH_SHARD_RATIO_YX", "0") or "0")
        use_v3_shards = (v3_shards_ratio_kw is not None) or (shard_ratio_yx > 0)

        # chunks_size will be finalised per-well (needs T,C,Z from final_shape).
        blending_exponent = kwargs.get("blending_exponent", 1.0)
        value_precision_bits = kwargs.get("value_precision_bits", 32)
        dtype_val = _resolve_value_dtype(value_precision_bits)
        print(f"[Dask Bands] chunks layout: tile_yx={ty_band} chunk_tc={chunk_tc} "
              f"v3_shards_ratio_kw={v3_shards_ratio_kw} "
              f"shard_ratio_yx_env={shard_ratio_yx} v3={use_v3_shards}")

        # Phase 1: Pre-create wells, compute Y-bands, build interleaved work queue
        work_queue = []
        fov_store_p = Path(input_store_path)

        for well_id in well_ids:
            well_shifts = grouped_shifts[well_id]
            final_shape_xy = get_output_shape(well_shifts, tile_shape[-2:])

            # Determine T,C,Z from tile metadata
            first_tile_key = next(iter(well_shifts.keys()))
            with open(fov_store_p / first_tile_key / "0" / ".zarray") as f:
                meta = json.load(f)
            first_tile_shape = tuple(meta["shape"])
            final_shape = first_tile_shape[:3] + final_shape_xy

            # Resolve per-well chunk geometry. With chunk_tc=True, collapse
            # T,C into a single chunk so each (Y, X) block on disk holds the
            # full leading-dim stack (one file per spatial chunk instead of
            # T*C files).
            tc_chunk = (final_shape[0], final_shape[1]) if chunk_tc else (1, 1)
            chunks_size = (tc_chunk[0], tc_chunk[1], 1, ty_band, tx_write)

            stitched_pos = output_store.create_position("A", well_id, "0")
            if use_v3_shards:
                # Prefer the channel-aware shards_ratio that outer stitch()
                # computed (v3-native path). Fall back to the env-var formula
                # for legacy users still passing STITCH_SHARD_RATIO_YX.
                if v3_shards_ratio_kw is not None:
                    # Match convert_v3's recipe exactly: chunks=(1,1,1,512,512)
                    # and shards_ratio computed by calculate_channel_based_shards.
                    # 512 must evenly divide the write-block dims; we ensure that
                    # by making ty_band/tx_write multiples of 512 below.
                    v3_chunks = (1, 1, 1, 512, 512)
                    v3_shards = tuple(v3_shards_ratio_kw)
                    # Round write block to nearest multiple of 512 so block
                    # writes align with chunk grid — otherwise sharded writes
                    # at sub-chunk granularity end up zeroing parts of shards.
                    if ty_band % 512:
                        new_ty_band = ((ty_band // 512) or 1) * 512
                        if well_id == well_ids[0]:
                            print(f"[v3-native] adjusting ty_band {ty_band} → {new_ty_band} "
                                  f"to align with chunk size 512")
                        ty_band = new_ty_band
                    if tx_write % 512:
                        new_tx_write = ((tx_write // 512) or 1) * 512
                        if well_id == well_ids[0]:
                            print(f"[v3-native] adjusting tx_write {tx_write} → {new_tx_write} "
                                  f"to align with chunk size 512")
                        tx_write = new_tx_write
                    chunks_size = (tc_chunk[0], tc_chunk[1], 1, ty_band, tx_write)
                else:
                    inner_yx = max(1, ty_band // shard_ratio_yx)
                    v3_chunks = (1, 1, 1, inner_yx, inner_yx)
                    v3_shards = (tc_chunk[0], tc_chunk[1], 1, ty_band, tx_write)

                # Use iohub create_zeros (it accepts shards_ratio in modern
                # iohub) so multiscales/transform metadata is written by the
                # NGFF helper and zarr_inspector_data sees a fully-formed
                # OME-Zarr v0.5 array.
                stitched_pos.create_zeros(
                    "0",
                    shape=final_shape,
                    chunks=v3_chunks,
                    shards_ratio=v3_shards,
                    dtype=dtype_val,
                    transform=(
                        [TransformationMeta(type="scale", scale=scale)]
                        if scale is not None
                        else None
                    ),
                )
                if well_id == well_ids[0]:
                    print(f"[Dask Bands] zarr v3 layout: chunks={v3_chunks} "
                          f"shards_ratio={v3_shards}")
            else:
                stitched_pos.create_zeros(
                    "0",
                    shape=final_shape,
                    chunks=chunks_size,
                    dtype=dtype_val,
                    transform=(
                        [TransformationMeta(type="scale", scale=scale)]
                        if scale is not None
                        else None
                    ),
                )

            # Override compressor (zarr v2 path only): lz4 is 3x faster than
            # zstd on dense float16 image data (5ms vs 16ms per 8MB chunk)
            # with the same compression ratio (~2.1x). iohub hardcodes zstd;
            # patch .zarray. Zarr v3 has its own codec block — leave it.
            zarray_path = (
                Path(output_store_path) / "A" / well_id / "0" / "0" / ".zarray"
            )
            if not use_v3_shards and zarray_path.exists():
                with open(zarray_path) as f:
                    zmeta = json.load(f)
                old_cname = zmeta.get("compressor", {}).get("cname", "?")
                zmeta["compressor"] = {
                    "id": "blosc",
                    "cname": "lz4",
                    "clevel": 1,
                    "shuffle": 2,  # BITSHUFFLE
                    "blocksize": 0,
                }
                with open(zarray_path, "w") as f:
                    json.dump(zmeta, f)
                if well_id == well_ids[0]:
                    print(f"[Dask Bands] Compressor: {old_cname} -> lz4 (3x faster on dense float16)")

            # Pre-compute tile metadata (same logic as assemble_streaming)
            dtype_idx = _resolve_shift_dtype(well_shifts, tile_shape[-2:], base_bits=32)
            tile_shapes = {}
            for tname in well_shifts.keys():
                with open(fov_store_p / tname / "0" / ".zarray") as f:
                    tmeta = json.load(f)
                tile_shapes[tname] = tuple(tmeta["shape"])

            tile_meta = []
            for tile_name, shift in well_shifts.items():
                ts = tile_shapes[tile_name]
                t_end = min(ts[0], final_shape[0])
                c_end = min(ts[1], final_shape[1])
                z_end = min(ts[2], final_shape[2])
                ys = int(shift[0])
                ye = int(shift[0] + tile_shape[-2])
                xs = int(shift[1])
                xe = int(shift[1] + tile_shape[-1])
                tile_meta.append((tile_name, t_end, c_end, z_end, ys, ye, xs, xe))

            # Pre-identify Y-bands and their contributing tiles
            total_y = final_shape[-2]
            n_bands = len(range(0, total_y, ty_band))
            for band_idx, y0 in enumerate(range(0, total_y, ty_band)):
                y1 = min(total_y, y0 + ty_band)
                y_tiles = [
                    (nm, t_end, c_end, z_end, ys, ye, xs, xe)
                    for (nm, t_end, c_end, z_end, ys, ye, xs, xe) in tile_meta
                    if not (ye <= y0 or ys >= y1)
                ]
                is_last = (band_idx == n_bands - 1)
                work_queue.append((well_id, band_idx, y0, y1, y_tiles, final_shape, is_last))

            print(f"[Dask Bands] Well {well_id}: shape={final_shape}, {n_bands} bands")

        # Interleave work queue: w1b1, w2b1, w3b1, w1b2, w2b2, w3b2, ...
        # Keeps consecutive tasks on different wells → less filesystem cache thrashing
        work_queue.sort(key=lambda x: (x[1], well_ids.index(x[0])))
        t_phase1_elapsed = time.time() - t_phase1
        print(f"[Dask Bands] Total work units: {len(work_queue)} (interleaved)")
        print(f"[Dask Bands] Phase 1 (store creation + metadata): {t_phase1_elapsed:.1f}s")

        # Capture CUDA_PATH for worker processes (needed for CuPy JIT compilation)
        cuda_path = os.environ.get('CUDA_PATH', os.environ.get('CUDA_HOME', ''))

        # Close the output store before spawning workers
        output_store.close()
        del output_store

        # Phase 2: Launch Dask LocalCluster with band-level workers
        t_phase2 = time.time()

        # Detect available GPUs and scale workers across all devices.
        # Peak GPU per worker depends on band size: numer + denom + norm temp
        # + block + working ≈ 4 × (T*C*Z × ty_band × total_x × itemsize). For
        # track/pheno T*C*Z ~5-10 → ~10-15 GB; for ISS T*C*Z ~50-70 → ~20-25 GB.
        # We compute peak from the actual work queue (max final_shape across
        # wells) so packing is correct for the real dataset rather than a
        # hardcoded 18 GB divisor.
        from ops_utils.hpc.gpu_utils import _setup_gpu_environment
        from ops_utils.hpc.parallel_utils import MultiGPUCluster
        available_gpus = _setup_gpu_environment()
        n_gpus = len(available_gpus)

        if n_gpus > 0 and _USING_CUPY:
            # Query per-device VRAM from the first visible GPU
            per_gpu_mb = xp.cuda.Device(0).mem_info[1]
            per_gpu_gb = per_gpu_mb / 1e9

            # Peak band size across the whole work_queue: each item carries
            # final_shape; band height is at most ty_band; band width is the
            # full stitched x extent (final_shape[-1]). When STITCH_T_CHUNK is
            # set, the per-worker peak is bounded by the chunk size in the
            # leading-channel dim, not the full C — so the divisor shrinks
            # proportionally and we can safely pack more workers per GPU.
            itemsize = int(np.dtype(dtype_val).itemsize)
            t_chunk_env = int(os.environ.get("STITCH_T_CHUNK", "0") or "0")
            peak_band_bytes = 0
            for _wid, _bidx, _y0, _y1, _yts, _fshape, _is_last in work_queue:
                t = int(_fshape[0])
                c = int(_fshape[1])
                z = int(_fshape[2])
                effective_t = min(t_chunk_env, t) if t_chunk_env > 0 else t
                t_c_z = effective_t * c * z
                band_h = int(min(ty_band, _fshape[-2] - _y0))
                band_w = int(_fshape[-1])
                array_bytes = t_c_z * band_h * band_w * itemsize
                if array_bytes > peak_band_bytes:
                    peak_band_bytes = array_bytes
            # Per-worker peak: numer + denom + norm + block/working ≈ 4×.
            # Add 4 GB headroom for cuda context, cupy memory-pool overhead,
            # and the prefetched-tile cache.
            peak_gb_per_worker = (4 * peak_band_bytes) / 1e9 + 4.0
            workers_per_gpu = max(1, int(per_gpu_gb // peak_gb_per_worker))
            # STITCH_WORKERS_PER_GPU overrides the auto-derived count — useful
            # when you want a small number of workers fed by deep read prefetch
            # / large write window (saturating GPU with less host-RAM
            # contention than packing more workers).
            override = os.environ.get("STITCH_WORKERS_PER_GPU", "")
            try:
                override_n = int(override) if override else 0
            except ValueError:
                override_n = 0
            if override_n > 0:
                print(f"[Dask Bands] STITCH_WORKERS_PER_GPU override: "
                      f"{workers_per_gpu} → {override_n}")
                workers_per_gpu = override_n
            n_dask_workers = min(workers_per_gpu * n_gpus, len(work_queue))
            print(f"[Dask Bands] {n_gpus} GPU(s), {per_gpu_gb:.0f}GB each, "
                  f"peak~{peak_gb_per_worker:.1f}GB/worker → "
                  f"{workers_per_gpu}/gpu × {n_gpus} = {n_dask_workers} workers")
        else:
            workers_per_gpu = min(4, len(work_queue))
            n_dask_workers = workers_per_gpu
            print(f"[Dask Bands] No GPU detected, using {n_dask_workers} CPU workers")

        # threads_per_worker=1: each worker runs one long-lived task (loop
        # over its assigned bands). Write pipelining + prefetch use module-
        # level state that is NOT thread-safe.
        if cuda_path:
            os.environ['CUDA_PATH'] = cuda_path  # workers inherit parent env

        # Clear parent CUDA_VISIBLE_DEVICES — MultiGPUCluster sets it per-cluster
        parent_cuda_devices = os.environ.pop("CUDA_VISIBLE_DEVICES", None)

        # Disable Dask's per-worker memory limit — workers hold ~28-42GB each
        # (current tile_cache + prefetched tile_cache + norm_cpu + prev norm_cpu).
        # MultiGPUCluster creates one LocalCluster per GPU with CUDA_VISIBLE_DEVICES
        # set BEFORE worker spawn, ensuring proper GPU distribution.
        use_multi_gpu = n_gpus > 0
        if use_multi_gpu:
            multi_cluster = MultiGPUCluster(available_gpus, workers_per_gpu,
                                            memory_limit=0)
        else:
            cluster = LocalCluster(n_workers=n_dask_workers, threads_per_worker=1,
                                   memory_limit=0)
            client = Client(cluster)

        cpu_monitor_stop = None
        try:
            t_phase2_elapsed = time.time() - t_phase2
            print(f"[Dask Bands] Cluster started: {n_dask_workers} workers, 1 thread each ({t_phase2_elapsed:.1f}s)")

            # Start CPU/IO monitor (reads /proc for per-worker stats every 5s)
            n_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", 32))
            cpu_monitor_stop = _start_cpu_monitor(interval=5.0, n_cores=n_cores)

            # Distribute work round-robin so each worker gets interleaved bands
            worker_chunks = [[] for _ in range(n_dask_workers)]
            for i, item in enumerate(work_queue):
                worker_chunks[i % n_dask_workers].append(item)

            chunk_sizes = [len(c) for c in worker_chunks]
            print(f"[Dask Bands] Chunks per worker: {chunk_sizes} (prefetch enabled)")

            submit_kwargs = dict(
                output_store_path=output_store_path,
                input_store_path=input_store_path,
                tile_size=tile_shape[-2:],
                tx=tx_write,
                flipud=flipud,
                fliplr=fliplr,
                rot90=rot90,
                blending_method=blending_method,
                blending_exponent=blending_exponent,
                value_precision_bits=value_precision_bits,
                cuda_path=cuda_path,
            )

            futures = []
            for chunk in worker_chunks:
                if not chunk:
                    continue
                if use_multi_gpu:
                    future = multi_cluster.submit(
                        _stitch_bands_loop_worker,
                        band_list=chunk,
                        **submit_kwargs,
                    )
                else:
                    future = client.submit(
                        _stitch_bands_loop_worker,
                        band_list=chunk,
                        **submit_kwargs,
                    )
                futures.append(future)

            # Gather results from all worker loops
            completed = []
            failed = []
            for future in futures:
                try:
                    results_list = future.result()
                    for well_id, band_idx, success in results_list:
                        if success:
                            completed.append((well_id, band_idx))
                        else:
                            failed.append((well_id, band_idx))
                except Exception as e:
                    print(f"[Dask Bands] Worker exception: {e}")
                    import traceback
                    traceback.print_exc()

        finally:
            if cpu_monitor_stop is not None:
                cpu_monitor_stop.set()
            try:
                if use_multi_gpu:
                    multi_cluster.close()
                else:
                    client.close()
                    cluster.close(timeout=120)
            except Exception:
                pass
            # Restore parent CUDA_VISIBLE_DEVICES
            if parent_cuda_devices is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = parent_cuda_devices

        print(f"[Dask Bands] Completed: {len(completed)}/{len(work_queue)} bands")
        if failed:
            print(f"[Dask Bands] Failed bands: {failed}")
            raise RuntimeError(f"Failed to process {len(failed)} bands: {failed}")

        print(f"[Dask Bands] All {len(work_queue)} bands processed successfully!")

    elif use_thread_wells:
        # THREADED WELL PROCESSING — wells overlap on the same GPU.
        # Heavy work releases the GIL: zarr/NFS reads, CuPy kernels, NumPy
        # ops, tensorstore writes, blosc compression. The "GIL-limited"
        # legacy comment was conservative; in practice we see good overlap.
        global _PARALLEL_WELLS_ACTIVE
        _PARALLEL_WELLS_ACTIVE = True
        well_lock = threading.Lock()
        max_workers = min(4, num_wells)

        print(f"[Thread Wells] Processing {num_wells} wells with {max_workers} thread workers")
        print(f"[Thread Wells] Wells: {well_ids}")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for well_id in well_ids:
                shifts = grouped_shifts[well_id]
                future = executor.submit(
                    _process_single_well,
                    well_id, shifts, output_store, input_store_path, tile_shape,
                    flipud, fliplr, rot90, kwargs, blending_method, chunks_size, scale,
                    well_lock, parallel_y_bands=False, n_workers=None
                )
                futures.append((well_id, future))

            completed_wells = []
            failed_wells = []
            for well_id, future in futures:
                try:
                    result_well_id, success = future.result()
                    if success:
                        completed_wells.append(result_well_id)
                    else:
                        failed_wells.append(result_well_id)
                except Exception as e:
                    print(f"[Thread Wells] Exception in well {well_id}: {e}")
                    failed_wells.append(well_id)

        print(f"[Thread Wells] Completed: {len(completed_wells)} wells: {completed_wells}")
        if failed_wells:
            print(f"[Thread Wells] Failed: {len(failed_wells)} wells: {failed_wells}")
            raise RuntimeError(f"Failed to process {len(failed_wells)} wells: {failed_wells}")

        print(f"[Thread Wells] All {num_wells} wells processed successfully!")
        _PARALLEL_WELLS_ACTIVE = False

    elif use_process_wells:
        # MULTIPROCESS WELL PROCESSING — separate Python interpreters.
        # Each well gets its own CUDA context + tensorstore handle. CPU work
        # truly parallel (no GIL). GPU work multiplexed by the driver.
        import multiprocessing as _mp
        from concurrent.futures import ProcessPoolExecutor as _PPE

        print(f"[Process Wells] Processing {num_wells} wells with multiprocess pool")
        print(f"[Process Wells] Wells: {well_ids}")

        # Pre-create positions in the parent so iohub's plate metadata is
        # flushed to disk before workers open the store (otherwise iohub's
        # _detect_layout raises KeyError on a plate without 'plate' attrs).
        for well_id in well_ids:
            try:
                pos = output_store.create_position("A", well_id, "0")
                v3_shards = kwargs.get("v3_shards_ratio")
                if v3_shards is not None:
                    pos._v3_shards_ratio = v3_shards
            except Exception as e:
                print(f"[Process Wells] WARN: pre-create position {well_id} failed: {e}")

        # Workers re-open the output store; close the parent's handle.
        output_store.close()
        del output_store

        # Forward the env knobs that workers care about.
        env_to_forward = {
            k: v for k, v in os.environ.items()
            if k.startswith("STITCH_") or k.startswith("CUDA_") or k.startswith("OMP_")
            or k.startswith("MKL_") or k.startswith("OPENBLAS_") or k.startswith("NUMEXPR_")
        }

        args_list = [
            (well_id, grouped_shifts[well_id], output_store_path, input_store_path,
             tile_shape, flipud, fliplr, rot90, kwargs, blending_method,
             chunks_size, scale, env_to_forward)
            for well_id in well_ids
        ]

        ctx = _mp.get_context("spawn")
        # Cap worker count: more workers ≠ better when contending for one GPU
        # and one NFS link. STITCH_PROCESS_WELLS_WORKERS overrides default.
        max_workers = int(os.environ.get("STITCH_PROCESS_WELLS_WORKERS", str(min(num_wells, 4))))
        completed_wells = []
        failed_wells = []
        with _PPE(max_workers=max_workers, mp_context=ctx) as ex:
            for well_id, ok in ex.map(_process_well_subprocess_entry, args_list):
                if ok:
                    completed_wells.append(well_id)
                else:
                    failed_wells.append(well_id)

        print(f"[Process Wells] Completed: {len(completed_wells)} wells: {completed_wells}")
        if failed_wells:
            print(f"[Process Wells] Failed: {len(failed_wells)} wells: {failed_wells}")
            raise RuntimeError(f"Failed to process {len(failed_wells)} wells: {failed_wells}")

    elif use_stripe_workers:
        # SHARD-STRIPE MULTIPROCESS — partition each well's Y range into
        # chunk-aligned stripes and run them across a worker pool.
        # Workers never write to the same shard rows → no commit conflict;
        # tile reads partition cleanly → much less NFS contention than
        # whole-well multiprocess; memory per worker scales with stripe
        # size (smaller buffers per worker = fits more workers in RAM).
        import multiprocessing as _mp
        from concurrent.futures import ProcessPoolExecutor as _PPE

        stripes_per_well = max(1, int(os.environ.get("STITCH_STRIPES_PER_WELL", "4")))
        max_workers_env = int(os.environ.get(
            "STITCH_STRIPE_WORKERS",
            str(min(num_wells * stripes_per_well, 6))
        ))
        # Pre-create plate, positions, AND output arrays in the parent so
        # workers don't race on iohub writes. Workers open the array as
        # raw zarr.
        for well_id in well_ids:
            try:
                pos = output_store.create_position("A", well_id, "0")
                v3_shards_in = kwargs.get("v3_shards_ratio")
                if v3_shards_in is not None:
                    pos._v3_shards_ratio = v3_shards_in
                # Pre-create the output array so workers can open as r+
                final_shape = output_store.get_well_final_shape("A", well_id) \
                    if hasattr(output_store, "get_well_final_shape") else None
                # Fall back to computing it from the shifts the worker would use.
                if final_shape is None:
                    well_shifts = grouped_shifts[well_id]
                    final_shape_xy = get_output_shape(well_shifts, tile_shape[-2:])
                    # Match the pattern used inside assemble_streaming
                    final_shape = (
                        tile_shape[0], tile_shape[1], tile_shape[2],
                        final_shape_xy[0], final_shape_xy[1],
                    )
                from iohub.ngff import TransformationMeta as _TM
                create_kwargs = dict(
                    name="0",
                    shape=final_shape,
                    chunks=chunks_size,
                    dtype=_resolve_value_dtype(kwargs.get("value_precision_bits", 32)),
                    transform=([_TM(type="scale", scale=scale)] if scale is not None else None),
                )
                if v3_shards_in is not None:
                    wb_y, wb_x = chunks_size[3], chunks_size[4]
                    rsy = max(1, wb_y // 512); rsx = max(1, wb_x // 512)
                    create_kwargs["shards_ratio"] = (
                        v3_shards_in[0], v3_shards_in[1], v3_shards_in[2],
                        max(1, v3_shards_in[3] // rsy),
                        max(1, v3_shards_in[4] // rsx),
                    )
                try:
                    pos.create_zeros(**create_kwargs)
                except Exception as e:
                    print(f"[Stripes] WARN: pre-create array {well_id} failed: {e}")
            except Exception as e:
                print(f"[Stripes] WARN: pre-create position {well_id} failed: {e}")
        output_store.close()
        del output_store

        env_to_forward = {
            k: v for k, v in os.environ.items()
            if k.startswith("STITCH_") or k.startswith("CUDA_") or k.startswith("OMP_")
            or k.startswith("MKL_") or k.startswith("OPENBLAS_") or k.startswith("NUMEXPR_")
        }

        # Build (well, stripe) tasks. CRITICAL: stripe boundaries must
        # snap to SHARD CELL height, not chunk height. Multiple stripes
        # writing into the same shard race on commit (each worker has its
        # own tensorstore Transaction; commits are not coordinated, so
        # the second commit overwrites the first). Aligning to shard
        # cells ensures each shard is fully owned by exactly one worker.
        v3_shards_in = kwargs.get("v3_shards_ratio")
        ty_band = chunks_size[3]
        if v3_shards_in is not None:
            wb_y = chunks_size[3]
            rsy = max(1, wb_y // 512)
            shard_ratio_y = max(1, v3_shards_in[3] // rsy)
            shard_cell_y = ty_band * shard_ratio_y
        else:
            shard_cell_y = ty_band
        all_tasks = []
        for well_id in well_ids:
            well_shifts = grouped_shifts[well_id]
            final_shape_xy = get_output_shape(well_shifts, tile_shape[-2:])
            total_y_well = int(final_shape_xy[0])
            # Cap stripes at the number of shard rows available
            n_shard_rows = max(1, (total_y_well + shard_cell_y - 1) // shard_cell_y)
            n_stripes_eff = min(stripes_per_well, n_shard_rows)
            if n_stripes_eff < stripes_per_well:
                print(f"[Stripes] Capped well {well_id} from {stripes_per_well} to "
                      f"{n_stripes_eff} stripes (only {n_shard_rows} shard rows)")
            # Distribute shard rows across stripes. Each stripe owns a
            # contiguous block of shard rows; last stripe takes any leftover.
            shards_per_stripe = n_shard_rows // n_stripes_eff
            cuts = [i * shards_per_stripe * shard_cell_y for i in range(n_stripes_eff)]
            cuts.append(total_y_well)
            for y_start, y_end in zip(cuts[:-1], cuts[1:]):
                if y_end <= y_start:
                    continue
                # Pass FULL shifts so assemble_streaming computes the correct
                # global final_shape. Filtering shifts to "tiles in this
                # stripe" makes get_output_shape return a smaller total_y,
                # which in turn clamps y_range and corrupts stripe writes.
                # The band-vs-tile intersection filter inside the band loop
                # only loads tiles that actually overlap each band, so
                # passing full shifts costs little — most tiles outside the
                # stripe are never loaded.
                all_tasks.append((
                    well_id, y_start, y_end, well_shifts, output_store_path,
                    input_store_path, tile_shape, flipud, fliplr, rot90,
                    kwargs, blending_method, chunks_size, scale, env_to_forward,
                ))

        print(f"[Stripes] Dispatching {len(all_tasks)} tasks across {max_workers_env} workers "
              f"({stripes_per_well} stripes/well × {num_wells} wells)")

        ctx = _mp.get_context("spawn")
        completed = []
        failed = []
        with _PPE(max_workers=max_workers_env, mp_context=ctx) as ex:
            for result in ex.map(_process_stripe_subprocess_entry, all_tasks):
                well_id, y_start, y_end, ok = result
                if ok:
                    completed.append((well_id, y_start, y_end))
                else:
                    failed.append((well_id, y_start, y_end))

        print(f"[Stripes] Completed {len(completed)}/{len(all_tasks)} tasks")
        if failed:
            print(f"[Stripes] Failed: {failed}")
            raise RuntimeError(f"Failed to process {len(failed)} stripes: {failed}")

    else:
        # SEQUENTIAL WELL PROCESSING with optional parallel Y-bands
        well_lock = threading.Lock()
        print(f"[Sequential Wells] Processing {num_wells} wells: {well_ids}")

        completed_wells = []
        failed_wells = []

        for well_id in well_ids:
            shifts = grouped_shifts[well_id]
            result_well_id, success = _process_single_well(
                well_id, shifts, output_store, input_store_path, tile_shape,
                flipud, fliplr, rot90, kwargs, blending_method, chunks_size, scale,
                well_lock, parallel_y_bands=use_parallel_y_bands, n_workers=n_workers
            )
            if success:
                completed_wells.append(result_well_id)
            else:
                failed_wells.append(result_well_id)

        # Report final results
        print(f"[Sequential Wells] Completed: {len(completed_wells)} wells: {completed_wells}")
        if failed_wells:
            print(f"[Sequential Wells] Failed: {len(failed_wells)} wells: {failed_wells}")
            raise RuntimeError(f"Failed to process {len(failed_wells)} wells: {failed_wells}")

        print(f"[Sequential Wells] All {num_wells} wells processed successfully!")

    return


def estimate_stitch(
    input_store_path: str,
    output_config_path: Path,
    flipud: bool,
    fliplr: bool,
    rot90: int,
    tile_size: tuple = (2048, 2048),
    overlap: int = 150,
    x_guess: Optional[dict] = None,
    limit_positions: Optional[int] = None,
    channel: int = 0,
    timepoint: int = 0,
    timepoint_per_well: Optional[dict] = None,
    use_clahe: bool = False,
    clahe_clip_limit: float = 0.02,
    verbose: bool = False,
):
    """Mimic of Biahub estimate stitch function

    Args:
        channel: Channel index to use for registration (default: 0)
        timepoint: Timepoint index to use for registration (default: 0)
        timepoint_per_well: Optional dict mapping well names (e.g., "A/2") to timepoint indices.
                           Overrides the default timepoint for specific wells.
        use_clahe: Apply CLAHE preprocessing for better registration (default: False)
        clahe_clip_limit: CLAHE contrast limit (default: 0.02)
        verbose: Print confidence scores as edges are computed (default: False)
    """

    if verbose:
        print(f"[assemble.estimate_stitch] Configuration:")
        print(f"  input_store_path  = {input_store_path}")
        print(f"  output_config_path = {output_config_path}")
        print(f"  flipud={flipud}, fliplr={fliplr}, rot90={rot90}")
        print(f"  tile_size={tile_size}, overlap={overlap}")
        print(f"  channel={channel}, timepoint={timepoint}")
        print(f"  timepoint_per_well={timepoint_per_well}")
        print(f"  use_clahe={use_clahe}, clahe_clip_limit={clahe_clip_limit}")
        print(f"  limit_positions={limit_positions}")
        print(f"  x_guess={x_guess}")

    store = open_ome_zarr(input_store_path)
    if limit_positions is not None and int(limit_positions) > 0:
        print(f"[assemble.estimate_stitch] DEBUG MODE: Limiting to {limit_positions} positions")
    else:
        print(f"[assemble.estimate_stitch] Processing ALL positions (full stitching)")
    # Discover positions with an optional centered selection to avoid scanning entire store in debug
    if limit_positions is not None and int(limit_positions) > 0:
        # Build a true centered m×m block for EACH well from the filesystem (no fallback)
        root = Path(input_store_path)
        rows = sorted([d.name for d in root.iterdir() if d.is_dir()])
        if not rows:
            raise RuntimeError("No row directories found for centered grid selection")
        k = int(limit_positions)
        side = int(max(1, int(k**0.5)))
        target = min(k, side * side)

        def _rc_from_name(name: str):
            digits = "".join(ch for ch in name if ch.isdigit())
            if len(digits) >= 6:
                return int(digits[:3]), int(digits[3:6])
            half = len(digits) // 2
            return int(digits[:half] or 0), int(digits[half:] or 0)

        position_list = []
        for row in rows:
            cols = sorted([d.name for d in (root / row).iterdir() if d.is_dir()])
            for col in cols:
                well_dir = root / row / col
                tiles = sorted([d.name for d in well_dir.iterdir() if d.is_dir()])
                if not tiles:
                    continue
                rc = [_rc_from_name(t) for t in tiles]
                rs = sorted(set(r for r, _ in rc))
                cs = sorted(set(c for _, c in rc))
                if not rs or not cs:
                    continue
                rmid = rs[len(rs) // 2]
                cmid = cs[len(cs) // 2]
                selected = []
                radius = 0
                while len(selected) < target and radius <= max(len(rs), len(cs)):
                    rmin, rmax = rmid - radius, rmid + radius
                    cmin, cmax = cmid - radius, cmid + radius
                    for name, (rr, cc) in zip(tiles, rc):
                        if (
                            rmin <= rr <= rmax
                            and cmin <= cc <= cmax
                            and name not in selected
                        ):
                            selected.append(name)
                            if len(selected) >= target:
                                break
                    radius += 1
                tiles_sel = selected[:target]
                # Append positions for this well
                position_list.extend([f"{row}/{col}/{t}" for t in tiles_sel])
        print(
            f"[assemble.estimate_stitch] Using centered grid {side}x{side} per well; total tiles={len(position_list)}"
        )
    else:
        # Full list (may be large) - try fast discovery first
        position_list = _discover_positions_fast(Path(input_store_path))

        if position_list is not None:
            print(f"[assemble.estimate_stitch] Fast discovery found {len(position_list)} positions")
        else:
            # Not HCS layout or fast discovery failed - use slower iterator
            print(f"[assemble.estimate_stitch] Not HCS layout, falling back to iterator-based discovery")
            position_list = [
                p for p, _ in tqdm(store.positions(), desc="Getting positions")
            ]

    grouped_positions = defaultdict(list)
    for a in position_list:
        group = a[:3]
        grouped_positions[group].append(a)

    def _process_well(well_id):
        """Worker function to process a single well."""
        well_positions = grouped_positions[well_id]
        tile_lut = {t[4:]: i for i, t in enumerate(well_positions)}

        # Determine timepoint for this well
        well_timepoint = timepoint
        if timepoint_per_well is not None and well_id in timepoint_per_well:
            well_timepoint = timepoint_per_well[well_id]
            print(f"[assemble.estimate_stitch] Using timepoint {well_timepoint} for well {well_id}")

        edge_list, confidence_dict = pairwise_shifts(
            well_positions,
            input_store_path,
            well=well_id,
            flipud=flipud,
            fliplr=fliplr,
            rot90=rot90,
            overlap=overlap,
            channel=channel,
            timepoint=well_timepoint,
            use_clahe=use_clahe,
            clahe_clip_limit=clahe_clip_limit,
            verbose=verbose,
        )

        opt_shift_dict = optimal_positions(edge_list, tile_lut, well_id, tile_size, x_guess)

        return well_id, opt_shift_dict, confidence_dict

    running_opt_shift_dict = {}
    running_confidence_dict = {}

    # Parallel processing of wells using ThreadPoolExecutor
    num_wells = len(grouped_positions)
    max_workers = min(4, num_wells)  # Use up to 4 parallel wells
    print(f"[estimate_stitch] Processing {num_wells} wells with {max_workers} parallel workers")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all well computations
        futures = {
            executor.submit(_process_well, well_id): well_id
            for well_id in grouped_positions.keys()
        }

        # Collect results
        for future in tqdm(as_completed(futures), total=num_wells, desc="Processing wells"):
            well_id, opt_shift_dict, confidence_dict = future.result()
            running_opt_shift_dict = running_opt_shift_dict | opt_shift_dict
            running_confidence_dict[well_id] = confidence_dict

    to_write = {
        "total_translation": running_opt_shift_dict,
        "confidence": running_confidence_dict,
    }
    # Ensure parent directory exists before writing
    try:
        Path(output_config_path).parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    with open(output_config_path, "w") as f:
        yaml.dump(to_write, f)

    return opt_shift_dict


def array_apply(
    *in_arrays: ArrayLike,
    func: Callable,
    out_array: ArrayLike,
    axis: Union[Tuple[int], int] = 0,
    **kwargs,
) -> ArrayLike:
    """Apply a function over a given dimension of an array.

    adapted from Ultrack.apply_array
    """
    name = func.__name__ if hasattr(func, "__name__") else type(func).__name__

    try:
        in_shape = [arr.shape for arr in in_arrays]
        xp.broadcast_shapes(out_array.shape, *in_shape)
    except ValueError as e:
        print(
            f"Warning: if you are not using multichannel operations, "
            f"this can be an error. {e}."
        )

    if isinstance(axis, int):
        axis = (axis,)

    stub_slicing = [slice(None) for _ in range(out_array.ndim)]
    multi_indices = list(itertools.product(*[range(out_array.shape[i]) for i in axis]))
    for indices in tqdm(multi_indices, f"Applying {name} ..."):
        for a, i in zip(axis, indices):
            stub_slicing[a] = i
        indexing = tuple(stub_slicing)

        func_result = func(*[a[indexing] for a in in_arrays], **kwargs)
        output_shape = out_array[indexing].shape
        out_array[indexing] = xp.broadcast_to(func_result, output_shape)

    return out_array


def divide_tile(
    *in_arrays: ArrayLike,
    func: Callable,
    out_array: ArrayLike,
    tile: tuple,
    overlap: tuple = (0, 0),
):

    final_shape = out_array.shape[-2:]

    tiling_start = list(
        product(
            *[
                range(o, size + 2 * o, t + o)  # t + o step, because of blending map
                for size, t, o in zip(final_shape, tile, overlap)
            ]
        )
    )
    for start_indices in tqdm(tiling_start, "Applying function to tiles"):
        slicing = (...,) + tuple(
            slice(start - o, start + t + o)
            for start, t, o in zip(start_indices, tile, overlap)
        )
        out_array[slicing] = func(*[a[slicing] for a in in_arrays])

    return out_array