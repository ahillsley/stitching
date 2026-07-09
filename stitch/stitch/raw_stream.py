"""Raw-shard, row-streaming stitch estimate (acquire-zarr back-end).

A convert-free alternative to the Hilbert-ordered ``pairwise_shifts``: instead of
reading per-FOV tiles from a converted HCS store, this reads directly from the
raw per-camera ``[t, g, y, x]`` acquire-zarr arrays, **whole shards at a time, in
snake (``g``) order**, and emits the same horizontal/vertical neighbor edges via
the existing :class:`~stitch.stitch.tile.Edge` / ``offset`` machinery. The global
solve (:func:`~stitch.stitch.tile.optimal_positions`) is unchanged.

Design notes live in
``ops_process/ops_analysis/livescreen/planning/shard_aware_stitch_estimate_plan.md``
and the raw layout in ``.../livescreen/raw_zarr_structure.md``. Key facts this
relies on:

- The raw ``g`` axis already runs row-wise snake, so ``g`` ascending == snake
  order == on-disk shard order; no Hilbert curve is needed.
- A shard packs ``shard_g`` (256) FOVs; reading any frame in it is cheapest done
  by reading the whole shard once. The frame cache is sized to ~2 scan rows so
  g-order access reads each shard exactly once and vertical (row-to-row) edges
  find both rows resident.

This is the v1 estimate engine; geometry (the ``g <-> (row,col)`` map, frame
size, per-camera flips) is computed by the caller and passed in — the submodule
stays ``mda.yaml``-agnostic. Coexists with the Hilbert/HCS path (unchanged).
"""
from __future__ import annotations

import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Event, Lock

import zarr
from tqdm import tqdm

from stitch.stitch.tile import Edge, augment_tile


class _FrameLRU(OrderedDict):
    """Insertion-ordered LRU. Oldest-*loaded* frames evict first — which, under
    g-order streaming, means oldest scan rows drop once we move past them."""

    def __init__(self, max_size: int):
        super().__init__()
        self.max_size = max_size

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if len(self) > self.max_size:
            self.popitem(last=False)


class RawShardTileSource:
    """Tile provider over ONE raw camera array (the estimate channel).

    Indexed by HCS-style ``"RRRCCC"`` names (like ``TileCache``) so
    :class:`~stitch.stitch.tile.Edge` works unchanged. On a cache miss it reads
    the whole shard containing the requested ``g`` and scatters its frames into a
    frame-LRU sized to ``cache_rows`` scan rows.

    Args:
        store_path: path to the camera's array dir ``<cam>.ome.zarr/0`` ([t,g,y,x]).
        order: list mapping ``g -> (row, col)`` (e.g. from ``positions.csv``).
        per_cam_flip: ``{"hflip": bool, "vflip": bool}`` for this camera — applied
            at read time (the convert step used to bake this in). ``None`` = none.
        flipud/fliplr/rot90: engine ``augment_tile`` transforms (same as today).
        timepoint: ``t`` index to read.
        shard_g: FOVs per shard along ``g`` (raw outer-chunk size; 256 here).
        n_cols: scan-row width (FOVs/row); inferred from ``order`` if None.
        cache_rows: frame-LRU depth in scan rows (2 = sliding 2-row window).
    """

    def __init__(self, store_path, order, *, per_cam_flip=None,
                 flipud=False, fliplr=False, rot90=0, timepoint=0,
                 shard_g=256, n_cols=None, cache_rows=2):
        self.arr = zarr.open(str(store_path), mode="r")  # [t, g, y, x]
        self.timepoint = int(timepoint)
        self.shard_g = int(shard_g)
        self.flipud, self.fliplr, self.rot90 = flipud, fliplr, rot90
        pc = per_cam_flip or {}
        self.hflip, self.vflip = bool(pc.get("hflip")), bool(pc.get("vflip"))

        self.g_of = {(r, c): g for g, (r, c) in enumerate(order)}
        if n_cols is None:
            n_cols = max(c for _, c in order) + 1
        self.n_cols = int(n_cols)
        self.cache = _FrameLRU(max_size=cache_rows * self.n_cols + 2 * self.shard_g)
        self.n_shard_reads = 0  # diagnostics: should equal #shards touched, once each
        # Cache access lock. Shard I/O (the slow part — NFS + blosc decompress)
        # runs OUTSIDE the lock so distinct shards load in parallel; only the
        # cache read + frame-scatter is serialized.
        self._lock = Lock()
        # Per-shard "load in progress" table. When a thread misses cache on a
        # shard that another thread is already loading, we wait on that shard's
        # Event instead of duplicating the NFS read. This eliminates the "18×
        # wasted reads" pattern that dominated the assemble load phase.
        self._loading = {}  # shard_g0 -> threading.Event set when load completes
        # Profiling counters (aggregate seconds; ~thread-safe via GIL, exact
        # values under the lock in _ensure). Read from the caller after the
        # run finishes to see where shard-fetch wall time went.
        self.t_nfs_read = 0.0        # zarr[t, g0:g1] — NFS + blosc decompress
        self.t_augment = 0.0         # per-frame flip/rot90 on the shard block
        self.t_populate_lock = 0.0   # cache-populate time under the lock
        self.n_wasted_shard_reads = 0  # thread found cache populated before it could
        self.n_waited_for_peer = 0   # thread deferred to a peer's in-progress load

    def _augment(self, frame):
        # Per-camera flip first (was baked at convert), then engine augment.
        if self.hflip:
            frame = frame[:, ::-1]
        if self.vflip:
            frame = frame[::-1, :]
        return augment_tile(frame, self.flipud, self.fliplr, self.rot90)

    def _ensure(self, g: int):
        # Fast path: cache hit under the lock, return immediately.
        with self._lock:
            frame = self.cache.get(g)
            if frame is not None:
                return frame
        # Slow path with per-shard single-loader synchronization: exactly one
        # thread reads a given shard; concurrent misses on the same shard wait
        # on that thread's Event instead of duplicating the ~1 GB NFS read.
        shard_g0 = (g // self.shard_g) * self.shard_g
        my_event = None
        peer_event = None
        with self._lock:
            # Recheck: maybe another thread populated between our two locks.
            frame = self.cache.get(g)
            if frame is not None:
                return frame
            if shard_g0 in self._loading:
                # Someone else is already loading this shard — wait for them.
                peer_event = self._loading[shard_g0]
                self.n_waited_for_peer += 1
            else:
                # We're the designated loader for this shard.
                my_event = Event()
                self._loading[shard_g0] = my_event

        if peer_event is not None:
            peer_event.wait()
            with self._lock:
                frame = self.cache.get(g)
                if frame is not None:
                    return frame
                # Peer might have failed to populate (edge case); fall through
                # to load ourselves. This is not idempotent — a second loader
                # will race with any recovery attempts — but reasonably safe.
                my_event = Event()
                self._loading[shard_g0] = my_event

        # We are the loader for this shard.
        try:
            g1 = min(shard_g0 + self.shard_g, self.arr.shape[1])
            t0 = time.perf_counter()
            block = self.arr[self.timepoint, shard_g0:g1]  # NFS + blosc
            t_read = time.perf_counter() - t0
            t0 = time.perf_counter()
            augmented = [self._augment(block[i]) for i in range(block.shape[0])]
            t_aug = time.perf_counter() - t0
            t0 = time.perf_counter()
            with self._lock:
                self.t_nfs_read += t_read
                self.t_augment += t_aug
                # Recheck once more (defensive; a peer_event fallthrough could
                # have populated us by now).
                frame = self.cache.get(g)
                if frame is None:
                    self.n_shard_reads += 1
                    for i, f in enumerate(augmented):
                        self.cache[shard_g0 + i] = f
                    frame = self.cache[g]
                else:
                    self.n_wasted_shard_reads += 1
                self.t_populate_lock += time.perf_counter() - t0
                return frame
        finally:
            # Signal any peer threads waiting on this shard's load.
            with self._lock:
                self._loading.pop(shard_g0, None)
            my_event.set()

    def fetch(self, row: int, col: int):
        """Single-threaded warm-up of one tile (loads its shard if needed)."""
        return self._ensure(self.g_of[(row, col)])

    def __getitem__(self, name: str):
        """``name`` = ``"RRRCCC"`` (Edge calls this). Read-only once warmed."""
        return self._ensure(self.g_of[(int(name[:3]), int(name[3:]))])


def _neighbor_edges(present):
    """Ordered neighbor edges, identical set+orientation to ``connectivity``.

    For each present ``(r, c)`` (row-major): a horizontal edge
    ``(r, c-1)->(r, c)`` and a vertical edge ``(r-1, c)->(r, c)``. The ``b``
    endpoint is always ``(r, c)``, so grouping by ``b``'s row gives the edges to
    compute once a given row (and the one above it) is resident.
    """
    edges = []
    for (r, c) in sorted(present):
        if (r, c - 1) in present:           # horizontal (within row)
            edges.append(((r, c - 1), (r, c)))
        if (r - 1, c) in present:           # vertical (row-to-row)
            edges.append(((r - 1, c), (r, c)))
    return edges


def pairwise_shifts_streaming(
    source: RawShardTileSource,
    positions_rc,
    overlap,
    max_workers: int = 16,
    verbose: bool = False,
):
    """Row-streaming neighbor-shift estimation.

    Emits the same edge set/orientation as ``connectivity`` (see
    :func:`_neighbor_edges`), but in row order: warm a row's tiles
    single-threaded (one shard read each), then compute that row's edge offsets
    in parallel against the resident 2-row window before advancing. The previous
    row stays resident for the vertical edges, then ages out.

    Returns ``(edge_list, confidence_dict)`` matching ``pairwise_shifts``.
    """
    present = set(positions_rc)
    rows = sorted({r for r, _ in present})

    edges_by_row = {}
    for a, b in _neighbor_edges(present):
        edges_by_row.setdefault(b[0], []).append((a, b))  # b's row = "current" row

    edge_list = []
    confidence_dict = {}
    idx = 0

    def _work(a_key, b_key):
        e = Edge(a_key, b_key, source, overlap=overlap)
        conf = [list(map(int, a_key)), list(map(int, b_key)), float(e.model.confidence)]
        return e, conf

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for row in tqdm(rows, desc="Streaming rows"):
            # Warm this row (loads its shards; row-1 already resident from last iter).
            for (r, c) in present:
                if r == row:
                    source.fetch(row, c)

            # All endpoints resident -> parallel offsets are pure cache reads.
            futures = [ex.submit(_work, a, b) for a, b in edges_by_row.get(row, [])]
            for f in as_completed(futures):
                e, conf = f.result()
                edge_list.append(e)
                confidence_dict[str(idx)] = conf
                idx += 1
                if verbose:
                    cf = conf[2]
                    tag = " [LOW]" if cf < 0.3 else (" [WARN]" if cf < 0.5 else "")
                    print(f"  Edge {conf[0]} <-> {conf[1]} confidence={cf:.4f}{tag}")

    return edge_list, confidence_dict
