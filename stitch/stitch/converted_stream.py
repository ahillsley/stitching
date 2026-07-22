"""Per-FOV chunk tile source for the shard_to_hcs converted store.

Drop-in replacement for ``RawShardTileSource`` when the raw acquire-zarr shards
have already been converted to a per-FOV HCS zarr on local scratch. Estimate
reads from ``/tmp/livescreen_JOBID/livescreen_converted.zarr`` instead of
``/hpc/instruments/.../*.ome.zarr``.

Why: raw shards are 640 MB blosc-encoded blocks on NFS. To fetch one FOV you
touch the whole shard (256 FOVs × 4 cams). With a per-FOV converted store on
local xfs, each fetch is a single 5-MB chunk file at 2 GB/s local disk. Cuts
estimate's read wall by ~5-8x.

Interface matches ``RawShardTileSource``:
    src = ConvertedTileSource(store_root, channel=0, flipud=..., ...)
    tile = src["003005"]      # -> numpy uint16 (frame_y, frame_x)

Layout expected (from ``ops_analysis.livescreen.stitch.shard_to_hcs``):
    <store_root>/A/1/<RRRCCC>/0/zarr.json
    <store_root>/A/1/<RRRCCC>/0/c/0/<C>/0/0/0    (blosc-encoded chunk file)
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

import numcodecs
import numpy as np

# Inlined from stitch.stitch.tile.augment_tile — that module transitively
# imports `dexp` (rapids-dask-cuda etc), which bloats the import graph for
# any lightweight consumer of ConvertedTileSource (Dragon workers, tests).
# augment_tile itself is a couple of numpy calls; keeping a local copy here
# avoids pulling in the whole reconstruction stack.
def augment_tile(tile, flipud: bool, fliplr: bool, rot90: int):
    if flipud:
        tile = np.flip(tile, axis=-2)
    if fliplr:
        tile = np.flip(tile, axis=-1)
    if rot90:
        tile = np.rot90(tile, k=rot90, axes=(-2, -1))
    return tile


class ConvertedTileSource:
    """Reads one channel of a per-FOV HCS converted store from local disk.

    Codec is auto-detected from a sample tile's ``zarr.json`` (currently
    ``numcodecs.Blosc(cname='zstd', clevel=1, shuffle=1)`` from the writer's
    2026-07-10 codec swap; historical stores used plain zstd — those are NOT
    handled by this reader since convert now emits Blosc uniformly).

    Per-cam flips are NOT applied here — the writer (`shard_to_hcs.py`) bakes
    them in during ingest. This class only applies the engine-level
    ``flipud/fliplr/rot90`` (matches ``RawShardTileSource``'s post-per-cam step).

    Args:
        store_root: path to the converted store (e.g. ``.../livescreen_converted.zarr``).
        channel: index into the writer's ``cams`` list. LiveScreen estimate
            typically uses channel 0 (``z0c0_PHX5``).
        flipud/fliplr/rot90: engine augmentation. Matches
            ``ops_analysis.livescreen.config.LivescreenConfig.stitch``.
    """

    def __init__(self, store_root, *, channel: int = 0,
                 flipud: bool = False, fliplr: bool = False, rot90: int = 0,
                 frame_shape: tuple[int, int] | None = None,
                 dtype: str = "uint16",
                 well_id: str = "A/1"):
        self.store_root = Path(store_root)
        self.channel = int(channel)
        self.flipud = bool(flipud)
        self.fliplr = bool(fliplr)
        self.rot90 = int(rot90) % 4
        # Which HCS well this source reads (single-well per instance). Slash
        # form ("A/1") — split into (row, col) for path construction.
        _wparts = well_id.strip().split("/", 1)
        if len(_wparts) != 2:
            raise ValueError(f"well_id must be '<row>/<col>' (got {well_id!r})")
        self.well_id = f"{_wparts[0]}/{_wparts[1]}"
        self._well_path = self.store_root / _wparts[0] / _wparts[1]

        # Peek at a sample tile's zarr.json to confirm chunk shape + dtype.
        # Matches assemble_gpu's _build_direct_chunk_tile_loader flow.
        if frame_shape is None:
            sample_dir = next(self._well_path.glob("*/0"))
            with open(sample_dir / "zarr.json") as fh:
                meta = json.load(fh)
            chunk_shape = tuple(meta["chunk_grid"]["configuration"]["chunk_shape"])
            # chunk_shape is (T, C, Z, Y, X) — take Y, X.
            self.frame_shape = (chunk_shape[-2], chunk_shape[-1])
            self.dtype = np.dtype(meta["data_type"])
        else:
            self.frame_shape = tuple(frame_shape)
            self.dtype = np.dtype(dtype)

        # Cache of already-loaded tiles. Populated on-demand by __getitem__ or
        # in bulk by load_many(). Callers manage eviction via evict() to keep
        # memory bounded — a full 199-row plate would be ~830 GB decompressed.
        self._cache: dict[str, np.ndarray] = {}
        self._cache_lock = Lock()

        # Diagnostics
        self.n_reads = 0

    def _fetch_one(self, name: str) -> np.ndarray:
        """Actually read + decode + augment. Not cached — for direct call sites."""
        chunk_path = (
            self._well_path / name / "0"
            / "c" / "0" / str(self.channel) / "0" / "0" / "0"
        )
        # numcodecs.Blosc holds an internal decompression context that is NOT
        # thread-safe (per the note in assemble_gpu.py). Instantiate per call —
        # construction is <1 ms — so multi-threaded callers stay safe.
        codec = numcodecs.Blosc()
        with open(chunk_path, "rb") as fh:
            raw = fh.read()
        decoded = codec.decode(raw)
        arr = np.frombuffer(decoded, dtype=self.dtype).reshape(self.frame_shape)
        # Convert doesn't apply flipud/fliplr/rot90 (those are engine-level
        # augment ops), so we apply them here to match RawShardTileSource's
        # post-per-cam-flip augment_tile call.
        arr = augment_tile(arr, self.flipud, self.fliplr, self.rot90)
        return np.ascontiguousarray(arr)

    def __getitem__(self, name: str) -> np.ndarray:
        """Fetch one FOV by ``RRRCCC`` name. Serves from cache if pre-loaded."""
        with self._cache_lock:
            cached = self._cache.get(name)
        if cached is not None:
            return cached
        # Not cached — fetch synchronously.
        self.n_reads += 1
        return self._fetch_one(name)

    def load_many(self, names, max_workers: int = 16) -> None:
        """Parallel-preload the given ``names`` into the cache.

        Callers use this at the start of a row-block: fan out N tile reads
        across ``max_workers`` threads so ~5-10× parallel /tmp reads keep the
        downstream GPU pipeline fed. Serial ``__getitem__`` would leave the
        GPU idle waiting for each tile in turn.

        Idempotent: names already in cache are skipped.
        """
        with self._cache_lock:
            to_load = [n for n in names if n not in self._cache]
        if not to_load:
            return
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for name, arr in zip(to_load, pool.map(self._fetch_one, to_load)):
                with self._cache_lock:
                    self._cache[name] = arr
        self.n_reads += len(to_load)

    def evict(self, names) -> None:
        """Drop the given names from cache to free memory. Missing names OK."""
        with self._cache_lock:
            for n in names:
                self._cache.pop(n, None)

    def cache_size(self) -> int:
        with self._cache_lock:
            return len(self._cache)


# --------------------------------------------------------------------------
# Dragon-aware variant — reads chunks from LOCAL /tmp when the FOV lives on
# this node, or fetches over Dragon HSTA when it lives on another node.
# --------------------------------------------------------------------------

def dragon_fetch_server(local_out_path: str, in_queue, ready_event,
                        well_id: str = "A/1"):
    """Runs on each allocated node as a Dragon Process. Serves incoming
    FOV chunk requests by reading the requested channel's chunk file
    from LOCAL /tmp and returning raw (still-Blosc-encoded) bytes.

    Wire format:
      request  = ("batch_get", [pos_name, ...], channel, response_queue)
      request  = ("stop", None, None, None)
      response = (server_host, {pos_name: raw_bytes, ...})   # streamed
      response = (server_host, None)                          # end-of-batch

    Sends raw (encoded) bytes back — the caller does the Blosc decode
    on the reader side to keep per-server work small and let decode
    parallelize across worker threads.

    Streams tiles as small dicts (DRAGON_FETCH_MSG_TILES per message)
    followed by a `None` sentinel, so a single put() never carries more
    than a few hundred MB. One-shot puts of full batch dicts (~4.5 GB
    for 300 tiles) blew Dragon's dynheap and deadlocked the loader.
    """
    import socket
    from pathlib import Path
    from concurrent.futures import as_completed

    host = socket.gethostname()
    _wparts = well_id.strip().split("/", 1)
    root = Path(local_out_path) / _wparts[0] / _wparts[1]
    ready_event.set()
    n_served = 0
    # Multi-threaded per-batch reads. Peer nodes send batch_gets of hundreds of
    # tiles; reading them sequentially with a single thread bounded us at
    # ~50 μs/open × N files = ~50 ms per batch of 1000 tiles. Using a
    # ThreadPoolExecutor to run open+read in parallel scales with NVMe queue
    # depth (typically 16-32 concurrent readers is optimal). Env-configurable.
    import os as _os
    from concurrent.futures import ThreadPoolExecutor as _TPE
    n_read_workers = int(_os.environ.get("DRAGON_FETCH_READ_WORKERS", "16"))
    msg_tiles = int(_os.environ.get("DRAGON_FETCH_MSG_TILES", "8"))
    _pool = _TPE(max_workers=n_read_workers)
    while True:
        msg = in_queue.get()
        op = msg[0]
        if op == "stop":
            _pool.shutdown(wait=False)
            return
        if op != "batch_get":
            continue
        _, names, channel, response_queue = msg
        def _read_one(name):
            chunk_path = (root / name / "0"
                          / "c" / "0" / str(channel) / "0" / "0" / "0")
            with open(chunk_path, "rb") as fh:
                return name, fh.read()
        futs = [_pool.submit(_read_one, name) for name in names]
        batch: dict[str, bytes] = {}
        for fut in as_completed(futs):
            name, raw = fut.result()
            batch[name] = raw
            if len(batch) >= msg_tiles:
                response_queue.put((host, batch))
                batch = {}
        if batch:
            response_queue.put((host, batch))
        response_queue.put((host, None))
        n_served += len(names)


class DragonAwareConvertedTileSource(ConvertedTileSource):
    """ConvertedTileSource that reads local FOVs from local /tmp and remote
    FOVs via Dragon HSTA. Drop-in replacement for ConvertedTileSource;
    same `__getitem__`/`load_many`/`evict` interface.

    Args (beyond parent's):
      shard_map: {pos_name: (owner_host, local_root_path_on_owner)}
      server_queues: {host: mp.Queue} — the request queue each host's
                     `dragon_fetch_server` process is polling.
      my_host: hostname of the current process (default: socket.gethostname())

    The base class's `store_root` is still used for local reads; when
    `shard_map[name]` says the FOV lives here, we read directly from disk
    (unchanged fast path). When it lives on another host, we send a
    `batch_get` to that host's queue and decode the returned raw bytes.

    Batches remote requests per-host: one message per (host, load_many)
    call, not one per FOV. Cuts Dragon round-trips from O(N_fovs) to
    O(N_hosts) per block.
    """

    def __init__(self, *args, shard_map=None, server_queues=None,
                 my_host: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        import socket
        self.shard_map = dict(shard_map or {})
        self.server_queues = dict(server_queues or {})
        self._my_host = my_host or socket.gethostname()
        # Per-worker-process response queue for remote fetches. Created
        # lazily to avoid Dragon channel creation cost when the caller
        # never triggers a remote fetch.
        self._response_queue = None

    def _get_response_queue(self):
        """Lazy singleton — one mp.Queue per DragonAware instance for
        collecting batched remote responses."""
        if self._response_queue is None:
            import multiprocessing as mp
            self._response_queue = mp.Queue()
        return self._response_queue

    def _split_by_host(self, names):
        """Return (local_names, {remote_host: [names]})."""
        from collections import defaultdict
        # OPS_ASSEMBLE_FORCE_LOCAL=1: skip shard_map routing; treat every FOV
        # as local. Safe only when the caller guarantees each node's /tmp has
        # the full converted store (see Y-partitioned / full-mirror convert).
        import os as _os
        if _os.environ.get("OPS_ASSEMBLE_FORCE_LOCAL", "0") == "1":
            return list(names), {}
        local: list[str] = []
        remote: dict[str, list[str]] = defaultdict(list)
        for name in names:
            entry = self.shard_map.get(name)
            if entry is None or entry[0] == self._my_host:
                local.append(name)
            else:
                remote[entry[0]].append(name)
        return local, dict(remote)

    def load_many(self, names, max_workers: int = 16) -> None:
        """Load names into cache. Local names use the parent's threaded
        read path; remote names batch-request to each owner host and
        decode the returned encoded bytes.

        Multi-stream: `MULTI_FETCH_STREAMS` env (default 1) splits each
        per-host batch into K parallel batch_get requests. Single Dragon
        channel between two nodes tops out at ~1 GB/s; K=4-8 aggregates
        to ~3 GB/s per node pair (measured in HSTA stream sweep, jobs
        34842934 / 34843612). Diminishing returns past K≈8.
        """
        import os as _os
        with self._cache_lock:
            to_load = [n for n in names if n not in self._cache]
        if not to_load:
            return

        local, remote_by_host = self._split_by_host(to_load)

        # 1) Local reads via the parent's ThreadPoolExecutor path.
        if local:
            super().load_many(local, max_workers=max_workers)

        # 2) Remote reads: one or K batch requests per host, in parallel.
        if remote_by_host:
            n_streams = max(1, int(_os.environ.get("MULTI_FETCH_STREAMS", "1")))
            resp_q = self._get_response_queue()
            outstanding = 0
            for host, host_names in remote_by_host.items():
                # Split host_names into n_streams sub-batches (round-robin
                # so each stream has ~equal work). Small hosts collapse to
                # min(n_streams, len(names)) streams.
                k = min(n_streams, len(host_names))
                for s in range(k):
                    sub = host_names[s::k]
                    if not sub:
                        continue
                    self.server_queues[host].put(
                        ("batch_get", sub, self.channel, resp_q))
                    outstanding += 1
            # Decode responses as they arrive. Fetch server streams: each
            # outstanding request emits ≥1 result chunk + a (host, None)
            # sentinel. Count sentinels, not messages.
            codec = numcodecs.Blosc()
            sentinels_seen = 0
            while sentinels_seen < outstanding:
                _host, chunk = resp_q.get()
                if chunk is None:
                    sentinels_seen += 1
                    continue
                for name, raw in chunk.items():
                    decoded = codec.decode(raw)
                    arr = np.frombuffer(decoded, dtype=self.dtype).reshape(
                        self.frame_shape)
                    arr = augment_tile(
                        arr, self.flipud, self.fliplr, self.rot90)
                    with self._cache_lock:
                        self._cache[name] = np.ascontiguousarray(arr)
            self.n_reads += sum(len(v) for v in remote_by_host.values())

    def _fetch_one(self, name: str) -> np.ndarray:
        """Single-FOV fetch — routes to remote if needed. Used by
        `__getitem__` when the cache misses (uncommon in practice —
        callers pre-load via load_many)."""
        entry = self.shard_map.get(name)
        if entry is None or entry[0] == self._my_host:
            return super()._fetch_one(name)
        # Remote: one-off batch_get of size 1. Fetch server now streams
        # results as (host, {names: raw, ...}) chunks terminated by
        # (host, None) — drain until sentinel.
        host = entry[0]
        resp_q = self._get_response_queue()
        self.server_queues[host].put(
            ("batch_get", [name], self.channel, resp_q))
        merged: dict = {}
        while True:
            _host, chunk = resp_q.get()
            if chunk is None:
                break
            merged.update(chunk)
        raw = merged[name]
        codec = numcodecs.Blosc()
        decoded = codec.decode(raw)
        arr = np.frombuffer(decoded, dtype=self.dtype).reshape(self.frame_shape)
        arr = augment_tile(arr, self.flipud, self.fliplr, self.rot90)
        return np.ascontiguousarray(arr)
