"""
GeoTesseraZarr — read embeddings from a Tessera zarr store.

Provides ``GeoTesseraZarr`` for store-level access (zone routing, point
sampling, region reading) and a ``tessera`` xarray accessor for per-zone
operations.  All spatial indexing uses xarray coordinate-based selection
(``sel(method='nearest')``), with no manual affine math.

Usage::

    from geotessera.store import GeoTesseraZarr

    gt = GeoTesseraZarr()  # default public store
    X = gt.sample_points([(-2.97, 53.44), (-2.96, 53.43)], year=2025)
    mosaic, transform, crs = gt.read_region(bbox, year=2025)

    # Direct zone access
    ds = gt.open_zone(lon=-2.97)
    emb = ds.tessera.sample_at(-2.97, 53.44, year=2025)
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import math
import struct
import threading
from typing import List, Optional, Tuple

import numpy as np
import rasterio.transform
import xarray as xr
import zarr
from pyproj import Transformer
from rich.progress import track

log = logging.getLogger(__name__)

DEFAULT_STORE = "https://s3.us-west-2.amazonaws.com/tessera-embeddings/v1/zarr"

# Shard-aligned chunk sizes so dask tasks match zarr shards
SHARD_CHUNKS = {"time": 1, "band": 128, "y": 4096, "x": 4096}


def enable_http_logging(level: int = logging.DEBUG) -> None:
    """Enable fsspec HTTP request logging for debugging.

    Call before opening a store to see every HTTP request::

        from geotessera.store import enable_http_logging
        enable_http_logging()
    """
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("fsspec.http").setLevel(level)
    log.setLevel(level)


def _zone_for_lon(lon: float) -> int:
    """UTM zone number (1-60) for a WGS84 longitude."""
    return max(1, min(60, int(math.floor((lon + 180) / 6)) + 1))


def _zone_for_point(x: float, y: float, crs: str = "EPSG:4326") -> int:
    """Determine UTM zone for a point in any CRS.

    For UTM EPSG codes (326xx/327xx), extracts the zone directly.
    Otherwise projects to WGS84 and computes from longitude.
    """
    from pyproj import CRS as ProjCRS

    if crs != "EPSG:4326":
        proj_crs = ProjCRS.from_user_input(crs)
        utm_zone = proj_crs.utm_zone
        if utm_zone is not None:
            return int(utm_zone[:-1])
        lon, _ = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(x, y)
    else:
        lon = x
    return _zone_for_lon(lon)


# ---------------------------------------------------------------------------
# Concurrent partial-shard reader
#
# Zarr's sharding codec decodes the inner chunks of a shard serially, so a
# small region read over a high-latency store (e.g. S3 across an ocean) pays
# one round-trip *per inner chunk, in sequence* — tens of round-trips for a
# 1 km box.  These helpers fetch only the inner chunks a window overlaps, but
# issue the byte-range requests concurrently over one pooled HTTPS session,
# collapsing N serial round-trips into ~1.  They transparently raise (and the
# caller falls back to the plain xarray read) for any store that is not a
# Zarr v3 sharded + blosc array reachable over http(s).
# ---------------------------------------------------------------------------

_zone_group_cache: dict = {}


def _zone_group(store_url: str, zone: str):
    """Open (and cache) the raw zarr group for a zone, for metadata access."""
    key = (store_url, zone)
    if key not in _zone_group_cache:
        _zone_group_cache[key] = zarr.open_group(store_url, mode="r")[zone]
    return _zone_group_cache[key]


def _shard_params(arr):
    """Extract sharding geometry, or raise ValueError if not fast-readable.

    Returns ``(shard_shape, inner_shape, inner_grid, index_suffix_len)``.
    """
    m = arr.metadata.to_dict()
    try:
        shard = tuple(m["chunk_grid"]["configuration"]["chunk_shape"])
        sc = next(c for c in m["codecs"] if c["name"] == "sharding_indexed")
    except (KeyError, StopIteration):
        raise ValueError("array is not a Zarr v3 sharded store")
    cfg = sc["configuration"]
    inner = tuple(cfg["chunk_shape"])
    if not any(c["name"] == "blosc" for c in cfg["codecs"]):
        raise ValueError("inner codec is not blosc")
    if cfg.get("index_location", "end") != "end":
        raise ValueError("shard index is not at end")
    grid = tuple(s // i for s, i in zip(shard, inner))
    has_crc = any(c["name"] == "crc32c" for c in cfg.get("index_codecs", []))
    return shard, inner, grid, int(np.prod(grid)) * 16 + (4 if has_crc else 0)


async def _read_window_async(base_url, arr, sel, concurrency, session):
    """Return ``arr[sel]`` by fetching only the overlapping inner chunks.

    ``sel`` is a tuple of int / slice, one entry per array dimension.
    """
    from numcodecs import Blosc

    blosc = Blosc()
    shard, inner, grid, idx_len = _shard_params(arr)
    nd = arr.ndim
    sentinel = (1 << 64) - 1

    starts, stops, squeeze = [], [], []
    for d in range(nd):
        s = sel[d]
        if isinstance(s, slice):
            starts.append(0 if s.start is None else int(s.start))
            stops.append(arr.shape[d] if s.stop is None else int(s.stop))
            squeeze.append(False)
        else:
            starts.append(int(s))
            stops.append(int(s) + 1)
            squeeze.append(True)

    out = np.zeros([stops[d] - starts[d] for d in range(nd)], dtype=arr.dtype)
    sem = asyncio.Semaphore(concurrency)
    tasks = []

    async def fetch(url, off, ln, dst, src):
        async with sem, session.get(url, headers={"Range": f"bytes={off}-{off + ln - 1}"}) as r:
            raw = await r.read()
        chunk = np.frombuffer(blosc.decode(raw), dtype=arr.dtype).reshape(inner)
        out[dst] = chunk[src]

    shard_ranges = [
        range(starts[d] // shard[d], (stops[d] - 1) // shard[d] + 1) for d in range(nd)
    ]
    for sh in itertools.product(*shard_ranges):
        url = base_url + "/c/" + "/".join(map(str, sh))
        async with sem, session.get(url, headers={"Range": f"bytes=-{idx_len}"}) as r:
            idx = await r.read()
        inner_ranges = []
        for d in range(nd):
            base = sh[d] * shard[d]
            lo, hi = max(starts[d], base), min(stops[d], base + shard[d])
            inner_ranges.append(range((lo - base) // inner[d], (hi - 1 - base) // inner[d] + 1))
        for ic in itertools.product(*inner_ranges):
            off, ln = struct.unpack_from("<QQ", idx, int(np.ravel_multi_index(ic, grid)) * 16)
            if off == sentinel:
                continue  # unwritten chunk → fill_value (already zeros)
            dst, src = [], []
            for d in range(nd):
                g0 = sh[d] * shard[d] + ic[d] * inner[d]
                a, b = max(starts[d], g0), min(stops[d], g0 + inner[d])
                dst.append(slice(a - starts[d], b - starts[d]))
                src.append(slice(a - g0, b - g0))
            tasks.append(fetch(url, off, ln, tuple(dst), tuple(src)))

    await asyncio.gather(*tasks)
    sq = tuple(d for d in range(nd) if squeeze[d])
    return out.squeeze(axis=sq) if sq else out


def _run_coro(coro):
    """Run an async coroutine to completion from sync code, even inside a loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    box = {}
    t = threading.Thread(target=lambda: box.setdefault("v", asyncio.run(coro)))
    t.start()
    t.join()
    return box["v"]


def _concurrent_read(store_url, zone, name, arr, sel, concurrency):
    """Concurrently read ``arr[sel]`` from an http(s) sharded store."""
    if not store_url.startswith(("http://", "https://")):
        raise ValueError("concurrent read only supports http(s) stores")
    import aiohttp

    base = f"{store_url.rstrip('/')}/{zone}/{name}"

    async def go():
        connector = aiohttp.TCPConnector(limit=concurrency)
        async with aiohttp.ClientSession(connector=connector) as session:
            return await _read_window_async(base, arr, sel, concurrency, session)

    return _run_coro(go())


def open_zone(
    store_url: str = DEFAULT_STORE,
    *,
    zone: Optional[int] = None,
    lon: Optional[float] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    **kwargs,
) -> xr.Dataset:
    """Open a tessera zone as an xarray Dataset.

    Provide exactly one of ``zone``, ``lon``, or ``bbox`` to select the
    UTM zone.  Returns a Dataset with the ``.tessera`` accessor.

    Args:
        store_url: Zarr store URL or local path.
        zone: UTM zone number (1-60).
        lon: A longitude — zone is derived automatically.
        bbox: (min_lon, min_lat, max_lon, max_lat) — zone from centre.

    Example::

        from geotessera.store import open_zone
        ds = open_zone(lon=-2.97)
        ds = open_zone(bbox=(-3.0, 53.4, -2.9, 53.5))
        ds = open_zone(zone=30)
    """
    match (zone, lon, bbox):
        case (int(), None, None):
            z = zone
        case (None, float(), None):
            z = _zone_for_lon(lon)
        case (None, None, tuple()):
            z = _zone_for_lon((bbox[0] + bbox[2]) / 2)
        case _:
            raise TypeError("Provide exactly one of zone=, lon=, or bbox=")

    log.debug("open_zone: utm%02d from %s", z, store_url)
    ds = xr.open_zarr(
        store_url,
        group=f"utm{z:02d}",
        zarr_format=3,
        consolidated=True,
        chunks=SHARD_CHUNKS,
        **kwargs,
    )

    # Attach computed piecewise tile transform — reproduces the tile
    # generation geometry (0.1° WGS84 → UTM projection) to compute
    # exact per-pixel coordinates without stored metadata.
    from .tile_transform import TesseraTileTransform

    try:
        transform = TesseraTileTransform.from_zone_attrs(ds.attrs)
        idx = xr.indexes.CoordinateTransformIndex(transform)
        ds = ds.assign_coords(xr.Coordinates.from_xindex(idx))
        log.debug("Attached TesseraTileTransform for EPSG:%d", transform.epsg)
    except Exception as exc:
        log.debug("Could not attach tile transform: %s", exc)

    # Record the source so the tessera accessor can issue concurrent
    # partial-shard reads against the raw store (see read_region).
    ds.attrs["geoemb:store_url"] = store_url
    ds.attrs["geoemb:zone"] = f"utm{z:02d}"

    return ds


# ---------------------------------------------------------------------------
# xarray accessor
# ---------------------------------------------------------------------------


@xr.register_dataset_accessor("tessera")
class TesseraAccessor:
    """Tessera-aware methods on an xarray Dataset from a zarr zone.

    Uses coordinate-based selection (``sel(method='nearest')``) for all
    spatial lookups — no manual affine math.  Reads ``proj:code`` and
    ``spatial:transform`` from Dataset attrs, years from the time coordinate.
    """

    def __init__(self, ds: xr.Dataset):
        self._ds = ds
        attrs = ds.attrs
        self._epsg: int = int(attrs["proj:code"].split(":")[1])
        self._to_utm = Transformer.from_crs(
            "EPSG:4326",
            f"EPSG:{self._epsg}",
            always_xy=True,
        )
        # Derive years from the time coordinate
        if "time" in ds.coords:
            self._years: list[int] = [int(v) for v in ds.coords["time"].values]
        else:
            self._years = []
        # Read n_bands from geoemb:dimensions if available, else from band dim
        self._n_bands: int = int(
            attrs.get("geoemb:dimensions", ds.sizes.get("band", 128))
        )
        t = attrs["spatial:transform"]
        self._px: float = float(t[0])
        log.debug("TesseraAccessor: EPSG:%d, years=%s", self._epsg, self._years)

    # -- Properties ---------------------------------------------------------

    @property
    def crs(self) -> str:
        """CRS string, e.g. ``'EPSG:32630'``."""
        return f"EPSG:{self._epsg}"

    @property
    def pixel_size(self) -> float:
        """Pixel size in CRS units (metres for UTM)."""
        return self._px

    @property
    def years(self) -> list[int]:
        """Available years, matching the time dimension order."""
        return self._years

    @property
    def n_bands(self) -> int:
        """Number of embedding bands."""
        return self._n_bands

    # -- Dequantisation -----------------------------------------------------

    @staticmethod
    def dequantise(emb_int8: np.ndarray, scales: np.ndarray) -> np.ndarray:
        """Dequantise int8 embeddings: ``(B,H,W)`` + ``(H,W)`` → ``(H,W,B)`` float32.

        Non-finite scales (NaN = water, +inf = no data) produce NaN rows.
        """
        valid = np.isfinite(scales)
        safe = np.where(valid, scales, 0.0)
        f32 = emb_int8.astype(np.float32) * safe[np.newaxis, :, :]
        f32[:, ~valid] = np.nan
        return f32.transpose(1, 2, 0)

    # -- Point sampling -----------------------------------------------------

    def sample_at(
        self,
        x: float,
        y: float,
        year: int,
        *,
        crs: str = "EPSG:4326",
    ) -> np.ndarray:
        """Sample a single dequantised embedding.  Returns ``(B,)`` float32.

        Args:
            x: Easting or longitude.
            y: Northing or latitude.
            year: Year to sample.
            crs: Input coordinate CRS (default WGS84).  Accepts any
                pyproj-compatible CRS string.
        """
        if crs == f"EPSG:{self._epsg}":
            e, n = x, y
        elif crs == "EPSG:4326":
            e, n = self._to_utm.transform(x, y)
        else:
            proj = Transformer.from_crs(crs, f"EPSG:{self._epsg}", always_xy=True)
            e, n = proj.transform(x, y)

        log.debug("sample_at(%.6f, %.6f, crs=%s) → UTM(%.1f, %.1f)", x, y, crs, e, n)

        # Use tile transform coords (xc/yc) if available, else regular (x/y)
        if "xc" in self._ds.coords:
            pixel = self._ds.sel(
                time=year,
                xc=xr.DataArray(e),
                yc=xr.DataArray(n),
                method="nearest",
            )
        else:
            pixel = self._ds.sel(time=year, x=e, y=n, method="nearest")
        scale = float(pixel["scales"].values)
        if not np.isfinite(scale):
            return np.full(self.n_bands, np.nan, dtype=np.float32)
        return pixel["embeddings"].values.astype(np.float32) * scale

    def sample_points(
        self,
        coords: List[Tuple[float, float]],
        year: int,
        *,
        crs: str = "EPSG:4326",
        progress: bool = True,
    ) -> np.ndarray:
        """Sample embeddings at points.  Returns ``(N, B)`` float32.

        Args:
            coords: List of (x, y) tuples in the given CRS.
            crs: Input coordinate CRS (default WGS84).
        """
        it = coords
        if progress:
            it = track(coords, description="Sampling points...", transient=True)
        return np.array([self.sample_at(x, y, year, crs=crs) for x, y in it])

    # -- Region reading -----------------------------------------------------

    def read_region(
        self,
        bbox: Tuple[float, float, float, float],
        year: int,
        *,
        crs: str = "EPSG:4326",
        progress: bool = False,
        concurrent: bool = True,
        concurrency: int = 32,
    ) -> Tuple[np.ndarray, rasterio.transform.Affine]:
        """Read and dequantise a bbox region.

        Args:
            bbox: (x_min, y_min, x_max, y_max) in the given CRS.
            crs: Input bbox CRS (default WGS84).
            concurrent: Fetch the overlapping shard chunks concurrently
                (default).  Dramatically faster on high-latency remote stores;
                falls back automatically to a plain serial read if the store
                is not an http(s) Zarr v3 sharded+blosc array.
            concurrency: Max simultaneous range requests for the fast path.

        Returns ``(mosaic, transform)`` where mosaic is ``(H, W, B)``
        float32 and transform is a rasterio Affine for the window.
        """
        zone_crs = f"EPSG:{self._epsg}"
        if crs == zone_crs:
            e_nw, n_nw = bbox[0], bbox[3]
            e_se, n_se = bbox[2], bbox[1]
        elif crs == "EPSG:4326":
            e_nw, n_nw = self._to_utm.transform(bbox[0], bbox[3])
            e_se, n_se = self._to_utm.transform(bbox[2], bbox[1])
        else:
            proj = Transformer.from_crs(crs, zone_crs, always_xy=True)
            e_nw, n_nw = proj.transform(bbox[0], bbox[3])
            e_se, n_se = proj.transform(bbox[2], bbox[1])
        e_min, e_max = min(e_nw, e_se), max(e_nw, e_se)
        n_min, n_max = min(n_nw, n_se), max(n_nw, n_se)

        # y is descending (north→south), so slice is (n_max, n_min)
        sub = self._ds.sel(time=year, x=slice(e_min, e_max), y=slice(n_max, n_min))
        h, w = int(sub.sizes["y"]), int(sub.sizes["x"])
        log.info(
            "read_region: %d x %d pixels (%s), %.0fm resolution",
            h,
            w,
            f"{h * w:,}",
            self._px,
        )

        emb_int8 = scales = None
        if concurrent and h and w:
            try:
                emb_int8, scales = self._read_region_concurrent(
                    year, e_min, e_max, n_max, n_min, concurrency
                )
            except Exception as exc:  # noqa: BLE001 — any failure → serial fallback
                log.debug("concurrent read unavailable (%s); using serial read", exc)

        if emb_int8 is None:
            if progress:
                from dask.diagnostics import ProgressBar

                with ProgressBar():
                    scales = sub["scales"].values
                    emb_int8 = sub["embeddings"].values
            else:
                scales = sub["scales"].values
                emb_int8 = sub["embeddings"].values

        mosaic = self.dequantise(emb_int8, scales)

        # Build affine from the selected window's coordinate values
        x0 = float(sub["x"].values[0]) - 0.5 * self._px  # pixel centre → corner
        y0 = float(sub["y"].values[0]) + 0.5 * self._px
        transform = rasterio.transform.Affine(self._px, 0, x0, 0, -self._px, y0)
        return mosaic, transform

    def _read_region_concurrent(self, year, e_min, e_max, n_max, n_min, concurrency):
        """Fast path: concurrent partial-shard read of the integer window.

        Returns ``(emb_int8 (B,H,W), scales (H,W))`` matching the serial
        ``.values`` reads.  Raises if the store is not fast-readable; the
        caller then falls back to the plain xarray read.
        """
        store_url = self._ds.attrs["geoemb:store_url"]
        zone = self._ds.attrs["geoemb:zone"]
        xs = self._ds.get_index("x").slice_indexer(e_min, e_max)
        ys = self._ds.get_index("y").slice_indexer(n_max, n_min)
        time_vals = np.asarray(self._ds.get_index("time"))
        ti = int(np.argwhere(time_vals == year)[0][0])
        grp = _zone_group(store_url, zone)
        x_sl, y_sl = slice(xs.start, xs.stop), slice(ys.start, ys.stop)
        emb = _concurrent_read(
            store_url, zone, "embeddings", grp["embeddings"],
            (ti, slice(None), y_sl, x_sl), concurrency,
        )
        scales = _concurrent_read(
            store_url, zone, "scales", grp["scales"],
            (ti, y_sl, x_sl), concurrency,
        )
        return emb, scales


# ---------------------------------------------------------------------------
# GeoTesseraZarr — store-level API with zone routing
# ---------------------------------------------------------------------------


class GeoTesseraZarr:
    """Read embeddings from a Tessera zarr store.

    Routes geographic queries to the correct UTM zone automatically.
    For single-zone work, use :func:`open_zone` directly.

    Args:
        store_url: Zarr store URL or local path.  Defaults to the public
            TESSERA store at ``s3.us-west-2.amazonaws.com/tessera-embeddings``.

    Example::

        from geotessera.store import GeoTesseraZarr

        gt = GeoTesseraZarr()
        print(gt.years)  # [2017, 2018, ..., 2025]

        # Sample embeddings at points
        X = gt.sample_points([(-2.97, 53.44)], year=2025)

        # Read a region
        mosaic, transform, crs = gt.read_region(
            (-3.0, 53.4, -2.9, 53.5), year=2025,
        )
    """

    def __init__(self, store_url: str = DEFAULT_STORE):
        self.url = store_url.rstrip("/")
        root = zarr.open_group(self.url, mode="r")
        root_attrs = dict(root.attrs)
        self.model_version: str = root_attrs.get("geoemb:model", "")
        self.build_version: str = root_attrs.get("geoemb:build_version", "")
        # Derive years from the first zone's time coordinate array
        self.years: list[int] = []
        for member_name in sorted(root.keys()):
            if member_name.startswith("utm"):
                try:
                    zone_grp = root[member_name]
                    time_arr = zone_grp["time"][:]
                    self.years = [int(v) for v in time_arr]
                    break
                except Exception:
                    continue
        self._cache: dict[int, xr.Dataset] = {}
        log.info(
            "GeoTesseraZarr: %s, years=%s, model=%s",
            self.url,
            self.years,
            self.model_version,
        )

    def __repr__(self) -> str:
        return f"GeoTesseraZarr({self.url!r}, years={self.years})"

    # -- Zone access --------------------------------------------------------

    def open_zone(
        self,
        *,
        zone: Optional[int] = None,
        lon: Optional[float] = None,
        bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> xr.Dataset:
        """Open a zone Dataset with the ``.tessera`` accessor.

        Provide exactly one of ``zone``, ``lon``, or ``bbox``.
        Datasets are cached for the lifetime of this instance.
        """
        match (zone, lon, bbox):
            case (int(), None, None):
                z = zone
            case (None, float(), None):
                z = _zone_for_lon(lon)
            case (None, None, tuple()):
                z = _zone_for_lon((bbox[0] + bbox[2]) / 2)
            case _:
                raise TypeError("Provide exactly one of zone=, lon=, or bbox=")

        if z not in self._cache:
            ds = open_zone(self.url, zone=z)
            self._cache[z] = ds
        return self._cache[z]

    # -- Point sampling (cross-zone) ----------------------------------------

    def sample_at(
        self,
        x: float,
        y: float,
        year: int,
        *,
        crs: str = "EPSG:4326",
    ) -> np.ndarray:
        """Sample a single embedding, routing to the correct zone.

        Args:
            x: Easting or longitude.
            y: Northing or latitude.
            crs: Input CRS (default WGS84).

        Returns ``(B,)`` float32.
        """
        z = _zone_for_point(x, y, crs)
        ds = self.open_zone(zone=z)
        return ds.tessera.sample_at(x, y, year, crs=crs)

    def sample_points(
        self,
        coords: List[Tuple[float, float]],
        year: int,
        *,
        crs: str = "EPSG:4326",
        progress: bool = True,
    ) -> np.ndarray:
        """Sample embeddings at points, routing each to its zone.

        Args:
            coords: List of (x, y) tuples in the given CRS.
            crs: Input CRS (default WGS84).

        Returns ``(N, B)`` float32.  Points outside coverage get NaN rows.
        """
        it = coords
        if progress:
            it = track(coords, description="Sampling points...", transient=True)
        return np.array([self.sample_at(x, y, year, crs=crs) for x, y in it])

    # -- Region reading (dominant zone) -------------------------------------

    def read_region(
        self,
        bbox: Tuple[float, float, float, float],
        year: int,
        *,
        crs: str = "EPSG:4326",
        progress: bool = False,
    ) -> Tuple[np.ndarray, rasterio.transform.Affine, str]:
        """Read and dequantise a bbox region.

        Args:
            bbox: (x_min, y_min, x_max, y_max) in the given CRS.
            crs: Input bbox CRS (default WGS84).

        Uses the dominant UTM zone (from bbox centre).  Returns
        ``(mosaic, transform, crs)`` where mosaic is ``(H, W, B)`` float32.
        """
        # Convert bbox centre to WGS84 for zone routing
        cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        z = _zone_for_point(cx, cy, crs)

        ds = self.open_zone(zone=z)
        mosaic, transform = ds.tessera.read_region(
            bbox, year, crs=crs, progress=progress
        )
        return mosaic, transform, ds.tessera.crs
