"""Snapshot-bound Icechunk → geoembeddings Zarr v3 migration.

Data lives in the destination; inventories, ownership and commit receipts
live in a separate local/S3 state directory. A receipt is published only
after every selected array in a spatial shard has been written successfully.
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import io
import json
import logging
import math
import os
import re
import signal
import tempfile
import threading
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr

from . import icechunk_source as source
from ._migration_state import (
    BusyError,
    State,
    canonical_url,
    digest,
    json_bytes,
    owner_stopped,
)
from .zarr import GEOEMB_CONVENTION, StoreLocation, _geo_convention_attrs

FORMAT_VERSION = 1
RESUMABLE = 75
MASK_SOURCE = "geotessera:mask_source"
MIGRATION_ID = "geotessera:migration_id"
logger = logging.getLogger(__name__)


def event(event_name, **fields):
    logger.info(json_bytes({"event": event_name, **fields}).decode())


def _versions():
    return {
        n: importlib.metadata.version(n)
        for n in ("geotessera", "icechunk", "zarr", "numcodecs", "pcodec")
    }


def _implementation():
    return digest(
        b"".join(
            (Path(__file__).parent / n).read_bytes()
            for n in ("transcode.py", "icechunk_source.py", "_migration_state.py")
        )
    )


def _planning_options(request):
    """Return user-visible options; code revisions do not change the plan."""
    return {
        key: value
        for key, value in request.items()
        if key not in ("implementation", "versions")
    }


def _partition_bytes(units):
    table = pa.table(
        {
            name: pa.array([row[i] for row in units], type=pa.int32())
            for i, name in enumerate(("year", "row", "col"))
        }
    )
    output = io.BytesIO()
    pq.write_table(table, output, compression="zstd")
    return output.getvalue()


def create_plan(
    source_url,
    output,
    destination,
    *,
    snapshot_id=None,
    branch="main",
    zones=None,
    years=None,
    arrays=None,
    shard_size=4096,
    inner_chunk=32,
    source_options=None,
    state_options=None,
):
    """Inventory a source without writing to the destination.

    ``output`` is a durable directory, not a local job checkpoint. Repeated
    preparation uses planning.json's pinned snapshot even if main advances.
    """
    if shard_size < 1 or inner_chunk < 1 or shard_size % inner_chunk:
        raise ValueError("shard_size must be a positive multiple of inner_chunk")
    if zones is not None and (not zones or any(z < 1 or z > 60 for z in zones)):
        raise ValueError("zones must be in 1..60")
    source_url, output, destination = map(
        canonical_url, (source_url, output, destination)
    )
    for a, b in (
        (output, destination),
        (source_url, destination),
        (source_url, output),
    ):
        if a == b or a.startswith(b + "/") or b.startswith(a + "/"):
            raise ValueError(
                "Source, destination and state must be separate, non-nested prefixes"
            )
    state = State(output, state_options)
    request = {
        "source": source_url,
        "destination": destination,
        "state": output,
        "branch": branch,
        "snapshot_requested": snapshot_id,
        "zones": sorted(set(zones)) if zones is not None else None,
        "years": sorted(set(years)) if years is not None else None,
        "arrays": sorted(set(arrays)) if arrays is not None else None,
        "shard_size": shard_size,
        "inner_chunk": inner_chunk,
        "format_version": FORMAT_VERSION,
        "implementation": _implementation(),
        "versions": _versions(),
    }
    with state.owner("planner"):
        old, _ = state.get("planning.json")
        if old is not None:
            header = json.loads(old)
            if _planning_options(header["request"]) != _planning_options(request):
                raise ValueError(
                    "Preparation options changed; use a new state and destination prefix"
                )
            # Preserve the exact original header so the migration id and
            # completed zone inventories remain stable across compatible fixes.
            request = header["request"]
            snapshot_id = header["snapshot_id"]
        session, root = source.open_source(
            source_url, snapshot_id=snapshot_id, branch=branch, options=source_options
        )
        header = {"request": request, "snapshot_id": session.snapshot_id}
        state.immutable("planning.json", json_bytes(header))
        root_attrs = dict(root.attrs)
        if (
            root_attrs.get("geoemb:dimensions") != 128
            or root_attrs.get("geoemb:data_type") != "int8"
        ):
            raise ValueError("Expected a 128-dimensional int8 geoembeddings source")
        names = sorted(root.group_keys())
        if any(not re.fullmatch(r"(?:0[1-9]|[1-5][0-9]|60)[NS]", n) for n in names):
            raise ValueError("Expected only hemisphere groups named 01N through 60S")
        available = sorted({int(n[:2]) for n in names})
        selected = request["zones"] or available
        if not selected or not set(selected) <= set(available):
            raise ValueError("Requested zones are absent from the source")
        partitions = {}
        for zone in selected:
            key = f"inventory/utm{zone:02d}.json"
            previous, _ = state.get(key)
            if previous is not None:
                zone_doc = json.loads(previous)
                if zone_doc["planning_hash"] != digest(json_bytes(header)):
                    raise ValueError(f"Conflicting inventory for zone {zone}")
                raw, _ = state.get(zone_doc["inventory"])
                if raw is None or digest(raw) != zone_doc["inventory_sha256"]:
                    raise ValueError(f"Corrupt inventory for zone {zone}")
            else:
                groups = [
                    source.inspect_group(root[n], n, arrays)
                    for n in names
                    if int(n[:2]) == zone
                ]
                available_years = sorted({y for g in groups for y in g["years"]})
                selected_years = request["years"] or available_years
                if not selected_years or not set(selected_years) <= set(
                    available_years
                ):
                    raise ValueError(f"Zone {zone}: requested years absent from source")
                grid = source.zone_grid(groups, selected_years, shard_size)
                schemas, static = _output_schema(groups, grid)
                units, counts = asyncio.run(
                    source.inventory(session, root, groups, grid, shard_size, event)
                )
                raw = _partition_bytes(units)
                inventory_key = f"inventory/utm{zone:02d}.parquet"
                state.immutable(inventory_key, raw)
                zone_doc = {
                    "zone": zone,
                    "grid": grid,
                    "groups": groups,
                    "arrays": schemas,
                    "static": static,
                    "unit_count": len(units),
                    "source_chunks": counts,
                    "inventory": inventory_key,
                    "inventory_sha256": digest(raw),
                    "planning_hash": digest(json_bytes(header)),
                }
                state.immutable(key, json_bytes(zone_doc))
            partitions[str(zone)] = {
                "path": key,
                "sha256": digest(json_bytes(zone_doc)),
                "units": zone_doc["unit_count"],
            }
            event("zone_planned", zone=zone, units=zone_doc["unit_count"])
        plan = {**header, "source_attrs": root_attrs, "partitions": partitions}
        plan["migration_id"] = digest(json_bytes(plan))
        state.immutable("plan.json", json_bytes(plan))
        event(
            "plan_ready",
            migration_id=plan["migration_id"],
            snapshot_id=session.snapshot_id,
            units=sum(p["units"] for p in partitions.values()),
        )
        return plan


def _output_schema(groups, grid):
    schemas, static = {}, {}
    for g in groups:
        for name, spec in g["arrays"].items():
            ds = spec["output_dims"]
            src_shape = dict(zip(spec["dims"], spec["shape"]))
            shape = [
                {
                    "time": len(grid["years"]),
                    "y": grid["height"],
                    "x": grid["width"],
                }.get(d, src_shape[d])
                for d in ds
            ]
            attrs = dict(spec["attrs"])
            # Spatial/coordinate references refer to destination axes now.
            if "coordinates" in attrs:
                attrs["coordinates"] = " ".join(
                    source.SPATIAL_NAMES.get(d, d) for d in attrs["coordinates"].split()
                )
            item = {
                "dims": ds,
                "shape": shape,
                "dtype": spec["dtype"],
                "fill": spec["fill"],
                "attrs": attrs,
            }
            if name in schemas and schemas[name] != item:
                raise ValueError(f"Hemisphere schemas differ for {name}")
            schemas[name] = item
        for name, spec in g["static"].items():
            # Time-dependent static arrays are combined by explicit year.
            ds = spec["dims"]
            if name not in static:
                static[name] = {
                    k: v for k, v in spec.items() if k not in ("data", "shape")
                }
                static[name]["shape"] = [
                    len(grid["years"]) if d == "time" else s
                    for d, s in zip(ds, spec["shape"])
                ]
                static[name]["_pieces"] = {}
            current = static[name]
            if any(current[k] != spec[k] for k in ("dims", "dtype", "fill", "attrs")):
                raise ValueError(f"Hemisphere schemas differ for {name}")
            data = np.asarray(spec["data"], dtype=spec["dtype"])
            if "time" in ds:
                axis = ds.index("time")
                pieces = {
                    year: np.take(data, i, axis=axis).tolist()
                    for i, year in enumerate(g["years"])
                    if year in grid["years"]
                }
            else:
                pieces = {"all": data.tolist()}
            for key, value in pieces.items():
                if key in current["_pieces"] and current["_pieces"][key] != value:
                    raise ValueError(
                        f"Hemisphere coordinate values differ for {name}/{key}"
                    )
                current["_pieces"][key] = value
    for name, spec in static.items():
        pieces = spec.pop("_pieces")
        if "time" in spec["dims"]:
            if any(y not in pieces for y in grid["years"]):
                raise ValueError(f"Missing annual coordinate values for {name}")
            spec["data"] = np.stack(
                [pieces[y] for y in grid["years"]], axis=spec["dims"].index("time")
            ).tolist()
        else:
            spec["data"] = pieces["all"]
    return schemas, static


class Migration:
    def __init__(
        self,
        plan_url,
        *,
        state_options=None,
        store_options=None,
        source_options=None,
        destination=None,
    ):
        if str(plan_url).endswith("/plan.json"):
            plan_url = str(plan_url)[:-10]
        self.state = State(plan_url, state_options)
        self.plan = self.state.read("plan.json")
        unsigned = {k: v for k, v in self.plan.items() if k != "migration_id"}
        if digest(json_bytes(unsigned)) != self.plan.get("migration_id"):
            raise ValueError("Migration plan checksum mismatch")
        self.request = self.plan["request"]
        if self.request["format_version"] != FORMAT_VERSION:
            raise ValueError("Unsupported migration plan format")
        if self.request["state"] != self.state.url:
            raise ValueError("Plan was moved to a different state prefix")
        if (
            destination is not None
            and canonical_url(destination) != self.request["destination"]
        ):
            raise ValueError("Destination differs from immutable migration plan")
        self.store = StoreLocation(self.request["destination"], store_options)
        self.source_options = source_options
        self.shard_size = self.request["shard_size"]
        self.zones = sorted(int(z) for z in self.plan["partitions"])
        self.id = self.plan["migration_id"]
        self._source = None
        self._object_index = None
        self._completion_invalidated = False
        self._invalidation_lock = threading.Lock()
        # Hemisphere-seam conflict handling (mtelvers fork). "error" (default)
        # keeps upstream behaviour: abort on any N/S disagreement at the
        # equatorial overlap. "north"/"south" instead keep that hemisphere's
        # values, emit a seam_conflict event, and continue -- so the migration
        # can complete despite known upstream source-data inconsistencies. The
        # logged seam_conflict events are the exact redo manifest once the data
        # is fixed upstream (reported to dClimate).
        self.seam_conflict = os.environ.get("GEOTESSERA_SEAM_CONFLICT", "error")
        if self.seam_conflict not in ("error", "north", "south"):
            raise ValueError(
                "GEOTESSERA_SEAM_CONFLICT must be one of error, north, south; "
                f"got {self.seam_conflict!r}"
            )

    @property
    def source(self):
        if self._source is None:
            _, self._source = source.open_source(
                self.request["source"],
                snapshot_id=self.plan["snapshot_id"],
                options=self.source_options,
            )
        return self._source

    def selected_zones(self, zones=None):
        result = self.zones if zones is None else sorted(set(zones))
        if not result or not set(result) <= set(self.zones):
            raise ValueError("Requested zones are not in this migration plan")
        return result

    def work_items(self, zones=None, years=None):
        """Validate the full selection before acquiring any year ownership."""
        items = []
        for zone in self.selected_zones(zones):
            available = self.zone(zone)["grid"]["years"]
            selected = available if years is None else sorted(set(years))
            if not selected or not set(selected) <= set(available):
                raise ValueError(
                    f"Zone {zone}: requested years are not in the migration plan"
                )
            items.extend((zone, year) for year in selected)
        return items

    def zone(self, zone):
        part = self.plan["partitions"][str(zone)]
        data, _ = self.state.get(part["path"])
        if data is None or digest(data) != part["sha256"]:
            raise ValueError(f"Zone {zone}: metadata checksum mismatch")
        return json.loads(data)

    def units(self, zone_doc, years=None):
        raw, _ = self.state.get(zone_doc["inventory"])
        if raw is None or digest(raw) != zone_doc["inventory_sha256"]:
            raise ValueError("Inventory checksum mismatch")
        if years is not None and not set(years) <= set(zone_doc["grid"]["years"]):
            raise ValueError("Requested years are not in the migration plan")
        table = pq.read_table(io.BytesIO(raw))
        for batch in table.to_batches(max_chunksize=4096):
            for row in zip(*(c.to_pylist() for c in batch.columns)):
                if years is None or row[0] in years:
                    yield tuple(row)

    def root_attrs(self):
        attrs = dict(self.plan["source_attrs"])
        attrs.update(
            {
                "zarr_conventions": [GEOEMB_CONVENTION],
                MIGRATION_ID: self.id,
                MASK_SOURCE: "source_nodata",
                "geoemb:landmask": False,
                "geoemb:build_version": self.request["versions"]["geotessera"],
                "geotessera:source_snapshot": self.plan["snapshot_id"],
                "geotessera:source_store": self.request["source"],
            }
        )
        return attrs

    def zone_attrs(self, doc):
        g = doc["grid"]
        x0, y0 = g["x"] - 5, g["y"] + 5
        attrs = _geo_convention_attrs(
            ["y", "x"],
            f"EPSG:{32600 + doc['zone']}",
            [x0, y0 - g["height"] * 10, x0 + g["width"] * 10, y0],
            [10, 0, x0, 0, -10, y0],
            [g["height"], g["width"]],
        )
        attrs.update(
            {
                MASK_SOURCE: "source_nodata",
                MIGRATION_ID: self.id,
                "geotessera:source_groups": {
                    s["name"]: s["attrs"] for s in doc["groups"]
                },
            }
        )
        return attrs

    def configs(self, doc):
        from zarr.codecs import BloscCodec

        specs = dict(doc["arrays"])
        specs.update(doc["static"])
        grid = doc["grid"]
        for n, size, dtype in (
            ("x", grid["width"], "<f8"),
            ("y", grid["height"], "<f8"),
            ("time", len(grid["years"]), "<i4"),
            ("band", 128, "<i4"),
        ):
            specs[n] = {
                "shape": [size],
                "dims": [n],
                "dtype": dtype,
                "fill": 0,
                "attrs": {},
            }
        for name, spec in specs.items():
            shape, ds = spec["shape"], spec["dims"]
            spatial = name in doc["arrays"]
            chunks = tuple(
                1
                if d == "time" and spatial
                else self.request["inner_chunk"]
                if d in ("y", "x") and spatial
                else s
                for d, s in zip(ds, shape)
            )
            shards = (
                tuple(
                    1 if d == "time" else self.shard_size if d in ("y", "x") else s
                    for d, s in zip(ds, shape)
                )
                if spatial
                else None
            )
            yield (
                name,
                {
                    "shape": tuple(shape),
                    "dtype": spec["dtype"],
                    "chunks": chunks,
                    "shards": shards,
                    "fill_value": source.fill_value(spec["fill"]),
                    "dimension_names": ds,
                    "attributes": spec["attrs"],
                    "compressors": BloscCodec(cname="zstd", clevel=3),
                },
            )

    @contextmanager
    def controller(self):
        with ExitStack() as stack:
            stack.enter_context(self.state.owner("controller"))
            for zone, year in self.work_items():
                stack.enter_context(self.state.owner(f"utm{zone:02d}-{year}"))
            yield

    @contextmanager
    def worker(self, zone, year):
        with self.state.owner(f"utm{zone:02d}-{year}"):
            controller, _ = self.state.get("owners/controller.json")
            if controller is not None and not owner_stopped(json.loads(controller)):
                raise BusyError("Initialization/finalization owns the store")
            if self.state.read("initialized.json") != {"migration_id": self.id}:
                raise ValueError("Initialization barrier does not match this migration")
            yield

    def initialize(self):
        with self.controller():
            try:
                root = self.store.open_group(mode="r+")
            except (FileNotFoundError, zarr.errors.GroupNotFoundError):
                root = zarr.open_group(
                    self.store.as_zarr_store(),
                    mode="w-",
                    zarr_format=3,
                    attributes=self.root_attrs(),
                    use_consolidated=False,
                )
            if dict(root.attrs) != self.root_attrs():
                raise ValueError(
                    "Destination root metadata conflicts with this migration"
                )
            for zone in self.zones:
                doc = self.zone(zone)
                name = f"utm{zone:02d}"
                try:
                    group = root[name]
                except KeyError:
                    group = root.create_group(name, attributes=self.zone_attrs(doc))
                if dict(group.attrs) != self.zone_attrs(doc):
                    raise ValueError(f"{name}: conflicting group metadata")
                for array_name, config in self.configs(doc):
                    try:
                        array = group[array_name]
                    except KeyError:
                        array = group.create_array(array_name, **config)
                    _check_array(array, config)
                    data = coordinate_data(doc, array_name)
                    if data is not None:
                        array[:] = data
                self.state.write(f"init/{name}.json", {"migration_id": self.id})
                event("zone_initialized", zone=zone)
            self.state.immutable(
                "initialized.json", json_bytes({"migration_id": self.id})
            )

    def validate_zone(self, doc, group, *, coordinates=False):
        if dict(group.attrs) != self.zone_attrs(doc):
            raise ValueError("Destination zone metadata changed")
        for name, config in self.configs(doc):
            _check_array(group[name], config)
            if coordinates:
                expected = coordinate_data(doc, name)
                if expected is not None and not np.array_equal(
                    group[name][:], expected, equal_nan=True
                ):
                    raise ValueError(f"{group.path}/{name}: coordinate values changed")

    def receipt_key(self, zone, unit):
        year, row, col = unit
        return f"commits/utm{zone:02d}/{year}/{row}-{col}.json"

    def object_key(self, doc, name, unit):
        year, row, col = unit
        ds = doc["arrays"][name]["dims"]
        indices = [
            {"time": doc["grid"]["years"].index(year), "y": row, "x": col}.get(d, 0)
            for d in ds
        ]
        return f"utm{doc['zone']:02d}/{name}/c/" + "/".join(map(str, indices))

    def object_info(self, key):

        if self._object_index is not None:
            return self._object_index.get(key)
        return self._object_head(key)

    def _object_head(self, key):
        from . import remote

        if not self.store.is_remote:
            try:
                stat = (Path(self.store.url) / key).stat()
            except FileNotFoundError:
                return None
            return {"size": stat.st_size, "identity": str(stat.st_mtime_ns)}
        fs = remote.get_fs(self.store.url, self.store.storage_options)
        url = self.store.join(key)
        fs.invalidate_cache(url)
        try:
            info = fs.info(url)
        except FileNotFoundError:
            return None
        return {
            "size": info["size"],
            "identity": str(info.get("ETag") or info.get("mtime") or ""),
        }

    @contextmanager
    def object_inventory(self, zone, year=None):
        """List a zone, or only each array's selected-year prefix for workers."""
        index = {}
        prefixes = [f"utm{zone:02d}/"]
        if year is not None:
            doc = self.zone(zone)
            time_index = doc["grid"]["years"].index(year)
            prefixes = [
                f"utm{zone:02d}/{name}/c/{time_index}/" for name in doc["arrays"]
            ]
        if self.store.is_remote:
            from urllib.parse import urlparse

            from botocore.exceptions import ClientError

            from ._migration_state import s3_client

            url = urlparse(self.store.url)
            base = url.path.strip("/")
            client = s3_client(self.store.storage_options)
            kwargs = {
                "Bucket": url.netloc,
            }
            if (self.store.storage_options or {}).get("requester_pays"):
                kwargs["RequestPayer"] = "requester"
            try:
                for prefix in prefixes:
                    kwargs["Prefix"] = f"{base}/{prefix}" if base else prefix
                    for page in client.get_paginator("list_objects_v2").paginate(
                        **kwargs
                    ):
                        for obj in page.get("Contents", []):
                            key = obj["Key"][len(base) + 1 :] if base else obj["Key"]
                            index[key] = {"size": obj["Size"], "identity": obj["ETag"]}
            except ClientError as e:
                if e.response["Error"]["Code"] not in ("AccessDenied", "403"):
                    raise
                index = None
        else:
            base = Path(self.store.url)
            for prefix in prefixes:
                for path in (base / prefix).rglob("*"):
                    if path.is_file():
                        stat = path.stat()
                        index[path.relative_to(base).as_posix()] = {
                            "size": stat.st_size,
                            "identity": str(stat.st_mtime_ns),
                        }
        self._object_index = index
        try:
            yield
        finally:
            self._object_index = None

    def statuses(self, doc, units, checksum=False):
        from concurrent.futures import ThreadPoolExecutor
        from itertools import islice

        units = iter(units)
        # Bounded submission avoids millions of outstanding Future objects.
        with ThreadPoolExecutor(max_workers=16) as pool:
            while batch := list(islice(units, 256)):
                statuses = pool.map(
                    lambda u: self.unit_status(doc, u, checksum=checksum), batch
                )
                yield from zip(batch, statuses)

    def unit_status(self, doc, unit, *, checksum=False):
        data, _ = self.state.get(self.receipt_key(doc["zone"], unit))
        if data is None:
            # Without a receipt the unit must be written anyway. A role that
            # cannot list may receive 403 for absent keys: don't turn that
            # ambiguity into an existence test or a false completion claim.
            present = self._object_index is not None and any(
                self.object_info(self.object_key(doc, n, unit)) is not None
                for n in doc["arrays"]
            )
            return "partial" if present else "missing"
        receipt = json.loads(data)
        if (
            receipt.get("migration_id") != self.id
            or receipt.get("unit") != list(unit)
            or receipt.get("zone") != doc["zone"]
        ):
            raise ValueError("Conflicting shard receipt")
        if set(receipt["objects"]) != set(doc["arrays"]):
            return "partial"
        for name, record in receipt["objects"].items():
            key = self.object_key(doc, name, unit)
            if record["key"] != key or self.object_info(key) != record["info"]:
                return "partial"
            if (
                checksum
                and record["info"] is not None
                and digest(self.store.read_bytes(key)) != record["sha256"]
            ):
                return "partial"
        return "complete"

    def scan(self, zones=None, years=None, *, checksum=False):
        root = self.store.open_group(mode="r")
        if dict(root.attrs) != self.root_attrs():
            raise ValueError("Destination root metadata changed")
        for zone in self.selected_zones(zones):
            doc = self.zone(zone)
            self.validate_zone(doc, root[f"utm{zone:02d}"], coordinates=True)
            with self.object_inventory(
                zone, years[0] if years is not None and len(years) == 1 else None
            ):
                for unit, status in self.statuses(
                    doc, self.units(doc, years), checksum=checksum
                ):
                    yield {
                        "zone": zone,
                        "year": unit[0],
                        "row": unit[1],
                        "col": unit[2],
                        "status": status,
                    }

    def transcode(
        self,
        zones=None,
        *,
        years=None,
        workers=1,
        spill_dir=None,
        max_shards=None,
        max_seconds=None,
        read_rows=256,
        io_concurrency=4,
        stop=None,
    ):
        if workers < 1 or read_rows < 1 or io_concurrency < 1:
            raise ValueError("Worker/read concurrency and read_rows must be positive")
        if (
            max_shards is not None
            and max_shards < 0
            or max_seconds is not None
            and max_seconds <= 0
        ):
            raise ValueError("Invalid shard/runtime limit")
        from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

        started, written = time.monotonic(), 0
        stop = stop or threading.Event()
        # Zarr settings multiply process-level parallelism: explicitly bound them.
        with zarr.config.set(
            {
                "async.concurrency": io_concurrency,
                "threading.max_workers": io_concurrency,
            }
        ):
            for zone, year in self.work_items(zones, years):
                with self.worker(zone, year):
                    doc = self.zone(zone)
                    group = self.store.open_group(mode="r", path=f"utm{zone:02d}")
                    self.validate_zone(doc, group, coordinates=True)
                    # Initialize the shared read-only session before threads start.
                    _ = self.source
                    with self.object_inventory(zone, year):
                        missing = [
                            u
                            for u, status in self.statuses(doc, self.units(doc, [year]))
                            if status != "complete"
                        ]
                    units = iter(missing)
                    year_written = 0
                    exhausted = False
                    pending = set()
                    with ThreadPoolExecutor(max_workers=workers) as pool:
                        while pending or not exhausted:
                            limit = stop.is_set() or (
                                max_seconds is not None
                                and time.monotonic() - started >= max_seconds
                            )
                            while (
                                not exhausted and not limit and len(pending) < workers
                            ):
                                if (
                                    max_shards is not None
                                    and written + len(pending) >= max_shards
                                ):
                                    limit = True
                                    break
                                try:
                                    unit = next(units)
                                except StopIteration:
                                    exhausted = True
                                    break
                                pending.add(
                                    pool.submit(
                                        self.write_unit, doc, unit, spill_dir, read_rows
                                    )
                                )
                            if pending:
                                done, pending = wait(
                                    pending, return_when=FIRST_COMPLETED
                                )
                                for future in done:
                                    future.result()
                                    written += 1
                                    year_written += 1
                            elif limit:
                                break
                    # Limits may have stopped just after the last missing unit.
                    complete = year_written == len(missing)
                    event(
                        "year_attempt_finished",
                        zone=zone,
                        year=year,
                        written=written,
                        complete=complete,
                    )
                    if not complete:
                        return RESUMABLE
        return 0

    def write_unit(self, doc, unit, spill_dir=None, read_rows=256):
        start = time.monotonic()
        receipt_key = self.receipt_key(doc["zone"], unit)
        # A partial receipt makes interruption visible without requiring
        # DeleteObject permission on the state bucket.
        self.state.write(
            receipt_key,
            {
                "migration_id": self.id,
                "zone": doc["zone"],
                "unit": list(unit),
                "objects": {},
            },
        )
        self.invalidate_completion()
        raw_store = self.store.as_zarr_store()
        if isinstance(raw_store, str):
            raw_store = zarr.storage.LocalStore(raw_store)
        audited = AuditStore(raw_store)
        group = zarr.open_group(
            audited, mode="r+", path=f"utm{doc['zone']:02d}", use_consolidated=False
        )
        objects = {}
        names = sorted(
            doc["arrays"], key=lambda n: (n == "embeddings", n != "scales", n)
        )
        for name in names:
            with self.assemble(doc, name, unit, spill_dir, read_rows) as values:
                array = group[name].with_config({"write_empty_chunks": False})
                array[unit_selection(doc, name, unit, self.shard_size)] = values
            key = self.object_key(doc, name, unit)
            info = None if key in audited.deleted else self.object_info(key)
            encoded_hash = audited.hashes.get(key)
            if (info is not None) != (encoded_hash is not None):
                raise RuntimeError(
                    f"Zarr did not publish the expected shard object {key}"
                )
            objects[name] = {"key": key, "info": info, "sha256": encoded_hash}
        self.state.write(
            receipt_key,
            {
                "migration_id": self.id,
                "zone": doc["zone"],
                "unit": list(unit),
                "objects": objects,
            },
        )
        event(
            "shard_committed",
            zone=doc["zone"],
            year=unit[0],
            row=unit[1],
            col=unit[2],
            bytes=sum(o["info"]["size"] for o in objects.values() if o["info"]),
            seconds=round(time.monotonic() - start, 3),
        )

    def invalidate_completion(self):
        """Replace final markers before the first write; never delete state."""
        with self._invalidation_lock:
            if self._completion_invalidated:
                return
            marker = {"migration_id": self.id, "invalidated": True}
            for key in ("verified.json", "complete.json", "publication.json"):
                if self.state.get(key)[0] is not None:
                    self.state.write(key, marker)
            self._completion_invalidated = True

    @contextmanager
    def assemble(self, doc, name, unit, spill_dir=None, read_rows=256):
        """Yield one output array shard, reading bounded strips from each source.

        Embedding vectors are indivisible: finite-scale zero vectors count
        as data. Ancillary overlap merges equal or sole non-fill values and
        rejects conflicting non-fill values. No geographic interpolation.
        """
        spec = doc["arrays"][name]
        ds = spec["dims"]
        shape = tuple(
            1 if d == "time" else self.shard_size if d in ("y", "x") else s
            for d, s in zip(ds, spec["shape"])
        )
        if spill_dir:
            Path(spill_dir).mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=spill_dir, prefix="transcode-") as tmp:
            values = (
                np.memmap(
                    Path(tmp) / "values", mode="w+", shape=shape, dtype=spec["dtype"]
                )
                if spill_dir
                else np.empty(shape, dtype=spec["dtype"])
            )
            values[:] = source.fill_value(spec["fill"])
            occupied_shape = (
                (self.shard_size, self.shard_size) if name == "embeddings" else shape
            )
            occupied = (
                np.memmap(
                    Path(tmp) / "occupied", mode="w+", shape=occupied_shape, dtype=bool
                )
                if spill_dir
                else np.zeros(occupied_shape, dtype=bool)
            )
            try:
                self._assemble_into(doc, name, unit, values, occupied, read_rows)
                yield values
            finally:
                # Drop mappings before TemporaryDirectory cleanup on Windows.
                for a in (values, occupied):
                    if isinstance(a, np.memmap):
                        a._mmap.close()

    def _hemi_wins(self, group_name):
        """Under seam_conflict=north/south, whether this hemisphere group's
        values win at the equatorial overlap (robust to N/S write order)."""
        if self.seam_conflict == "north":
            return str(group_name).endswith("N")
        if self.seam_conflict == "south":
            return str(group_name).endswith("S")
        return True

    def _assemble_into(self, doc, name, unit, values, occupied, read_rows):
        year, row, col = unit
        r0, c0 = row * self.shard_size, col * self.shard_size
        output_dims = doc["arrays"][name]["dims"]
        for g in doc["groups"]:
            if name not in g["arrays"] or year not in g["years"]:
                continue
            spec = g["arrays"][name]
            ds = spec["dims"]
            top, left = max(r0, g["row_offset"]), max(c0, g["col_offset"])
            bottom = min(r0 + self.shard_size, g["row_offset"] + g["height"])
            right = min(c0 + self.shard_size, g["col_offset"] + g["width"])
            if top >= bottom or left >= right:
                continue
            t = g["years"].index(year)
            for start in range(top, bottom, read_rows):
                end = min(bottom, start + read_rows)
                slices = {
                    "time": slice(t, t + 1),
                    "y": slice(start - g["row_offset"], end - g["row_offset"]),
                    "x": slice(left - g["col_offset"], right - g["col_offset"]),
                }
                data = np.asarray(
                    self.source[g["name"]][name][
                        tuple(slices.get(d, slice(None)) for d in ds)
                    ]
                )
                data = data.transpose(tuple(ds.index(d) for d in output_dims))
                out_slices = {
                    "time": slice(0, 1),
                    "y": slice(start - r0, end - r0),
                    "x": slice(left - c0, right - c0),
                }
                out = tuple(out_slices.get(d, slice(None)) for d in output_dims)
                view = values[out]
                if name == "embeddings":
                    scale_dims = g["arrays"]["scales"]["dims"]
                    scales = np.asarray(
                        self.source[g["name"]]["scales"][
                            tuple(slices[d] for d in scale_dims)
                        ]
                    )
                    scales = scales.transpose(
                        tuple(scale_dims.index(d) for d in ("time", "y", "x"))
                    )[0]
                    nonfill = np.isfinite(scales) | np.any(data[0] != 0, axis=0)
                    seen = occupied[out_slices["y"], out_slices["x"]]
                    conflict = seen & nonfill & np.any(view[0] != data[0], axis=0)
                    if conflict.any():
                        if self.seam_conflict == "error":
                            raise ValueError(
                                f"Conflicting hemisphere embeddings: zone={doc['zone']} unit={unit}"
                            )
                        event(
                            "seam_conflict",
                            zone=doc["zone"],
                            year=year,
                            row=row,
                            col=col,
                            array=name,
                            group=g["name"],
                            pixels=int(conflict.sum()),
                            resolution=self.seam_conflict,
                        )
                        if not self._hemi_wins(g["name"]):
                            nonfill = nonfill & ~seen
                    np.copyto(view[0], data[0], where=nonfill[None])
                else:
                    fill = source.fill_value(spec["fill"])
                    nonfill = (
                        ~np.isnan(data)
                        if isinstance(fill, float) and math.isnan(fill)
                        else data != fill
                    )
                    seen = occupied[out]
                    equal = (
                        (view == data) | (np.isnan(view) & np.isnan(data))
                        if data.dtype.kind == "f"
                        else view == data
                    )
                    conflict = seen & nonfill & ~equal
                    if np.any(conflict):
                        if self.seam_conflict == "error":
                            raise ValueError(
                                f"Conflicting hemisphere {name}: zone={doc['zone']} unit={unit}"
                            )
                        event(
                            "seam_conflict",
                            zone=doc["zone"],
                            year=year,
                            row=row,
                            col=col,
                            array=name,
                            group=g["name"],
                            pixels=int(np.sum(conflict)),
                            resolution=self.seam_conflict,
                        )
                        if not self._hemi_wins(g["name"]):
                            nonfill = nonfill & ~seen
                    np.copyto(view, data, where=nonfill)
                seen |= nonfill

    def verify(
        self,
        zones=None,
        years=None,
        *,
        samples=8,
        seed=0,
        full=False,
        spill_dir=None,
        read_rows=256,
    ):
        """Compare all values in selected shards; stratify by zone AND year.

        Boundary shards and the equatorial overlap are always included. A
        full run also verifies encoded receipt checksums for every unit.
        """
        if samples < 1:
            raise ValueError("samples must be positive")
        rng = np.random.default_rng(seed)
        checked = 0
        for zone in self.selected_zones(zones):
            doc = self.zone(zone)
            group = self.store.open_group(mode="r", path=f"utm{zone:02d}")
            self.validate_zone(doc, group, coordinates=True)
            by_year = {}
            for unit in self.units(doc, years):
                by_year.setdefault(unit[0], []).append(unit)
            for year, units in by_year.items():
                if full:
                    chosen = units
                else:
                    indices = set(
                        rng.choice(
                            len(units), min(samples, len(units)), replace=False
                        ).tolist()
                    ) | {0, len(units) - 1}
                    equator_row = int(doc["grid"]["y"] / 10) // self.shard_size
                    indices.update(
                        i for i, u in enumerate(units) if abs(u[1] - equator_row) <= 1
                    )
                    chosen = [units[i] for i in sorted(indices)]
                for unit in chosen:
                    if self.unit_status(doc, unit, checksum=full) != "complete":
                        raise ValueError(
                            f"Cannot verify incomplete shard {zone}/{unit}"
                        )
                    for name in doc["arrays"]:
                        with self.assemble(
                            doc, name, unit, spill_dir, read_rows
                        ) as expected:
                            actual = group[name][
                                unit_selection(doc, name, unit, self.shard_size)
                            ]
                            if not np.array_equal(expected, actual, equal_nan=True):
                                raise ValueError(
                                    f"Source mismatch: zone={zone} unit={unit} array={name}"
                                )
                    checked += 1
                # Also check implicit-fill corners if outside the sparse inventory.
                present = set(units)
                corners = {
                    (year, r, c)
                    for r in (0, doc["grid"]["height"] // self.shard_size - 1)
                    for c in (0, doc["grid"]["width"] // self.shard_size - 1)
                }
                for unit in corners - present:
                    for name in doc["arrays"]:
                        if (
                            self.object_info(self.object_key(doc, name, unit))
                            is not None
                        ):
                            raise ValueError(
                                f"Unexpected object in source-fill-only corner {zone}/{unit}/{name}"
                            )
                event("year_verified", zone=zone, year=year, shards=len(chosen))
        return {
            "migration_id": self.id,
            "checked_shards": checked,
            "full": full,
            "seed": seed,
        }

    def finalize(self, *, samples=8, full=False, spill_dir=None):
        with self.controller():
            if self.state.read("initialized.json") != {"migration_id": self.id}:
                raise ValueError("Migration is not initialized")
            counts = {"complete": 0, "partial": 0, "missing": 0}
            for row in self.scan():
                counts[row["status"]] += 1
            if counts["missing"] or counts["partial"]:
                raise ValueError(f"Migration incomplete: {counts}")
            report = self.verify(samples=samples, full=full, spill_dir=spill_dir)
            self.state.write("verified.json", report)
            # The controller holds every zone/year while touching the shared root.
            zarr.consolidate_metadata(self.store.as_zarr_store())
            self.state.write(
                "complete.json",
                {
                    **report,
                    "counts": counts,
                    "source_snapshot": self.plan["snapshot_id"],
                },
            )
            event("migration_complete", **counts)
            return report

    def verify_public(self, public_url):
        """Read metadata and one small patch per zone/year through the reader API."""
        from . import GeoTesseraZarr

        reader = GeoTesseraZarr(public_url)
        checked = 0
        for zone in self.zones:
            doc = self.zone(zone)
            ds = reader.open_zone(zone=zone)
            if ds.attrs.get(MIGRATION_ID) != self.id:
                raise ValueError(f"Public metadata is stale for zone {zone}")
            group = self.store.open_group(mode="r", path=f"utm{zone:02d}")
            seen = set()
            for year, row, col in self.units(doc):
                if year in seen:
                    continue
                seen.add(year)
                window = {
                    "time": slice(
                        doc["grid"]["years"].index(year),
                        doc["grid"]["years"].index(year) + 1,
                    ),
                    "y": slice(
                        row * self.shard_size,
                        row * self.shard_size + self.request["inner_chunk"],
                    ),
                    "x": slice(
                        col * self.shard_size,
                        col * self.shard_size + self.request["inner_chunk"],
                    ),
                }
                for name, spec in doc["arrays"].items():
                    expected = group[name][
                        tuple(window.get(d, slice(None)) for d in spec["dims"])
                    ]
                    actual = ds[name].isel(window).values
                    if not np.array_equal(expected, actual, equal_nan=True):
                        raise ValueError(
                            f"Public data mismatch: zone={zone} year={year} array={name}"
                        )
                checked += 1
            ds.close()
        return {
            "migration_id": self.id,
            "public_url": public_url,
            "checked_public_patches": checked,
        }


class AuditStore(zarr.storage.WrapperStore):
    """Hash encoded objects while Zarr already holds their upload buffers."""

    def __init__(self, store):
        super().__init__(store)
        self.hashes = {}
        self.deleted = set()

    async def set(self, key, value):
        # Every worker thread's store traffic is funnelled onto Zarr's single
        # global event loop thread, so anything synchronous here stalls every
        # other worker's reads and writes as well. sha256 over a 2 GiB encoded
        # shard is ~10s of CPU: hand it to a worker thread and overlap it with
        # the upload instead of running it, inline, ahead of the upload.
        # Both sides only read the buffer, and Zarr does not reuse it once it
        # has been handed to `set`.
        checksum = asyncio.ensure_future(
            asyncio.to_thread(digest, memoryview(value.as_numpy_array()))
        )
        try:
            await self._store.set(key, value)
        except BaseException:
            checksum.cancel()
            raise
        self.hashes[key] = await checksum
        self.deleted.discard(key)

    async def delete(self, key):
        # Zarr asks to delete an all-fill shard even when it was never stored.
        # Avoid requiring DeleteObject for that normal fresh-store case, but
        # retain deletion when an old object really exists.
        if await self._store.exists(key):
            await self._store.delete(key)
        self.hashes.pop(key, None)
        self.deleted.add(key)


def coordinate_data(doc, name):
    g = doc["grid"]
    if name == "x":
        return g["x"] + np.arange(g["width"], dtype=np.float64) * 10
    if name == "y":
        return g["y"] - np.arange(g["height"], dtype=np.float64) * 10
    if name == "time":
        return np.array(g["years"], dtype=np.int32)
    if name == "band":
        return np.arange(128, dtype=np.int32)
    if name in doc["static"]:
        return np.array(doc["static"][name]["data"], dtype=doc["static"][name]["dtype"])
    return None


def _check_array(array, config):
    reference = zarr.create_array(store=zarr.storage.MemoryStore(), **config)
    if array.metadata.to_dict() != reference.metadata.to_dict():
        raise ValueError(
            f"{array.path}: destination array metadata differs from the plan"
        )


def unit_selection(doc, name, unit, shard_size):
    year, row, col = unit
    t = doc["grid"]["years"].index(year)
    return tuple(
        {
            "time": slice(t, t + 1),
            "y": slice(row * shard_size, (row + 1) * shard_size),
            "x": slice(col * shard_size, (col + 1) * shard_size),
        }.get(d, slice(None))
        for d in doc["arrays"][name]["dims"]
    )


@contextmanager
def termination_event():
    stop = threading.Event()
    previous = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    try:
        yield stop
    finally:
        signal.signal(signal.SIGTERM, previous)
