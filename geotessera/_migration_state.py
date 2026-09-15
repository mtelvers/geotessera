"""Atomic local/S3 migration state and exclusive, recoverable ownership.

An S3 owner is never stolen on a timer: its ECS task must be STOPPED.
Local owners can be recovered only on the same host, after their PID exits.
The state prefix is separate from the data prefix and may share its bucket.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import socket
import tempfile
import urllib.request
import uuid
from pathlib import Path
from urllib.parse import unquote, urlparse


class BusyError(RuntimeError):
    """Another process owns this operation; retry later."""


def json_bytes(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_url(url: str) -> str:
    parsed = urlparse(str(url))
    if parsed.scheme in ("", "file"):
        return str(
            Path(unquote(parsed.path) if parsed.scheme else url).expanduser().resolve()
        )
    if parsed.scheme != "s3" or not parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError(
            "Migration locations must be local paths or s3://bucket/prefix URLs"
        )
    return str(url).rstrip("/")


def s3_client(options=None):
    """Translate the CLI's existing fsspec options to an independent SDK client."""
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    options = options or {}
    config = dict(options.get("config_kwargs", {}))
    config.setdefault("retries", {"mode": "adaptive", "max_attempts": 10})
    # One State client is shared by every worker thread and by the 16-way scan
    # pool, but botocore pools only 10 connections per client: the surplus has
    # its connection discarded and reopened, paying a TLS handshake per
    # request. Same default and environment variable as the fsspec pool in
    # remote.py, so both are raised together.
    config.setdefault(
        "max_pool_connections",
        int(os.environ.get("GEOTESSERA_MAX_POOL_CONNECTIONS", "32")),
    )
    if options.get("anon"):
        config["signature_version"] = UNSIGNED
    kwargs = dict(options.get("client_kwargs", {}))
    if options.get("endpoint_url"):
        kwargs["endpoint_url"] = options["endpoint_url"]
    for key, sdk in (
        ("key", "aws_access_key_id"),
        ("secret", "aws_secret_access_key"),
        ("token", "aws_session_token"),
    ):
        if options.get(key):
            kwargs[sdk] = options[key]
    return boto3.Session(profile_name=options.get("profile")).client(
        "s3", config=Config(**config), **kwargs
    )


class State:
    """Small durable objects. Missing means 404, never access denied."""

    def __init__(self, url, options=None):
        self.url = canonical_url(url)
        self.options = options or {}
        p = urlparse(self.url)
        self.bucket = p.netloc if p.scheme == "s3" else None
        self.prefix = p.path.strip("/")
        self.client = s3_client(options) if self.bucket else None

    def _key(self, key):
        if (
            not key
            or key.startswith("/")
            or any(p in ("", ".", "..") for p in key.split("/"))
        ):
            raise ValueError(f"Invalid state key: {key!r}")
        return f"{self.prefix}/{key}" if self.prefix else key

    def get(self, key):
        """Return (bytes, opaque compare-and-swap token), or (None, None)."""
        self._key(key)
        if not self.bucket:
            try:
                data = (Path(self.url) / key).read_bytes()
            except FileNotFoundError:
                return None, None
            return data, digest(data)
        from botocore.exceptions import ClientError

        try:
            obj = self.client.get_object(Bucket=self.bucket, Key=self._key(key))
        except ClientError as e:
            if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
                return None, None
            raise
        with obj["Body"] as body:
            return body.read(), obj["ETag"]

    def read(self, key):
        data, _ = self.get(key)
        if data is None:
            raise FileNotFoundError(f"{self.url}/{key}")
        return json.loads(data)

    def put(self, key, data: bytes, *, absent=False, match=None):
        """Atomic publish; conditional conflicts raise BusyError."""
        self._key(key)
        if self.bucket:
            from botocore.exceptions import ClientError

            kwargs = dict(self.options.get("s3_additional_kwargs", {}))
            if absent:
                kwargs["IfNoneMatch"] = "*"
            if match is not None:
                kwargs["IfMatch"] = match
            try:
                self.client.put_object(
                    Bucket=self.bucket,
                    Key=self._key(key),
                    Body=data,
                    ChecksumAlgorithm="SHA256",
                    **kwargs,
                )
            except ClientError as e:
                if e.response["Error"]["Code"] in (
                    "PreconditionFailed",
                    "ConditionalRequestConflict",
                    "412",
                    "409",
                ):
                    raise BusyError(f"State changed concurrently: {key}") from e
                raise
            return
        path = Path(self.url) / key
        path.parent.mkdir(parents=True, exist_ok=True)
        # Serialize local CAS, including publication/deletion, across processes.
        with self._local_guard(path):
            old, token = self.get(key)
            if (absent and old is not None) or (match is not None and token != match):
                raise BusyError(f"State changed concurrently: {key}")
            fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".publish-")
            try:
                with os.fdopen(fd, "wb") as f:
                    f.write(data)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, path)
            finally:
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(tmp)

    @contextlib.contextmanager
    def _local_guard(self, path):
        # OS locks disappear on SIGKILL. The small guard file is persistent.
        with open(str(path) + ".guard", "a+b") as f:
            if os.name == "nt":
                import msvcrt

                f.seek(0)
                f.write(b"\0")
                f.flush()
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.flock(f, fcntl.LOCK_EX)
            try:
                yield
            finally:
                if os.name == "nt":
                    f.seek(0)
                    msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(f, fcntl.LOCK_UN)

    def write(self, key, value, **kwargs):
        self.put(key, json_bytes(value), **kwargs)

    def immutable(self, key, data):
        """Publish once, allowing an identical retry."""
        try:
            self.put(key, data, absent=True)
        except BusyError:
            if self.get(key)[0] != data:
                raise ValueError(
                    f"Conflicting immutable state: {self.url}/{key}"
                ) from None

    def delete(self, key, *, match=None):
        self._key(key)
        if self.bucket:
            from botocore.exceptions import ClientError

            kwargs = {"IfMatch": match} if match is not None else {}
            try:
                self.client.delete_object(
                    Bucket=self.bucket, Key=self._key(key), **kwargs
                )
            except ClientError as e:
                if e.response["Error"]["Code"] in ("PreconditionFailed", "412"):
                    raise BusyError(f"State changed concurrently: {key}") from e
                raise
        else:
            path = Path(self.url) / key
            path.parent.mkdir(parents=True, exist_ok=True)
            with self._local_guard(path):
                if match is not None and self.get(key)[1] != match:
                    raise BusyError(f"State changed concurrently: {key}")
                with contextlib.suppress(FileNotFoundError):
                    path.unlink()

    @contextlib.contextmanager
    def owner(self, name):
        key = f"owners/{name}.json"
        identity = owner_identity()
        data = json_bytes(identity)
        for _ in range(3):
            previous, token = self.get(key)
            if previous is not None and not owner_stopped(json.loads(previous)):
                raise BusyError(
                    f"Active or unverified owner of {name}: {previous.decode()}"
                )
            try:
                self.put(key, data, absent=previous is None, match=token)
                break
            except BusyError:
                continue
        else:
            raise BusyError(f"Could not acquire {name}")
        token = self.get(key)[1]
        try:
            yield identity
        finally:
            self.put(
                key,
                json_bytes({**identity, "released": True}),
                match=token,
            )


def owner_identity():
    identity = {
        "id": uuid.uuid4().hex,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
    }
    if os.environ.get("AWS_BATCH_JOB_ID"):
        identity["job_id"] = os.environ["AWS_BATCH_JOB_ID"]
        identity["attempt"] = os.environ.get("AWS_BATCH_JOB_ATTEMPT", "1")
        metadata = os.environ.get("ECS_CONTAINER_METADATA_URI_V4")
        if not metadata:
            raise RuntimeError("Batch ownership requires ECS_CONTAINER_METADATA_URI_V4")
        with urllib.request.urlopen(metadata + "/task", timeout=5) as response:
            task = json.load(response)
        identity.update(task_arn=task["TaskARN"], cluster=task["Cluster"])
    return identity


def owner_stopped(owner):
    if owner.get("released") is True:
        return True
    if owner.get("task_arn"):
        import boto3

        region = owner["task_arn"].split(":")[3]
        result = boto3.client("ecs", region_name=region).describe_tasks(
            cluster=owner["cluster"], tasks=[owner["task_arn"]]
        )
        if result.get("tasks"):
            return all(t["lastStatus"] == "STOPPED" for t in result["tasks"])
        # ECS stops retaining old task descriptions. A terminal Batch job is
        # an independent positive confirmation; an absent task alone is not.
        if owner.get("job_id"):
            jobs = boto3.client("batch", region_name=region).describe_jobs(
                jobs=[owner["job_id"]]
            )["jobs"]
            return bool(jobs) and jobs[0]["status"] in ("SUCCEEDED", "FAILED")
        return False
    if owner.get("hostname") != socket.gethostname():
        return False
    try:
        os.kill(int(owner["pid"]), 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False
