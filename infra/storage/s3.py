"""Resumable, conflict-rejecting dataset snapshots in S3."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

from botocore.exceptions import ClientError


INDEX = "_TRANSFER.json"


def parse_uri(uri: str) -> tuple[str, str]:
    parsed = urlsplit(uri)
    prefix = parsed.path.strip("/")
    if parsed.scheme != "s3" or not parsed.netloc or not prefix or parsed.query or parsed.fragment:
        raise ValueError("Expected s3://bucket/run-prefix (not a bucket root)")
    safe_relative(prefix)
    return parsed.netloc, prefix


def safe_relative(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
        raise ValueError(f"Unsafe dataset path: {name}")
    if path.as_posix() != name:
        raise ValueError(f"Non-canonical dataset path: {name}")
    return path


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def remote_digest(client: Any, bucket: str, key: str) -> str | None:
    try:
        response = client.get_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        if exc.response["Error"]["Code"] in {"NoSuchKey", "404", "NotFound"}:
            return None
        raise
    checksum = hashlib.sha256()
    body = response["Body"]
    try:
        for block in iter(lambda: body.read(1024 * 1024), b""):
            checksum.update(block)
    finally:
        body.close()
    return checksum.hexdigest()


def push(client: Any, source: Path, uri: str) -> int:
    """Upload an immutable dataset snapshot, publishing frame markers last."""
    bucket, prefix = parse_uri(uri)
    source = source.resolve()
    if not (source / "manifest.json").is_file() or not (source / "rig.yaml").is_file():
        raise ValueError("Dataset must include manifest.json and rig.yaml")
    if not (source / "calib").is_dir() or not (source / "frames").is_dir():
        raise ValueError("Dataset must include calib/ and frames/")
    entries: dict[str, dict[str, Any]] = {}
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Dataset cannot contain symlinks: {path}")
        if not path.is_file():
            continue
        name = path.relative_to(source).as_posix()
        if name == INDEX or path.name == "_DONE":
            continue
        safe_relative(name)
        entries[name] = {"sha256": digest(path), "size": path.stat().st_size}
    frames = sorted({PurePosixPath(name).parts[1] for name in entries
                     if name.startswith("frames/") and len(PurePosixPath(name).parts) > 2})
    if not frames:
        raise ValueError("Dataset contains no frames")
    for frame in frames:
        if not frame.isdecimal() or f"frames/{frame}/meta.json" not in entries:
            raise ValueError(f"Frame {frame} requires a numeric directory and meta.json")
    payload = json.dumps({"version": 1, "files": entries, "frames": frames}, sort_keys=True).encode()
    index_hash = hashlib.sha256(payload).hexdigest()
    remote_index = remote_digest(client, bucket, f"{prefix}/{INDEX}")
    if remote_index is not None and remote_index != index_hash:
        raise ValueError("Remote snapshot differs; use a new S3 run prefix")
    pending = []
    for name, entry in entries.items():
        remote = remote_digest(client, bucket, f"{prefix}/{name}")
        if remote is not None and remote != entry["sha256"]:
            raise ValueError(f"Conflicting remote file: {name}; use a new S3 run prefix")
        if remote is None:
            pending.append(name)
    for name in pending:
        if digest(source / name) != entries[name]["sha256"]:
            raise ValueError(f"Source changed during transfer: {name}")
        client.upload_file(str(source / name), bucket, f"{prefix}/{name}")
        if remote_digest(client, bucket, f"{prefix}/{name}") != entries[name]["sha256"]:
            raise ValueError(f"Uploaded checksum mismatch: {name}")
    client.put_object(Bucket=bucket, Key=f"{prefix}/{INDEX}", Body=payload)
    for frame in frames:
        client.put_object(Bucket=bucket, Key=f"{prefix}/frames/{frame}/_DONE", Body=b"")
    return len(pending)


def pull(client: Any, uri: str, destination: Path, frames: tuple[int, int] | None = None) -> int:
    """Download verified files atomically; frame selection is a half-open index range."""
    bucket, prefix = parse_uri(uri)
    response = client.get_object(Bucket=bucket, Key=f"{prefix}/{INDEX}")
    body = response["Body"]
    try:
        index = json.loads(body.read())
    finally:
        body.close()
    if index.get("version") != 1 or not isinstance(index.get("files"), dict):
        raise ValueError("Unsupported transfer index")
    entries = index["files"]
    selected: dict[str, dict[str, Any]] = {}
    selected_frames = set()
    for name, entry in entries.items():
        relative = safe_relative(name)
        if not isinstance(entry, dict) or not isinstance(entry.get("sha256"), str):
            raise ValueError(f"Invalid checksum entry: {name}")
        if len(entry["sha256"]) != 64 or any(char not in "0123456789abcdef" for char in entry["sha256"]):
            raise ValueError(f"Invalid checksum: {name}")
        if relative.parts[0] == "frames":
            if len(relative.parts) < 3 or not relative.parts[1].isdecimal():
                raise ValueError(f"Invalid frame path: {name}")
            frame = relative.parts[1]
            if frames is not None and not frames[0] <= int(frame) < frames[1]:
                continue
            selected_frames.add(frame)
        selected[name] = entry
    if not {"manifest.json", "rig.yaml"}.issubset(selected):
        raise ValueError("Transfer index is missing dataset headers")
    if frames is not None and not selected_frames:
        raise ValueError("No frames match the requested range")
    for frame in selected_frames:
        if remote_digest(client, bucket, f"{prefix}/frames/{frame}/_DONE") is None:
            raise ValueError(f"Frame {frame} is incomplete in S3; retry after upload completes")
    destination = destination.resolve()
    for frame in selected_frames:
        marker = destination / "frames" / frame / "_DONE"
        if marker.is_symlink() or marker.resolve() != marker:
            raise ValueError(f"Unsafe completion marker: {marker}")
    pending = []
    for name, entry in selected.items():
        target = destination / name
        if target.resolve() != target:
            raise ValueError(f"Destination contains a symlink: {target}")
        if target.exists():
            if not target.is_file() or digest(target) != entry["sha256"]:
                raise ValueError(f"Conflicting local file: {target}")
        else:
            pending.append(name)
    for name in pending:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".synthrig-part")
        if temporary.is_symlink():
            raise ValueError(f"Unsafe temporary path: {temporary}")
        try:
            client.download_file(bucket, f"{prefix}/{name}", str(temporary))
            if digest(temporary) != selected[name]["sha256"]:
                raise ValueError(f"Downloaded checksum mismatch: {name}")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    for frame in selected_frames:
        (destination / "frames" / frame / "_DONE").touch()
    return len(pending)