import io
import json
from pathlib import Path

import pytest
from botocore.exceptions import ClientError

from infra.storage.s3 import INDEX, parse_uri, pull, push


class MemoryS3:
    def __init__(self):
        self.objects = {}
        self.writes = []

    def get_object(self, *, Bucket, Key):
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return {"Body": io.BytesIO(self.objects[Key])}

    def put_object(self, *, Bucket, Key, Body):
        self.objects[Key] = Body
        self.writes.append(Key)

    def upload_file(self, filename, bucket, key):
        self.put_object(Bucket=bucket, Key=key, Body=Path(filename).read_bytes())

    def download_file(self, bucket, key, filename):
        Path(filename).write_bytes(self.objects[key])


@pytest.fixture
def dataset(tmp_path):
    source = tmp_path / "source"
    for name in ["manifest.json", "rig.yaml", "calib/camera.yml",
                 "frames/000000/meta.json", "frames/000000/cam0/image.png",
                 "frames/000001/meta.json", "frames/000001/cam0/image.png"]:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    return source


def test_roundtrip_resume_and_subset(dataset, tmp_path):
    client = MemoryS3()
    uri = "s3://bucket/datasets/run"
    assert push(client, dataset, uri) == 7
    assert all(name.endswith("_DONE") for name in client.writes[-2:])
    assert push(client, dataset, uri) == 0
    destination = tmp_path / "download"
    assert pull(client, uri, destination, (0, 1)) == 5
    assert pull(client, uri, destination, (0, 1)) == 0
    assert (destination / "manifest.json").is_file()
    assert (destination / "frames/000000/_DONE").is_file()
    assert not (destination / "frames/000001").exists()


def test_conflicting_push_does_not_write(dataset):
    client = MemoryS3()
    push(client, dataset, "s3://bucket/run")
    before = len(client.writes)
    (dataset / "rig.yaml").write_text("changed")
    with pytest.raises(ValueError, match="differs"):
        push(client, dataset, "s3://bucket/run")
    assert len(client.writes) == before


def test_interrupted_push_resumes(dataset):
    client = MemoryS3()
    client.objects["run/manifest.json"] = (dataset / "manifest.json").read_bytes()
    assert push(client, dataset, "s3://bucket/run") == 6


def test_download_rejects_corruption_and_incomplete_frame(dataset, tmp_path):
    client = MemoryS3()
    push(client, dataset, "s3://bucket/run")
    del client.objects["run/frames/000000/_DONE"]
    with pytest.raises(ValueError, match="incomplete"):
        pull(client, "s3://bucket/run", tmp_path / "download")
    client.objects["run/frames/000000/_DONE"] = b""
    client.objects["run/frames/000000/meta.json"] = b"corrupt"
    with pytest.raises(ValueError, match="checksum mismatch"):
        pull(client, "s3://bucket/run", tmp_path / "download")
    assert not (tmp_path / "download/frames/000000/_DONE").exists()


def test_download_rejects_traversal(dataset, tmp_path):
    client = MemoryS3()
    push(client, dataset, "s3://bucket/run")
    index = json.loads(client.objects[f"run/{INDEX}"])
    index["files"]["../escape"] = {"sha256": "0" * 64}
    client.objects[f"run/{INDEX}"] = json.dumps(index).encode()
    with pytest.raises(ValueError, match="Unsafe"):
        pull(client, "s3://bucket/run", tmp_path / "download")


@pytest.mark.parametrize("uri", ["s3://bucket", "https://bucket/run", "s3://bucket/../run"])
def test_invalid_uri(uri):
    with pytest.raises(ValueError):
        parse_uri(uri)