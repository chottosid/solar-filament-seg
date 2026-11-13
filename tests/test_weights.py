"""Weight download: extraction, checksum verification and the unpublished-release message (offline, file:// URLs)."""

import hashlib
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from filament.weights import download, ensure, problems, read_manifest  # noqa: E402

FILES = {"a.pt": b"detector", "b.pt": b"refiner"}
MANIFEST = {n: hashlib.sha256(b).hexdigest() for n, b in FILES.items()}


def _zip(tmp: Path, files: dict) -> str:
    path = tmp / "w.zip"
    with zipfile.ZipFile(path, "w") as z:
        for n, b in files.items():
            z.writestr(f"solar-filament-seg-weights/{n}", b)  # nested folder, as in the release zip
        z.writestr("README.txt", b"ignored")
    return path.as_uri()


def test_download_extracts_flat_and_verifies():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        download(tmp / "out", _zip(tmp, FILES), MANIFEST)
        assert sorted(p.name for p in (tmp / "out").iterdir()) == ["a.pt", "b.pt"] and not problems(tmp / "out", MANIFEST)
        try:
            download(tmp / "bad", _zip(tmp, {**FILES, "b.pt": b"tampered"}), MANIFEST)
        except SystemExit as e:
            assert "b.pt" in str(e)
        else:
            raise AssertionError("accepted a corrupted file")
        try:
            download(tmp / "none", (tmp / "missing.zip").as_uri(), MANIFEST)
        except SystemExit as e:
            assert "could not download" in str(e)
        else:
            raise AssertionError("missing archive not reported")


def test_ensure_skips_when_present_and_manifest_is_complete():
    with tempfile.TemporaryDirectory() as t:
        for n, b in FILES.items():
            (Path(t) / n).write_bytes(b)
        ensure(Path(t), tuple(FILES), url="file:///nonexistent")  # nothing missing -> no download attempted
    manifest = read_manifest()
    assert len(manifest) == 17 and all(len(h) == 64 for h in manifest.values())
    assert {f"refiner_extent_f{k}.pt" for k in range(5)} <= set(manifest)


if __name__ == "__main__":
    test_download_extracts_flat_and_verifies()
    test_ensure_skips_when_present_and_manifest_is_complete()
    print("ok")
