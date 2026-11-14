"""Download and verify the 17 trained weight files.

    python -m filament.weights --dest weights                 # download if needed, then verify
    python -m filament.weights --dest weights --verify-only
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

# The one place to change once the release exists.
WEIGHTS_URL = "https://github.com/chottosid/solar-filament-seg/releases/download/v1.0/solar-filament-seg-weights.zip"
MANIFEST = Path(__file__).resolve().parents[1] / "data" / "weights.sha256"


def read_manifest(path: Path = MANIFEST) -> dict[str, str]:
    """{file name: sha256} from a `sha256sum`-style manifest."""
    return {name: digest for digest, name in (line.split() for line in path.read_text().splitlines() if line.strip())}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def problems(dest: Path, manifest: dict[str, str]) -> list[str]:
    """Missing or corrupted files under dest, as readable strings (empty = all good)."""
    out = []
    for name, digest in manifest.items():
        f = dest / name
        if not f.exists():
            out.append(f"{name}: missing")
        elif sha256(f) != digest:
            out.append(f"{name}: checksum differs from data/weights.sha256")
    return out


def download(dest: Path, url: str = WEIGHTS_URL, manifest: dict[str, str] | None = None) -> None:
    """Download the zip, extract the manifest's files flat into dest, verify them."""
    manifest = manifest or read_manifest()
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "weights.zip"
        try:
            with urllib.request.urlopen(url) as src, archive.open("wb") as out:
                shutil.copyfileobj(src, out)
        except urllib.error.HTTPError as e:
            hint = " (the release has probably not been published yet)" if e.code == 404 else ""
            raise SystemExit(f"could not download the weights from {url}: HTTP {e.code}{hint}") from e
        except urllib.error.URLError as e:
            raise SystemExit(f"could not download the weights from {url}: {e.reason}") from e
        with zipfile.ZipFile(archive) as z:
            for member in z.namelist():
                name = Path(member).name
                if name in manifest:  # flat extraction; anything else in the zip is ignored
                    with z.open(member) as src, (dest / name).open("wb") as out:
                        shutil.copyfileobj(src, out)
    bad = problems(dest, manifest)
    if bad:
        raise SystemExit("downloaded weights failed verification:\n  " + "\n  ".join(bad))


def ensure(dest: Path, names: tuple[str, ...], url: str = WEIGHTS_URL) -> None:
    """Make sure the named weight files exist in dest, downloading the whole set if any is missing."""
    if all((dest / n).exists() for n in names):
        return
    print(f"weights missing in {dest}; downloading from {url}", flush=True)
    download(dest, url)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", type=Path, default=Path("weights"))
    ap.add_argument("--url", default=WEIGHTS_URL)
    ap.add_argument("--verify-only", action="store_true", help="only hash the files already in --dest")
    args = ap.parse_args()
    manifest = read_manifest()
    if not args.verify_only and problems(args.dest, manifest):
        download(args.dest, args.url, manifest)
    bad = problems(args.dest, manifest)
    if bad:
        raise SystemExit("weights in %s are not the released ones:\n  %s" % (args.dest, "\n  ".join(bad)))
    print(f"{len(manifest)} weight files in {args.dest} match data/weights.sha256")


if __name__ == "__main__":
    main()
