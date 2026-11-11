"""Download, verify and safely extract the ARCADE dataset from Zenodo.

Source: ARCADE (Automatic Region-based Coronary Artery Disease diagnostics using
x-ray angiography imagEs), Zenodo record 10390295, version "COCO", CC0 1.0.

Security notes
* The archive is untrusted input. It is verified against the MD5 that Zenodo
  publishes and against a SHA-256 pinned below before anything is extracted.
* Extraction is allow-listed: only .png and .json members, no absolute paths,
  no parent-directory parts, no symlinks, a cap on total size and on the
  compression ratio (zip bomb guard). Nothing from the archive is executed or
  unpickled; images are decoded with Pillow and annotations parsed with json.

Usage: python -m angiolens.data.download [--no-extract]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
import sys
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

from angiolens.paths import ARCADE_DIR, ARCADE_EXTRACT_DIR, ARCADE_ZIP_DIR, ensure

DATASET = {
    "name": "ARCADE",
    "title": (
        "ARCADE: Automatic Region-based Coronary Artery Disease diagnostics "
        "using x-ray angiography imagEs Dataset"
    ),
    "zenodo_record": 10390295,
    "zenodo_version": "COCO",
    "doi": "10.5281/zenodo.10390295",
    "url": "https://zenodo.org/api/records/10390295/files/arcade.zip/content",
    "landing_page": "https://zenodo.org/records/10390295",
    "file_name": "arcade.zip",
    "size_bytes": 451573813,
    # Published by Zenodo in the record metadata.
    "md5": "c5b1973ade06f7dff210f878161e1a76",
    # Computed by this script on the first verified download (2026-11-03).
    "sha256": "a396cdea7c92c55dc97bbf3dd8e3df517d76872b289a8bcb45513bdb3350837f",
    "license": "CC0 1.0 Universal (Zenodo record license id: cc-zero)",
    "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
    "citation": (
        "Popov M, Amanturdieva A, Zhaksylyk N, et al. Dataset for Automatic "
        "Region-based Coronary Artery Disease Diagnostics Using X-Ray Angiography "
        "Images. Scientific Data 11, 20 (2024). doi:10.1038/s41597-023-02871-z"
    ),
}

ALLOWED_SUFFIXES = {".png", ".json"}
# Only the vessel tree task ("syntax") is used. The stenosis task has
# different images and only stenosis boxes, so it is left in the archive.
ALLOWED_PREFIX = "arcade/syntax/"
MAX_TOTAL_UNCOMPRESSED = 3 * 1024**3  # 3 GiB, the real archive is far smaller
MAX_MEMBERS = 20000
MAX_RATIO = 200.0  # PNGs barely compress; a huge ratio means something is wrong
CHUNK = 1 << 20


def _hashes(path: Path) -> tuple[str, str]:
    md5 = hashlib.md5()
    sha = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(CHUNK)
            if not block:
                break
            md5.update(block)
            sha.update(block)
    return md5.hexdigest(), sha.hexdigest()


def download(dest_dir: Path = ARCADE_ZIP_DIR) -> Path:
    ensure(dest_dir)
    target = dest_dir / DATASET["file_name"]
    if target.exists():
        print(f"found {target}, verifying")
    else:
        part = target.with_suffix(".part")
        print(f"downloading {DATASET['url']}")
        req = urllib.request.Request(DATASET["url"], headers={"User-Agent": "angiolens/0.1"})
        with urllib.request.urlopen(req, timeout=60) as resp, open(part, "wb") as out:
            total = 0
            while True:
                block = resp.read(CHUNK)
                if not block:
                    break
                out.write(block)
                total += len(block)
                if total > DATASET["size_bytes"] * 1.01:
                    part.unlink(missing_ok=True)
                    raise RuntimeError("download is larger than the published size, aborting")
                if total % (50 * CHUNK) < CHUNK:
                    print(f"  {total / 1e6:.0f} MB")
        part.rename(target)
    verify(target)
    return target


def verify(path: Path) -> dict:
    size = path.stat().st_size
    md5, sha = _hashes(path)
    problems = []
    if size != DATASET["size_bytes"]:
        problems.append(f"size {size} != {DATASET['size_bytes']}")
    if md5 != DATASET["md5"]:
        problems.append(f"md5 {md5} != {DATASET['md5']}")
    if DATASET["sha256"] is not None and sha != DATASET["sha256"]:
        problems.append(f"sha256 {sha} != {DATASET['sha256']}")
    if problems:
        # Fail closed: a file that does not match is never extracted.
        raise RuntimeError("checksum verification failed: " + "; ".join(problems))
    print(f"verified {path.name}: {size} bytes, md5 {md5}, sha256 {sha}")
    if DATASET["sha256"] is None:
        print("note: sha256 not pinned yet; pin the value above in DATASET['sha256']")
    return {"size_bytes": size, "md5": md5, "sha256": sha}


def _safe_member_path(name: str) -> PurePosixPath | None:
    """Return a safe relative path for an archive member, or None to skip it."""
    if "\\" in name or name.startswith("/") or ":" in name:
        raise RuntimeError(f"unsafe member name in archive: {name!r}")
    p = PurePosixPath(name)
    if any(part == ".." for part in p.parts):
        raise RuntimeError(f"path traversal in archive member: {name!r}")
    if p.parts and p.parts[0] == "__MACOSX":
        return None
    if p.name.startswith("."):
        return None
    if p.suffix.lower() not in ALLOWED_SUFFIXES:
        return None
    if not name.startswith(ALLOWED_PREFIX):
        return None
    return p


def extract(zip_path: Path, dest: Path = ARCADE_EXTRACT_DIR) -> dict:
    ensure(dest)
    dest_resolved = dest.resolve()
    counts = {"extracted": 0, "skipped": 0, "bytes": 0}
    with zipfile.ZipFile(zip_path) as zf:
        infos = zf.infolist()
        if len(infos) > MAX_MEMBERS:
            raise RuntimeError(f"archive has {len(infos)} members, limit {MAX_MEMBERS}")
        total = sum(i.file_size for i in infos)
        if total > MAX_TOTAL_UNCOMPRESSED:
            raise RuntimeError(f"archive expands to {total} bytes, limit {MAX_TOTAL_UNCOMPRESSED}")
        for info in infos:
            if info.is_dir():
                continue
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise RuntimeError(f"symlink in archive: {info.filename!r}")
            rel = _safe_member_path(info.filename)
            if rel is None:
                counts["skipped"] += 1
                continue
            if info.compress_size and info.file_size / info.compress_size > MAX_RATIO:
                raise RuntimeError(f"suspicious compression ratio for {info.filename!r}")
            out_path = (dest / Path(*rel.parts)).resolve()
            if dest_resolved not in out_path.parents:
                raise RuntimeError(f"member escapes destination: {info.filename!r}")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(out_path, "wb") as dst:
                shutil.copyfileobj(src, dst, CHUNK)
            counts["extracted"] += 1
            counts["bytes"] += info.file_size
    print(f"extracted {counts['extracted']} files ({counts['bytes'] / 1e6:.1f} MB), skipped {counts['skipped']}")
    return counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-extract", action="store_true")
    args = ap.parse_args(argv)
    zip_path = download()
    if not args.no_extract:
        counts = extract(zip_path)
        manifest = {"dataset": DATASET, "extraction": counts}
        (ARCADE_DIR / "download_manifest.json").write_text(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
