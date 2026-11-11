import stat
import zipfile

import pytest

from angiolens.data import download as dl


def _zip(path, members):
    with zipfile.ZipFile(path, "w") as zf:
        for name, data, mode in members:
            info = zipfile.ZipInfo(name)
            if mode is not None:
                info.external_attr = mode << 16
            zf.writestr(info, data)
    return path


def test_member_filter():
    assert dl._safe_member_path("arcade/syntax/train/images/1.png") is not None
    assert dl._safe_member_path("arcade/syntax/train/annotations/train.json") is not None
    assert dl._safe_member_path("arcade/stenosis/train/images/1.png") is None  # not used
    assert dl._safe_member_path("arcade/syntax/train/run.py") is None  # not allow-listed
    assert dl._safe_member_path("__MACOSX/arcade/syntax/x.png") is None
    for bad in ("../evil.png", "arcade/../../evil.png", "/abs/evil.png", "arcade\\syntax\\x.png", "C:/x.png"):
        with pytest.raises(RuntimeError):
            dl._safe_member_path(bad)


def test_extract_only_writes_allowed_files(tmp_path):
    z = _zip(tmp_path / "ok.zip", [
        ("arcade/syntax/train/images/1.png", b"png-bytes", None),
        ("arcade/syntax/train/notes.txt", b"skip me", None),
        ("arcade/stenosis/val/images/2.png", b"skip me too", None),
    ])
    dest = tmp_path / "out"
    counts = dl.extract(z, dest)
    assert counts == {"extracted": 1, "skipped": 2, "bytes": 9}
    files = sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file())
    assert files == ["arcade/syntax/train/images/1.png"]


def test_extract_rejects_traversal_and_symlinks(tmp_path):
    z1 = _zip(tmp_path / "trav.zip", [("arcade/syntax/../../../escape.png", b"x", None)])
    with pytest.raises(RuntimeError):
        dl.extract(z1, tmp_path / "out1")
    assert not (tmp_path / "escape.png").exists()
    z2 = _zip(tmp_path / "link.zip", [("arcade/syntax/link.png", b"/etc/passwd", stat.S_IFLNK | 0o777)])
    with pytest.raises(RuntimeError):
        dl.extract(z2, tmp_path / "out2")


def test_verify_fails_closed_on_checksum_mismatch(tmp_path, monkeypatch):
    f = tmp_path / "arcade.zip"
    f.write_bytes(b"not the real archive")
    monkeypatch.setitem(dl.DATASET, "size_bytes", f.stat().st_size)
    with pytest.raises(RuntimeError, match="md5"):
        dl.verify(f)


def test_pinned_checksums_are_present():
    assert len(dl.DATASET["md5"]) == 32
    assert dl.DATASET["sha256"] and len(dl.DATASET["sha256"]) == 64
    assert dl.DATASET["url"].startswith("https://zenodo.org/")
