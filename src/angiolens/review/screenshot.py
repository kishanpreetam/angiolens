"""Save a picture of the review tool on the first case of a manifest.

The viewer is built hidden (never shown), so this runs without a visible
window. Nothing is written to the review output; the session uses a
temporary directory.

Usage: python -m angiolens.review.screenshot <manifest.csv> <out.png> [--window]
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from angiolens.review.napari_app import build_viewer
from angiolens.review.session import ReviewSession, load_manifest


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) < 2:
        print(__doc__.strip().splitlines()[-1])
        return 2
    manifest, out = Path(argv[0]), Path(argv[1])
    whole_window = "--window" in argv[2:]
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        session = ReviewSession(load_manifest(manifest), Path(tmp))
        viewer = build_viewer(session, show=False)
        if "ground truth" in viewer.layers:
            viewer.layers["ground truth"].visible = True
        viewer.window._qt_window.resize(1500, 950)
        viewer.screenshot(canvas_only=False, flash=False)  # lays out the hidden window
        viewer.reset_view()
        viewer.screenshot(path=str(out), canvas_only=not whole_window, flash=False)
        viewer.close()
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
