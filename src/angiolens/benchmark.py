"""Inference latency for one 512 x 512 frame: U-Net on the Apple GPU (MPS) and
CPU, and the Frangi baseline on CPU.

Usage: python -m angiolens.benchmark [run_name]

Timing: 10 warm-up passes, then the median and 90th percentile of 50 timed
passes (batch size 1, torch.no_grad, device synchronized before each clock
read). Frangi: median over 20 test frames. Other programs running on the
machine add noise, so treat the numbers as indicative.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time

import numpy as np
import torch

from angiolens.paths import RESULTS


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def time_unet(model: torch.nn.Module, device: torch.device, x: np.ndarray, warmup: int = 10, runs: int = 50) -> dict:
    model = model.to(device).eval()
    t = torch.from_numpy(x[None, None]).to(device)
    times = []
    with torch.no_grad():
        for i in range(warmup + runs):
            _sync(device)
            t0 = time.perf_counter()
            torch.sigmoid(model(t))
            _sync(device)
            if i >= warmup:
                times.append((time.perf_counter() - t0) * 1000)
    return {"device": str(device), "median_ms": float(np.median(times)), "p90_ms": float(np.percentile(times, 90)),
            "runs": runs, "warmup": warmup}


def cpu_name() -> str:
    try:
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True,
                              timeout=5).stdout.strip()
    except Exception:
        return platform.processor()


def main(argv: list[str] | None = None) -> int:
    from angiolens.baseline.frangi import vesselness
    from angiolens.data.dataset import normalize
    from angiolens.predict import load_model
    from angiolens.train import data_rows

    argv = sys.argv[1:] if argv is None else argv
    name = argv[0] if argv else "unet_scratch"
    images, _, rows, _ = data_rows("arcade", "test")
    frame = normalize(np.asarray(images[rows[0]]))
    model, _ = load_model(name, device=torch.device("cpu"))
    torch.set_num_threads(max(1, torch.get_num_threads()))
    results = {"model": name, "input": "1 x 1 x 512 x 512 float32", "unet": []}
    if torch.backends.mps.is_available():
        results["unet"].append(time_unet(model, torch.device("mps"), frame))
    if torch.cuda.is_available():
        results["unet"].append(time_unet(model, torch.device("cuda"), frame))
    results["unet"].append(time_unet(model, torch.device("cpu"), frame))
    ftimes = []
    for r in rows[:20]:
        img = np.asarray(images[r])
        t0 = time.perf_counter()
        vesselness(img)
        ftimes.append(time.perf_counter() - t0)
    results["frangi_cpu_median_s"] = float(np.median(ftimes))
    results["machine"] = {"cpu": cpu_name(), "platform": platform.platform(), "torch": torch.__version__,
                          "cpu_threads": torch.get_num_threads()}
    results["note"] = "measured while other programs were running; indicative only"
    (RESULTS / "benchmark.json").write_text(json.dumps(results, indent=1))
    for u in results["unet"]:
        print(f"U-Net on {u['device']}: median {u['median_ms']:.1f} ms, p90 {u['p90_ms']:.1f} ms")
    print(f"Frangi on CPU: median {results['frangi_cpu_median_s']:.2f} s per frame")
    return 0


if __name__ == "__main__":
    sys.exit(main())
