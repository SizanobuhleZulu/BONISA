"""Measure g from a pendulum video, end to end.

    python scripts/measure.py my_pendulum.mp4 --object "orange ball" --length 0.80

Writes everything to runs/<video name>/: the converted clip, the raw SAM 3.1
tracks (cached, so re-running doesn't call the API again), a results JSON and
a plot.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

from bonisa import physics, track, verify, video  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--object", default="pendulum bob", help='what the bob looks like, e.g. "orange ball"')
    ap.add_argument("--sheet", default="white sheet of paper", help="prompt for the A4 scale sheet")
    ap.add_argument("--length", type=float, help="string length in metres, pivot to bob centre")
    ap.add_argument("--refresh", action="store_true", help="call SAM again even if cached")
    args = ap.parse_args()

    load_dotenv()
    src = Path(args.video)
    out = Path("runs") / src.stem
    out.mkdir(parents=True, exist_ok=True)

    clip = out / "clip.mp4"
    if not clip.exists() or args.refresh:
        print("Converting video...")
        video.normalise(src, clip)
    times, width, height = video.frame_info(clip)
    print(f"{len(times)} frames, {width}x{height}, {times[-1]:.1f} s")

    cache = out / "sam_tracks.json"
    if cache.exists() and not args.refresh:
        raw = json.loads(cache.read_text())
        print("Using cached SAM 3.1 results.")
    else:
        first = video.grab_frame(clip, out / "frame0.png", 0)
        print(f'SAM 3.1: finding "{args.sheet}" in the first frame...')
        sam = track
        if os.environ.get("BONISA_FAKE_SAM"):  # offline testing only
            from tests import fake_sam as sam
        sheet = track.run(sam.find_in_image(first, args.sheet))
        print(f'SAM 3.1: tracking "{args.object}" through the video...')
        bob = track.run(sam.track_object(clip, args.object))
        raw = {"sheet": [asdict(b) for b in sheet], "bob": [asdict(b) for b in bob]}
        cache.write_text(json.dumps(raw))

    bob = track.best_track([track.Box(**b) for b in raw["bob"]])
    if len(bob) < 10:
        sys.exit(f'SAM 3.1 found the "{args.object}" in only {len(bob)} frames. Try a different --object description.')
    frames = sorted(bob)
    t = np.array([times[f] for f in frames if f < len(times)])
    x = np.array([bob[f].cx for f in frames if f < len(times)])
    y = np.array([bob[f].cy for f in frames if f < len(times)])

    sheets = sorted((track.Box(**b) for b in raw["sheet"]),
                    key=lambda b: (b.right - b.left) * (b.bottom - b.top), reverse=True)
    if sheets:
        scale, scale_err, aspect = track.scale_from_sheet(sheets[0])
    else:
        scale, scale_err, aspect = None, 0.0, None
        if not args.length:
            sys.exit("No A4 sheet found and no --length given, so there's no way to get real units.")
        print("No A4 sheet found: lengths from the video won't be checked.")

    fit = physics.fit_pendulum(t, x, y, m_per_px=scale or 1.0, m_per_px_err=scale_err,
                               length_user_m=args.length)
    if scale is None:
        fit.length_video_m = fit.length_video_err_m = None

    dt = np.diff(times)
    fps = 1 / np.median(dt)
    cv = float(np.std(dt) / np.mean(dt))
    checks = verify.run_checks(fit, len(times), len(t), fps, cv, aspect)
    v = verify.verdict(checks)

    result = {"verdict": v, "fit": fit.to_dict(), "checks": verify.checks_to_dicts(checks),
              "video": {"frames": len(times), "fps": fps, "duration_s": float(times[-1]),
                        "width": width, "height": height},
              "scale_m_per_px": scale, "sheet_aspect": aspect}
    (out / "result.json").write_text(json.dumps(result, indent=2))

    print()
    if v == "rejected":
        print("REJECTED: Bonisa doesn't trust this measurement.")
    else:
        print(f"g = {fit.g:.3f} ± {fit.g_err:.3f} m/s²   ({v.replace('_', ' ')})")
    print(f"T = {fit.period_s:.4f} ± {fit.period_err_s:.4f} s, L = {fit.length_used_m*100:.1f} cm ({fit.length_source})")
    for c in checks:
        print(f"  [{c.status:4}] {c.name}: {c.value}. {c.message}")

    plot(out, t, x, y, fit)
    print(f"\nSaved to {out}/")


def plot(out: Path, t, x, y, fit) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(x, y, ".", ms=3, label="SAM 3.1 track")
    if fit.pivot_px:
        ax[0].plot(*fit.pivot_px, "r+", ms=12, label="fitted pivot")
    ax[0].invert_yaxis()
    ax[0].set_aspect("equal", adjustable="datalim")
    ax[0].set(title="Bob path (pixels)", xlabel="x", ylabel="y")
    ax[0].legend()

    t0 = t - t[0]
    ax[1].plot(t0, x, ".", ms=3, label="tracked x")
    ax[1].set(title=f"g = {fit.g:.3f} ± {fit.g_err:.3f} m/s²", xlabel="time (s)", ylabel="x (px)")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(out / "plot.png", dpi=130)


if __name__ == "__main__":
    main()
