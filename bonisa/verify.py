"""Physics checks that decide whether Bonisa trusts its own measurement.

Each check returns pass / warn / fail with a plain-language reason. Any fail
means Bonisa refuses to report g as a result and explains what to fix.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .physics import PendulumFit

A4_RATIO = 297 / 210  # 1.414


@dataclass
class Check:
    name: str
    status: str  # "pass" | "warn" | "fail"
    value: str
    message: str


def _c(name, ok, warn, value, msg_ok, msg_warn, msg_fail) -> Check:
    if ok:
        return Check(name, "pass", value, msg_ok)
    if warn:
        return Check(name, "warn", value, msg_warn)
    return Check(name, "fail", value, msg_fail)


def run_checks(
    fit: PendulumFit,
    n_frames_total: int,
    n_frames_tracked: int,
    fps: float,
    frame_interval_cv: float,
    sheet_aspect: float | None,
) -> list[Check]:
    checks: list[Check] = []

    cov = n_frames_tracked / max(n_frames_total, 1)
    checks.append(_c(
        "Tracking coverage", cov >= 0.9, cov >= 0.75, f"{cov:.0%} of frames",
        "The bob was found in almost every frame.",
        "The bob was lost in some frames. Keep it fully in view and avoid busy backgrounds.",
        "The bob was lost in too many frames to trust the timing. Re-film with the whole swing in view.",
    ))

    spp = fps * fit.period_s
    checks.append(_c(
        "Time resolution", spp >= 20 and frame_interval_cv < 0.05, spp >= 10, f"{fps:.0f} fps, {spp:.0f} frames per swing",
        "Enough frames per swing to time it precisely.",
        "Few frames per swing, or uneven frame timing. Use a longer pendulum or a higher frame rate.",
        "Too few frames per swing to measure the period. Use a longer pendulum or film at 60 fps.",
    ))

    checks.append(_c(
        "Swings recorded", fit.n_periods >= 5, fit.n_periods >= 3, f"{fit.n_periods:.1f} periods",
        "Enough complete swings to average out timing errors.",
        "Only a few swings recorded. Film for at least 20 seconds.",
        "Fewer than 3 swings recorded. Film for at least 20 seconds.",
    ))

    checks.append(_c(
        "Model fit", fit.r_squared >= 0.98, fit.r_squared >= 0.9, f"R² = {fit.r_squared:.3f}",
        "The motion matches a pendulum closely.",
        "The motion only roughly matches a pendulum. The camera may have moved, or the bob twisted.",
        "The motion does not look like a simple pendulum. Steady the camera and release the bob without pushing it.",
    ))

    if fit.amplitude_deg is not None:
        a = fit.amplitude_deg
        checks.append(_c(
            "Swing angle", a <= 15, a <= 30, f"{a:.1f}°",
            "Small swing, so the simple pendulum formula holds (corrected anyway).",
            "Large swing. Bonisa corrected for it, but a release under 15° is more accurate.",
            "Swing too large for the pendulum formula. Release the bob from under 15°.",
        ))

    if sheet_aspect is not None:
        dev = abs(sheet_aspect - A4_RATIO) / A4_RATIO
        checks.append(_c(
            "Scale reference", dev <= 0.05, dev <= 0.12, f"A4 aspect {sheet_aspect:.2f} (expected 1.41)",
            "The A4 sheet looks undistorted, so the scale is reliable.",
            "The A4 sheet looks slightly distorted. Point the camera straight at the wall.",
            "The A4 sheet looks strongly distorted, so the camera is at an angle and the scale is wrong. Film square-on.",
        ))
    else:
        checks.append(Check("Scale reference", "warn", "not found",
                            "No A4 sheet was found, so lengths from the video can't be checked."))

    if fit.length_video_m is not None and fit.length_source == "measured by user":
        diff = abs(fit.length_video_m - fit.length_used_m)
        sig = np.hypot(fit.length_video_err_m or 0, fit.length_used_err_m) or 1e-9
        rel = diff / fit.length_used_m
        nsig = diff / sig
        checks.append(_c(
            "Length cross-check", rel <= 0.05 or nsig <= 2, rel <= 0.12 or nsig <= 3,
            f"video {fit.length_video_m*100:.1f} ± {(fit.length_video_err_m or 0)*100:.1f} cm vs tape {fit.length_used_m*100:.1f} cm",
            "The length seen in the video agrees with your tape measure.",
            "The video and tape-measure lengths differ a little. Measure to the centre of the bob.",
            "The video and tape-measure lengths disagree. Check you measured from the pivot to the bob's centre.",
        ))

    rel_err = fit.g_err / fit.g
    checks.append(_c(
        "Precision", rel_err <= 0.02, rel_err <= 0.05, f"±{rel_err:.1%}",
        "The uncertainty is small.",
        "The uncertainty is moderate. More swings and a longer pendulum will help.",
        "The uncertainty is too large to be useful.",
    ))

    return checks


def verdict(checks: list[Check]) -> str:
    if any(c.status == "fail" for c in checks):
        return "rejected"
    if any(c.status == "warn" for c in checks):
        return "verified_with_warnings"
    return "verified"


def checks_to_dicts(checks: list[Check]) -> list[dict]:
    return [asdict(c) for c in checks]
