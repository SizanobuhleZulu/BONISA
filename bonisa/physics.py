"""Pendulum physics: turn a tracked bob trajectory into a measured g.

Inputs are pixel positions of the bob per frame, the real time of each frame,
and a metres-per-pixel scale. Everything here is plain NumPy/SciPy so it can be
tested without calling any API.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy.optimize import curve_fit, least_squares


@dataclass
class PendulumFit:
    period_s: float
    period_err_s: float
    damping_per_s: float
    amplitude_px: float
    r_squared: float
    residual_rms_px: float
    n_periods: float
    pivot_px: tuple[float, float] | None
    length_video_m: float | None
    length_video_err_m: float | None
    amplitude_deg: float | None
    length_used_m: float
    length_used_err_m: float
    length_source: str
    amplitude_correction: float
    g: float
    g_err: float

    def to_dict(self) -> dict:
        return asdict(self)


def damped_cosine(t, a, gamma, omega, phi, c):
    return a * np.exp(-gamma * t) * np.cos(omega * t + phi) + c


def _initial_frequency(t: np.ndarray, x: np.ndarray) -> float:
    """Dominant angular frequency of x(t), from an FFT on a uniform resample."""
    n = len(t)
    tu = np.linspace(t[0], t[-1], n)
    xu = np.interp(tu, t, x - np.mean(x))
    xu = xu * np.hanning(n)
    spec = np.abs(np.fft.rfft(xu))
    freqs = np.fft.rfftfreq(n, d=(tu[1] - tu[0]))
    spec[0] = 0.0
    k = int(np.argmax(spec))
    return 2 * np.pi * float(freqs[k])


def fit_circle(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """Geometric circle fit (seeded by the algebraic Kasa fit).

    Returns centre x, centre y, radius and the radius's standard error. On a
    short arc the radius is poorly constrained, and the error says so.
    """
    a = np.column_stack([x, y, np.ones_like(x)])
    b = x**2 + y**2
    sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    xc0, yc0 = sol[0] / 2, sol[1] / 2
    r0 = float(np.sqrt(max(sol[2] + xc0**2 + yc0**2, 1.0)))

    def resid(p):
        return np.hypot(x - p[0], y - p[1]) - p[2]

    out = least_squares(resid, [xc0, yc0, r0], method="lm")
    xc, yc, r = out.x
    dof = max(len(x) - 3, 1)
    s2 = float(np.sum(out.fun**2) / dof)
    try:
        cov = np.linalg.inv(out.jac.T @ out.jac) * s2
        r_err = float(np.sqrt(abs(cov[2, 2])))
    except np.linalg.LinAlgError:
        r_err = float("inf")
    return float(xc), float(yc), float(abs(r)), r_err


def fit_pendulum(
    t: np.ndarray,
    x_px: np.ndarray,
    y_px: np.ndarray,
    m_per_px: float,
    m_per_px_err: float = 0.0,
    length_user_m: float | None = None,
    length_user_err_m: float = 0.003,
) -> PendulumFit:
    t = np.asarray(t, float) - float(t[0])
    x = np.asarray(x_px, float)
    y = np.asarray(y_px, float)

    # 1. Period from horizontal motion x(t) (robust even if the arc is short).
    w0 = _initial_frequency(t, x)
    a0 = (np.max(x) - np.min(x)) / 2
    c0 = float(np.mean(x))
    best = None
    for phi0 in np.linspace(-np.pi, np.pi, 8, endpoint=False):
        try:
            p, cov = curve_fit(
                damped_cosine, t, x, p0=[a0, 0.01, w0, phi0, c0],
                bounds=([0, -0.5, 0.5 * w0, -10, -np.inf], [np.inf, 5, 1.5 * w0, 10, np.inf]),
                maxfev=20000,
            )
        except RuntimeError:
            continue
        res = x - damped_cosine(t, *p)
        sse = float(np.sum(res**2))
        if best is None or sse < best[2]:
            best = (p, cov, sse, res)
    if best is None:
        raise ValueError("Could not fit an oscillation to the tracked motion.")
    p, cov, sse, res = best
    amp, gamma, omega_d, _, _ = p
    omega_err = float(np.sqrt(cov[2, 2]))
    gamma_err = float(np.sqrt(abs(cov[1, 1])))

    # Natural frequency of the undamped pendulum.
    omega0 = float(np.sqrt(omega_d**2 + gamma**2))
    omega0_err = float(np.hypot(omega_d * omega_err, gamma * gamma_err) / omega0)
    period = 2 * np.pi / omega0
    period_err = 2 * np.pi * omega0_err / omega0**2

    ss_tot = float(np.sum((x - np.mean(x)) ** 2))
    r2 = 1 - sse / ss_tot if ss_tot > 0 else 0.0
    n_periods = float((t[-1] - t[0]) / period)

    # 2. Pivot and string length from the arc the bob traces.
    pivot = None
    l_video = l_video_err = amp_deg = None
    try:
        xc, yc, r_px, r_err_px = fit_circle(x, y)
        if yc < np.mean(y) and r_px < 20 * (np.ptp(x) + 1):  # pivot above bob, sane radius
            pivot = (xc, yc)
            l_video = r_px * m_per_px
            l_video_err = float(np.hypot(r_err_px * m_per_px, r_px * m_per_px_err))
            amp_deg = float(np.degrees(np.arcsin(min(amp / r_px, 1.0))))
    except np.linalg.LinAlgError:
        pass

    # 3. Which length to trust: the tape measure if given, else the video.
    if length_user_m:
        length, length_err, source = length_user_m, length_user_err_m, "measured by user"
    elif l_video:
        length, length_err, source = l_video, l_video_err, "measured from video"
    else:
        raise ValueError("No pendulum length: enter it, or keep the pivot's arc in view.")

    if amp_deg is None:
        amp_deg_est = float(np.degrees(np.arcsin(min(amp * m_per_px / length, 1.0))))
    else:
        amp_deg_est = amp_deg

    # 4. g = 4 pi^2 L / T^2, with the finite-amplitude correction
    #    T = T0 (1 + theta0^2 / 16)  =>  T0 = T / (1 + theta0^2/16).
    theta0 = np.radians(amp_deg_est)
    corr = 1 + theta0**2 / 16
    t0 = period / corr
    g = 4 * np.pi**2 * length / t0**2
    g_err = g * float(np.hypot(length_err / length, 2 * period_err / period))

    return PendulumFit(
        period_s=float(period), period_err_s=float(period_err), damping_per_s=float(gamma),
        amplitude_px=float(amp), r_squared=float(r2),
        residual_rms_px=float(np.sqrt(np.mean(res**2))), n_periods=n_periods,
        pivot_px=pivot, length_video_m=l_video, length_video_err_m=l_video_err,
        amplitude_deg=amp_deg_est, length_used_m=float(length), length_used_err_m=float(length_err),
        length_source=source, amplitude_correction=float(corr), g=float(g), g_err=float(g_err),
    )
