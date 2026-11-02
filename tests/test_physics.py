"""Simulate a filmed pendulum with known g and check Bonisa recovers it."""
import numpy as np
from scipy.integrate import solve_ivp

from bonisa.physics import fit_pendulum
from bonisa.verify import run_checks, verdict


def simulate(L=0.80, g=9.81, theta0_deg=10, gamma=0.02, fps=30, secs=25, m_per_px=0.001,
             pivot=(640, 100), noise_px=1.0, seed=0):
    rng = np.random.default_rng(seed)
    def rhs(t, s):
        th, w = s
        return [w, -(g / L) * np.sin(th) - 2 * gamma * w]
    t = np.arange(0, secs, 1 / fps) + rng.normal(0, 0.0005, int(secs * fps))  # slight VFR jitter
    t = np.sort(t - t[0])
    sol = solve_ivp(rhs, (0, t[-1]), [np.radians(theta0_deg), 0], t_eval=t, rtol=1e-10, atol=1e-12)
    th = sol.y[0]
    r_px = L / m_per_px
    x = pivot[0] + r_px * np.sin(th) + rng.normal(0, noise_px, len(t))
    y = pivot[1] + r_px * np.cos(th) + rng.normal(0, noise_px, len(t))
    return t, x, y


def test_recovers_g_with_tape_length():
    t, x, y = simulate()
    fit = fit_pendulum(t, x, y, m_per_px=0.001, length_user_m=0.80)
    assert abs(fit.g - 9.81) < 3 * fit.g_err + 0.02, (fit.g, fit.g_err)
    assert abs(fit.length_video_m - 0.80) < 0.01
    checks = run_checks(fit, len(t), len(t), 30, 0.01, 1.41)
    assert verdict(checks) == "verified", [(c.name, c.status) for c in checks]


def test_large_swing_is_corrected():
    t, x, y = simulate(theta0_deg=25)
    fit = fit_pendulum(t, x, y, m_per_px=0.001, length_user_m=0.80)
    assert abs(fit.g - 9.81) / 9.81 < 0.01, fit.g


def test_short_video_rejected():
    t, x, y = simulate(secs=3)
    fit = fit_pendulum(t, x, y, m_per_px=0.001, length_user_m=0.80)
    checks = run_checks(fit, len(t), len(t), 30, 0.01, 1.41)
    assert verdict(checks) == "rejected"


def test_tilted_camera_rejected():
    t, x, y = simulate()
    fit = fit_pendulum(t, x, y, m_per_px=0.001, length_user_m=0.80)
    checks = run_checks(fit, len(t), len(t), 30, 0.01, 1.15)
    assert verdict(checks) == "rejected"


if __name__ == "__main__":
    t, x, y = simulate()
    f = fit_pendulum(t, x, y, m_per_px=0.001, length_user_m=0.80)
    print(f"g = {f.g:.3f} ± {f.g_err:.3f}  T = {f.period_s:.4f} s  L_video = {f.length_video_m:.4f} m  amp = {f.amplitude_deg:.1f} deg  R2 = {f.r_squared:.4f}")
    for c in run_checks(f, len(t), len(t), 30, 0.01, 1.41):
        print(f"  [{c.status}] {c.name}: {c.value}")
