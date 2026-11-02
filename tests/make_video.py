"""Render a fake phone video of a pendulum with an A4 sheet on the wall."""
import sys
import cv2
import numpy as np
from scipy.integrate import solve_ivp

L, g, th0, fps, secs = 0.70, 9.81, np.radians(12), 30, 22
m_per_px = 0.0016                     # 1 px = 1.6 mm
pivot = np.array([500, 60])
sol = solve_ivp(lambda t, s: [s[1], -(g / L) * np.sin(s[0]) - 0.03 * s[1]], (0, secs), [th0, 0],
                t_eval=np.arange(0, secs, 1 / fps), rtol=1e-10, atol=1e-12)
w = cv2.VideoWriter(sys.argv[1], cv2.VideoWriter_fourcc(*"mp4v"), fps, (960, 600))
a4 = (np.array([297, 210]) / 1000 / m_per_px).astype(int)
for th in sol.y[0]:
    img = np.full((600, 960, 3), (90, 110, 120), np.uint8)
    cv2.rectangle(img, (650, 330), (650 + a4[0], 330 + a4[1]), (250, 250, 250), -1)
    bob = pivot + (L / m_per_px) * np.array([np.sin(th), np.cos(th)])
    cv2.line(img, tuple(pivot), tuple(bob.astype(int)), (30, 30, 30), 2)
    cv2.circle(img, tuple(bob.astype(int)), 16, (0, 140, 255), -1)
    w.write(img)
w.release()
print("rendered", len(sol.t), "frames")
