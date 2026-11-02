"""Video helpers: real frame timestamps, format conversion and frame grabs.

Uses the ffmpeg binary bundled with the `imageio-ffmpeg` package, so nothing
extra needs installing on Windows. Falls back to a system ffmpeg.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import numpy as np


def ffmpeg() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        exe = shutil.which("ffmpeg")
        if not exe:
            raise RuntimeError("ffmpeg not found: run `pip install imageio-ffmpeg`.")
        return exe


def normalise(src: str | Path, dst: str | Path, max_height: int = 720) -> Path:
    """Re-encode to H.264 MP4 (what SAM 3.1 accepts): applies phone rotation,
    downscales for a smaller upload, drops audio, keeps every original frame."""
    dst = Path(dst)
    subprocess.run([ffmpeg(), "-y", "-v", "error", "-i", str(src), "-an",
                    "-vf", f"scale=-2:'min({max_height},ih)'", "-fps_mode", "passthrough",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
                    str(dst)], check=True)
    return dst


def frame_info(path: str | Path) -> tuple[np.ndarray, int, int]:
    """Timestamp (s, from 0) of every frame, plus width and height.

    Phones often record at a variable frame rate, so we use each frame's real
    timestamp instead of assuming a constant fps."""
    proc = subprocess.run([ffmpeg(), "-v", "info", "-i", str(path), "-vf", "showinfo",
                           "-f", "null", "-"], capture_output=True, text=True, check=True)
    times = [float(m) for m in re.findall(r"pts_time:\s*([0-9.eE+-]+)", proc.stderr)]
    size = re.search(r"Stream #.*Video:.*?(\d{2,5})x(\d{2,5})", proc.stderr)
    w, h = (int(size.group(1)), int(size.group(2))) if size else (0, 0)
    t = np.array(times)
    return t - t[0], w, h


def grab_frame(src: str | Path, dst: str | Path, index: int = 0) -> Path:
    dst = Path(dst)
    subprocess.run([ffmpeg(), "-y", "-v", "error", "-i", str(src), "-vf",
                    f"select=eq(n\\,{index})", "-frames:v", "1", str(dst)], check=True)
    return dst
