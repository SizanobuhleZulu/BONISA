"""Object tracking with SAM 3.1 through the Meta Model API.

SAM 3.1 takes a short noun phrase ("orange ball") and returns a box and mask
for every match in every frame. We use the box centre as the bob's position,
and one detection of the A4 sheet to turn pixels into metres.
"""

from __future__ import annotations

import asyncio
import base64
import os
from dataclasses import dataclass
from pathlib import Path

from openai import AsyncOpenAI
from meta_sam_parser import (
    image_segmentation_format,
    parse_responses_stream,
    video_segmentation_format,
)

BASE_URL = "https://api.meta.ai/v1"
A4_LONG_M, A4_SHORT_M = 0.297, 0.210


def client() -> AsyncOpenAI:
    key = os.environ.get("MODEL_API_KEY")
    if not key:
        raise RuntimeError("Set MODEL_API_KEY (put it in a .env file, never in the code).")
    return AsyncOpenAI(base_url=BASE_URL, api_key=key)


@dataclass
class Box:
    frame: int
    object_id: str
    left: float
    top: float
    right: float
    bottom: float
    confidence: float | None

    @property
    def cx(self) -> float:
        return (self.left + self.right) / 2

    @property
    def cy(self) -> float:
        return (self.top + self.bottom) / 2


async def _stream_boxes(content: list[dict], media: str) -> list[Box]:
    c = client()
    stream = await c.responses.create(
        model="sam-3.1",
        input=[{"type": "message", "role": "user", "content": content}],
        metadata={"include_confidence": "true"},
        stream=True,
    )
    fmt = video_segmentation_format() if media == "video" else image_segmentation_format()
    parsed = parse_responses_stream(stream, fmt)
    result = await parsed.final_result()
    boxes = []
    for r in result.records:
        if r.kind != "box":
            continue
        boxes.append(Box(frame=r.frame.frame_index if r.frame else 0, object_id=r.object_id,
                         left=r.left, top=r.top, right=r.right, bottom=r.bottom,
                         confidence=r.confidence))
    return boxes


async def upload_video(path: str | Path) -> str:
    c = client()
    with open(path, "rb") as f:
        up = await c.files.create(file=f, purpose="user_data")
    return up.id


async def track_object(video_path: str | Path, prompt: str) -> list[Box]:
    """Track `prompt` through the whole video. Returns every box SAM reports."""
    file_id = await upload_video(video_path)
    content = [{"type": "input_text", "text": prompt},
               {"type": "input_video", "file_id": file_id}]
    return await _stream_boxes(content, "video")


async def find_in_image(image_path: str | Path, prompt: str) -> list[Box]:
    data = base64.b64encode(Path(image_path).read_bytes()).decode()
    suffix = Path(image_path).suffix.lstrip(".").replace("jpg", "jpeg")
    content = [{"type": "input_text", "text": prompt},
               {"type": "input_image", "image_url": f"data:image/{suffix};base64,{data}"}]
    return await _stream_boxes(content, "image")


def best_track(boxes: list[Box]) -> dict[int, Box]:
    """If SAM found several objects, keep the one seen in the most frames,
    one box per frame (highest confidence)."""
    by_obj: dict[str, dict[int, Box]] = {}
    for b in boxes:
        frames = by_obj.setdefault(b.object_id, {})
        old = frames.get(b.frame)
        if old is None or (b.confidence or 0) > (old.confidence or 0):
            frames[b.frame] = b
    if not by_obj:
        return {}
    return max(by_obj.values(), key=len)


def scale_from_sheet(sheet: Box) -> tuple[float, float, float]:
    """Metres per pixel from an A4 sheet's box, its error, and the sheet's aspect
    ratio (should be 1.414 if the camera faces the wall squarely)."""
    w, h = sheet.right - sheet.left, sheet.bottom - sheet.top
    long_px, short_px = max(w, h), min(w, h)
    s1, s2 = A4_LONG_M / long_px, A4_SHORT_M / short_px
    scale = (s1 + s2) / 2
    err = abs(s1 - s2) / 2 + scale * (2 / long_px)  # disagreement + ~2 px box-edge error
    return scale, err, long_px / short_px


def run(coro):
    return asyncio.run(coro)
