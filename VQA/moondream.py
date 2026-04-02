"""
For each image/video under IMAGE_DIR, use moondream2 (via Ollama) to generate
a short descriptive phrase and save results to paligemma_phrases.txt.

moondream2 is a lightweight 1.9B vision model — well-suited for CPU-only machines.

Before running:
  ollama pull moondream

Output format (one line per file):
  <relative/path/to/file> | <phrase>
"""

import base64
import json
import threading
import time
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image

# Shared stop flag — set when Ctrl+C is pressed
_stop = threading.Event()

# ── Configuration ──────────────────────────────────────────────────────────────

IMAGE_DIR      = r"C:\Users\renov\Codes\fire_factory"
OLLAMA_URL     = "http://127.0.0.1:11434/api/generate"
MODEL_NAME     = "moondream"
OLLAMA_TIMEOUT = 600

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}

FRAMES_PER_VIDEO = 4
MAX_IMAGE_PX     = 512

PROMPT = (
    "Describe this image in one short phrase focusing on "
    "whether fire is present, its nature, and the environment. "
    "Reply with the phrase only."
)

# ── Helpers ────────────────────────────────────────────────────────────────────

def resize_image(img: Image.Image, max_px: int = MAX_IMAGE_PX) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_px:
        return img
    scale = max_px / max(w, h)
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def image_to_b64(path: Path) -> str:
    img = resize_image(Image.open(path).convert("RGB"))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def pil_to_b64(img: Image.Image) -> str:
    img = resize_image(img.convert("RGB"))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def ollama_phrase(image_b64: str) -> str:
    """Streams response in a daemon thread so Ctrl+C fires within ~0.5 s."""
    payload = {
        "model":   MODEL_NAME,
        "prompt":  PROMPT,
        "images":  [image_b64],
        "stream":  True,
        "options": {"temperature": 0.2},
    }

    result: list[str]       = []
    error:  list[Exception] = []

    def _worker():
        try:
            with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=OLLAMA_TIMEOUT) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if _stop.is_set():
                        break
                    if not line:
                        continue
                    data = json.loads(line.decode("utf-8") if isinstance(line, bytes) else line)
                    result.append(data.get("response", ""))
                    if data.get("done"):
                        break
        except Exception as exc:
            error.append(exc)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    while t.is_alive():
        t.join(timeout=0.5)
        if _stop.is_set():
            raise KeyboardInterrupt

    if error:
        raise error[0]
    return "".join(result).strip()


def extract_frames(video_path: Path, n: int = FRAMES_PER_VIDEO) -> list[Image.Image]:
    cap   = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return []
    frames = []
    for idx in np.linspace(0, total - 1, n, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


# ── Startup check ──────────────────────────────────────────────────────────────

def check_ollama():
    base = OLLAMA_URL.rsplit("/", 2)[0]
    try:
        r = requests.get(f"{base}/api/tags", timeout=10)
        r.raise_for_status()
        models  = [m["name"] for m in r.json().get("models", [])]
        matched = [m for m in models if MODEL_NAME.split(":")[0] in m]
        if not matched:
            raise SystemExit(
                f"ERROR: '{MODEL_NAME}' not found in Ollama.\n"
                f"  Run:  ollama pull {MODEL_NAME}\n"
                f"  Available models: {models}"
            )
        print(f"Ollama OK — model '{matched[0]}' ready.")
    except requests.exceptions.ConnectionError:
        raise SystemExit(
            f"ERROR: Cannot connect to Ollama at {base}\n"
            "  Make sure Ollama is running (check system tray or run: ollama serve)"
        )
    except requests.exceptions.Timeout:
        raise SystemExit(f"ERROR: Ollama at {base} did not respond within 10 s.")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    check_ollama()
    root = Path(IMAGE_DIR)

    all_files = sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS
    )

    print(f"\nFound {len(all_files)} file(s) under {root}\n{'─'*60}")

    out_path = root / "moondream_phrases.txt"
    try:
        with open(out_path, "w", encoding="utf-8") as out:

            for idx, path in enumerate(all_files, 1):
                rel = path.relative_to(root)
                ext = path.suffix.lower()
                print(f"[{idx}/{len(all_files)}] {rel}")

                try:
                    t0 = time.time()

                    if ext in IMAGE_EXTS:
                        phrase = ollama_phrase(image_to_b64(path))
                        print(f"  → {phrase!r}  ({time.time()-t0:.1f}s)")
                        out.write(f"{rel} | {phrase}\n")

                    else:  # video
                        frames = extract_frames(path)
                        for i, frame in enumerate(frames):
                            phrase = ollama_phrase(pil_to_b64(frame))
                            print(f"  frame {i+1}/{len(frames)} → {phrase!r}  ({time.time()-t0:.1f}s)")
                            out.write(f"{rel} [frame {i+1}] | {phrase}\n")
                            t0 = time.time()

                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    print(f"  ERROR: {e}")
                    out.write(f"{rel} | ERROR: {e}\n")

                out.flush()

    except KeyboardInterrupt:
        _stop.set()
        print("\n\nStopped by user (Ctrl+C). Progress saved so far.")
        return

    print(f"\nDone. Phrases saved → {out_path}")


if __name__ == "__main__":
    main()
