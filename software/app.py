import base64
import io
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import requests
import torch
from flask import Flask, render_template, request
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
from transformers import CLIPModel, CLIPProcessor
from ultralytics import YOLO
from waitress import serve


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
YOLO_WEIGHTS_PATH = MODEL_DIR / "firedetect-11s.pt"
YOLO_WEIGHTS_URL = (
    "https://huggingface.co/leeyunjai/yolo11-firedetect/resolve/main/firedetect-11s.pt"
)

YOLO_CONF_THRESHOLD = 0.25
DANGEROUS_FIRE_THRESHOLD = 0.14
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

FIRE_CAPTIONS = [
    "Flaming structure in a fenced area with debris",
    "Building engulfed in large flames and thick smoke",
    "Grilled meat over an open flame on a barbecue grill",
    "Gas stove with blue flame cooking",
    "Fireworks explosion in the night sky",
    "Campfire in a forest at dusk",
    "Candle flame burning in a dark environment",
    "Wood-burning fireplace in a cozy living room",
    "Man exhales smoke from a lit cigarette, emitting visible white vapor",
    "Sunset illuminates modern building, vibrant clouds, calm water reflecting hues",
    "Bright streetlamps and dense smoke create a hazy atmosphere",
]

DANGEROUS_CAPTION_INDICES = [0, 1]

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

_yolo_model: Optional[YOLO] = None
_clip_model: Optional[CLIPModel] = None
_clip_processor: Optional[CLIPProcessor] = None
_device: Optional[str] = None


@dataclass
class FireDetectionResult:
    filename: str
    is_dangerous: bool
    yolo_detected: bool
    yolo_conf: float
    clip_danger_score: float
    danger_score: float
    best_caption: str
    best_caption_probability: float
    annotated_image: Optional[str]
    error: Optional[str] = None


def device() -> str:
    global _device
    if _device is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
    return _device


def download_yolo_weights() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    if YOLO_WEIGHTS_PATH.exists() and YOLO_WEIGHTS_PATH.stat().st_size > 0:
        return

    with requests.get(YOLO_WEIGHTS_URL, stream=True, timeout=120) as response:
        response.raise_for_status()
        with YOLO_WEIGHTS_PATH.open("wb") as weights_file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    weights_file.write(chunk)


def load_models() -> tuple[YOLO, CLIPModel, CLIPProcessor]:
    global _yolo_model, _clip_model, _clip_processor

    if _yolo_model is None:
        download_yolo_weights()
        _yolo_model = YOLO(str(YOLO_WEIGHTS_PATH))

    if _clip_model is None or _clip_processor is None:
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device())
        _clip_model.eval()
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    return _yolo_model, _clip_model, _clip_processor


def allowed_image(filename: str) -> bool:
    return Path(filename).suffix.lower() in IMAGE_EXTENSIONS


def display_filename(filename: str) -> str:
    cleaned = filename.replace("\\", "/").strip("/")
    return cleaned or f"image-{uuid.uuid4().hex}.jpg"


def image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def draw_detections(image: Image.Image, boxes, combined_score: float) -> str:
    annotated = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()

    for box in boxes:
        confidence = float(box.conf.detach().cpu().item())
        xyxy = box.xyxy.detach().cpu().numpy()[0]
        x1, y1, x2, y2 = [int(v) for v in xyxy]

        draw.rectangle((x1, y1, x2, y2), outline=(255, 46, 46), width=4)
        label = f"fire {confidence:.2f}"
        bbox = draw.textbbox((x1, y1), label, font=font)
        label_h = bbox[3] - bbox[1] + 8
        label_w = bbox[2] - bbox[0] + 10
        y_label = max(0, y1 - label_h)
        draw.rectangle((x1, y_label, x1 + label_w, y_label + label_h), fill=(255, 46, 46))
        draw.text((x1 + 5, y_label + 4), label, fill=(255, 255, 255), font=font)

    footer = f"danger score {combined_score:.3f}"
    footer_bbox = draw.textbbox((0, 0), footer, font=font)
    draw.rectangle(
        (0, annotated.height - 28, footer_bbox[2] + 18, annotated.height),
        fill=(20, 24, 31),
    )
    draw.text((9, annotated.height - 21), footer, fill=(255, 255, 255), font=font)
    return image_to_data_url(annotated)


def clip_caption_probabilities(image: Image.Image) -> torch.Tensor:
    _, clip_model, clip_processor = load_models()
    inputs = clip_processor(
        text=FIRE_CAPTIONS,
        images=image,
        return_tensors="pt",
        padding=True,
    )
    inputs = {key: value.to(device()) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = clip_model(**inputs)
        return outputs.logits_per_image.softmax(dim=-1).detach().cpu()[0]


def analyze_image(filename: str, image: Image.Image) -> FireDetectionResult:
    yolo_model, _, _ = load_models()
    rgb_image = image.convert("RGB")

    yolo_results = yolo_model.predict(
        source=rgb_image,
        conf=YOLO_CONF_THRESHOLD,
        device=device(),
        verbose=False,
        save=False,
    )
    yolo_result = yolo_results[0]
    boxes = yolo_result.boxes

    if boxes is None or len(boxes) == 0:
        return FireDetectionResult(
            filename=filename,
            is_dangerous=False,
            yolo_detected=False,
            yolo_conf=0.0,
            clip_danger_score=0.0,
            danger_score=0.0,
            best_caption="No YOLO fire detection",
            best_caption_probability=0.0,
            annotated_image=None,
        )

    yolo_conf = float(boxes.conf.detach().cpu().max().item())
    probabilities = clip_caption_probabilities(rgb_image)
    clip_danger_score = float(probabilities[DANGEROUS_CAPTION_INDICES].sum().item())
    danger_score = yolo_conf * clip_danger_score
    best_index = int(torch.argmax(probabilities).item())
    is_dangerous = danger_score >= DANGEROUS_FIRE_THRESHOLD

    return FireDetectionResult(
        filename=filename,
        is_dangerous=is_dangerous,
        yolo_detected=True,
        yolo_conf=yolo_conf,
        clip_danger_score=clip_danger_score,
        danger_score=danger_score,
        best_caption=FIRE_CAPTIONS[best_index],
        best_caption_probability=float(probabilities[best_index].item()),
        annotated_image=draw_detections(rgb_image, boxes, danger_score) if is_dangerous else None,
    )


def analyze_uploaded_files(files) -> List[FireDetectionResult]:
    results: List[FireDetectionResult] = []

    for uploaded_file in files:
        if not uploaded_file.filename:
            continue

        filename = display_filename(uploaded_file.filename)
        if not allowed_image(filename):
            continue

        try:
            image = Image.open(uploaded_file.stream)
            results.append(analyze_image(filename, image))
        except UnidentifiedImageError:
            results.append(
                FireDetectionResult(
                    filename=filename,
                    is_dangerous=False,
                    yolo_detected=False,
                    yolo_conf=0.0,
                    clip_danger_score=0.0,
                    danger_score=0.0,
                    best_caption="",
                    best_caption_probability=0.0,
                    annotated_image=None,
                    error="Could not read this image",
                )
            )

    return results


@app.route("/", methods=["GET", "POST"])
def index():
    results: List[FireDetectionResult] = []
    result_layout = "images"
    error = None

    if request.method == "POST":
        result_layout = request.form.get("result_layout", "images")
        files = request.files.getlist("images")

        if not files or all(not file.filename for file in files):
            error = "Select at least one image file or a folder that contains images."
        else:
            try:
                results = analyze_uploaded_files(files)
                if not results:
                    error = "No supported image files were found in the selected files or folder."
            except Exception as exc:
                error = f"Detection failed: {exc}"

    dangerous_results = [result for result in results if result.is_dangerous and not result.error]
    return render_template(
        "index.html",
        results=results,
        dangerous_results=dangerous_results,
        result_layout=result_layout,
        error=error,
        threshold=DANGEROUS_FIRE_THRESHOLD,
        yolo_threshold=YOLO_CONF_THRESHOLD,
        captions=FIRE_CAPTIONS,
        device=device(),
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    host = os.environ.get("HOST", "127.0.0.1")
    print(f"Serving Dangerous Fire Detection at http://{host}:{port}")
    serve(app, host=host, port=port)
