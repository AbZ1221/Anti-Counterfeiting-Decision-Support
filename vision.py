# === Type annotations ===
# Postpone evaluation of type hints instead of resolving them when functions are defined.
from __future__ import annotations

# === Python standard library: filenames, configuration and result data ===
# Generate the SHA-256 suffix used in saved heatmap filenames.
import hashlib
# Read model settings and scoring weights from the JSON configuration file.
import json
# dataclass generates result-container methods; asdict converts its fields to a dictionary.
from dataclasses import asdict, dataclass
# Build filesystem paths, resolve image locations and create output directories.
from pathlib import Path

# === Image processing and numerical calculations ===
# Work with pixel arrays and calculate histograms, similarities and quality scores.
import numpy as np
# Pillow image utilities:
# Image: open images, convert color modes, resize and composite the heatmap overlay.
# ImageChops: calculate pixel-by-pixel differences between listing and reference images.
# ImageEnhance: increase contrast to make image differences easier to see.
# ImageFilter: apply Gaussian blur to smooth the heatmap.
# ImageOps: correct EXIF orientation and fit images to a common size.
from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ImageOps

# Anchor configuration, image paths and generated outputs to this module's folder.
ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "cv_config.json"
CACHE_DIR = ROOT / "outputs" / "vision"


@dataclass
class VisionAssessment:
    # Scores range from 0 to 1: higher similarity/quality is better; higher risk is worse.
    # This result also carries supporting evidence for display in the application.
    visual_risk: float
    embedding_similarity: float
    structural_similarity: float
    ocr_similarity: float
    quality_score: float
    detected_text: str
    object_detections: list[dict]
    backend: str
    heatmap_path: str
    reasons: list[str]

    def as_dict(self):
        # Convert the dataclass into a plain dictionary for downstream consumers.
        return asdict(self)


def _resolve(path: str) -> Path:
    # Preserve absolute paths; interpret relative paths from the project folder.
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _normalise(image: Image.Image, size=(384, 384)) -> Image.Image:
    # Apply camera orientation metadata and use a consistent three-channel format.
    image = ImageOps.exif_transpose(image).convert("RGB")
    # Resize and center-crop so both images have matching dimensions for comparison.
    return ImageOps.fit(image, size, method=Image.Resampling.LANCZOS)


def _histogram_similarity(a: Image.Image, b: Image.Image) -> float:
    # Compare RGB color distributions. This fallback does not capture object layout.
    ha = np.asarray(a.histogram(), dtype=float)
    hb = np.asarray(b.histogram(), dtype=float)
    ha /= max(ha.sum(), 1); hb /= max(hb.sum(), 1)
    # Cosine similarity measures histogram alignment; epsilon prevents division by zero.
    return float(np.clip(np.dot(ha, hb) / (np.linalg.norm(ha) * np.linalg.norm(hb) + 1e-9), 0, 1))


def _structural_similarity(a: Image.Image, b: Image.Image) -> float:
    # Multi-scale luminance comparison; deterministic fallback when OpenCV/SSIM is unavailable.
    scores = []
    for size in (16, 32, 64):
        # Compare grayscale brightness at several resolutions, scaled to [0, 1].
        aa = np.asarray(a.convert("L").resize((size, size)), dtype=float) / 255
        bb = np.asarray(b.convert("L").resize((size, size)), dtype=float) / 255
        # Convert average pixel difference into similarity (identical images score 1).
        scores.append(1 - np.mean(np.abs(aa - bb)))
    return float(np.clip(np.mean(scores), 0, 1))


def _quality_score(image: Image.Image) -> float:
    # Estimate quality using neighboring-pixel changes (sharpness) and brightness spread.
    gray = np.asarray(image.convert("L"), dtype=float)
    gx = np.abs(np.diff(gray, axis=1)).mean()
    gy = np.abs(np.diff(gray, axis=0)).mean()
    # These scaling constants and weights are heuristics, not a calibrated quality model.
    sharpness = np.clip((gx + gy) / 32, 0, 1)
    contrast = np.clip(gray.std() / 64, 0, 1)
    return float(0.55 * sharpness + 0.45 * contrast)


def _save_heatmap(listing: Image.Image, reference: Image.Image, listing_id: str) -> str:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # Highlight pixel differences, boosting contrast and smoothing the visualization.
    diff = ImageChops.difference(listing, reference).convert("L")
    diff = ImageEnhance.Contrast(diff).enhance(2.5).filter(ImageFilter.GaussianBlur(3))
    arr = np.asarray(diff, dtype=np.uint8)
    # Red-yellow anomaly overlay; brighter regions differ more from the verified reference.
    heat = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)
    # The final axis holds red, green, blue and alpha (opacity) channels.
    heat[..., 0] = np.minimum(255, arr * 2)
    heat[..., 1] = np.where(arr > 100, arr, 0)
    heat[..., 3] = np.clip(arr, 0, 190)
    overlay = Image.alpha_composite(listing.convert("RGBA"), Image.fromarray(heat, "RGBA"))
    # Create a repeatable filename suffix from the listing ID and its pixel sum.
    # This is not a hash of the full image contents, so different images can share it.
    digest = hashlib.sha256((listing_id + str(np.asarray(listing).sum())).encode()).hexdigest()[:10]
    path = CACHE_DIR / f"{listing_id}_{digest}_heatmap.png"
    overlay.save(path)
    return str(path)


class AdvancedVisionEngine:
    """Reference-based CV engine with optional CLIP, OCR and YOLO integrations."""

    def __init__(self, config_path: Path = CONFIG_PATH):
        # Load scoring settings, then attempt to initialize the optional model backends.
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        self.clip_model = self.clip_processor = None
        self.yolo_model = None
        self.backend = "multiscale-reference-fallback"
        self._load_optional_clip()
        self._load_optional_yolo()

    def _load_optional_clip(self):
        # CLIP encodes images as feature vectors for a learned visual comparison.
        # Loading pretrained weights may download them if they are not cached locally.
        try:
            # === Optional CLIP dependencies: imported only when loading this backend ===
            # PyTorch runs tensor inference and disables gradient tracking via no_grad().
            import torch
            # CLIPModel produces image features; CLIPProcessor prepares model input images.
            from transformers import CLIPModel, CLIPProcessor
            name = self.config["embedding_model"]
            self.clip_model = CLIPModel.from_pretrained(name)
            self.clip_processor = CLIPProcessor.from_pretrained(name)
            # Use inference behavior rather than training behavior (for example, dropout).
            self.clip_model.eval()
            self.torch = torch
            self.backend = f"CLIP:{name}"
        except Exception:
            # The fully local fallback keeps the demonstration runnable offline.
            self.clip_model = self.clip_processor = None

    def _load_optional_yolo(self):
        # Object detection requires both an enabled setting and an existing weights file.
        if not self.config.get("enable_yolo", False):
            return
        weights = _resolve(self.config.get("yolo_weights", "models/logo_detector.pt"))
        if not weights.exists():
            return
        try:
            # === Optional object detection dependency ===
            # Load YOLO weights and predict labeled bounding boxes in listing images.
            from ultralytics import YOLO
            self.yolo_model = YOLO(str(weights))
        except Exception:
            self.yolo_model = None

    def _detect_objects(self, image_path: str) -> list[dict]:
        if self.yolo_model is None:
            return []
        results = self.yolo_model.predict(str(_resolve(image_path)), verbose=False)[0]
        detections = []
        for box in results.boxes:
            # Return readable labels, confidence scores and [x1, y1, x2, y2] pixel boxes.
            cls_id = int(box.cls.item())
            detections.append({
                "class": results.names[cls_id], "confidence": round(float(box.conf.item()), 4),
                "xyxy": [round(float(v), 1) for v in box.xyxy[0].tolist()],
            })
        return detections

    def _embedding_similarity(self, a: Image.Image, b: Image.Image) -> float:
        if self.clip_model is None:
            return _histogram_similarity(a, b)
        inputs = self.clip_processor(images=[a, b], return_tensors="pt")
        # Process both images together; disable gradient tracking to save inference memory.
        with self.torch.no_grad():
            output = self.clip_model.get_image_features(**inputs)
            # transformers releases may return either a Tensor or a structured
            # BaseModelOutputWithPooling. Normalise both formats here.
            if hasattr(output, "norm"):
                features = output
            elif getattr(output, "image_embeds", None) is not None:
                features = output.image_embeds
            elif getattr(output, "pooler_output", None) is not None:
                features = output.pooler_output
            elif isinstance(output, (tuple, list)) and len(output):
                features = output[0]
            else:
                raise TypeError(f"Unsupported CLIP image-feature output: {type(output).__name__}")
            # Unit-length vectors make their dot product equal to cosine similarity.
            features = features / features.norm(dim=-1, keepdim=True)
        return float(np.clip((features[0] @ features[1]).item(), 0, 1))

    def _ocr(self, image: Image.Image) -> str:
        if not self.config.get("enable_ocr", True):
            return ""
        try:
            # === Optional text recognition dependency ===
            # Python interface to the Tesseract OCR executable for extracting package text.
            import pytesseract
            # Page segmentation mode 6 treats the image as one uniform block of text.
            return pytesseract.image_to_string(image, config="--psm 6").strip()
        except Exception:
            # Missing Tesseract or an OCR failure leaves text evidence unavailable.
            return ""

    def _ocr_similarity(self, text: str, expected: str) -> float:
        if not text:
            return 0.5  # unknown, not automatically adverse
        # Match unique, case-insensitive whitespace-separated words longer than 2 characters.
        # Punctuation is retained; the score is the fraction of expected words found.
        observed = {w.lower() for w in text.split() if len(w) > 2}
        target = {w.lower() for w in expected.split() if len(w) > 2}
        return len(observed & target) / max(len(target), 1)

    def analyse(self, listing_id: str, image_path: str, reference_path: str, expected_text: str) -> VisionAssessment:
        # Prepare the listing and verified reference using the same image transformations.
        image = _normalise(Image.open(_resolve(image_path)))
        reference = _normalise(Image.open(_resolve(reference_path)))
        # Collect visual, text and quality signals independently.
        embedding = self._embedding_similarity(image, reference)
        structural = _structural_similarity(image, reference)
        quality = _quality_score(image)
        detected_text = self._ocr(image)
        # Detections are supporting evidence only; they do not contribute to the risk score.
        object_detections = self._detect_objects(image_path)
        ocr_similarity = self._ocr_similarity(detected_text, expected_text)
        weights = self.config["weights"]
        # Invert each similarity/quality score into an anomaly score, then apply weights.
        # The result is a heuristic risk indicator, not a probability of counterfeiting.
        risk = (
            weights["embedding_anomaly"] * (1 - embedding)
            + weights["structural_anomaly"] * (1 - structural)
            + weights["ocr_anomaly"] * (1 - ocr_similarity)
            + weights["quality_anomaly"] * (1 - quality)
        )
        # Explain weak signals using fixed thresholds, independently of the weighted total.
        reasons = []
        if embedding < 0.75: reasons.append("Deep visual similarity to the verified reference is low.")
        if structural < 0.75: reasons.append("Packaging structure or layout differs from the reference.")
        if ocr_similarity < 0.50: reasons.append("Extracted package text does not adequately match expected brand text.")
        if quality < 0.30: reasons.append("Image quality is too low for reliable visual verification.")
        if not reasons: reasons.append("No strong visual anomaly was found against the supplied reference image.")
        # Bound the final risk, round display values and save a visual difference overlay.
        return VisionAssessment(
            visual_risk=round(float(np.clip(risk, 0, 1)), 4),
            embedding_similarity=round(embedding, 4), structural_similarity=round(structural, 4),
            ocr_similarity=round(ocr_similarity, 4), quality_score=round(quality, 4),
            detected_text=detected_text, object_detections=object_detections, backend=self.backend,
            heatmap_path=_save_heatmap(image, reference, listing_id), reasons=reasons,
        )
