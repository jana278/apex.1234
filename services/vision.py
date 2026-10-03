import os
import re
import io
import base64
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

# 1. Device Setup Guard
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VISION_MODEL = "dima806/car_models_image_detection"

car_vision_model = None
img_processor = None

try:
    print(f"[*] Loading Vision Transformer: '{VISION_MODEL}' on {DEVICE}...")
    img_processor = AutoImageProcessor.from_pretrained(VISION_MODEL)
    car_vision_model = AutoModelForImageClassification.from_pretrained(VISION_MODEL).to(DEVICE)
    car_vision_model.eval()
except Exception as e:
    print(f"[!] Critical Vision Load Failure: {e}")

# 2. Comprehensive Brand Normalization Mapping
BRAND_RENAME = {
    "mercedes-benz": "mercedes",
    "vw": "volkswagen",
    "chevy": "chevrolet",
    "alfa-romeo": "alfa romeo",
    "land-rover": "land rover",
    "aston-martin": "aston martin"
}

KNOWN_MULTIWORD_BRANDS = [
    "alfa romeo", "land rover", "aston martin", "mercedes benz", "rolls royce"
]

def prepare_image(file_storage):
    try:
        raw = file_storage.read()
        if not raw or len(raw) == 0:
            return None, "INVALID_IMAGE: Uploaded file is empty."
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        return img, None
    except Exception as ex:
        return None, f"INVALID_IMAGE: Image decoding failed: {ex}"

def predict_top5_cars_from_image(image_input, top_k: int = 5):
    if car_vision_model is None or img_processor is None:
        return []

    try:
        if isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        elif isinstance(image_input, bytes):
            image = Image.open(io.BytesIO(image_input)).convert("RGB")
        else:
            return []

        inputs = img_processor(images=image, return_tensors="pt").to(DEVICE)
        
        with torch.inference_mode():
            logits = car_vision_model(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=-1)[0]

        top_k_val = min(top_k, len(car_vision_model.config.id2label))
        top_probs, top_indices = torch.topk(probs, k=top_k_val)

        candidates = []
        for p, idx in zip(top_probs, top_indices):
            raw_label = car_vision_model.config.id2label[idx.item()].replace("_", " ").strip()
            label_lower = raw_label.lower()

            detected_brand = None
            detected_model = ""

            for mb in KNOWN_MULTIWORD_BRANDS:
                if label_lower.startswith(mb):
                    detected_brand = BRAND_RENAME.get(mb, mb)
                    detected_model = raw_label[len(mb):].strip()
                    break

            if not detected_brand:
                tokens = raw_label.split()
                first_token = tokens[0].lower()
                detected_brand = BRAND_RENAME.get(first_token, first_token)
                detected_model = " ".join(tokens[1:]) if len(tokens) > 1 else raw_label

            candidates.append({
                "label": raw_label,
                "confidence": round(float(p.item()) * 100, 2),
                "brand": detected_brand.capitalize(),
                "model": detected_model.capitalize()
            })

        return candidates
    except Exception as exc:
        print(f"[!] Vision Inference Runtime Exception: {exc}")
        return []

def classify_car_image(image_input):
    candidates = predict_top5_cars_from_image(image_input, top_k=5)
    if not candidates:
        return ""
    labels_combined = " ".join([c['label'].lower() for c in candidates])
    selected_brand = candidates[0]['brand']
    selected_model = candidates[0]['model']
    if any(k in labels_combined for k in ["subaru", "brz", "toyota", "gr86", "gt86", "86"]):
        for c in candidates:
            if any(k in c['label'].lower() for k in ["subaru", "brz", "toyota", "gr86", "gt86", "86"]):
                selected_brand = c['brand']
                selected_model = c['model']
                break
    clean_model = re.sub(r"\b(class|series|sedan|suv|coupe)\b", "", selected_model, flags=re.IGNORECASE).strip()
    return f"{selected_brand} {clean_model}".strip()

def detect_vehicle_rois(img):
    w, h = img.size
    return {
        "is_car": True,
        "rois": [{"crop_id": 0, "bbox": [0, 0, w, h], "confidence": 0.99}]
    }

def extract_visual_embedding(img):
    return [0.0] * 100

def crop_to_bytes(img, bbox):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return buf.getvalue()

def crop_to_b64(img, bbox):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=82)
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"

def run_vision_pipeline(pil_img, img_bytes, target_bbox=None):
    label = classify_car_image(pil_img)
    candidates = predict_top5_cars_from_image(pil_img, top_k=5)
    
    if not label or not candidates:
        return {"vehicle_detected": False}
        
    top = candidates[0]
    make = top["brand"]
    model = top["model"]
    conf = top["confidence"] / 100.0
    
    alternatives = []
    for alt in candidates[1:]:
        alternatives.append({
            "label": alt["label"],
            "make": alt["brand"],
            "model": alt["model"],
            "generation": None,
            "body_type": None,
            "confidence": alt["confidence"]
        })
        
    return {
        "vehicle_detected": True,
        "identification": {
            "make": make,
            "model": model,
            "generation": None,
            "year_estimate": None,
            "body_type": None,
            "trim": None
        },
        "confidence": {
            "overall": conf,
            "make": conf,
            "model": conf,
            "generation": 0.0,
            "year": 0.0,
            "trim": 0.0
        },
        "visual_evidence": [],
        "ocr_evidence": [],
        "alternatives": alternatives,
        "needs_confirmation": conf < 0.80,
        "debug_info": {"engine": "local_pytorch_vit", "provider": "Apex Vision"}
    }
