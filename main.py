import os
import json
import base64
import logging
import re
from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ImageEnhance
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from openai import OpenAI

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("OrderParser")

# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Antigravity Order Image Parser",
    description="Upload a handwritten order image → get structured JSON",
    version="1.0.0",
)

# ── Gemini client (OpenAI-compatible) ─────────────────────────────────────
API_KEY      = os.environ.get("GEMINI_API_KEY", "")
BASE_URL     = os.environ.get("GEMINI_BASE_URL", "https://apidev.navigatelabsai.com")
MODEL_NAME   = os.environ.get("GEMINI_MODEL",   "gemini-2.5-flash-preview-04-17")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ── System prompt ─────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
You are an expert industrial order-processing AI for Antigravity Hardware Supply Company.
Carefully analyse the handwritten order sheet — which may include rough sketches, arrows,
crossed-out items, abbreviations, and diagrams — and extract structured order data.

STRICT OUTPUT RULES:
1. Respond ONLY with a valid JSON object — no markdown fences, no explanation.
2. Follow this exact schema:
{
  "customer_intent": "order | inquiry | unclear",
  "customer_name": "<name if visible, else null>",
  "order_date": "<date if visible, else null>",
  "items": [
    {
      "product_name": "<exact or best-guess product name>",
      "quantity": "<numeric quantity or null if unclear>",
      "unit": "<unit e.g. pcs, kg, m, box — or null>",
      "notes": "<any special notes, specs, or modifiers>",
      "confidence": <float 0.0–1.0>
    }
  ],
  "additional_notes": "<general order notes, delivery instructions, etc.>",
  "extraction_warnings": ["<list any unclear regions or assumptions made>"]
}
3. Crossed-out items are CANCELLED — do NOT include them.
4. Arrows pointing to quantities/products must be resolved and linked.
5. Common hardware abbreviations: pcs=pieces, m=metres, kg=kilograms, dz=dozen.
6. If confidence < 0.5, still include the item — just set confidence accurately.
"""


# ── Image helpers ──────────────────────────────────────────────────────────
def load_image(file_bytes: bytes, filename: str) -> Image.Image:
    """Load image from bytes; handles JPG, PNG, PDF (first page)."""
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext == "pdf":
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise HTTPException(status_code=500, detail="PDF support requires PyMuPDF. Install it.")
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        page = doc[0]
        pix = page.get_pixmap(dpi=200)
        return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return Image.open(BytesIO(file_bytes)).convert("RGB")


def preprocess_image(pil_image: Image.Image, target_min_width: int = 1200) -> Image.Image:
    """Grayscale → contrast enhance → bilateral denoise → adaptive threshold → RGB."""
    img = pil_image.copy()
    w, h = img.size

    # Upscale if too small
    if w < target_min_width:
        scale = target_min_width / w
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    gray = img.convert("L")
    gray = ImageEnhance.Contrast(gray).enhance(2.0)
    gray = ImageEnhance.Sharpness(gray).enhance(2.5)

    cv_img = np.array(gray)
    cv_img = cv2.bilateralFilter(cv_img, 9, 75, 75)
    thresh = cv2.adaptiveThreshold(
        cv_img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    return Image.fromarray(thresh).convert("RGB")


def pil_to_base64(pil_image: Image.Image, fmt: str = "JPEG") -> str:
    buf = BytesIO()
    pil_image.save(buf, format=fmt, quality=95)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── Gemini call ────────────────────────────────────────────────────────────
def call_gemini(original: Image.Image, processed: Image.Image) -> str:
    """Send both original + preprocessed images to Gemini; return raw text."""
    orig_b64 = pil_to_base64(original)
    proc_b64 = pil_to_base64(processed)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{orig_b64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{proc_b64}"}},
                    {"type": "text",      "text": "Extract all order information. Return valid JSON only."},
                ],
            },
        ],
        temperature=0.1,
        max_tokens=4096,
    )
    return response.choices[0].message.content


# ── JSON parser ────────────────────────────────────────────────────────────
def parse_response(raw: str) -> dict:
    """Parse model output → dict, with fallbacks for imperfect JSON."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"```\s*$",          "", text, flags=re.MULTILINE)
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass

    logger.error("Failed to parse JSON from model output.")
    return {
        "customer_intent": "unclear",
        "customer_name": None,
        "order_date": None,
        "items": [],
        "additional_notes": "PARSE ERROR — model did not return valid JSON",
        "extraction_warnings": ["Failed to parse model output"],
    }


def validate_order(order: dict) -> dict:
    """Fill missing keys with safe defaults."""
    order.setdefault("customer_intent", "unclear")
    order.setdefault("customer_name",   None)
    order.setdefault("order_date",      None)
    order.setdefault("items",           [])
    order.setdefault("additional_notes", "")
    order.setdefault("extraction_warnings", [])
    for item in order["items"]:
        item.setdefault("product_name", "Unknown")
        item.setdefault("quantity",     None)
        item.setdefault("unit",         None)
        item.setdefault("notes",        "")
        item.setdefault("confidence",   0.5)
        item["confidence"] = max(0.0, min(1.0, float(item["confidence"])))
    return order


# ── Routes ─────────────────────────────────────────────────────────────────
@app.get("/")
def health():
    return {"status": "ok", "service": "Antigravity Order Image Parser"}


@app.post("/parse-order")
async def parse_order(file: UploadFile = File(...)):
    """
    Upload a handwritten order image (JPG / PNG / PDF).
    Returns a structured JSON order object.
    """
    allowed = {"jpg", "jpeg", "png", "pdf"}
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}. Use JPG, PNG, or PDF.")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    logger.info(f"Received file: {file.filename}  ({len(file_bytes)} bytes)")

    try:
        original  = load_image(file_bytes, file.filename)
        processed = preprocess_image(original)
        logger.info(f"Image preprocessed → size={processed.size}")

        raw = call_gemini(original, processed)
        logger.info(f"Gemini responded with {len(raw)} chars")

        order = validate_order(parse_response(raw))
        return JSONResponse(content=order)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error during order parsing")
        raise HTTPException(status_code=500, detail=str(e))
