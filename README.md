# Antigravity Order Image Parser

A production-ready **FastAPI** service that accepts a handwritten order image and returns a structured JSON order object — powered by **Gemini 2.5 Flash**.

---

## 📂 Project Structure

```
parsing/
├── main.py            ← FastAPI application
├── requirements.txt   ← Python dependencies
├── Procfile           ← Render start command
└── README.md
```

---

## 🚀 API Endpoints

### `GET /`
Health check.
```json
{ "status": "ok", "service": "Antigravity Order Image Parser" }
```

### `POST /parse-order`
Upload a handwritten order image → receive structured JSON.

**Request:** `multipart/form-data`
| Field | Type | Description |
|-------|------|-------------|
| `file` | file | JPG / PNG / PDF image |

**Response:**
```json
{
  "customer_intent": "order",
  "customer_name": "John Doe",
  "order_date": "2026-03-26",
  "items": [
    {
      "product_name": "MS Angle 40x40x5",
      "quantity": "10",
      "unit": "pcs",
      "notes": "galvanised",
      "confidence": 0.92
    }
  ],
  "additional_notes": "Deliver by Friday",
  "extraction_warnings": []
}
```

---

## ⚙️ Environment Variables

Set these on Render:

| Variable | Description |
|----------|-------------|
| `GEMINI_API_KEY` | Your API key |
| `GEMINI_BASE_URL` | `https://apidev.navigatelabsai.com` |
| `GEMINI_MODEL` | `gemini-2.5-flash-preview-04-17` |

---

## 🌐 Deploy on Render

1. Push the `parsing/` folder contents to a GitHub repo.
2. Go to [render.com](https://render.com) → **New Web Service**.
3. Connect your repo, set:
   - **Root Directory:** *(leave blank if repo root = parsing/)*
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Add the three environment variables above.
5. Click **Deploy**.

---

## 🧪 Test Locally

```bash
cd parsing
pip install -r requirements.txt
set GEMINI_API_KEY=your-key-here
uvicorn main:app --reload
```

Then POST to `http://localhost:8000/parse-order` with an image file.

Interactive docs: `http://localhost:8000/docs`
