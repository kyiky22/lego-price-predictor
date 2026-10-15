"""FastAPI service untuk prediksi harga set LEGO.
Jalankan:  uvicorn app:app --host 0.0.0.0 --port 8000
"""
import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from lego_features import FEATURE_COLUMNS, prepare_features

BASE = Path(__file__).parent
LOG_DIR = Path(os.getenv("LOG_DIR", BASE / "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

model = joblib.load(BASE / "artifacts" / "lego_price_model.joblib")
meta = json.loads((BASE / "artifacts" / "model_metadata.json").read_text())
_lock = threading.Lock()

app = FastAPI(title="LEGO Price Predictor", version=meta["model_version"])


class LegoSet(BaseModel):
    Set_ID: str = Field(..., min_length=1, examples=["75192-1"])
    Year: int = Field(..., ge=1949, le=2100)
    Theme: Optional[str] = None
    Theme_Group: Optional[str] = None
    Category: Optional[str] = None
    Packaging: Optional[str] = None
    Availability: Optional[str] = None
    Num_Instructions: Optional[float] = Field(None, ge=0)
    Pieces: Optional[float] = Field(None, ge=0)
    Minifigures: Optional[float] = Field(None, ge=0)
    Owned: Optional[float] = Field(None, ge=0)
    Rating: Optional[float] = Field(None, ge=0, le=5)
    USD_MSRP: Optional[float] = Field(None, ge=0)
    Total_Quantity: Optional[float] = Field(None, ge=0)


class Feedback(BaseModel):
    request_id: str
    actual_price_usd: float = Field(..., gt=0)


def _append(name, record):
    with _lock, open(LOG_DIR / name, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def _predict(items: List[LegoSet]):
    raw = pd.DataFrame([i.model_dump() for i in items])
    X = prepare_features(raw, meta["current_year"])
    pred_log = model.predict(X)
    out = []
    for (_, row), pl in zip(X.iterrows(), pred_log):
        rid = uuid.uuid4().hex
        price = float(np.expm1(pl))
        low = float(max(np.expm1(pl + meta["residual_q10"]), 0))
        high = float(np.expm1(pl + meta["residual_q90"]))
        _append("predictions.jsonl", {
            "request_id": rid, "ts": datetime.now(timezone.utc).isoformat(),
            "model_version": meta["model_version"],
            "features": {k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v))
                         for k, v in row.items()},
            "pred_log": float(pl), "pred_price_usd": price})
        out.append({"request_id": rid, "predicted_price_usd": round(price, 2),
                    "interval_80pct_usd": [round(low, 2), round(high, 2)],
                    "model_version": meta["model_version"]})
    return out


@app.get("/health")
def health():
    return {"status": "ok", "model_version": meta["model_version"],
            "trained_at": meta["trained_at"]}


@app.post("/predict")
def predict(item: LegoSet):
    return _predict([item])[0]


@app.post("/predict/batch")
def predict_batch(items: List[LegoSet]):
    if not 0 < len(items) <= 1000:
        raise HTTPException(422, "Batch harus berisi 1-1000 item")
    return _predict(items)


@app.post("/feedback")
def feedback(fb: Feedback):
    """Kirim harga aktual setelah diketahui -> dipakai untuk monitoring performa."""
    _append("actuals.jsonl", {"request_id": fb.request_id, "actual_price_usd": fb.actual_price_usd,
                              "ts": datetime.now(timezone.utc).isoformat()})
    return {"status": "recorded"}
