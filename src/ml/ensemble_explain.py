"""Plain-language ensemble call explanations (counterfactual attribution)."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

_NEUTRAL: dict[str, float] = {
    "financial_pred_proba": 0.5,
    "lstm_confidence": 0.0,
    "news_tfidf_pred_proba": 0.5,
    "tfidf_confidence": 0.0,
    "news_embeddings_pred_proba": 0.5,
    "emb_confidence": 0.0,
    "has_news": 0.0,
    "n_headlines": 0.0,
    "news_tfidf_x_has_news": 0.5,
    "news_emb_x_has_news": 0.5,
    "lstm_x_agree": 0.5,
    "spy_return_5d": 0.0,
    "all_agree": 0.0,
}


def _row_to_matrix(row: pd.Series) -> np.ndarray:
    from scripts.build_ensemble import META_FEATURES, _add_interaction_features, _ensure_derived_features

    df = pd.DataFrame([row.to_dict()])
    df = _ensure_derived_features(df)
    df = _add_interaction_features(df)
    for c in META_FEATURES:
        if c not in df.columns:
            df[c] = 0.0
    return df[META_FEATURES].to_numpy(dtype=np.float64)


def _score_matrix(model: Any, X: np.ndarray, temperature: float) -> float:
    p = float(model.predict_proba(X)[0, 1])
    eps = 1e-6
    p = float(np.clip(p, eps, 1 - eps))
    if abs(temperature - 1.0) < 1e-6:
        return p
    logits = np.log(p / (1 - p))
    return float(1.0 / (1.0 + np.exp(-logits / max(temperature, 1e-3))))


def _apply_neutral(row: pd.Series, feature: str) -> pd.Series:
    out = row.copy()
    out[feature] = _NEUTRAL[feature]
    if feature == "financial_pred_proba":
        out["financial_confidence"] = 0.0
        out["lstm_confidence"] = 0.0
    if feature == "lstm_confidence":
        out["lstm_confidence"] = 0.0
        out["financial_confidence"] = 0.0
    if feature == "news_tfidf_pred_proba":
        out["news_tfidf_confidence"] = 0.0
        out["tfidf_confidence"] = 0.0
    if feature == "news_embeddings_pred_proba":
        out["news_embeddings_confidence"] = 0.0
        out["emb_confidence"] = 0.0
    if feature == "all_agree":
        out["all_agree"] = 0.0
    if feature == "has_news":
        out["has_news"] = 0.0
        out["n_headlines"] = 0.0
    return out


def _lean_display(proba: float) -> tuple[str, str]:
    pct = proba * 100
    if proba >= 0.5:
        return f"{pct:.0f}% UP", "UP"
    return f"{pct:.0f}% UP ({100 - pct:.0f}% down)", "DOWN"


def explain_ensemble_row(row: pd.Series, meta_payload: dict) -> dict[str, Any]:
    """Build structured explanation for one (ticker, date) ensemble call."""
    model = meta_payload.get("meta") or meta_payload.get("model")
    if model is None:
        return {"error": "no meta model"}

    temperature = float(meta_payload.get("temperature", 1.0))
    importances = dict(meta_payload.get("importances") or [])

    X = _row_to_matrix(row)
    base_p = _score_matrix(model, X, temperature)
    direction = "UP" if base_p >= 0.5 else "DOWN"
    confidence = abs(base_p - 0.5) * 2.0

    lstm_p = float(row.get("financial_pred_proba", 0.5))
    tfidf_p = float(row.get("news_tfidf_pred_proba", 0.5))
    emb_p = float(row.get("news_embeddings_pred_proba", 0.5))
    lstm_conf = float(row.get("financial_confidence", abs(lstm_p - 0.5) * 2))
    agree = float(row.get("all_agree", 0)) >= 0.5
    spy = float(row.get("spy_return_5d", 0.0))

    simple_avg = (lstm_p + tfidf_p + emb_p) / 3.0

    route = meta_payload.get("route")
    has_news = float(row.get("has_news", 0)) >= 0.5

    base_votes = [
        {"model": "lstm", "label": "Price", "proba": lstm_p, "active": True, **_vote_fields(lstm_p)},
    ]
    if has_news:
        base_votes.extend([
            {"model": "tfidf", "label": "Keywords", "proba": tfidf_p, "active": True, **_vote_fields(tfidf_p)},
            {"model": "embeddings", "label": "FinBERT", "proba": emb_p, "active": True, **_vote_fields(emb_p)},
        ])
    else:
        base_votes.extend([
            {"model": "tfidf", "label": "Keywords", "proba": tfidf_p, "active": False,
             "display": "No headlines", "direction": "N/A"},
            {"model": "embeddings", "label": "FinBERT", "proba": emb_p, "active": False,
             "display": "No headlines", "direction": "N/A"},
        ])
    active_votes = [v for v in base_votes if v.get("active", True)]
    up_votes = sum(1 for v in active_votes if v.get("direction") == "UP")
    down_votes = sum(1 for v in active_votes if v.get("direction") == "DOWN")

    # Counterfactual drivers (interpretable subset)
    driver_specs = [
        ("all_agree", "Models agree"),
        ("financial_pred_proba", "Price model lean"),
        ("lstm_confidence", "Price conviction"),
        ("news_tfidf_pred_proba", "Keyword headlines"),
        ("news_embeddings_pred_proba", "FinBERT headlines"),
        ("has_news", "Headlines present"),
        ("n_headlines", "Headline count"),
        ("spy_return_5d", "SPY 5-day return"),
        ("news_tfidf_x_has_news", "Keywords × headlines"),
        ("news_emb_x_has_news", "FinBERT × headlines"),
        ("lstm_x_agree", "Price × agreement"),
    ]
    drivers: list[dict[str, Any]] = []
    for feat, label in driver_specs:
        neutral_row = _apply_neutral(row, feat)
        cf_p = _score_matrix(model, _row_to_matrix(neutral_row), temperature)
        effect = base_p - cf_p
        val = float(row.get(feat, _NEUTRAL.get(feat, 0)))
        if feat == "lstm_confidence":
            val = lstm_conf
        drivers.append({
            "feature": feat,
            "label": label,
            "value": val,
            "effect": effect,
            "direction": "up" if effect > 0.003 else "down" if effect < -0.003 else "neutral",
        })
    drivers.sort(key=lambda d: abs(d["effect"]), reverse=True)

    # Plain-English summary (kept short for UI)
    if direction == "DOWN":
        headline = f"Final call is **DOWN** ({base_p*100:.0f}% UP) — below the 50% threshold."
    else:
        headline = f"Final call is **UP** ({base_p*100:.0f}% UP) — above the 50% threshold."

    bullets: list[str] = [headline]
    if route == "no_news":
        bullets.append(
            "No headlines before 4 PM ET — the **price-only combiner** used the LSTM score and market context."
        )
    elif route == "has_news":
        bullets.append(
            "Headlines present — the **news-tuned combiner** weighted price, keywords, and FinBERT."
        )
    if has_news and up_votes and down_votes:
        bullets.append(
            f"Active models split **{up_votes} up / {down_votes} down**; simple avg **{simple_avg*100:.0f}% UP** "
            f"→ ensemble **{base_p*100:.0f}% UP**."
        )
    elif has_news and abs(simple_avg - base_p) > 0.03:
        bullets.append(
            f"Simple avg **{simple_avg*100:.0f}% UP** → ensemble adjusted to **{base_p*100:.0f}% UP**."
        )

    news_imp = (
        importances.get("news_tfidf_pred_proba", 0)
        + importances.get("news_embeddings_pred_proba", 0)
    )
    news_note = None
    if news_imp < 1e-6:
        news_note = (
            "Headlines did not move the combiner on this call — training found they add "
            "little globally, so keyword/meaning rows show as no impact."
        )

    conf_label = "Low" if confidence < 0.25 else "Moderate" if confidence < 0.45 else "Strong"
    conf_help = (
        "Confidence is distance from 50/50."
    )

    return {
        "ensemble_proba": base_p,
        "ensemble_direction": direction,
        "ensemble_confidence": confidence,
        "confidence_label": conf_label,
        "confidence_help": conf_help,
        "simple_average_proba": simple_avg,
        "summary": bullets[0],
        "bullets": bullets,
        "base_votes": base_votes,
        "drivers": drivers,
        "news_weight_note": news_note,
        "models_disagree": has_news and not agree,
        "ensemble_route": route,
        "has_news": has_news,
    }


def _vote_fields(proba: float) -> dict[str, str]:
    display, direction = _lean_display(proba)
    return {"display": display, "direction": direction}
