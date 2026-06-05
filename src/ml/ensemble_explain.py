"""Plain-language ensemble call explanations.

Attribution uses exact Shapley values (see ``_build_drivers``): each displayed
signal's contribution is its fair share of the gap between a "typical day"
baseline (every signal at its route-median) and the actual call. Contributions
sum exactly to ``final - baseline``, so the UI waterfall can never contradict the
call direction. Coalitions are scored from the raw row with only the baseline
members swapped to their median, so the full coalition reproduces the real call
(including HGB's native NaN handling for missing news scores).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _row_to_matrix(row: pd.Series, features: list[str] | None = None) -> np.ndarray:
    from scripts.build_ensemble import META_FEATURES, _add_interaction_features, _ensure_derived_features

    df = pd.DataFrame([row.to_dict()])
    df = _ensure_derived_features(df)
    df = _add_interaction_features(df)
    cols = features or META_FEATURES
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0
    return df[cols].to_numpy(dtype=np.float64)


def _temp_adjust(p: np.ndarray, temperature: float) -> np.ndarray:
    eps = 1e-6
    p = np.clip(p, eps, 1 - eps)
    if abs(temperature - 1.0) < 1e-6:
        return p
    logits = np.log(p / (1 - p))
    return 1.0 / (1.0 + np.exp(-logits / max(temperature, 1e-3)))


def _score_matrix(model: Any, X: np.ndarray, temperature: float) -> float:
    p = float(model.predict_proba(X)[0, 1])
    return float(_temp_adjust(np.array([p]), temperature)[0])


def _score_rows(model: Any, rows: list[dict], features: list[str], temperature: float) -> np.ndarray:
    """Score many feature rows in one predict_proba call (derived feats applied once)."""
    from scripts.build_ensemble import _add_interaction_features, _ensure_derived_features

    df = pd.DataFrame(rows)
    df = _add_interaction_features(_ensure_derived_features(df))
    for c in features:
        if c not in df.columns:
            df[c] = 0.0
    raw = model.predict_proba(df[features].to_numpy(dtype=np.float64))[:, 1]
    return _temp_adjust(raw, temperature)


def _safe_proba(value: Any, default: float = 0.5) -> float:
    """Coerce a probability cell to float, mapping NaN/None to a neutral 0.5."""
    try:
        p = float(value)
    except (TypeError, ValueError):
        return default
    return default if pd.isna(p) else p


def _lean_display(proba: float) -> tuple[str, str]:
    pct = proba * 100
    if proba >= 0.5:
        return f"{pct:.0f}% UP", "UP"
    return f"{pct:.0f}% UP ({100 - pct:.0f}% down)", "DOWN"


def explain_ensemble_row(
    row: pd.Series, meta_payload: dict, background: pd.DataFrame | None = None
) -> dict[str, Any]:
    """Build structured explanation for one (ticker, date) ensemble call.

    ``background`` should be the prediction rows on the same route (has_news vs
    no_news); their per-feature medians anchor the Shapley attribution.
    """
    model = meta_payload.get("meta") or meta_payload.get("model")
    if model is None:
        return {"error": "no meta model"}

    features: list[str] = list(meta_payload.get("features") or [])
    if not features:
        from scripts.build_ensemble import META_FEATURES
        features = list(META_FEATURES)

    temperature = float(meta_payload.get("temperature", 1.0))
    importances = dict(meta_payload.get("importances") or [])

    X = _row_to_matrix(row, features)
    base_p = _score_matrix(model, X, temperature)
    direction = "UP" if base_p >= 0.5 else "DOWN"
    confidence = abs(base_p - 0.5) * 2.0

    # Guard against missing/NaN base-model scores (e.g. a news row where a news
    # model couldn't produce a probability) — treat as a neutral 0.5 lean.
    lstm_p = _safe_proba(row.get("financial_pred_proba"))
    tfidf_p = _safe_proba(row.get("news_tfidf_pred_proba"))
    emb_p = _safe_proba(row.get("news_embeddings_pred_proba"))
    lstm_conf = float(row.get("financial_confidence", abs(lstm_p - 0.5) * 2))
    if pd.isna(lstm_conf):
        lstm_conf = abs(lstm_p - 0.5) * 2
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
            {"model": "embeddings", "label": "Sentiment", "proba": emb_p, "active": True, **_vote_fields(emb_p)},
        ])
    else:
        base_votes.extend([
            {"model": "tfidf", "label": "Keywords", "proba": tfidf_p, "active": False,
             "display": "No headlines", "direction": "N/A"},
            {"model": "embeddings", "label": "Sentiment", "proba": emb_p, "active": False,
             "display": "No headlines", "direction": "N/A"},
        ])
    active_votes = [v for v in base_votes if v.get("active", True)]
    up_votes = sum(1 for v in active_votes if v.get("direction") == "UP")
    down_votes = sum(1 for v in active_votes if v.get("direction") == "DOWN")

    # Per-call attribution via exact Shapley values over the displayed signals.
    #
    # Why not simpler? The combiner is a tiny per-route HGB whose output is
    # dominated by a regime baseline (a "typical" news day scores ~75% UP), with
    # individual signals only nudging it. Single-feature counterfactuals are
    # either non-discriminative (wash out toward the mean) or unstable (a tiny
    # input change crosses a tree split and swings 30pts). Shapley fairly splits
    # the gap between the baseline (all displayed signals at their route-median)
    # and the actual call, sums exactly, and is stable. The UI shows it as a
    # waterfall: baseline -> signed contributions -> final, so it can never
    # contradict the call.
    drivers, baseline_proba = _build_drivers(
        row, model, features, temperature, has_news, background
    )

    # base lean = simple mean of active model P(up)s — what users intuitively expect
    # flips_base_lean = ensemble direction disagrees with that mean (the confusing case)
    active_probas = [v["proba"] for v in active_votes]
    base_lean_proba = float(np.mean(active_probas)) if active_probas else simple_avg
    base_lean_dir = "UP" if base_lean_proba >= 0.5 else "DOWN"
    flips = base_lean_dir != direction

    # Signals leaning the same way as the final call (used to explain a flip).
    if direction == "DOWN":
        flip_drivers = [d for d in drivers if d["effect"] < -0.02]
    else:
        flip_drivers = [d for d in drivers if d["effect"] > 0.02]
    flip_drivers = sorted(flip_drivers, key=lambda d: abs(d["effect"]), reverse=True)[:3]

    if flips:
        names = ", ".join(d["label"] for d in flip_drivers) or "the combiner's learned regime weighting"
        disagreement_text = (
            f"Base models lean **{base_lean_dir}** ({base_lean_proba*100:.0f}% UP on average), "
            f"but the ensemble calls **{direction}** ({base_p*100:.0f}% UP). "
            f"The combiner is non-linear — it down-weighted the raw lean because of: {names}."
        )
    else:
        disagreement_text = (
            f"The ensemble call (**{direction}**, {base_p*100:.0f}% UP) agrees with the "
            f"base-model lean (**{base_lean_dir}**, {base_lean_proba*100:.0f}% UP)."
        )

    disagreement = {
        "base_lean_proba": base_lean_proba,
        "base_lean_direction": base_lean_dir,
        "ensemble_direction": direction,
        "ensemble_proba": base_p,
        "flips_base_lean": flips,
        "flip_drivers": flip_drivers,
        "explanation": disagreement_text,
    }

    # Plain-English summary (kept short for UI)
    if direction == "DOWN":
        headline = f"Final call is **DOWN** ({base_p*100:.0f}% UP) — below the 50% threshold."
    else:
        headline = f"Final call is **UP** ({base_p*100:.0f}% UP) — above the 50% threshold."

    bullets: list[str] = [headline]
    if flips:
        bullets.append(disagreement_text)
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
        "baseline_proba": baseline_proba,
        "disagreement": disagreement,
        "news_weight_note": news_note,
        "models_disagree": has_news and not agree,
        "ensemble_route": route,
        "has_news": has_news,
    }


def _vote_fields(proba: float) -> dict[str, str]:
    display, direction = _lean_display(proba)
    return {"display": display, "direction": direction}


# Signals attributed in "what weighted the call". For has-news rows we include
# news volume (n_headlines) — empirically the combiner leans on it heavily. For
# no-news rows only price + market regime vary.
_PLAYERS_NEWS = [
    ("financial_pred_proba", "Price model"),
    ("news_tfidf_pred_proba", "Keywords"),
    ("news_embeddings_pred_proba", "Sentiment"),
    ("spy_return_5d", "Market trend (5d)"),
    ("n_headlines", "News volume"),
]
_PLAYERS_NO_NEWS = [
    ("financial_pred_proba", "Price model"),
    ("spy_return_5d", "Market trend (5d)"),
]

# proba feature -> the confidence columns derived from it (kept consistent when
# we swap a feature to its baseline value).
_CONF_LINK = {
    "financial_pred_proba": ("financial_confidence", "lstm_confidence"),
    "news_tfidf_pred_proba": ("news_tfidf_confidence", "tfidf_confidence"),
    "news_embeddings_pred_proba": ("news_embeddings_confidence", "emb_confidence"),
}

# Effects (probability points) below this read as "no effect" in the UI.
_EFFECT_EPS = 0.01


def _build_drivers(
    row: pd.Series,
    model: Any,
    features: list[str],
    temperature: float,
    has_news: bool,
    background: pd.DataFrame | None,
) -> tuple[list[dict[str, Any]], float]:
    """Exact Shapley attribution of the call over the displayed signals.

    Returns ``(drivers, baseline_proba)`` where ``baseline_proba`` is the score
    with every player at its route-median and each driver ``effect`` is the
    signal's Shapley contribution (in probability units). They sum exactly from
    the baseline to the ensemble probability.
    """
    import itertools
    from math import factorial

    players = _PLAYERS_NEWS if has_news else _PLAYERS_NO_NEWS

    # Baseline ("typical day") value for each player: route median when we have
    # a background, else a neutral fallback.
    def _median(feat: str) -> float:
        if background is not None and feat in background.columns:
            col = pd.to_numeric(background[feat], errors="coerce").dropna()
            if len(col):
                return float(col.median())
        return 0.0 if feat in ("spy_return_5d", "n_headlines") else 0.5

    base_vals = {feat: _median(feat) for feat, _ in players}
    # Display value only (Shapley uses the raw row so the full coalition exactly
    # reproduces the real call, including HGB's native NaN handling).
    actual_disp = {feat: _safe_proba(row.get(feat), default=0.0) for feat, _ in players}

    n = len(players)
    keys = [feat for feat, _ in players]
    subsets = list(
        itertools.chain.from_iterable(
            itertools.combinations(range(n), r) for r in range(n + 1)
        )
    )

    # Build each coalition from the raw row, swapping only the *baseline* members
    # to their route-median. Members "in" the coalition keep the row's real value
    # (NaN included) — so the full coalition == the actual scored row and the
    # contributions reconcile exactly to (final - baseline).
    # Confidence columns are derived from the probas. Drop them from every
    # coalition dict so `_ensure_derived_features` recomputes them uniformly —
    # otherwise the full coalition (which never calls `_set_feature`) ends up
    # missing those keys, pandas injects NaN for it in the batch frame, and its
    # score diverges from the real call.
    _conf_cols = [c for cols in _CONF_LINK.values() for c in cols]
    coalition_rows = []
    for S in subsets:
        d = row.to_dict()
        for c in _conf_cols:
            d.pop(c, None)
        for i, feat in enumerate(keys):
            if i not in S:
                d[feat] = base_vals[feat]
        coalition_rows.append(d)

    scores = _score_rows(model, coalition_rows, features, temperature)
    vmap = {S: float(scores[i]) for i, S in enumerate(subsets)}
    baseline_proba = vmap[()]

    drivers: list[dict[str, Any]] = []
    for i, (feat, label) in enumerate(players):
        contrib = 0.0
        for S in subsets:
            if i in S:
                continue
            w = factorial(len(S)) * factorial(n - len(S) - 1) / factorial(n)
            contrib += w * (vmap[tuple(sorted(S + (i,)))] - vmap[S])
        drivers.append({
            "feature": feat,
            "label": label,
            "value": actual_disp[feat],
            "effect": contrib,
            "direction": "up" if contrib > _EFFECT_EPS else "down" if contrib < -_EFFECT_EPS else "neutral",
        })
    drivers.sort(key=lambda d: abs(d["effect"]), reverse=True)
    return drivers, baseline_proba
