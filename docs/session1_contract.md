# Session 1 Contract

## 1. Branch audit
stock-data branch audit:
- useful files: none
- decision: ignore
- reason: no diff and no unique commits relative to main

## 2. Label definition
prediction_date = market day t on which the prediction is made
label_binary = 1 if close(t+1) > close(t), else 0

## 3. Cutoff rule
News published before 4:00 PM ET on day t predicts the next trading day after t.
News published at or after 4:00 PM ET on day t predicts the next trading day after t+1.
Weekends and market holidays are skipped.

## 4. Shared prediction export schema
ticker
prediction_date
split
pred_proba
pred_binary
confidence
actual_binary
model_name
model_version

## 5. Confidence formula
confidence = abs(pred_proba - 0.5) * 2

## 6. News-model requirement
Both news models must train on cutoff-aligned ticker-day rows keyed by prediction_date, not on raw published_at calendar dates alone.

## 7. Integration rule for April 11
Join model outputs on ticker + prediction_date.
Do not assume split matches across price and news outputs unless explicitly verified.