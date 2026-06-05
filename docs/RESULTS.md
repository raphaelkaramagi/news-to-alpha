# Evaluation results

Held-out **test split** (n≈2,220). Regenerate after retrain:

```bash
python scripts/evaluate_predictions.py --horizon 1
cat data/processed/evaluation_summary.txt
```

Walk-forward (multi-fold, less optimistic):

```bash
python scripts/walk_forward_eval.py --all-targets
```

---

## Direction (test, latest `max_v2` retrain, n=2,183)

| Model | Acc | AUC | Note |
|-------|-----|-----|------|
| ensemble | 0.528 | **0.533** | best direction signal|
| lstm_price | 0.507 | 0.504 | calibration fixed (11.5k distinct probas, std 0.030) |
| previous_day_direction | 0.511 | 0.511 | momentum baseline |
| always_up (drift) | 0.508 | 0.500 | base rate, not skill |
| ensemble on `news_scored` (n=380) | 0.568 | **0.555** | slight edge on fresh-headline days |

**Walk-forward mean AUC (6 folds, less optimistic than the single split):** next_day 0.49, weekly 0.50, cross_sectional 0.49, large_move 0.43.
No direction target shows durable skill on these tickers — the single-split 0.53 is the optimistic end of the range.

---

## Volatility / expected move (test, n=2,183)

| Metric | Value | Meaning |
|--------|-------|---------|
| High-move AUC | **0.649** | ranks big vs small \|return\| days |
| MAE | **1.21%** | vs median move ~1.2% |

Separate from the direction ensemble; reported as the ±% band on the cards. Move *size* is more predictable than move *direction*.

---

## Subsets

| Subset | Use |
|--------|-----|
| `news_scored` | honest news OOS |
| `low_vol` / `high_vol` | direction by predicted move size |
| `high_conf` | ensemble confidence ≥ 0.3 |


---

## UI

| Surface | Signal |
|---------|--------|
| Markets grid | ±expected move (primary) + direction call |
| Why tab | base inputs + disagreement when ensemble flips |
| Advanced | 13 meta-features, counterfactual drivers |
