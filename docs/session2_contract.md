# Session 2 Contract

## Goal of this session
Finish all technical deliverables needed for the presentation so that Week 11 only requires Google Slides, app polish, bug fixes, and rehearsal.

## Locked ensemble formula
Use the following ensemble formula in `scripts/build_ensemble.py`:

ensemble_pred_proba = 0.45 * financial_pred_proba + 0.25 * news_tfidf_pred_proba + 0.30 * news_embeddings_pred_proba

Then compute:
- ensemble_pred_binary = 1 if ensemble_pred_proba >= 0.5 else 0
- ensemble_confidence = abs(ensemble_pred_proba - 0.5) * 2

## Integration rule
For April 11 integration, join model outputs on:
- ticker
- prediction_date

Do **not** require `split` to match across model outputs.

If needed:
- keep separate columns such as `price_split` and `news_split`, or
- assign one final split after overlapping rows are identified

## Required output files by end of session
The following files must exist by the end of April 11:

- `data/processed/eval_dataset.csv`
- `data/processed/evaluation_overall.csv`
- `data/processed/evaluation_by_ticker.csv`
- `data/processed/evaluation_summary.txt`
- `data/processed/final_ensemble_predictions.csv`

## Required metrics for slides
The metrics that should be computed and saved for later use in slides are:

- accuracy
- precision_up
- recall_up
- f1_up
- num_predictions

These should be available for:
- `lstm_price`
- `news_tfidf`
- `news_embeddings`
- `ensemble`
- `always_up`
- `previous_day_direction` only if it can be added cleanly without breaking scope

## Demo tickers
Prioritize these 3 tickers in the app and for the demo:
- AAPL
- NVDA
- TSLA

If one of these does not have a usable final prediction row in the overlapping dataset, replace it with:
- META

## App priority order
The app must display these things in this order:

1. Selected ticker
2. Latest available prediction_date
3. Final ensemble direction
4. Final ensemble probability
5. Final ensemble confidence
6. Individual model probabilities:
   - financial_pred_proba
   - news_tfidf_pred_proba
   - news_embeddings_pred_proba
7. top_headlines

## App data source
The app must read from:
- `data/processed/final_ensemble_predictions.csv`

Use saved outputs only. Do not retrain models or run live inference in the app.

## Shared output expectations
- Row key for final integration: `(ticker, prediction_date)`
- Confidence format: `abs(pred_proba - 0.5) * 2`
- `top_headlines` must remain a single field and not be expanded into multiple columns

## Session success condition
April 11 is successful only if the team leaves with:
- `eval_dataset.csv`
- `evaluation_overall.csv`
- `evaluation_by_ticker.csv`
- `evaluation_summary.txt`
- `final_ensemble_predictions.csv`
- and an app already connected to the final saved outputs

So that April 18 only requires:
- Google Slides
- app polish
- bug fixes
- rehearsal