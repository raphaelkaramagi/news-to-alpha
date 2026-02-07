# News x Stock Price Prediction Platform 

We collect stock prices + news, build a labeled dataset, train NLP models to predict next-day movement, and demo results in a simple app.

## Setup
1) Create a virtual environment
2) Install dependencies: `pip install -r requirements.txt`
3) Copy `.env.example` to `.env` and add API keys (do not commit `.env`)

## Folder structure
- `src/` all Python code (data collection, dataset, modeling, evaluation)
- `docs/` project decisions (ticker list, cutoff rule, data sources)
- `app/` demo app 
- `data/` local data (ignored by git)
- `notebooks/` experiments (optional)
