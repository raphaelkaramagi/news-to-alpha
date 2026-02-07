# News x Stock Price Prediction Platform

We collect stock prices + news, build a labeled dataset, train NLP models to predict next-day movement, and demo results in a simple app.

## Folder structure
- `src/` all Python code (data collection, dataset, modeling, evaluation)
- `docs/` project decisions (ticker list, cutoff rule, data sources)
- `app/` demo app
- `data/` local data (ignored by git)
- `notebooks/` experiments (optional)

## Setup (Mac)

1) Clone the repo and enter it:
```bash
git clone <REPO_URL>
cd news-to-alpha

2) Create a virtual environment:

python3 -m venv .venv


3) Activate the virtual environment:

source .venv/bin/activate


You should see (.venv) at the start of your terminal line.

4) Install dependencies:

pip install -r requirements.txt


5) Create your local environment file (API keys/settings):

cp .env.example .env


Open .env and fill in values as needed. Do not commit .env.


To exit the virtual environment later:

deactivate
```

## Setup (Windows)

1) Clone the repo and enter it:

git clone <REPO_URL>
cd news-to-alpha


2) Create a virtual environment:

python -m venv .venv


3) Activate the virtual environment:

PowerShell

.\.venv\Scripts\Activate.ps1


Command Prompt (cmd)

.\.venv\Scripts\activate.bat


4) Install dependencies:

pip install -r requirements.txt


5) Create your local environment file (API keys/settings):

PowerShell

copy .env.example .env


Command Prompt (cmd)

copy .env.example .env


Open .env and fill in values as needed. Do not commit .env.


To exit the virtual environment later:

deactivate
```