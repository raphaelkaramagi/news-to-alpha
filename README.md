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
Create a virtual environment:

python3 -m venv .venv
Activate the virtual environment:

source .venv/bin/activate
You should see (.venv) at the start of your terminal line.

Install dependencies:

pip install -r requirements.txt
Create your local environment file (API keys/settings):

cp .env.example .env
Open .env and fill in values as needed. Do not commit .env.

Quick sanity check (optional):

python -c "import pandas, sklearn, requests; print('ok')"
To exit the virtual environment later:

deactivate
Setup (Windows)
Clone the repo and enter it:

git clone <REPO_URL>
cd news-to-alpha
Create a virtual environment:

python -m venv .venv
Activate the virtual environment:

PowerShell

.\.venv\Scripts\Activate.ps1
Command Prompt (cmd)

.\.venv\Scripts\activate.bat
Install dependencies:

pip install -r requirements.txt
Create your local environment file (API keys/settings):

PowerShell

copy .env.example .env
Command Prompt (cmd)

copy .env.example .env
Open .env and fill in values as needed. Do not commit .env.

Quick sanity check (optional):

python -c "import pandas, sklearn, requests; print('ok')"
To exit the virtual environment later:

deactivate

