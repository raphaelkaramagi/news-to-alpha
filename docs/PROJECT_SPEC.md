## Data Sources
### Prices
Provider: Yahoo Finance
Fields needed: date, open, high, low, close, volume (adj_close optional)

### News
Provider: 
Fields needed: title, url, source, published_time

## Data Storage Standards 

### Global standards
- Ticker strings are always **uppercase** (e.g., AAPL).
- Price `date` is stored as **YYYY-MM-DD** (e.g., 2026-02-07).
- News `published_time` is stored as **ISO 8601 in ET with timezone offset**
  - Example: `2026-02-07T14:13:00-05:00`
- Missing values are stored as **NULL** (not empty strings).
- News duplicates are detected using **url** as the unique identifier.


### Prices table (daily)
Required columns:
- ticker (TEXT), date (TEXT), open (REAL), high (REAL), low (REAL), close (REAL), volume (INTEGER)

Uniqueness rule:
- One row per (ticker, date)


### Articles table (news)
Required columns:
- url (TEXT), title (TEXT), source (TEXT), published_time (TEXT), fetched_for_ticker (TEXT), ingested_at (TEXT)

Uniqueness rule:
- One row per url