import sys
import json
import random
import sqlite3
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import DATABASE_PATH, PROCESSED_DATA_DIR 

def main() -> None:
    #Loading in all 5 files
    X = np.load(PROCESSED_DATA_DIR / "X_sequences.npy")
    y = np.load(PROCESSED_DATA_DIR / "y_labels.npy")
    with open(PROCESSED_DATA_DIR / "sequence_dates.json") as f:
      sequence_dates = json.load(f)
    with open(PROCESSED_DATA_DIR / "split_info.json") as f:
        split_info = json.load(f)
    conn = sqlite3.connect(DATABASE_PATH)
    
    lines = []
    def log(text=""):
      """Print a line and also store it for the report file."""
      print(text)
      lines.append(text)
      
    #Checking length of the sequences
    log("PRICE ALIGNMENT QC")
    log(f"\nTotal sequences:  {len(X)}")
    log(f"Total labels:     {len(y)}")
    log(f"Total dates:      {len(sequence_dates)}")
    
    if not (len(X) == len(y) == len(sequence_dates)):
      log("ERROR: lengths don't match!")
    else:
      log("OK: all three arrays have the same length")
    
    #Checking number of sequences per split (train, validation, test)
    date_to_split = {}
    for split_name in ["train", "val", "test"]:
        for d in split_info["splits"][split_name]["dates"]:
            date_to_split[d] = split_name
    split_counts = {"train": 0, "val": 0, "test": 0, "unknown": 0}
    for ticker, pred_date in sequence_dates:
        split = date_to_split.get(pred_date, "unknown")
        split_counts[split] += 1
    log("\nSequences per split:")
    for split_name, count in split_counts.items():
        log(f"  {split_name:8s}  {count}")
    if split_counts["unknown"] > 0:
        log(f"WARNING: {split_counts['unknown']} sequences have dates not in any split")
        
    #Checking that each (ticker,date) pair appears only once
    seen = set()
    duplicates = []
    for ticker, pred_date in sequence_dates:
      key = (ticker, pred_date)
      if key in seen:
          duplicates.append(key)
      seen.add(key)
    log(f"\nDuplicate (ticker, date) pairs: {len(duplicates)}")
    if duplicates:
      log("WARNING: duplicates found — first 5:")
      for t, d in duplicates[:5]:
          log(f"  {t} {d}")
    else:
      log("OK: no duplicates")

    #Cheking if every (ticker, date) pair exists in the labels table
    missing_from_labels = []
    for ticker, pred_date in sequence_dates:
      row = conn.execute(
          "SELECT label_binary FROM labels WHERE ticker = ? AND date = ?",
          (ticker, pred_date),
      ).fetchone()
      if row is None:
          missing_from_labels.append((ticker, pred_date))
    log(f"\nMissing from labels table: {len(missing_from_labels)}")
    if missing_from_labels:
      log("WARNING: these (ticker, date) pairs have no label row — first 5:")
      for t, d in missing_from_labels[:5]:
          log(f"  {t} {d}")
    else:
      log("OK: every sequence has a matching label in the database")
      
    #Simply checking random rows
    n = len(sequence_dates)
    sample_indices = random.sample(range(n), min(10, n))
    log("\nSpot-check (10 random rows):")
    log(f"  {'idx':>6s}  {'ticker':5s}  {'pred_date':10s}  {'label':5s}  {'split':8s}")
    log(f"  {'-'*6}  {'-'*5}  {'-'*10}  {'-'*5}  {'-'*8}")
    for idx in sample_indices:
      ticker, pred_date = sequence_dates[idx]
      label = int(y[idx])
      split = date_to_split.get(pred_date, "unknown")
      log(f"  {idx:6d}  {ticker:5s}  {pred_date:10s}  {label:5d}  {split:8s}")
      
    #Saving report
    conn.close()
    report_path = PROCESSED_DATA_DIR / "price_alignment_report.txt"
    with open(report_path, "w") as f:
      f.write("\n".join(lines) + "\n")
    print(f"\nReport saved to {report_path}")
    
if __name__ == "__main__":
  main()