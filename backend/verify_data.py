# Verifies that all TSV data files load correctly and have expected columns
import csv
import os
import random

data_dir = os.path.join(os.path.dirname(__file__), "data")
files = ["a.tsv", "b.tsv", "c.tsv", "d.tsv", "e.tsv"]

all_rows = []
for fname in files:
    path = os.path.join(data_dir, fname)
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
        all_rows.extend(rows)
        print(f"{fname}: {len(rows)} rows")

print(f"\nTotal openings loaded: {len(all_rows)}")

print("\n3 sample rows:")
samples = random.sample(all_rows, 3)
for row in samples:
    print(f"  eco={row['eco']}, name={row['name']}, pgn={row['pgn'][:50]}")

print("DATA OK")
