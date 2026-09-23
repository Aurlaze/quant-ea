from pathlib import Path
import pandas as pd

RESULTS_DIR = Path("results")

required_any = {
    "EntryRegime",
    "Side",
    "EntryTime",
}

print("=" * 80)
print("SEARCHING EXISTING MARKOV TRADE FILES")
print("=" * 80)

found = []

for path in RESULTS_DIR.rglob("*.csv"):
    try:
        df = pd.read_csv(path, nrows=5)
    except Exception:
        continue

    cols = set(df.columns)

    score = len(required_any & cols)

    if score >= 2:
        found.append((path, score, sorted(cols)))

for path, score, cols in sorted(found, key=lambda x: str(x[0])):
    print(f"\n[{score}/{len(required_any)}] {path}")
    print("Columns:", cols)

print("\n" + "=" * 80)
print(f"Found {len(found)} possible trade files.")
print("=" * 80)