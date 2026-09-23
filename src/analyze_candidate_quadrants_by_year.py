from pathlib import Path
import pandas as pd
import numpy as np


BASE = Path(__file__).resolve().parents[1] / "results" / "instrument_regime_diagnostic"

CANDIDATES = {
    "EURUSDm": [
        ("TREND", "SHORT"),
    ],
    "XAUUSDm": [
        ("MR", "SHORT"),
        ("TREND", "LONG"),
    ],
    "BTCUSDm": [
        ("MR", "LONG"),
    ],
}


def profit_factor(pnl):
    gross_profit = pnl[pnl > 0].sum()
    gross_loss = -pnl[pnl < 0].sum()

    if gross_loss == 0:
        return np.inf if gross_profit > 0 else np.nan

    return gross_profit / gross_loss


def analyze_group(df):
    if len(df) == 0:
        return {
            "Trades": 0,
            "WinRate": np.nan,
            "PF": np.nan,
            "Expectancy": np.nan,
            "TotalPnL": 0.0,
            "AvgReturn": np.nan,
            "StopRate": np.nan,
        }

    pnl = pd.to_numeric(df["PnL"], errors="coerce").dropna()
    returns = pd.to_numeric(df["Return"], errors="coerce")

    wins = pnl > 0

    return {
        "Trades": len(pnl),
        "WinRate": wins.mean() * 100,
        "PF": profit_factor(pnl),
        "Expectancy": pnl.mean(),
        "TotalPnL": pnl.sum(),
        "AvgReturn": returns.mean() * 100,
        "StopRate": (
            df["ExitReason"].eq("Z_STOP").mean() * 100
            if "ExitReason" in df.columns
            else np.nan
        ),
    }


def main():

    all_results = []

    for symbol, candidates in CANDIDATES.items():

        path = BASE / symbol / "markov_trades_with_regime.csv"

        if not path.exists():
            print(f"\nMISSING: {path}")
            continue

        df = pd.read_csv(path)

        df["EntryTime"] = pd.to_datetime(
            df["EntryTime"],
            utc=True,
            errors="coerce",
        )

        df["Year"] = df["EntryTime"].dt.year

        # Use the diagnostic regime column created earlier.
        regime_col = "EntryRegime_Diagnostic"

        if regime_col not in df.columns:
            raise ValueError(
                f"{symbol}: missing {regime_col}. "
                f"Available columns: {list(df.columns)}"
            )

        print("\n" + "=" * 80)
        print(symbol)
        print("=" * 80)

        for regime, side in candidates:

            candidate = df[
                (df[regime_col] == regime)
                & (df["Side"] == side)
            ].copy()

            print(f"\nCandidate: {regime} + {side}")
            print("-" * 80)

            for year in range(2019, 2027):

                yearly = candidate[candidate["Year"] == year]

                stats = analyze_group(yearly)

                row = {
                    "Symbol": symbol,
                    "Regime": regime,
                    "Side": side,
                    "Year": year,
                    **stats,
                }

                all_results.append(row)

                print(
                    f"{year}: "
                    f"Trades={stats['Trades']:4d} | "
                    f"Win={stats['WinRate']:.2f}% | "
                    f"PF={stats['PF']:.3f} | "
                    f"Exp=${stats['Expectancy']:.3f} | "
                    f"PnL=${stats['TotalPnL']:.2f} | "
                    f"AvgRet={stats['AvgReturn']:.4f}% | "
                    f"Stop={stats['StopRate']:.2f}%"
                )

    results = pd.DataFrame(all_results)

    output_dir = BASE / "candidate_quadrants"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "candidate_quadrants_by_year.csv"
    results.to_csv(output_file, index=False)

    print("\n" + "=" * 80)
    print("SAVED")
    print("=" * 80)
    print(output_file)


if __name__ == "__main__":
    main()