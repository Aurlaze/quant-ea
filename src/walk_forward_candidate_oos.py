from pathlib import Path
import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1] / "results" / "instrument_regime_diagnostic"

CANDIDATES = {
    "EURUSDm": [
        ("TREND", "SHORT"),
    ],
    "XAUUSDm": [
        ("MR", "SHORT"),
        ("TREND", "LONG"),
    ],
}

YEARS = list(range(2019, 2027))


def profit_factor(pnl):
    pnl = pd.Series(pnl).dropna()

    gross_profit = pnl[pnl > 0].sum()
    gross_loss = -pnl[pnl < 0].sum()

    if gross_loss == 0:
        return np.inf if gross_profit > 0 else np.nan

    return gross_profit / gross_loss


def calculate_stats(df):

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

    return {
        "Trades": len(pnl),
        "WinRate": (pnl > 0).mean() * 100,
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

    print("=" * 90)
    print("WALK-FORWARD CANDIDATE OOS ANALYSIS")
    print("=" * 90)

    print(
        "\nIMPORTANT:"
        "\nThis first pass evaluates the previously discovered candidate"
        "\nquadrants on yearly OOS periods."
        "\nIt does NOT re-optimize parameters inside each year."
        "\nCandidate discovery itself used the full historical sample,"
        "\nso this is a validation diagnostic rather than a fully"
        "\nclean model-selection OOS experiment."
    )

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

        regime_col = "EntryRegime_Diagnostic"

        if regime_col not in df.columns:
            raise ValueError(
                f"{symbol}: missing {regime_col}"
            )

        for regime, side in candidates:

            candidate = df[
                (df[regime_col] == regime)
                & (df["Side"] == side)
            ].copy()

            print("\n" + "=" * 90)
            print(f"{symbol} | {regime} + {side}")
            print("=" * 90)

            for test_year in YEARS:

                # Training period = all years before OOS year.
                train = candidate[
                    candidate["Year"] < test_year
                ]

                # OOS period = exactly the test year.
                test = candidate[
                    candidate["Year"] == test_year
                ]

                train_stats = calculate_stats(train)
                test_stats = calculate_stats(test)

                row = {
                    "Symbol": symbol,
                    "Regime": regime,
                    "Side": side,
                    "TestYear": test_year,

                    "TrainTrades": train_stats["Trades"],
                    "TrainPF": train_stats["PF"],
                    "TrainPnL": train_stats["TotalPnL"],

                    "OOSTrades": test_stats["Trades"],
                    "OOSWinRate": test_stats["WinRate"],
                    "OOSPF": test_stats["PF"],
                    "OOSExpectancy": test_stats["Expectancy"],
                    "OOSPnL": test_stats["TotalPnL"],
                    "OOSAvgReturn": test_stats["AvgReturn"],
                    "OOSStopRate": test_stats["StopRate"],
                }

                all_results.append(row)

                print(
                    f"OOS {test_year}: "
                    f"Trades={test_stats['Trades']:4d} | "
                    f"Win={test_stats['WinRate']:.2f}% | "
                    f"PF={test_stats['PF']:.3f} | "
                    f"Exp=${test_stats['Expectancy']:.3f} | "
                    f"PnL=${test_stats['TotalPnL']:.2f} | "
                    f"Stop={test_stats['StopRate']:.2f}%"
                )

    results = pd.DataFrame(all_results)

    output_dir = BASE / "candidate_quadrants"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = (
        output_dir /
        "candidate_quadrants_walk_forward_oos.csv"
    )

    results.to_csv(output_file, index=False)

    print("\n" + "=" * 90)
    print("SAVED")
    print("=" * 90)
    print(output_file)


if __name__ == "__main__":
    main()