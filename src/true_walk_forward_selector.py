from pathlib import Path
import numpy as np
import pandas as pd


BASE = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "instrument_regime_diagnostic"
)

OUTPUT_DIR = BASE / "true_walk_forward"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# All possible regime/direction combinations.
QUADRANTS = [
    ("MR", "LONG"),
    ("MR", "SHORT"),
    ("TREND", "LONG"),
    ("TREND", "SHORT"),
]


SYMBOLS = [
    "EURUSDm",
    "GBPUSDm",
    "USDJPYm",
    "XAUUSDm",
    "XAGUSDm",
    "USTECm",
    "JP225m",
    "BTCUSDm",
    "ETHUSDm",
    "USOILm",
]


# We need enough history for a meaningful initial training sample.
INITIAL_TRAIN_END = 2021

# Sequential OOS years.
OOS_YEARS = [2022, 2023, 2024, 2025, 2026]

MIN_TRAIN_TRADES = 30
MIN_TRAIN_PF = 1.00


def profit_factor(pnl):

    pnl = pd.Series(pnl).dropna()

    gross_profit = pnl[pnl > 0].sum()
    gross_loss = -pnl[pnl < 0].sum()

    if gross_loss == 0:
        if gross_profit > 0:
            return np.inf
        return np.nan

    return gross_profit / gross_loss


def stats(df):

    if df.empty:
        return {
            "Trades": 0,
            "WinRate": np.nan,
            "PF": np.nan,
            "Expectancy": np.nan,
            "PnL": 0.0,
            "AvgReturn": np.nan,
            "StopRate": np.nan,
        }

    pnl = pd.to_numeric(
        df["PnL"],
        errors="coerce",
    ).dropna()

    returns = pd.to_numeric(
        df["Return"],
        errors="coerce",
    )

    return {
        "Trades": len(pnl),
        "WinRate": (pnl > 0).mean() * 100,
        "PF": profit_factor(pnl),
        "Expectancy": pnl.mean(),
        "PnL": pnl.sum(),
        "AvgReturn": returns.mean() * 100,
        "StopRate": (
            df["ExitReason"].eq("Z_STOP").mean() * 100
            if "ExitReason" in df.columns
            else np.nan
        ),
    }


def get_quadrant(df, regime, side):

    return df[
        (df["EntryRegime_Diagnostic"] == regime)
        & (df["Side"] == side)
    ].copy()


def select_quadrant(training_df):

    candidates = []

    for regime, side in QUADRANTS:

        q = get_quadrant(
            training_df,
            regime,
            side,
        )

        s = stats(q)

        if s["Trades"] < MIN_TRAIN_TRADES:
            continue

        if not np.isfinite(s["PF"]):
            continue

        if s["PF"] < MIN_TRAIN_PF:
            continue

        if s["Expectancy"] <= 0:
            continue

        candidates.append(
            {
                "Regime": regime,
                "Side": side,
                **s,
            }
        )

    if not candidates:
        return None, candidates

    # Primary selection criterion:
    # highest training PF.
    #
    # Secondary:
    # higher expectancy.
    #
    # Tertiary:
    # more trades.
    candidates = sorted(
        candidates,
        key=lambda x: (
            x["PF"],
            x["Expectancy"],
            x["Trades"],
        ),
        reverse=True,
    )

    return candidates[0], candidates


def main():

    all_oos_results = []
    all_selection_results = []

    print("=" * 100)
    print("TRUE EXPANDING-WINDOW WALK-FORWARD SELECTOR")
    print("=" * 100)

    print(
        f"""
Initial training: 2019-{INITIAL_TRAIN_END}
OOS years: {OOS_YEARS}

Selection rules:
    Minimum training trades = {MIN_TRAIN_TRADES}
    Minimum training PF     = {MIN_TRAIN_PF}
    Training expectancy     > 0

Selection is performed using TRAINING DATA ONLY.
The selected quadrant is then frozen for the OOS year.
"""
    )

    for symbol in SYMBOLS:

        path = (
            BASE
            / symbol
            / "markov_trades_with_regime.csv"
        )

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

        print("\n" + "=" * 100)
        print(symbol)
        print("=" * 100)

        for test_year in OOS_YEARS:

            train = df[
                df["Year"] <= INITIAL_TRAIN_END
            ].copy()

            # Expanding window.
            #
            # For 2023, training becomes 2019-2022.
            # For 2024, training becomes 2019-2023, etc.
            train = df[
                df["Year"] < test_year
            ].copy()

            test = df[
                df["Year"] == test_year
            ].copy()

            selected, qualifying = select_quadrant(
                train
            )

            print(
                f"\nTraining: 2019-{test_year - 1}"
                f"  |  OOS: {test_year}"
            )

            # Save all qualifying training candidates.
            for candidate in qualifying:

                all_selection_results.append(
                    {
                        "Symbol": symbol,
                        "TestYear": test_year,
                        "TrainingEnd": test_year - 1,
                        **candidate,
                    }
                )

            if selected is None:

                print(
                    "  SELECTED: NO TRADE "
                    "(no quadrant passed training criteria)"
                )

                all_oos_results.append(
                    {
                        "Symbol": symbol,
                        "TestYear": test_year,
                        "SelectedRegime": "NONE",
                        "SelectedSide": "NONE",
                        "TrainTrades": 0,
                        "TrainPF": np.nan,
                        "TrainExpectancy": np.nan,
                        "OOSTrades": 0,
                        "OOSWinRate": np.nan,
                        "OOSPF": np.nan,
                        "OOSExpectancy": np.nan,
                        "OOSPnL": 0.0,
                        "OOSAvgReturn": np.nan,
                        "OOSStopRate": np.nan,
                    }
                )

                continue

            regime = selected["Regime"]
            side = selected["Side"]

            print(
                f"  SELECTED: {regime} + {side}"
                f" | Train trades={selected['Trades']}"
                f" | Train PF={selected['PF']:.3f}"
                f" | Train Exp=${selected['Expectancy']:.3f}"
            )

            oos = get_quadrant(
                test,
                regime,
                side,
            )

            oos_stats = stats(oos)

            print(
                f"  OOS: trades={oos_stats['Trades']}"
                f" | win={oos_stats['WinRate']:.2f}%"
                f" | PF={oos_stats['PF']:.3f}"
                f" | Exp=${oos_stats['Expectancy']:.3f}"
                f" | PnL=${oos_stats['PnL']:.2f}"
                f" | stop={oos_stats['StopRate']:.2f}%"
            )

            all_oos_results.append(
                {
                    "Symbol": symbol,
                    "TestYear": test_year,
                    "SelectedRegime": regime,
                    "SelectedSide": side,

                    "TrainTrades": selected["Trades"],
                    "TrainPF": selected["PF"],
                    "TrainExpectancy": selected["Expectancy"],

                    "OOSTrades": oos_stats["Trades"],
                    "OOSWinRate": oos_stats["WinRate"],
                    "OOSPF": oos_stats["PF"],
                    "OOSExpectancy": oos_stats["Expectancy"],
                    "OOSPnL": oos_stats["PnL"],
                    "OOSAvgReturn": oos_stats["AvgReturn"],
                    "OOSStopRate": oos_stats["StopRate"],
                }
            )

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------

    oos_df = pd.DataFrame(
        all_oos_results
    )

    selection_df = pd.DataFrame(
        all_selection_results
    )

    oos_file = (
        OUTPUT_DIR
        / "true_walk_forward_oos.csv"
    )

    selection_file = (
        OUTPUT_DIR
        / "training_selection_candidates.csv"
    )

    oos_df.to_csv(
        oos_file,
        index=False,
    )

    selection_df.to_csv(
        selection_file,
        index=False,
    )

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    if not oos_df.empty:

        valid_oos = oos_df[
            oos_df["OOSTrades"] > 0
        ]

        print(
            f"\nTotal OOS observations: "
            f"{len(oos_df)}"
        )

        print(
            f"OOS observations with trades: "
            f"{len(valid_oos)}"
        )

        if len(valid_oos) > 0:

            positive_pf = (
                valid_oos["OOSPF"] > 1
            ).sum()

            positive_pnl = (
                valid_oos["OOSPnL"] > 0
            ).sum()

            print(
                f"OOS periods PF > 1: "
                f"{positive_pf}/{len(valid_oos)}"
            )

            print(
                f"OOS periods PnL > 0: "
                f"{positive_pnl}/{len(valid_oos)}"
            )

            print(
                f"Total OOS PnL: "
                f"${valid_oos['OOSPnL'].sum():.2f}"
            )

    print("\nSaved:")
    print(oos_file)
    print(selection_file)


if __name__ == "__main__":
    main()