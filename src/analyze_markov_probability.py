from pathlib import Path
import sys
import argparse

import numpy as np
import pandas as pd


# =========================================================
# PATHS
# =========================================================

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from walk_forward_kalman import (
    walk_forward_kalman,
    calculate_zscore,
)

from markov_regime import add_markov_regime


# =========================================================
# CONFIG
# =========================================================

DATA_DIR = PROJECT_ROOT / "data" / "mt5_h1"

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "markov_probability"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


INSTRUMENTS = [
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


# =========================================================
# LOCKED KALMAN PARAMETERS
# =========================================================

PHI = 0.999
Q = 0.25
R = 0.25

MEAN_WINDOW = 168
Z_WINDOW = 36

ENTRY_Z = 2.25


# =========================================================
# LOCKED HMM PARAMETERS
# =========================================================

HMM_WINDOW = 1000
HMM_MIN_OBSERVATIONS = 300
HMM_REFIT_EVERY = 24


# =========================================================
# FORWARD HORIZONS
# =========================================================

HORIZONS = [
    1,
    3,
    6,
    12,
    24,
]


START_DATE = pd.Timestamp(
    "2019-01-01",
    tz="UTC",
)

END_DATE = pd.Timestamp(
    "2026-09-19",
    tz="UTC",
)


# =========================================================
# LOAD DATA
# =========================================================

def load_data(symbol):

    path = (
        DATA_DIR
        / f"{symbol}_H1_2019_2026.csv"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    raw = pd.read_csv(path)

    # Handle normal MT5 CSV naming
    time_col = next(
        (
            c for c in raw.columns
            if str(c).lower()
            in ["time", "timestamp", "datetime", "date"]
        ),
        None,
    )

    if time_col is None:
        raise ValueError(
            f"{symbol}: time column not found"
        )

    required = [
        "Open",
        "High",
        "Low",
        "Close",
    ]

    missing = [
        c for c in required
        if c not in raw.columns
    ]

    if missing:
        raise ValueError(
            f"{symbol}: missing {missing}"
        )

    df = pd.DataFrame()

    df["Time"] = pd.to_datetime(
        raw[time_col],
        utc=True,
        errors="coerce",
    )

    for column in required:
        df[column] = pd.to_numeric(
            raw[column],
            errors="coerce",
        )

    df = (
        df
        .dropna()
        .sort_values("Time")
        .drop_duplicates("Time")
    )

    df = df[
        (df["Time"] >= START_DATE)
        &
        (df["Time"] <= END_DATE)
    ].copy()

    return df.reset_index(drop=True)


# =========================================================
# KALMAN
# =========================================================

def add_kalman(df):

    (
        states,
        variances,
        residuals,
        means,
    ) = walk_forward_kalman(
        df["Close"],
        phi=PHI,
        q=Q,
        r=R,
        mean_window=MEAN_WINDOW,
    )

    residual_series = pd.Series(
        residuals,
        index=df.index,
    )

    zscore = calculate_zscore(
        residual_series,
        window=Z_WINDOW,
    )

    out = df.copy()

    out["KalmanState"] = states
    out["KalmanVariance"] = variances
    out["Residual"] = residuals
    out["RollingMean"] = means
    out["ZScore"] = zscore

    return out


# =========================================================
# SIGNALS
# =========================================================

def add_signals(df):

    out = df.copy()

    out["Signal"] = "NONE"

    out.loc[
        out["ZScore"] <= -ENTRY_Z,
        "Signal",
    ] = "LONG"

    out.loc[
        out["ZScore"] >= ENTRY_Z,
        "Signal",
    ] = "SHORT"

    return out


# =========================================================
# FORWARD RETURNS
# =========================================================

def add_forward_returns(df):

    out = df.copy()

    for h in HORIZONS:

        future = out["Close"].shift(-h)

        raw_return = (
            future
            / out["Close"]
            - 1.0
        )

        directional = raw_return.copy()

        directional.loc[
            out["Signal"] == "SHORT"
        ] *= -1

        out[
            f"DirectionalReturn_{h}h"
        ] = directional

    return out


# =========================================================
# PROBABILITY BUCKET
# =========================================================

def probability_bucket(prob):

    if pd.isna(prob):
        return None

    if prob < 0.40:
        return "<0.40"

    if prob < 0.60:
        return "0.40-0.60"

    if prob < 0.80:
        return "0.60-0.80"

    return ">=0.80"


# =========================================================
# STATISTICS
# =========================================================

def stats(values):

    values = (
        pd.Series(values)
        .dropna()
    )

    if len(values) == 0:

        return {
            "N": 0,
            "Mean": np.nan,
            "Median": np.nan,
            "WinRate": np.nan,
            "Std": np.nan,
        }

    return {
        "N": len(values),

        "Mean":
            values.mean(),

        "Median":
            values.median(),

        "WinRate":
            (values > 0).mean(),

        "Std":
            values.std(),
    }


# =========================================================
# ANALYZE SYMBOL
# =========================================================

def analyze_symbol(symbol):

    print()
    print("=" * 110)
    print(f"ANALYZING {symbol}")
    print("=" * 110)

    df = load_data(symbol)

    print(
        f"Bars: {len(df):,}"
    )

    print(
        "Calculating Kalman..."
    )

    df = add_kalman(df)

    df = add_signals(df)

    signal_count = (
        df["Signal"]
        .isin(["LONG", "SHORT"])
        .sum()
    )

    print(
        f"Kalman signals: "
        f"{signal_count:,}"
    )

    print(
        "Calculating causal HMM..."
    )

    df = add_markov_regime(
        df,
        hmm_window=HMM_WINDOW,
        min_observations=HMM_MIN_OBSERVATIONS,
        refit_every=HMM_REFIT_EVERY,
    )

    print(
        "Calculating forward returns..."
    )

    df = add_forward_returns(df)

    signals = df[
        df["Signal"]
        .isin(["LONG", "SHORT"])
        &
        df["P_MeanReverting"].notna()
    ].copy()

    signals["ProbabilityBucket"] = (
        signals["P_MeanReverting"]
        .apply(probability_bucket)
    )

    signals["Symbol"] = symbol

    # -----------------------------------------------------
    # Save signal-level data
    # -----------------------------------------------------

    symbol_dir = (
        RESULTS_DIR
        / symbol
    )

    symbol_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    signal_columns = [
        "Time",
        "Close",
        "ZScore",
        "Signal",
        "P_MeanReverting",
        "P_Trending",
        "ProbabilityBucket",
    ]

    for h in HORIZONS:
        signal_columns.append(
            f"DirectionalReturn_{h}h"
        )

    signals[
        signal_columns
    ].to_csv(
        symbol_dir
        / "signals.csv",
        index=False,
    )

    # -----------------------------------------------------
    # Aggregate statistics
    # -----------------------------------------------------

    results = []

    buckets = [
        "<0.40",
        "0.40-0.60",
        "0.60-0.80",
        ">=0.80",
    ]

    for bucket in buckets:

        for side in [
            "LONG",
            "SHORT",
        ]:

            subset = signals[
                (signals["ProbabilityBucket"] == bucket)
                &
                (signals["Signal"] == side)
            ]

            for h in HORIZONS:

                values = subset[
                    f"DirectionalReturn_{h}h"
                ]

                result = stats(values)

                results.append({
                    "Symbol": symbol,
                    "ProbabilityBucket": bucket,
                    "Side": side,
                    "HorizonHours": h,
                    **result,
                })

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        symbol_dir
        / "probability_statistics.csv",
        index=False,
    )

    # -----------------------------------------------------
    # Console
    # -----------------------------------------------------

    print()
    print(
        f"{symbol} PROBABILITY BUCKETS"
    )

    print("-" * 120)

    print(
        f"{'Bucket':<12}"
        f"{'Side':<8}"
        f"{'H':>5}"
        f"{'N':>8}"
        f"{'Mean':>14}"
        f"{'Median':>14}"
        f"{'WinRate':>12}"
    )

    print("-" * 120)

    for _, row in results_df.iterrows():

        print(
            f"{row['ProbabilityBucket']:<12}"
            f"{row['Side']:<8}"
            f"{int(row['HorizonHours']):>5}h"
            f"{int(row['N']):>8}"
            f"{row['Mean'] * 100:>13.4f}%"
            f"{row['Median'] * 100:>13.4f}%"
            f"{row['WinRate'] * 100:>11.2f}%"
        )

    return results_df


# =========================================================
# MAIN
# =========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--symbol",
        default="XAUUSDm",
    )

    args = parser.parse_args()

    if args.symbol.upper() == "ALL":
        symbols = INSTRUMENTS
    else:
        symbols = [args.symbol]

    all_results = []

    print()
    print("=" * 110)
    print(
        "KALMAN + MARKOV PROBABILITY DIAGNOSTIC"
    )
    print("=" * 110)

    print()
    print("No optimization.")
    print("No parameter changes.")
    print()

    for symbol in symbols:

        try:

            result = analyze_symbol(
                symbol
            )

            if len(result) > 0:
                all_results.append(result)

        except Exception as e:

            print()
            print(
                f"ERROR: {symbol}"
            )

            print(
                repr(e)
            )

    if all_results:

        combined = pd.concat(
            all_results,
            ignore_index=True,
        )

        combined.to_csv(
            RESULTS_DIR
            / "all_probability_statistics.csv",
            index=False,
        )

        print()
        print("=" * 110)
        print("SAVED")
        print("=" * 110)

        print(
            RESULTS_DIR
            / "all_probability_statistics.csv"
        )


if __name__ == "__main__":
    main()