from pathlib import Path
import sys
import argparse

import numpy as np
import pandas as pd


# =========================================================
# PATHS
# =========================================================

SRC_DIR = (
    Path(__file__)
    .resolve()
    .parent
)

PROJECT_ROOT = (
    SRC_DIR.parent
)

if str(SRC_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_DIR),
    )


from walk_forward_kalman import (
    walk_forward_kalman,
    calculate_zscore,
)

from markov_regime import (
    add_markov_regime,
)


# =========================================================
# CONFIGURATION
# =========================================================

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "mt5_h1"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "kalman_markov_expectancy"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# INSTRUMENTS
# =========================================================

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
# KALMAN PARAMETERS
# =========================================================

PHI = 0.999

Q = 0.25

R = 0.25

MEAN_WINDOW = 168

Z_WINDOW = 36

ENTRY_Z = 2.25


# =========================================================
# HMM PARAMETERS
# =========================================================

HMM_WINDOW = 1000

HMM_MIN_OBSERVATIONS = 300

HMM_REFIT_EVERY = 24


# =========================================================
# FORWARD HORIZONS
# =========================================================

FORWARD_HORIZONS = [
    1,
    3,
    6,
    12,
    24,
]


# =========================================================
# PERIOD
# =========================================================

START_DATE = pd.Timestamp(
    "2019-01-01",
    tz="UTC",
)

END_DATE = pd.Timestamp(
    "2026-09-19",
    tz="UTC",
)


# =========================================================
# DATA LOADING
# =========================================================

def find_column(
    df,
    candidates,
):

    lower_map = {
        str(column)
        .strip()
        .lower():
            column
        for column in df.columns
    }

    for candidate in candidates:

        key = (
            candidate
            .lower()
        )

        if key in lower_map:

            return lower_map[key]

    return None


def load_mt5_h1(
    symbol,
):

    path = (
        DATA_DIR
        / f"{symbol}_H1_2019_2026.csv"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Missing file:\n{path}"
        )

    raw = pd.read_csv(
        path
    )

    time_col = find_column(
        raw,
        [
            "time",
            "timestamp",
            "datetime",
            "date",
        ],
    )

    open_col = find_column(
        raw,
        ["open"],
    )

    high_col = find_column(
        raw,
        ["high"],
    )

    low_col = find_column(
        raw,
        ["low"],
    )

    close_col = find_column(
        raw,
        ["close"],
    )

    required = {
        "time": time_col,
        "open": open_col,
        "high": high_col,
        "low": low_col,
        "close": close_col,
    }

    missing = [
        key
        for key, value
        in required.items()
        if value is None
    ]

    if missing:

        raise ValueError(
            f"{symbol}: missing "
            f"columns {missing}"
        )

    df = pd.DataFrame()

    df["Time"] = pd.to_datetime(
        raw[time_col],
        utc=True,
        errors="coerce",
    )

    df["Open"] = pd.to_numeric(
        raw[open_col],
        errors="coerce",
    )

    df["High"] = pd.to_numeric(
        raw[high_col],
        errors="coerce",
    )

    df["Low"] = pd.to_numeric(
        raw[low_col],
        errors="coerce",
    )

    df["Close"] = pd.to_numeric(
        raw[close_col],
        errors="coerce",
    )

    df = (
        df
        .dropna()
        .sort_values("Time")
        .drop_duplicates(
            subset=["Time"],
            keep="first",
        )
    )

    df = df[
        (df["Time"] >= START_DATE)
        &
        (df["Time"] <= END_DATE)
    ].copy()

    df = df[
        (df["Open"] > 0)
        &
        (df["High"] > 0)
        &
        (df["Low"] > 0)
        &
        (df["Close"] > 0)
    ]

    df = df[
        (df["High"] >= df["Low"])
        &
        (df["High"] >= df["Open"])
        &
        (df["High"] >= df["Close"])
        &
        (df["Low"] <= df["Open"])
        &
        (df["Low"] <= df["Close"])
    ]

    return df.reset_index(
        drop=True
    )


# =========================================================
# KALMAN FEATURES
# =========================================================

def add_kalman_features(
    df,
):

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
        dtype=float,
    )

    zscore = calculate_zscore(
        residual_series,
        window=Z_WINDOW,
    )

    out = df.copy()

    out[
        "KalmanState"
    ] = states

    out[
        "KalmanVariance"
    ] = variances

    out[
        "Residual"
    ] = residuals

    out[
        "RollingMean"
    ] = means

    out[
        "ZScore"
    ] = zscore

    return out


# =========================================================
# GENERATE KALMAN SIGNALS
# =========================================================

def generate_signals(
    df,
):

    out = df.copy()

    out["Signal"] = "NONE"

    out.loc[
        out["ZScore"]
        <= -ENTRY_Z,
        "Signal",
    ] = "LONG"

    out.loc[
        out["ZScore"]
        >= ENTRY_Z,
        "Signal",
    ] = "SHORT"

    return out


# =========================================================
# CALCULATE FORWARD RETURNS
# =========================================================

def add_forward_returns(
    df,
):

    out = df.copy()

    close = (
        out["Close"]
        .astype(float)
    )

    for horizon in (
        FORWARD_HORIZONS
    ):

        future_close = (
            close.shift(
                -horizon
            )
        )

        # ---------------------------------------------
        # Raw forward return
        # ---------------------------------------------

        out[
            f"ForwardReturn_{horizon}h"
        ] = (
            future_close
            / close
            - 1
        )

        # ---------------------------------------------
        # Direction-adjusted return
        #
        # LONG:
        #   + price increase
        #
        # SHORT:
        #   + price decrease
        #
        # This is the more important metric.
        # ---------------------------------------------

        directional_return = (
            out[
                f"ForwardReturn_{horizon}h"
            ].copy()
        )

        directional_return.loc[
            out["Signal"]
            == "SHORT"
        ] *= -1

        out[
            f"DirectionalReturn_{horizon}h"
        ] = (
            directional_return
        )

    return out


# =========================================================
# EXPECTANCY STATISTICS
# =========================================================

def calculate_statistics(
    values,
):

    values = (
        pd.Series(values)
        .dropna()
    )

    if len(values) == 0:

        return {
            "Count": 0,
            "Mean": np.nan,
            "Median": np.nan,
            "Std": np.nan,
            "WinRate": np.nan,
            "Best": np.nan,
            "Worst": np.nan,
        }

    return {
        "Count":
            int(len(values)),

        "Mean":
            float(values.mean()),

        "Median":
            float(values.median()),

        "Std":
            float(values.std()),

        "WinRate":
            float(
                (
                    values > 0
                ).mean()
            ),

        "Best":
            float(values.max()),

        "Worst":
            float(values.min()),
    }


# =========================================================
# ANALYZE ONE GROUP
# =========================================================

def analyze_group(
    df,
    symbol,
    regime,
    side,
):

    subset = df.copy()

    # -----------------------------------------------------
    # Signal filter
    # -----------------------------------------------------

    subset = subset[
        subset["Signal"] == side
    ]

    # -----------------------------------------------------
    # Regime filter
    # -----------------------------------------------------

    if regime == "MR":

        subset = subset[
            subset[
                "P_MeanReverting"
            ]
            >= 0.60
        ]

    elif regime == "TREND":

        subset = subset[
            subset[
                "P_Trending"
            ]
            >= 0.60
        ]

    elif regime == "ALL":

        subset = subset[
            subset[
                "P_MeanReverting"
            ].notna()
        ]

    # -----------------------------------------------------
    # Result records
    # -----------------------------------------------------

    records = []

    for horizon in (
        FORWARD_HORIZONS
    ):

        values = subset[
            f"DirectionalReturn_{horizon}h"
        ]

        stats = calculate_statistics(
            values
        )

        record = {
            "Symbol":
                symbol,

            "Regime":
                regime,

            "Side":
                side,

            "HorizonHours":
                horizon,

            **stats,
        }

        records.append(
            record
        )

    return records


# =========================================================
# SIGNAL-LEVEL DATA
# =========================================================

def build_signal_dataset(
    df,
    symbol,
):

    signals = df[
        df["Signal"]
        .isin(
            [
                "LONG",
                "SHORT",
            ]
        )
        &
        df[
            "P_MeanReverting"
        ].notna()
    ].copy()

    if len(signals) == 0:

        return pd.DataFrame()

    signals[
        "Regime"
    ] = np.where(
        signals[
            "P_MeanReverting"
        ] >= 0.60,
        "MR",
        "TREND",
    )

    signals[
        "Symbol"
    ] = symbol

    return signals


# =========================================================
# RUN SYMBOL
# =========================================================

def analyze_symbol(
    symbol,
):

    print()
    print(
        "=" * 110
    )

    print(
        f"ANALYZING {symbol}"
    )

    print(
        "=" * 110
    )

    # -----------------------------------------------------
    # Load
    # -----------------------------------------------------

    df = load_mt5_h1(
        symbol
    )

    print(
        f"Bars: {len(df):,}"
    )

    # -----------------------------------------------------
    # Kalman
    # -----------------------------------------------------

    print(
        "Calculating Kalman..."
    )

    df = add_kalman_features(
        df
    )

    # -----------------------------------------------------
    # Signals
    # -----------------------------------------------------

    df = generate_signals(
        df
    )

    signal_count = int(
        (
            df["Signal"]
            != "NONE"
        ).sum()
    )

    print(
        f"Kalman signals: "
        f"{signal_count:,}"
    )

    # -----------------------------------------------------
    # HMM
    # -----------------------------------------------------

    print(
        "Calculating causal HMM..."
    )

    df = add_markov_regime(
        df,
        hmm_window=HMM_WINDOW,
        min_observations=HMM_MIN_OBSERVATIONS,
        refit_every=HMM_REFIT_EVERY,
    )

    valid_hmm = int(
        df[
            "P_MeanReverting"
        ]
        .notna()
        .sum()
    )

    print(
        f"HMM valid bars: "
        f"{valid_hmm:,}"
    )

    # -----------------------------------------------------
    # Forward returns
    # -----------------------------------------------------

    print(
        "Calculating forward returns..."
    )

    df = add_forward_returns(
        df
    )

    # -----------------------------------------------------
    # Signal dataset
    # -----------------------------------------------------

    signal_df = (
        build_signal_dataset(
            df,
            symbol,
        )
    )

    if len(signal_df) == 0:

        print(
            "No valid signals."
        )

        return (
            pd.DataFrame(),
            pd.DataFrame(),
        )

    # -----------------------------------------------------
    # Save raw signal dataset
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
        "HMMState",
        "MR_Autocorrelation",
        "Trend_Autocorrelation",
    ]

    for horizon in (
        FORWARD_HORIZONS
    ):

        signal_columns.extend(
            [
                f"ForwardReturn_{horizon}h",
                f"DirectionalReturn_{horizon}h",
            ]
        )

    signal_columns = [
        column
        for column in signal_columns
        if column in signal_df.columns
    ]

    signal_df[
        signal_columns
    ].to_csv(
        symbol_dir
        / "signal_expectancy.csv",
        index=False,
    )

    # =====================================================
    # STATISTICS
    # =====================================================

    all_records = []

    regimes = [
        "ALL",
        "MR",
        "TREND",
    ]

    sides = [
        "LONG",
        "SHORT",
    ]

    for regime in regimes:

        for side in sides:

            records = analyze_group(
                df,
                symbol,
                regime,
                side,
            )

            all_records.extend(
                records
            )

    statistics_df = pd.DataFrame(
        all_records
    )

    statistics_df.to_csv(
        symbol_dir
        / "expectancy_statistics.csv",
        index=False,
    )

    # =====================================================
    # CONSOLE OUTPUT
    # =====================================================

    print()
    print(
        f"{symbol} EXPECTANCY"
    )

    print(
        "-" * 120
    )

    print(
        f"{'Regime':<10}"
        f"{'Side':<10}"
        f"{'Horizon':<10}"
        f"{'N':>8}"
        f"{'Mean':>14}"
        f"{'Median':>14}"
        f"{'WinRate':>12}"
    )

    print(
        "-" * 120
    )

    for _, row in (
        statistics_df.iterrows()
    ):

        print(
            f"{row['Regime']:<10}"
            f"{row['Side']:<10}"
            f"{int(row['HorizonHours']):>6}h"
            f"{int(row['Count']):>8}"
            f"{row['Mean'] * 100:>13.4f}%"
            f"{row['Median'] * 100:>13.4f}%"
            f"{row['WinRate'] * 100:>11.2f}%"
        )

    return (
        statistics_df,
        signal_df,
    )


# =========================================================
# MAIN
# =========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--symbol",
        type=str,
        default="ALL",
        help=(
            "Instrument to analyze. "
            "Use ALL for all instruments."
        ),
    )

    args = parser.parse_args()

    if (
        args.symbol.upper()
        == "ALL"
    ):

        symbols = (
            INSTRUMENTS
        )

    else:

        if (
            args.symbol
            not in INSTRUMENTS
        ):

            raise ValueError(
                f"Unknown symbol: "
                f"{args.symbol}"
            )

        symbols = [
            args.symbol
        ]

    print()
    print(
        "=" * 110
    )

    print(
        "KALMAN + MARKOV CONDITIONAL EXPECTANCY ANALYSIS"
    )

    print(
        "=" * 110
    )

    print()
    print(
        "Locked Kalman parameters:"
    )

    print(
        f"  PHI        = {PHI}"
    )

    print(
        f"  Q          = {Q}"
    )

    print(
        f"  R          = {R}"
    )

    print(
        f"  Mean       = {MEAN_WINDOW}"
    )

    print(
        f"  Z window   = {Z_WINDOW}"
    )

    print(
        f"  Entry Z    = {ENTRY_Z}"
    )

    print()
    print(
        "Markov:"
    )

    print(
        f"  Window     = {HMM_WINDOW}"
    )

    print(
        f"  Refit      = {HMM_REFIT_EVERY}h"
    )

    print(
        "  Threshold  = 0.60"
    )

    print()
    print(
        "No optimization."
    )

    all_statistics = []

    all_signals = []

    # =====================================================
    # LOOP
    # =====================================================

    for symbol in symbols:

        try:

            (
                statistics_df,
                signal_df,
            ) = analyze_symbol(
                symbol
            )

            if len(
                statistics_df
            ) > 0:

                all_statistics.append(
                    statistics_df
                )

            if len(
                signal_df
            ) > 0:

                all_signals.append(
                    signal_df
                )

        except Exception as error:

            print()
            print(
                f"ERROR: {symbol}"
            )

            print(
                repr(error)
            )

    # =====================================================
    # COMBINE
    # =====================================================

    if len(
        all_statistics
    ) > 0:

        combined_statistics = (
            pd.concat(
                all_statistics,
                ignore_index=True,
            )
        )

        combined_statistics.to_csv(
            RESULTS_DIR
            / "all_expectancy_statistics.csv",
            index=False,
        )

    if len(
        all_signals
    ) > 0:

        combined_signals = (
            pd.concat(
                all_signals,
                ignore_index=True,
            )
        )

        combined_signals.to_csv(
            RESULTS_DIR
            / "all_signal_expectancy.csv",
            index=False,
        )

    # =====================================================
    # FINAL CROSS-ASSET SUMMARY
    # =====================================================

    if len(
        all_statistics
    ) == 0:

        print(
            "No results."
        )

        return

    combined = (
        pd.concat(
            all_statistics,
            ignore_index=True,
        )
    )

    print()
    print()
    print(
        "=" * 130
    )

    print(
        "CROSS-ASSET EXPECTANCY SUMMARY"
    )

    print(
        "=" * 130
    )

    # -----------------------------------------------------
    # Focus on MR vs Trend
    # -----------------------------------------------------

    focus = combined[
        combined["Regime"]
        .isin(
            [
                "MR",
                "TREND",
            ]
        )
        &
        (combined["HorizonHours"] == 6)
    ]

    if len(focus) > 0:

        print()
        print(
            "6-HOUR DIRECTIONAL EXPECTANCY"
        )

        print(
            "-" * 110
        )

        print(
            f"{'Symbol':<12}"
            f"{'Side':<10}"
            f"{'Regime':<10}"
            f"{'N':>8}"
            f"{'Mean':>14}"
            f"{'Median':>14}"
            f"{'WinRate':>12}"
        )

        print(
            "-" * 110
        )

        for _, row in (
            focus.iterrows()
        ):

            print(
                f"{row['Symbol']:<12}"
                f"{row['Side']:<10}"
                f"{row['Regime']:<10}"
                f"{int(row['Count']):>8}"
                f"{row['Mean'] * 100:>13.4f}%"
                f"{row['Median'] * 100:>13.4f}%"
                f"{row['WinRate'] * 100:>11.2f}%"
            )

    # -----------------------------------------------------
    # Aggregate MR
    # -----------------------------------------------------

    print()
    print(
        "AGGREGATE MR vs TREND"
    )

    print(
        "-" * 100
    )

    aggregate = (
        combined[
            combined["Regime"]
            .isin(
                [
                    "MR",
                    "TREND",
                ]
            )
        ]
        .groupby(
            [
                "Regime",
                "HorizonHours",
            ]
        )
        .agg(
            Signals=(
                "Count",
                "sum",
            ),

            MeanReturn=(
                "Mean",
                "mean",
            ),

            MedianReturn=(
                "Median",
                "mean",
            ),

            MeanWinRate=(
                "WinRate",
                "mean",
            ),
        )
        .reset_index()
    )

    for _, row in (
        aggregate.iterrows()
    ):

        print(
            f"{row['Regime']:<8}"
            f"{int(row['HorizonHours']):>4}h"
            f" | N={int(row['Signals']):>6}"
            f" | Mean="
            f"{row['MeanReturn'] * 100:>9.4f}%"
            f" | Median="
            f"{row['MedianReturn'] * 100:>9.4f}%"
            f" | Win="
            f"{row['MeanWinRate'] * 100:>7.2f}%"
        )

    print()
    print(
        "=" * 130
    )

    print(
        "DONE"
    )

    print(
        f"Results saved to:"
        f"\n{RESULTS_DIR}"
    )


if __name__ == "__main__":

    main()