import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from data import load_exness_ticks, ticks_to_bars
from kalman import MeanRevertingKalman


# ============================================================
# CONFIG
# ============================================================

CSV_PATH = "../data/Exness_XAUUSDm_2026_09.csv"

PHI = 0.995
Q = 0.05
R = 0.25

Z_WINDOW = 48

ENTRY_Z = 2.0

HORIZONS = [1, 2, 4, 8, 12, 24]


# ============================================================
# LOAD + KALMAN
# ============================================================

def prepare_data():

    ticks = load_exness_ticks(CSV_PATH)

    bars = ticks_to_bars(
        ticks,
        timeframe="1h"
    )

    prices = bars["Close"].dropna()

    kalman = MeanRevertingKalman(
        phi=PHI,
        q=Q,
        r=R
    )

    states, variances, residuals = (
        kalman.fit_transform(prices)
    )

    bars = bars.loc[prices.index].copy()

    bars["KalmanState"] = states
    bars["KalmanVariance"] = variances
    bars["Residual"] = residuals

    # Rolling statistics
    bars["ResidualMean"] = (
        bars["Residual"]
        .rolling(Z_WINDOW)
        .mean()
    )

    bars["ResidualStd"] = (
        bars["Residual"]
        .rolling(Z_WINDOW)
        .std()
    )

    bars["ZScore"] = (
        (
            bars["Residual"]
            - bars["ResidualMean"]
        )
        / bars["ResidualStd"]
    )

    return bars


# ============================================================
# FUTURE RESIDUAL ANALYSIS
# ============================================================

def analyze_extreme_events(bars):

    print("\n")
    print("=" * 70)
    print("KALMAN RESIDUAL MEAN-REVERSION TEST")
    print("=" * 70)

    z = bars["ZScore"]
    residual = bars["Residual"]

    # Extreme observations
    positive_events = bars[
        z > ENTRY_Z
    ].copy()

    negative_events = bars[
        z < -ENTRY_Z
    ].copy()

    print("\nExtreme events")
    print("-" * 40)

    print(
        f"Positive Z > +{ENTRY_Z}: "
        f"{len(positive_events)}"
    )

    print(
        f"Negative Z < -{ENTRY_Z}: "
        f"{len(negative_events)}"
    )

    # --------------------------------------------------------
    # Analyze future movement
    # --------------------------------------------------------

    results = []

    for horizon in HORIZONS:

        # Future residual
        future_residual = (
            residual.shift(-horizon)
        )

        # Change in residual
        residual_change = (
            future_residual - residual
        )

        # Positive extreme
        positive_change = (
            residual_change.loc[
                positive_events.index
            ]
            .dropna()
        )

        # Negative extreme
        negative_change = (
            residual_change.loc[
                negative_events.index
            ]
            .dropna()
        )

        # For positive residual:
        # mean reversion means residual should FALL.
        positive_reversion_rate = (
            positive_change < 0
        ).mean()

        # For negative residual:
        # mean reversion means residual should RISE.
        negative_reversion_rate = (
            negative_change > 0
        ).mean()

        results.append({

            "Horizon": f"{horizon}H",

            "Positive_Event_Count":
                len(positive_change),

            "Positive_Mean_Residual_Change":
                positive_change.mean(),

            "Positive_Median_Residual_Change":
                positive_change.median(),

            "Positive_Reversion_Rate":
                positive_reversion_rate,

            "Negative_Event_Count":
                len(negative_change),

            "Negative_Mean_Residual_Change":
                negative_change.mean(),

            "Negative_Median_Residual_Change":
                negative_change.median(),

            "Negative_Reversion_Rate":
                negative_reversion_rate,
        })

    result_df = pd.DataFrame(results)

    return result_df


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(results):

    print("\n")
    print("=" * 70)
    print("FUTURE RESIDUAL BEHAVIOR")
    print("=" * 70)

    pd.set_option(
        "display.max_columns",
        None
    )

    print(
        results.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print("\n")
    print("Interpretation:")
    print("-" * 70)

    print(
        "Positive event:"
    )

    print(
        "Z > +2 followed by a NEGATIVE "
        "residual change = reversion."
    )

    print()

    print(
        "Negative event:"
    )

    print(
        "Z < -2 followed by a POSITIVE "
        "residual change = reversion."
    )


# ============================================================
# RESIDUAL ADF TEST
# ============================================================

def residual_stationarity_test(bars):

    residual = (
        bars["Residual"]
        .dropna()
    )

    statistic, pvalue, *_ = (
        adfuller(residual)
    )

    print("\n")
    print("=" * 70)
    print("RESIDUAL STATIONARITY TEST")
    print("=" * 70)

    print(
        f"ADF statistic: {statistic:.6f}"
    )

    print(
        f"p-value      : {pvalue:.8f}"
    )

    if pvalue < 0.05:

        print(
            "\nResult: Reject the unit-root "
            "hypothesis at the 5% level."
        )

        print(
            "The residual is statistically "
            "consistent with stationarity "
            "in this sample."
        )

    else:

        print(
            "\nResult: Cannot reject the "
            "unit-root hypothesis."
        )

        print(
            "The residual does not show "
            "strong evidence of stationarity "
            "in this sample."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("Loading XAUUSDm...")

    bars = prepare_data()

    print(
        f"Hourly observations: {len(bars)}"
    )

    results = analyze_extreme_events(
        bars
    )

    print_results(results)

    residual_stationarity_test(
        bars
    )

    # Save results
    results.to_csv(
        "../kalman_reversion_results.csv",
        index=False
    )

    print("\n")
    print(
        "Saved: ../kalman_reversion_results.csv"
    )


if __name__ == "__main__":
    main()