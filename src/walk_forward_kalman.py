import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from data import load_exness_ticks, ticks_to_bars


# ============================================================
# CONFIGURATION
# ============================================================

CSV_PATH = "../data/Exness_XAUUSDm_2026_09.csv"

TIMEFRAME = "1h"

# Kalman parameters
# These are still research parameters, NOT optimized.
PHI = 0.995
Q = 0.05
R = 0.25

# Historical observations used to estimate the local mean
MEAN_WINDOW = 168       # 7 days × 24 hours

# Historical residual observations used for Z-score
Z_WINDOW = 48           # 48 hours

# Extreme deviation threshold
ENTRY_Z = 2.0

# Future horizons to investigate
HORIZONS = [1, 2, 4, 8, 12, 24]


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("Loading Exness data...")

    ticks = load_exness_ticks(CSV_PATH)

    bars = ticks_to_bars(
        ticks,
        timeframe=TIMEFRAME
    )

    bars = bars.dropna(
        subset=["Close"]
    ).copy()

    return bars


# ============================================================
# WALK-FORWARD KALMAN FILTER
# ============================================================

def walk_forward_kalman(
    prices,
    phi=PHI,
    q=Q,
    r=R,
    mean_window=MEAN_WINDOW
):
    """
    Leakage-free Kalman filter.

    At time t, the long-run mean is calculated using
    observations BEFORE t.

    Therefore the current/future price cannot influence
    the mean used to generate the signal.
    """

    prices = np.asarray(
        prices,
        dtype=float
    )

    n = len(prices)

    states = np.full(n, np.nan)
    variances = np.full(n, np.nan)
    residuals = np.full(n, np.nan)
    means = np.full(n, np.nan)

    # Initial state
    state = prices[0]
    variance = r

    states[0] = state
    variances[0] = variance
    residuals[0] = 0.0

    for t in range(1, n):

        # ----------------------------------------------------
        # IMPORTANT:
        # Mean uses ONLY previous observations.
        # Current price is NOT included.
        # ----------------------------------------------------

        start = max(
            0,
            t - mean_window
        )

        historical_prices = prices[
            start:t
        ]

        if len(historical_prices) < 24:
            # Not enough history yet
            means[t] = np.nan
            continue

        mu = np.mean(
            historical_prices
        )

        means[t] = mu

        # ----------------------------------------------------
        # PREDICT
        # ----------------------------------------------------

        predicted_state = (
            phi * state
            + (1 - phi) * mu
        )

        predicted_variance = (
            phi ** 2 * variance
            + q
        )

        # ----------------------------------------------------
        # OBSERVATION / INNOVATION
        # ----------------------------------------------------

        innovation = (
            prices[t]
            - predicted_state
        )

        innovation_variance = (
            predicted_variance
            + r
        )

        # ----------------------------------------------------
        # KALMAN GAIN
        # ----------------------------------------------------

        kalman_gain = (
            predicted_variance
            / innovation_variance
        )

        # ----------------------------------------------------
        # UPDATE
        # ----------------------------------------------------

        state = (
            predicted_state
            + kalman_gain * innovation
        )

        variance = (
            (1 - kalman_gain)
            * predicted_variance
        )

        # ----------------------------------------------------
        # RESIDUAL
        # ----------------------------------------------------

        residual = (
            prices[t]
            - state
        )

        states[t] = state
        variances[t] = variance
        residuals[t] = residual

    return (
        states,
        variances,
        residuals,
        means
    )


# ============================================================
# ROLLING Z-SCORE
# ============================================================

def calculate_zscore(
    residuals,
    window=Z_WINDOW
):

    residual_series = pd.Series(
        residuals
    )

    # IMPORTANT:
    # Shift(1) means the current residual is NOT used
    # when calculating the mean/std used for its signal.

    previous_mean = (
        residual_series
        .rolling(window)
        .mean()
        .shift(1)
    )

    previous_std = (
        residual_series
        .rolling(window)
        .std()
        .shift(1)
    )

    zscore = (
        residual_series
        - previous_mean
    ) / previous_std

    return zscore


# ============================================================
# FUTURE PRICE RETURN ANALYSIS
# ============================================================

def analyze_future_returns(
    bars,
    entry_z=ENTRY_Z,
    horizons=HORIZONS
):

    results = []

    z = bars["ZScore"]

    positive_events = bars[
        z > entry_z
    ].index

    negative_events = bars[
        z < -entry_z
    ].index

    print("\n")
    print("=" * 80)
    print("FUTURE PRICE RETURN ANALYSIS")
    print("=" * 80)

    print(
        f"Positive events (Z > +{entry_z}): "
        f"{len(positive_events)}"
    )

    print(
        f"Negative events (Z < -{entry_z}): "
        f"{len(negative_events)}"
    )

    for horizon in horizons:

        future_price = (
            bars["Close"]
            .shift(-horizon)
        )

        future_return = (
            np.log(
                future_price
                / bars["Close"]
            )
        )

        positive_returns = (
            future_return
            .loc[positive_events]
            .dropna()
        )

        negative_returns = (
            future_return
            .loc[negative_events]
            .dropna()
        )

        # -----------------------------------------------
        # Positive extreme:
        # We SHORT, therefore negative future return
        # is favorable.
        # -----------------------------------------------

        positive_success = (
            positive_returns < 0
        ).mean()

        # -----------------------------------------------
        # Negative extreme:
        # We LONG, therefore positive future return
        # is favorable.
        # -----------------------------------------------

        negative_success = (
            negative_returns > 0
        ).mean()

        results.append({

            "Horizon_Hours": horizon,

            "Positive_N": len(
                positive_returns
            ),

            "Positive_Mean_Return": (
                positive_returns.mean()
            ),

            "Positive_Median_Return": (
                positive_returns.median()
            ),

            "Positive_Success_Rate": (
                positive_success
            ),

            "Negative_N": len(
                negative_returns
            ),

            "Negative_Mean_Return": (
                negative_returns.mean()
            ),

            "Negative_Median_Return": (
                negative_returns.median()
            ),

            "Negative_Success_Rate": (
                negative_success
            ),
        })

    return pd.DataFrame(results)


# ============================================================
# TIME TO REVERSION
# ============================================================

def calculate_time_to_reversion(
    bars,
    entry_z=ENTRY_Z,
    max_horizon=24
):

    z = bars["ZScore"].values

    times = []

    for i in range(len(z)):

        if not np.isfinite(z[i]):
            continue

        # Positive extreme
        if z[i] > entry_z:

            for h in range(
                1,
                max_horizon + 1
            ):

                j = i + h

                if j >= len(z):
                    break

                if np.isfinite(z[j]) and z[j] <= 0:

                    times.append({
                        "Type": "Positive",
                        "Bars_To_Reversion": h
                    })

                    break

        # Negative extreme
        elif z[i] < -entry_z:

            for h in range(
                1,
                max_horizon + 1
            ):

                j = i + h

                if j >= len(z):
                    break

                if np.isfinite(z[j]) and z[j] >= 0:

                    times.append({
                        "Type": "Negative",
                        "Bars_To_Reversion": h
                    })

                    break

    return pd.DataFrame(times)


# ============================================================
# RESIDUAL STATIONARITY
# ============================================================

def residual_stationarity_test(
    residuals
):

    residuals = pd.Series(
        residuals
    ).dropna()

    statistic, pvalue, *_ = (
        adfuller(
            residuals,
            autolag="AIC"
        )
    )

    print("\n")
    print("=" * 80)
    print("WALK-FORWARD RESIDUAL ADF TEST")
    print("=" * 80)

    print(
        f"ADF statistic: {statistic:.6f}"
    )

    print(
        f"p-value: {pvalue:.10f}"
    )

    if pvalue < 0.05:

        print(
            "\nADF conclusion:"
        )

        print(
            "Reject the unit-root null hypothesis "
            "at the 5% significance level."
        )

    else:

        print(
            "\nADF conclusion:"
        )

        print(
            "Cannot reject the unit-root "
            "null hypothesis at the 5% level."
        )


# ============================================================
# PLOT
# ============================================================

def plot_results(bars):

    import matplotlib.pyplot as plt

    # -----------------------------------------------
    # Price vs Kalman state
    # -----------------------------------------------

    plt.figure(
        figsize=(14, 6)
    )

    plt.plot(
        bars.index,
        bars["Close"],
        label="XAUUSD Mid"
    )

    plt.plot(
        bars.index,
        bars["KalmanState"],
        label="Walk-Forward Kalman State"
    )

    plt.title(
        "XAUUSDm — Leakage-Free Kalman Filter"
    )

    plt.xlabel("Time")
    plt.ylabel("Price")

    plt.legend()

    plt.tight_layout()

    plt.show()

    # -----------------------------------------------
    # Z-score
    # -----------------------------------------------

    plt.figure(
        figsize=(14, 5)
    )

    plt.plot(
        bars.index,
        bars["ZScore"],
        label="Z-score"
    )

    plt.axhline(
        ENTRY_Z,
        linestyle="--",
        label="+2 Entry"
    )

    plt.axhline(
        -ENTRY_Z,
        linestyle="--",
        label="-2 Entry"
    )

    plt.axhline(
        0,
        linestyle=":"
    )

    plt.title(
        "XAUUSDm — Walk-Forward Kalman Z-score"
    )

    plt.xlabel("Time")
    plt.ylabel("Z-score")

    plt.legend()

    plt.tight_layout()

    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    bars = load_data()

    print(
        f"\nHourly observations: {len(bars)}"
    )

    # --------------------------------------------------------
    # Walk-forward Kalman
    # --------------------------------------------------------

    (
        states,
        variances,
        residuals,
        means
    ) = walk_forward_kalman(
        bars["Close"].values
    )

    bars["KalmanState"] = states
    bars["KalmanVariance"] = variances
    bars["Residual"] = residuals
    bars["RollingMean"] = means

    # --------------------------------------------------------
    # Z-score
    # --------------------------------------------------------

    bars["ZScore"] = calculate_zscore(
        bars["Residual"]
    )

    # --------------------------------------------------------
    # Basic statistics
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("WALK-FORWARD KALMAN STATISTICS")
    print("=" * 80)

    print(
        f"Phi: {PHI}"
    )

    print(
        f"Q: {Q}"
    )

    print(
        f"R: {R}"
    )

    print(
        f"Mean window: {MEAN_WINDOW} hours"
    )

    print(
        f"Z-score window: {Z_WINDOW} hours"
    )

    print("\nResidual statistics:")

    print(
        bars["Residual"]
        .dropna()
        .describe()
    )

    print("\nZ-score statistics:")

    print(
        bars["ZScore"]
        .dropna()
        .describe()
    )

    # --------------------------------------------------------
    # ADF
    # --------------------------------------------------------

    residual_stationarity_test(
        bars["Residual"]
    )

    # --------------------------------------------------------
    # Future returns
    # --------------------------------------------------------

    return_results = (
        analyze_future_returns(
            bars
        )
    )

    print("\n")
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)

    print(
        return_results.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    # --------------------------------------------------------
    # Time to reversion
    # --------------------------------------------------------

    reversion_times = (
        calculate_time_to_reversion(
            bars
        )
    )

    print("\n")
    print("=" * 80)
    print("TIME TO ZERO-CROSSING")
    print("=" * 80)

    if len(reversion_times) > 0:

        print(
            reversion_times
            .groupby("Type")[
                "Bars_To_Reversion"
            ]
            .describe()
        )

        print(
            "\nOverall median time:",
            reversion_times[
                "Bars_To_Reversion"
            ].median(),
            "hours"
        )

    else:

        print(
            "No Z-score events crossed zero "
            "within the 24-hour observation window."
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    bars.to_csv(
        "../XAUUSDm_walk_forward_kalman.csv"
    )

    return_results.to_csv(
        "../XAUUSDm_future_return_results.csv",
        index=False
    )

    if len(reversion_times) > 0:

        reversion_times.to_csv(
            "../XAUUSDm_reversion_times.csv",
            index=False
        )

    print("\n")
    print(
        "Saved walk-forward research data."
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plot_results(bars)


if __name__ == "__main__":
    main()