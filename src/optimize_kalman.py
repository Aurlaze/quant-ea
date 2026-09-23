import itertools
import os
import numpy as np
import pandas as pd

from data import load_exness_ticks, ticks_to_bars
from walk_forward_kalman import (
    walk_forward_kalman,
    calculate_zscore,
)
from backtest import KalmanBacktester


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "data"
RESULTS_DIR = "results"

# ------------------------------------------------------------
# TRAINING PERIOD
# ------------------------------------------------------------
# Stage A uses 2019-2022 only.
#
# 2023-2024 = validation later
# 2025-2026 = untouched final test later
# ------------------------------------------------------------

TRAIN_YEARS = [2019, 2020, 2021, 2022]

TIMEFRAME = "1h"


# ============================================================
# CAPITAL / RISK SETTINGS
# ============================================================

INITIAL_CAPITAL = 10_000.0

RISK_PER_TRADE = 0.005

MAX_NOTIONAL_FRACTION = 1.0

ASSUMED_STOP_RETURN = 0.01


# ============================================================
# FIXED STRATEGY PARAMETERS
# ============================================================
# These are NOT optimized in Stage A.
#
# We are only optimizing:
#
#     PHI
#     Q/R
#
# Everything below stays exactly as the current strategy.
# ============================================================

MEAN_WINDOW = 168

Z_WINDOW = 48

ENTRY_Z = 2.0

EXIT_Z = 0.5

STOP_Z = 3.5

MAX_HOLD_HOURS = 24


# ============================================================
# KALMAN PARAMETER SEARCH
# ============================================================

# PHI controls how persistent the latent price state is.

PHI_VALUES = [
    0.970,
    0.980,
    0.985,
    0.990,
    0.995,
    0.997,
    0.999,
]


# ------------------------------------------------------------
# Effective process/measurement uncertainty ratio:
#
#     Q / R
#
# Instead of independently testing Q and R, we test their
# effective ratio.
# ------------------------------------------------------------

Q_R_VALUES = [
    0.001,
    0.005,
    0.010,
    0.025,
    0.050,
    0.100,
    0.200,
    0.500,
    1.000,
    2.000,
    5.000,
]


# ------------------------------------------------------------
# Keep R fixed.
#
# Q will be calculated as:
#
#     Q = R * (Q/R)
#
# ------------------------------------------------------------

R_FIXED = 0.25


# ============================================================
# LOAD ONE YEAR
# ============================================================

def load_year(year):
    """
    Load one Exness yearly tick file and convert it
    to hourly bars.
    """

    filename = f"Exness_XAUUSDm_{year}.csv"

    csv_path = os.path.join(
        DATA_DIR,
        filename,
    )

    if not os.path.exists(csv_path):

        raise FileNotFoundError(
            f"Could not find:\n{csv_path}"
        )

    print(
        f"\nLoading {filename}..."
    )

    ticks = load_exness_ticks(
        csv_path
    )

    print(
        f"  Ticks: {len(ticks):,}"
    )

    print(
        "  Converting ticks to hourly bars..."
    )

    bars = ticks_to_bars(
        ticks
    )

    print(
        f"  Hourly bars: {len(bars):,}"
    )

    if len(bars) > 0:

        print(
            f"  Period: "
            f"{bars.index.min()} "
            f"→ "
            f"{bars.index.max()}"
        )

    return bars


# ============================================================
# LOAD TRAINING DATA
# ============================================================

def load_training_data():
    """
    Load all training years.

    Raw tick files are converted to hourly bars first.
    Only the hourly bars are combined.
    """

    all_bars = []

    for year in TRAIN_YEARS:

        bars = load_year(year)

        if bars.empty:

            print(
                f"WARNING: {year} contains no bars."
            )

            continue

        all_bars.append(
            bars
        )

    if not all_bars:

        raise RuntimeError(
            "No training data was loaded."
        )

    combined = pd.concat(
        all_bars
    )

    # Remove duplicated timestamps
    combined = combined[
        ~combined.index.duplicated(
            keep="first"
        )
    ]

    # Chronological order
    combined = combined.sort_index()

    print(
        "\n" + "=" * 70
    )

    print(
        "TRAINING DATA"
    )

    print(
        "=" * 70
    )

    print(
        f"Years: {TRAIN_YEARS}"
    )

    print(
        f"Total hourly bars: "
        f"{len(combined):,}"
    )

    print(
        f"Start: "
        f"{combined.index.min()}"
    )

    print(
        f"End:   "
        f"{combined.index.max()}"
    )

    return combined


# ============================================================
# RUN ONE PARAMETER COMBINATION
# ============================================================

def run_parameter_set(
    bars,
    phi,
    q_r_ratio,
):
    """
    Run the existing Kalman strategy using one
    PHI and Q/R combination.

    R is fixed.

    Q = R * (Q/R)
    """

    # --------------------------------------------------------
    # Calculate Q
    # --------------------------------------------------------

    q = R_FIXED * q_r_ratio

    r = R_FIXED

    # --------------------------------------------------------
    # Kalman filter
    # --------------------------------------------------------

    states, variances, residuals, means = (
        walk_forward_kalman(
            bars["Close"],
            phi=phi,
            q=q,
            r=r,
            mean_window=MEAN_WINDOW,
        )
    )

    # --------------------------------------------------------
    # Z-score
    #
    # Keep datetime index aligned with bars.
    # --------------------------------------------------------

    residual_series = pd.Series(
        residuals,
        index=bars.index,
    )

    zscore = calculate_zscore(
        residual_series,
        window=Z_WINDOW,
    )

    test_bars = bars.copy()

    test_bars["KalmanState"] = states

    test_bars["Residual"] = residuals

    test_bars["ZScore"] = zscore

    # --------------------------------------------------------
    # Existing backtester
    # --------------------------------------------------------

    backtester = KalmanBacktester(
        entry_z=ENTRY_Z,
        exit_z=EXIT_Z,
        stop_z=STOP_Z,
        max_hold_hours=MAX_HOLD_HOURS,
        initial_capital=INITIAL_CAPITAL,
        risk_per_trade=RISK_PER_TRADE,
        max_notional_fraction=MAX_NOTIONAL_FRACTION,
        assumed_stop_return=ASSUMED_STOP_RETURN,
    )

    equity, trades = backtester.run(
        test_bars
    )

    if equity.empty:

        return None

    # ========================================================
    # PERFORMANCE
    # ========================================================

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    final_equity = equity[
        "Equity"
    ].iloc[-1]

    total_return = (
        final_equity
        / INITIAL_CAPITAL
        - 1
    ) * 100

    # --------------------------------------------------------
    # Maximum drawdown
    # --------------------------------------------------------

    running_max = equity[
        "Equity"
    ].cummax()

    drawdown = (
        equity["Equity"]
        / running_max
        - 1
    )

    max_drawdown = (
        drawdown.min()
        * 100
    )

    # --------------------------------------------------------
    # Sharpe
    # --------------------------------------------------------

    returns = (
        equity["Equity"]
        .pct_change()
        .dropna()
    )

    if (
        len(returns) > 1
        and returns.std() > 0
    ):

        sharpe = (
            returns.mean()
            / returns.std()
            * np.sqrt(24 * 252)
        )

    else:

        sharpe = np.nan

    # --------------------------------------------------------
    # Trade statistics
    # --------------------------------------------------------

    number_trades = len(
        trades
    )

    if number_trades > 0:

        winning_trades = (
            trades["PnL"] > 0
        ).sum()

        win_rate = (
            winning_trades
            / number_trades
            * 100
        )

        gross_profit = trades.loc[
            trades["PnL"] > 0,
            "PnL",
        ].sum()

        gross_loss = -trades.loc[
            trades["PnL"] < 0,
            "PnL",
        ].sum()

        if gross_loss > 0:

            profit_factor = (
                gross_profit
                / gross_loss
            )

        else:

            profit_factor = np.inf

    else:

        win_rate = np.nan

        profit_factor = np.nan

    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return {
        "PHI": phi,

        "Q_R_Ratio": q_r_ratio,

        "Q": q,

        "R": r,

        "Return_%": total_return,

        "Max_DD_%": max_drawdown,

        "Sharpe": sharpe,

        "Trades": number_trades,

        "WinRate_%": win_rate,

        "ProfitFactor": profit_factor,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "KALMAN PARAMETER OPTIMIZATION — STAGE A"
    )

    print(
        "=" * 70
    )

    print(
        "\nStage A:"
    )

    print(
        "Optimize PHI and Q/R only."
    )

    print(
        "\nFixed parameters:"
    )

    print(
        f"MEAN_WINDOW = {MEAN_WINDOW}"
    )

    print(
        f"Z_WINDOW    = {Z_WINDOW}"
    )

    print(
        f"ENTRY_Z     = {ENTRY_Z}"
    )

    print(
        f"EXIT_Z      = {EXIT_Z}"
    )

    print(
        f"STOP_Z      = {STOP_Z}"
    )

    print(
        f"MAX_HOLD    = {MAX_HOLD_HOURS}"
    )

    print(
        f"R_FIXED     = {R_FIXED}"
    )

    # ========================================================
    # LOAD DATA
    # ========================================================

    bars = load_training_data()

    # ========================================================
    # CREATE SEARCH SPACE
    # ========================================================

    combinations = list(
        itertools.product(
            PHI_VALUES,
            Q_R_VALUES,
        )
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "OPTIMIZATION SEARCH"
    )

    print(
        "=" * 70
    )

    print(
        f"PHI values: "
        f"{len(PHI_VALUES)}"
    )

    print(
        f"Q/R values: "
        f"{len(Q_R_VALUES)}"
    )

    print(
        f"\nTotal combinations: "
        f"{len(combinations)}"
    )

    # ========================================================
    # RUN
    # ========================================================

    results = []

    for i, (
        phi,
        q_r_ratio,
    ) in enumerate(
        combinations,
        start=1,
    ):

        result = run_parameter_set(
            bars,
            phi=phi,
            q_r_ratio=q_r_ratio,
        )

        if result is not None:

            results.append(
                result
            )

        if (
            i % 10 == 0
            or i == len(combinations)
        ):

            print(
                f"Progress: "
                f"{i}/"
                f"{len(combinations)}"
            )

    # ========================================================
    # RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    if results_df.empty:

        print(
            "\nNo valid optimization results."
        )

        return

    # --------------------------------------------------------
    # Sort by return for initial inspection.
    # --------------------------------------------------------

    results_df = results_df.sort_values(
        by="Return_%",
        ascending=False,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True,
    )

    output_path = os.path.join(
        RESULTS_DIR,
        "kalman_stage_A_optimization_v2.csv",
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "TOP 20 PARAMETER COMBINATIONS"
    )

    print(
        "=" * 70
    )

    print(
        results_df.head(20).to_string(
            index=False
        )
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "RESULT"
    )

    print(
        "=" * 70
    )

    print(
        f"Saved to:\n"
        f"{output_path}"
    )


# ============================================================
# RUN SCRIPT
# ============================================================

if __name__ == "__main__":

    main()