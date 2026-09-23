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
# VALIDATION PERIOD
# ------------------------------------------------------------
#
# These years were NOT used during Stage A/B optimization.
#
# Training:
#   2019-2022
#
# Validation:
#   2023-2024
#
# Final test later:
#   2025-2026
# ------------------------------------------------------------

VALIDATION_YEARS = [2023, 2024]

TIMEFRAME = "1h"


# ============================================================
# CAPITAL / RISK SETTINGS
# ============================================================

INITIAL_CAPITAL = 10_000.0

RISK_PER_TRADE = 0.005

MAX_NOTIONAL_FRACTION = 1.0

ASSUMED_STOP_RETURN = 0.01


# ============================================================
# FIXED STRATEGY PARAMETER
# ============================================================

MEAN_WINDOW = 168

MAX_HOLD_HOURS = 24


# ============================================================
# FROZEN STAGE-B CANDIDATES
# ============================================================
#
# DO NOT OPTIMIZE THESE AGAIN.
#
# These candidates come from the stable region found during
# Stage B on 2019-2022.
#
# We test them unchanged on 2023-2024.
# ============================================================

CANDIDATES = [
    {
        "Name": "Candidate_1",
        "PHI": 0.997,
        "Q_R_Ratio": 1.0,
        "Q": 0.25,
        "R": 0.25,
        "Z_Window": 36,
        "Entry_Z": 2.25,
        "Exit_Z": 0.25,
        "Stop_Z": 3.0,
    },

    {
        "Name": "Candidate_2",
        "PHI": 0.997,
        "Q_R_Ratio": 1.0,
        "Q": 0.25,
        "R": 0.25,
        "Z_Window": 36,
        "Entry_Z": 2.25,
        "Exit_Z": 0.25,
        "Stop_Z": 4.0,
    },

    {
        "Name": "Candidate_3",
        "PHI": 0.997,
        "Q_R_Ratio": 1.0,
        "Q": 0.25,
        "R": 0.25,
        "Z_Window": 36,
        "Entry_Z": 2.25,
        "Exit_Z": 0.25,
        "Stop_Z": 4.5,
    },

    {
        "Name": "Candidate_4",
        "PHI": 0.999,
        "Q_R_Ratio": 1.0,
        "Q": 0.25,
        "R": 0.25,
        "Z_Window": 36,
        "Entry_Z": 2.25,
        "Exit_Z": 0.25,
        "Stop_Z": 3.0,
    },
]


# ============================================================
# LOAD ONE YEAR
# ============================================================

def load_year(year):

    filename = (
        f"Exness_XAUUSDm_{year}.csv"
    )

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
# LOAD VALIDATION DATA
# ============================================================

def load_validation_data():

    all_bars = []

    for year in VALIDATION_YEARS:

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
            "No validation data was loaded."
        )

    combined = pd.concat(
        all_bars
    )

    # Remove duplicate timestamps
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
        "VALIDATION DATA"
    )

    print(
        "=" * 70
    )

    print(
        f"Years: {VALIDATION_YEARS}"
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
# MONTHLY METRICS
# ============================================================

def calculate_monthly_metrics(
    equity
):

    if equity.empty:

        return {
            "Profitable_Months": 0,
            "Losing_Months": 0,
            "Average_Monthly_Return_%": np.nan,
            "Median_Monthly_Return_%": np.nan,
            "Worst_Monthly_Return_%": np.nan,
        }

    equity_series = equity[
        "Equity"
    ].copy()

    monthly_equity = (
        equity_series
        .resample("ME")
        .last()
        .dropna()
    )

    if len(monthly_equity) < 2:

        return {
            "Profitable_Months": 0,
            "Losing_Months": 0,
            "Average_Monthly_Return_%": np.nan,
            "Median_Monthly_Return_%": np.nan,
            "Worst_Monthly_Return_%": np.nan,
        }

    monthly_returns = (
        monthly_equity
        .pct_change()
        .dropna()
        * 100
    )

    return {
        "Profitable_Months": int(
            (monthly_returns > 0).sum()
        ),

        "Losing_Months": int(
            (monthly_returns < 0).sum()
        ),

        "Average_Monthly_Return_%": (
            monthly_returns.mean()
        ),

        "Median_Monthly_Return_%": (
            monthly_returns.median()
        ),

        "Worst_Monthly_Return_%": (
            monthly_returns.min()
        ),
    }


# ============================================================
# RUN ONE FROZEN CANDIDATE
# ============================================================

def run_candidate(
    bars,
    candidate,
):

    print(
        "\n" + "-" * 70
    )

    print(
        f"Testing {candidate['Name']}"
    )

    print(
        "-" * 70
    )

    print(
        f"PHI       = {candidate['PHI']}"
    )

    print(
        f"Q/R       = {candidate['Q_R_Ratio']}"
    )

    print(
        f"Z Window  = {candidate['Z_Window']}"
    )

    print(
        f"Entry Z   = {candidate['Entry_Z']}"
    )

    print(
        f"Exit Z    = {candidate['Exit_Z']}"
    )

    print(
        f"Stop Z    = {candidate['Stop_Z']}"
    )

    # --------------------------------------------------------
    # Kalman
    # --------------------------------------------------------

    states, variances, residuals, means = (
        walk_forward_kalman(
            bars["Close"],
            phi=candidate["PHI"],
            q=candidate["Q"],
            r=candidate["R"],
            mean_window=MEAN_WINDOW,
        )
    )

    # --------------------------------------------------------
    # Z-score
    # --------------------------------------------------------

    residual_series = pd.Series(
        residuals,
        index=bars.index,
    )

    zscore = calculate_zscore(
        residual_series,
        window=candidate["Z_Window"],
    )

    test_bars = bars.copy()

    test_bars["KalmanState"] = states

    test_bars["Residual"] = residuals

    test_bars["ZScore"] = zscore

    # --------------------------------------------------------
    # Backtest
    # --------------------------------------------------------

    backtester = KalmanBacktester(
        entry_z=candidate["Entry_Z"],
        exit_z=candidate["Exit_Z"],
        stop_z=candidate["Stop_Z"],
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

        print(
            "No equity data."
        )

        return None, None

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
    # Monthly statistics
    # --------------------------------------------------------

    monthly_metrics = (
        calculate_monthly_metrics(
            equity
        )
    )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    result = {

        "Candidate": candidate["Name"],

        "PHI": candidate["PHI"],

        "Q_R_Ratio": candidate["Q_R_Ratio"],

        "Z_Window": candidate["Z_Window"],

        "Entry_Z": candidate["Entry_Z"],

        "Exit_Z": candidate["Exit_Z"],

        "Stop_Z": candidate["Stop_Z"],

        "Max_Hold_Hours": MAX_HOLD_HOURS,

        "Return_%": total_return,

        "Max_DD_%": max_drawdown,

        "Sharpe": sharpe,

        "Trades": number_trades,

        "WinRate_%": win_rate,

        "ProfitFactor": profit_factor,

        "Profitable_Months": (
            monthly_metrics[
                "Profitable_Months"
            ]
        ),

        "Losing_Months": (
            monthly_metrics[
                "Losing_Months"
            ]
        ),

        "Average_Monthly_Return_%": (
            monthly_metrics[
                "Average_Monthly_Return_%"
            ]
        ),

        "Median_Monthly_Return_%": (
            monthly_metrics[
                "Median_Monthly_Return_%"
            ]
        ),

        "Worst_Monthly_Return_%": (
            monthly_metrics[
                "Worst_Monthly_Return_%"
            ]
        ),
    }

    return result, equity


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "KALMAN STRATEGY — OUT-OF-SAMPLE VALIDATION"
    )

    print(
        "=" * 70
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Parameters are FROZEN."
    )

    print(
        "No optimization is performed on 2023-2024."
    )

    print(
        "\nTraining period:"
    )

    print(
        "2019-2022"
    )

    print(
        "\nValidation period:"
    )

    print(
        "2023-2024"
    )

    print(
        "\nFinal test later:"
    )

    print(
        "2025-2026"
    )

    # ========================================================
    # LOAD DATA
    # ========================================================

    bars = load_validation_data()

    # ========================================================
    # RUN CANDIDATES
    # ========================================================

    results = []

    equity_results = {}

    for candidate in CANDIDATES:

        result, equity = run_candidate(
            bars,
            candidate,
        )

        if result is not None:

            results.append(
                result
            )

            equity_results[
                candidate["Name"]
            ] = equity

    # ========================================================
    # RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    if results_df.empty:

        print(
            "\nNo validation results."
        )

        return

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True,
    )

    summary_path = os.path.join(
        RESULTS_DIR,
        "kalman_validation_2023_2024.csv",
    )

    results_df.to_csv(
        summary_path,
        index=False,
    )

    # --------------------------------------------------------
    # Save equity
    # --------------------------------------------------------

    equity_frames = []

    for name, equity in (
        equity_results.items()
    ):

        temp = equity.copy()

        temp["Candidate"] = name

        equity_frames.append(
            temp
        )

    if equity_frames:

        equity_df = pd.concat(
            equity_frames
        )

        equity_path = os.path.join(
            RESULTS_DIR,
            "kalman_validation_2023_2024_equity.csv",
        )

        equity_df.to_csv(
            equity_path
        )

    # ========================================================
    # DISPLAY
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "VALIDATION RESULTS"
    )

    print(
        "=" * 70
    )

    display_columns = [

        "Candidate",

        "PHI",

        "Q_R_Ratio",

        "Z_Window",

        "Entry_Z",

        "Exit_Z",

        "Stop_Z",

        "Return_%",

        "Max_DD_%",

        "Sharpe",

        "Trades",

        "WinRate_%",

        "ProfitFactor",

        "Profitable_Months",

        "Losing_Months",

        "Average_Monthly_Return_%",

        "Worst_Monthly_Return_%",
    ]

    print(
        results_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "FILES SAVED"
    )

    print(
        "=" * 70
    )

    print(
        f"Summary:\n{summary_path}"
    )

    if equity_frames:

        print(
            f"\nEquity:\n{equity_path}"
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()