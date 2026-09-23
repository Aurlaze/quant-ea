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

# Stage B still uses the SAME training period.
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
# KALMAN CONFIGURATIONS FROM STAGE A
# ============================================================
#
# These are the strongest/diverse configurations observed
# in Stage A.
#
# We are NOT optimizing Kalman parameters again here.
#
# Stage A:
#     PHI
#     Q/R
#
# Stage B:
#     Z-score trading parameters
#
# ============================================================

KALMAN_CONFIGS = [
    {
        "PHI": 0.980,
        "Q_R_Ratio": 0.10,
        "Q": 0.025,
        "R": 0.25,
    },

    {
        "PHI": 0.999,
        "Q_R_Ratio": 1.00,
        "Q": 0.250,
        "R": 0.25,
    },

    {
        "PHI": 0.997,
        "Q_R_Ratio": 1.00,
        "Q": 0.250,
        "R": 0.25,
    },

    {
        "PHI": 0.995,
        "Q_R_Ratio": 1.00,
        "Q": 0.250,
        "R": 0.25,
    },

    {
        "PHI": 0.990,
        "Q_R_Ratio": 1.00,
        "Q": 0.250,
        "R": 0.25,
    },
]


# ============================================================
# FIXED KALMAN PARAMETER
# ============================================================

MEAN_WINDOW = 168


# ============================================================
# Z-SCORE SEARCH SPACE
# ============================================================

Z_WINDOW_VALUES = [
    24,
    36,
    48,
    72,
    96,
]


ENTRY_Z_VALUES = [
    1.50,
    1.75,
    2.00,
    2.25,
    2.50,
]


EXIT_Z_VALUES = [
    0.25,
    0.50,
    0.75,
    1.00,
]


STOP_Z_VALUES = [
    3.0,
    3.5,
    4.0,
    4.5,
]


# ============================================================
# MAXIMUM HOLDING PERIOD
# ============================================================
#
# IMPORTANT:
# The assignment constraint is preserved.
#
# We are NOT allowing >24 trading hours.
# ============================================================

MAX_HOLD_HOURS = 24


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
# LOAD TRAINING DATA
# ============================================================

def load_training_data():

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

    # Remove duplicate timestamps
    combined = combined[
        ~combined.index.duplicated(
            keep="first"
        )
    ]

    # Sort chronologically
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
# CALCULATE MONTHLY STATISTICS
# ============================================================

def calculate_monthly_metrics(
    equity
):
    """
    Calculate monthly returns from the equity curve.

    This is used to evaluate the strategy against the
    assignment's monthly-performance requirement.
    """

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

    # Monthly final equity
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
# RUN ONE PARAMETER COMBINATION
# ============================================================

def run_parameter_set(
    bars,
    kalman_config,
    z_window,
    entry_z,
    exit_z,
    stop_z,
):

    phi = kalman_config["PHI"]

    q = kalman_config["Q"]

    r = kalman_config["R"]

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
    # --------------------------------------------------------

    residual_series = pd.Series(
        residuals,
        index=bars.index,
    )

    zscore = calculate_zscore(
        residual_series,
        window=z_window,
    )

    test_bars = bars.copy()

    test_bars["KalmanState"] = states

    test_bars["Residual"] = residuals

    test_bars["ZScore"] = zscore

    # --------------------------------------------------------
    # Existing backtester
    # --------------------------------------------------------

    backtester = KalmanBacktester(
        entry_z=entry_z,
        exit_z=exit_z,
        stop_z=stop_z,
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
    # Total return
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
    # Return
    # --------------------------------------------------------

    return {
        "PHI": phi,

        "Q_R_Ratio": (
            kalman_config[
                "Q_R_Ratio"
            ]
        ),

        "Q": q,

        "R": r,

        "Z_Window": z_window,

        "Entry_Z": entry_z,

        "Exit_Z": exit_z,

        "Stop_Z": stop_z,

        "Max_Hold_Hours": (
            MAX_HOLD_HOURS
        ),

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


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "KALMAN Z-SCORE OPTIMIZATION — STAGE B"
    )

    print(
        "=" * 70
    )

    print(
        "\nStage B:"
    )

    print(
        "Optimize Z-score trading parameters."
    )

    print(
        "\nKalman parameters are FIXED "
        "to selected Stage-A configurations."
    )

    print(
        "\nFixed:"
    )

    print(
        f"MEAN_WINDOW = {MEAN_WINDOW}"
    )

    print(
        f"MAX_HOLD    = {MAX_HOLD_HOURS}"
    )

    print(
        f"Kalman configurations: "
        f"{len(KALMAN_CONFIGS)}"
    )

    # ========================================================
    # LOAD DATA
    # ========================================================

    bars = load_training_data()

    # ========================================================
    # SEARCH SPACE
    # ========================================================

    z_combinations = list(
        itertools.product(
            Z_WINDOW_VALUES,
            ENTRY_Z_VALUES,
            EXIT_Z_VALUES,
            STOP_Z_VALUES,
        )
    )

    total_combinations = (
        len(KALMAN_CONFIGS)
        * len(z_combinations)
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
        f"Z-window values: "
        f"{len(Z_WINDOW_VALUES)}"
    )

    print(
        f"Entry values: "
        f"{len(ENTRY_Z_VALUES)}"
    )

    print(
        f"Exit values: "
        f"{len(EXIT_Z_VALUES)}"
    )

    print(
        f"Stop values: "
        f"{len(STOP_Z_VALUES)}"
    )

    print(
        f"\nZ-score combinations: "
        f"{len(z_combinations)}"
    )

    print(
        f"Kalman configurations: "
        f"{len(KALMAN_CONFIGS)}"
    )

    print(
        f"\nTotal backtests: "
        f"{total_combinations}"
    )

    # ========================================================
    # RUN
    # ========================================================

    results = []

    counter = 0

    for kalman_config in KALMAN_CONFIGS:

        print(
            "\n" + "-" * 70
        )

        print(
            "Kalman configuration:"
        )

        print(
            f"PHI={kalman_config['PHI']}, "
            f"Q/R={kalman_config['Q_R_Ratio']}"
        )

        print(
            "-" * 70
        )

        for (
            z_window,
            entry_z,
            exit_z,
            stop_z,
        ) in z_combinations:

            counter += 1

            result = run_parameter_set(
                bars,
                kalman_config,
                z_window,
                entry_z,
                exit_z,
                stop_z,
            )

            if result is not None:

                results.append(
                    result
                )

            if (
                counter % 50 == 0
                or counter == total_combinations
            ):

                print(
                    f"Progress: "
                    f"{counter}/"
                    f"{total_combinations}"
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
    # Save complete results
    # --------------------------------------------------------

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True,
    )

    all_results_path = os.path.join(
        RESULTS_DIR,
        "kalman_stage_B_zscore_all.csv",
    )

    results_df.to_csv(
        all_results_path,
        index=False,
    )

    # ========================================================
    # INITIAL RANKING
    # ========================================================
    #
    # We do NOT simply use total return.
    #
    # First filter for:
    #
    #     positive profit factor
    #     reasonable drawdown
    #     sufficient trades
    #
    # Then rank.
    # ========================================================

    candidates = results_df.copy()

    candidates = candidates[
        candidates["ProfitFactor"] > 1.0
    ]

    candidates = candidates[
        candidates["Trades"] >= 100
    ]

    candidates = candidates[
        candidates["Max_DD_%"] >= -30.0
    ]

    # --------------------------------------------------------
    # Sort candidates by:
    #
    # 1. Return
    # 2. Profit factor
    # 3. Drawdown
    #
    # This is only a screening stage.
    # Final selection happens after validation.
    # --------------------------------------------------------

    candidates = candidates.sort_values(
        by=[
            "Return_%",
            "ProfitFactor",
        ],
        ascending=[
            False,
            False,
        ],
    )

    top_path = os.path.join(
        RESULTS_DIR,
        "kalman_stage_B_zscore_candidates.csv",
    )

    candidates.to_csv(
        top_path,
        index=False,
    )

    # ========================================================
    # DISPLAY
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "TOP 30 STAGE-B CANDIDATES"
    )

    print(
        "=" * 70
    )

    if candidates.empty:

        print(
            "\nNo candidates passed the "
            "initial screening."
        )

        print(
            "\nThis is useful information: "
            "the current strategy may require "
            "further research rather than "
            "blind optimization."
        )

    else:

        display_columns = [
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
            candidates[
                display_columns
            ]
            .head(30)
            .to_string(
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
        f"All results:\n"
        f"{all_results_path}"
    )

    print(
        f"\nScreened candidates:\n"
        f"{top_path}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()