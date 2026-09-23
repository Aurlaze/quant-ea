"""
FINAL OUT-OF-SAMPLE TEST
========================

Locked Kalman parameters selected before this test.

IMPORTANT:
- DO NOT optimize parameters here.
- DO NOT change the strategy here.
- 2025-2026 data is treated as unseen/final test data.
"""

from pathlib import Path
import sys

import pandas as pd
import numpy as np


# ============================================================
# ALLOW IMPORTS FROM src/
# ============================================================

sys.path.append(str(Path(__file__).resolve().parent))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from data import load_exness_ticks, ticks_to_bars
from walk_forward_kalman import (
    walk_forward_kalman,
    calculate_zscore,
)
from backtest import KalmanBacktester


# ============================================================
# 1. LOCKED PARAMETERS
# ============================================================

# Kalman filter
PHI = 0.999
Q = 0.25
R = 0.25

# Mean / Z-score
MEAN_WINDOW = 168
Z_WINDOW = 36

# Trading rules
ENTRY_Z = 2.25
EXIT_Z = 0.25
STOP_Z = 3.0

# Maximum holding period
MAX_HOLD_HOURS = 24

# Portfolio / risk
INITIAL_CAPITAL = 10_000.0
RISK_PER_TRADE = 0.005
MAX_NOTIONAL_FRACTION = 1.0
ASSUMED_STOP_RETURN = 0.01


# ============================================================
# 2. PROJECT DIRECTORIES
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 3. FINAL TEST DATA
# ============================================================

FILES = [
    DATA_DIR / "Exness_XAUUSDm_2025.csv",
    DATA_DIR / "Exness_XAUUSDm_2026.csv",
]


# ============================================================
# 4. LOAD ONE YEAR OF DATA
# ============================================================

def load_year(file_path):

    print("\n" + "=" * 70)
    print(f"LOADING: {file_path.name}")
    print("=" * 70)

    # --------------------------------------------------------
    # Load and clean Exness ticks
    # --------------------------------------------------------

    ticks = load_exness_ticks(file_path)

    print(f"Ticks loaded: {len(ticks):,}")

    # --------------------------------------------------------
    # Convert ticks to hourly bars
    # --------------------------------------------------------

    bars = ticks_to_bars(
        ticks,
        timeframe="1h"
    )

    print(f"Hourly bars: {len(bars):,}")

    if len(bars) > 0:

        print(
            f"Start: {bars.index.min()}"
        )

        print(
            f"End:   {bars.index.max()}"
        )

    return bars


# ============================================================
# 5. PREPARE KALMAN + Z-SCORE
# ============================================================

def prepare_bars(bars):

    bars = bars.copy()

    print("\nCalculating Kalman filter...")

    # --------------------------------------------------------
    # Kalman filter
    # --------------------------------------------------------

    states, variances, residuals, means = walk_forward_kalman(
        bars["Close"],
        phi=PHI,
        q=Q,
        r=R,
        mean_window=MEAN_WINDOW,
    )

    bars["KalmanState"] = states
    bars["KalmanVariance"] = variances
    bars["Residual"] = residuals
    bars["KalmanMean"] = means

    # --------------------------------------------------------
    # Z-score
    #
    # IMPORTANT:
    # Pass the Series directly instead of .values.
    # This preserves the DatetimeIndex.
    # --------------------------------------------------------

    print("Calculating Z-score...")

    bars["ZScore"] = calculate_zscore(
        bars["Residual"],
        window=Z_WINDOW,
    )

    return bars


# ============================================================
# 6. CALCULATE PERFORMANCE METRICS
# ============================================================

def calculate_metrics(equity, trades):

    final_equity = equity["Equity"].iloc[-1]

    # --------------------------------------------------------
    # Total return
    # --------------------------------------------------------

    total_return = (
        final_equity / INITIAL_CAPITAL - 1
    ) * 100

    # --------------------------------------------------------
    # Maximum drawdown
    # --------------------------------------------------------

    running_max = equity["Equity"].cummax()

    drawdown = (
        equity["Equity"] / running_max - 1
    )

    max_drawdown = drawdown.min() * 100

    # --------------------------------------------------------
    # Trade statistics
    # --------------------------------------------------------

    if len(trades) > 0 and "PnL" in trades.columns:

        pnl = trades["PnL"]

        winning = pnl[pnl > 0]
        losing = pnl[pnl < 0]

        # Win rate
        win_rate = (
            len(winning)
            / len(pnl)
            * 100
        )

        # Gross profit / loss
        gross_profit = winning.sum()

        gross_loss = abs(
            losing.sum()
        )

        # Profit factor
        if gross_loss > 0:
            profit_factor = (
                gross_profit
                / gross_loss
            )
        else:
            profit_factor = np.inf

        # Expectancy
        expectancy = pnl.mean()

    else:

        win_rate = np.nan
        profit_factor = np.nan
        expectancy = np.nan

    # --------------------------------------------------------
    # Sharpe ratio
    # --------------------------------------------------------

    equity_returns = (
        equity["Equity"]
        .pct_change()
        .dropna()
    )

    if (
        len(equity_returns) > 1
        and equity_returns.std() > 0
    ):

        # Hourly observations.
        #
        # Approximate annualization:
        # 24 hours/day × 252 trading days.
        #
        sharpe = (
            equity_returns.mean()
            / equity_returns.std()
            * np.sqrt(24 * 252)
        )

    else:

        sharpe = np.nan

    # --------------------------------------------------------
    # Holding time
    # --------------------------------------------------------

    if len(trades) > 0:

        if "TradingHoursHeld" in trades.columns:

            avg_hold = (
                trades["TradingHoursHeld"]
                .mean()
            )

            max_hold = (
                trades["TradingHoursHeld"]
                .max()
            )

        elif "HoursHeld" in trades.columns:

            avg_hold = (
                trades["HoursHeld"]
                .mean()
            )

            max_hold = (
                trades["HoursHeld"]
                .max()
            )

        else:

            avg_hold = np.nan
            max_hold = np.nan

    else:

        avg_hold = np.nan
        max_hold = np.nan

    return {
        "InitialCapital": INITIAL_CAPITAL,
        "FinalEquity": final_equity,
        "ReturnPct": total_return,
        "MaxDrawdownPct": max_drawdown,
        "Sharpe": sharpe,
        "Trades": len(trades),
        "WinRatePct": win_rate,
        "ProfitFactor": profit_factor,
        "Expectancy": expectancy,
        "AvgTradingHoursHeld": avg_hold,
        "MaxTradingHoursHeld": max_hold,
    }


# ============================================================
# 7. RUN ONE FINAL TEST
# ============================================================

def run_test(bars, label):

    print("\n" + "=" * 70)
    print(f"FINAL TEST: {label}")
    print("=" * 70)

    # --------------------------------------------------------
    # Kalman + Z-score
    # --------------------------------------------------------

    bars = prepare_bars(bars)

    # --------------------------------------------------------
    # Remove observations where Z-score cannot be calculated
    # --------------------------------------------------------

    test_bars = bars.dropna(
        subset=["ZScore"]
    ).copy()

    print(
        f"\nUsable bars: {len(test_bars):,}"
    )

    if len(test_bars) == 0:

        print(
            "No valid Z-score observations."
        )

        return None, None

    # --------------------------------------------------------
    # Create backtester
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

    # --------------------------------------------------------
    # Run backtest
    # --------------------------------------------------------

    print("\nRunning backtest...")

    equity, trades = backtester.run(
        test_bars
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = calculate_metrics(
        equity,
        trades
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print(f"RESULTS: {label}")
    print("=" * 70)

    print(
        f"Initial Capital : "
        f"${metrics['InitialCapital']:,.2f}"
    )

    print(
        f"Final Equity    : "
        f"${metrics['FinalEquity']:,.2f}"
    )

    print(
        f"Return          : "
        f"{metrics['ReturnPct']:.3f}%"
    )

    print(
        f"Max Drawdown    : "
        f"{metrics['MaxDrawdownPct']:.3f}%"
    )

    print(
        f"Sharpe          : "
        f"{metrics['Sharpe']:.3f}"
    )

    print(
        f"Trades          : "
        f"{metrics['Trades']}"
    )

    print(
        f"Win Rate        : "
        f"{metrics['WinRatePct']:.2f}%"
    )

    print(
        f"Profit Factor   : "
        f"{metrics['ProfitFactor']:.3f}"
    )

    print(
        f"Expectancy      : "
        f"${metrics['Expectancy']:.3f}"
    )

    print(
        f"Avg Hold        : "
        f"{metrics['AvgTradingHoursHeld']:.2f} "
        f"trading hours"
    )

    print(
        f"Max Hold        : "
        f"{metrics['MaxTradingHoursHeld']:.2f} "
        f"trading hours"
    )

    # ========================================================
    # SAVE EQUITY
    # ========================================================

    equity_file = (
        RESULTS_DIR
        / f"final_test_{label}_equity.csv"
    )

    equity.to_csv(
        equity_file
    )

    # ========================================================
    # SAVE TRADES
    # ========================================================

    trades_file = (
        RESULTS_DIR
        / f"final_test_{label}_trades.csv"
    )

    trades.to_csv(
        trades_file,
        index=False
    )

    print("\nSaved:")

    print(
        f"  {equity_file}"
    )

    print(
        f"  {trades_file}"
    )

    # --------------------------------------------------------
    # Return summary
    # --------------------------------------------------------

    summary = {
        "Period": label,
        **metrics,
    }

    return summary, trades


# ============================================================
# 8. MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("LOCKED KALMAN FINAL OUT-OF-SAMPLE TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Display locked parameters
    # --------------------------------------------------------

    print("\nLOCKED PARAMETERS")
    print("-" * 70)

    print(
        f"PHI          = {PHI}"
    )

    print(
        f"Q            = {Q}"
    )

    print(
        f"R            = {R}"
    )

    print(
        f"MEAN_WINDOW  = {MEAN_WINDOW}"
    )

    print(
        f"Z_WINDOW     = {Z_WINDOW}"
    )

    print(
        f"ENTRY_Z      = {ENTRY_Z}"
    )

    print(
        f"EXIT_Z       = {EXIT_Z}"
    )

    print(
        f"STOP_Z       = {STOP_Z}"
    )

    print(
        f"MAX_HOLD     = {MAX_HOLD_HOURS}"
    )

    print(
        f"RISK/TRADE   = "
        f"{RISK_PER_TRADE * 100:.2f}%"
    )

    print(
        f"INITIAL CAPITAL = "
        f"${INITIAL_CAPITAL:,.2f}"
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    summaries = []

    # --------------------------------------------------------
    # Run each final-test year separately
    # --------------------------------------------------------

    for file_path in FILES:

        if not file_path.exists():

            print(
                f"\nWARNING: File not found:"
                f"\n{file_path}"
            )

            continue

        # Load ticks → hourly bars
        bars = load_year(
            file_path
        )

        if len(bars) == 0:

            print(
                "No bars available."
            )

            continue

        # Extract year
        label = (
            file_path.stem
            .replace(
                "Exness_XAUUSDm_",
                ""
            )
        )

        # Run final test
        summary, trades = run_test(
            bars,
            label
        )

        if summary is not None:

            summaries.append(
                summary
            )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    if summaries:

        summary_df = pd.DataFrame(
            summaries
        )

        print("\n\n")
        print("=" * 70)
        print("FINAL TEST SUMMARY")
        print("=" * 70)

        print(
            summary_df.to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.3f}"
            )
        )

        # ----------------------------------------------------
        # Save summary
        # ----------------------------------------------------

        summary_file = (
            RESULTS_DIR
            / "final_test_2025_2026_summary.csv"
        )

        summary_df.to_csv(
            summary_file,
            index=False
        )

        print("\nSaved summary:")

        print(
            f"  {summary_file}"
        )

    else:

        print(
            "\nNo final-test results were generated."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()