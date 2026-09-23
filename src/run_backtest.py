import numpy as np

from data import (
    load_exness_ticks,
    ticks_to_bars
)

from walk_forward_kalman import (
    walk_forward_kalman,
    calculate_zscore
)

from backtest import (
    KalmanBacktester,
    performance_report,
    plot_equity,
    plot_drawdown
)


# ============================================================
# CONFIG
# ============================================================

CSV_PATH = "data/Exness_XAUUSDm_2025_09.csv"

PHI = 0.995
Q = 0.05
R = 0.25

MEAN_WINDOW = 168
Z_WINDOW = 48


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # 1. LOAD EXNESS TICK DATA
    # ========================================================

    ticks = load_exness_ticks(
        CSV_PATH
    )

    print(
        f"Ticks loaded: {len(ticks):,}"
    )

    # ========================================================
    # 2. AGGREGATE TO HOURLY BARS
    # ========================================================

    bars = ticks_to_bars(
        ticks,
        timeframe="1h"
    )

    print(
        f"Hourly bars: {len(bars)}"
    )

    # ========================================================
    # 3. TRUE WALK-FORWARD KALMAN
    # ========================================================

    (
        state,
        variance,
        residual,
        rolling_mean
    ) = walk_forward_kalman(

        bars["Close"].values,

        phi=PHI,

        q=Q,

        r=R,

        mean_window=MEAN_WINDOW
    )

    bars["KalmanState"] = state

    bars["KalmanVariance"] = variance

    bars["Residual"] = residual

    bars["RollingMean"] = rolling_mean

    # ========================================================
    # 4. LEAKAGE-FREE Z-SCORE
    # ========================================================

    bars["ZScore"] = calculate_zscore(
        bars["Residual"],
        window=Z_WINDOW
    )

    # ========================================================
    # 5. DISPLAY SIGNAL STATISTICS
    # ========================================================

    valid_z = bars[
        "ZScore"
    ].dropna()

    print("\n")
    print("=" * 70)
    print("SIGNAL STATISTICS")
    print("=" * 70)

    print(
        f"Valid Z-score observations: "
        f"{len(valid_z)}"
    )

    print(
        f"Z-score mean: "
        f"{valid_z.mean():.4f}"
    )

    print(
        f"Z-score std: "
        f"{valid_z.std():.4f}"
    )

    print(
        f"Minimum Z-score: "
        f"{valid_z.min():.4f}"
    )

    print(
        f"Maximum Z-score: "
        f"{valid_z.max():.4f}"
    )

    print(
        f"Long signals: "
        f"{(valid_z <= -2.0).sum()}"
    )

    print(
        f"Short signals: "
        f"{(valid_z >= 2.0).sum()}"
    )

    # ========================================================
    # 6. BACKTEST
    # ========================================================

    backtester = KalmanBacktester(

        # Signal
        entry_z=2.0,

        exit_z=0.5,

        stop_z=3.5,

        max_hold_hours=24,

        # Account
        initial_capital=10_000.0,

        # Risk
        risk_per_trade=0.005,

        # No more than 1x account notional
        max_notional_fraction=1.0,

        # Temporary research sizing assumption
        assumed_stop_return=0.01
    )

    equity, trades = (
        backtester.run(bars)
    )

    # ========================================================
    # 7. PERFORMANCE
    # ========================================================

    performance_report(
        equity,
        trades
    )

    # ========================================================
    # 8. SAVE
    # ========================================================

    equity.to_csv(
        "results/XAUUSDm_walkforward_equity.csv"
    )

    trades.to_csv(
        "results/XAUUSDm_walkforward_trades.csv",
        index=False
    )

    bars.to_csv(
        "results/XAUUSDm_walkforward_backtest_data.csv"
    )

    print("\nSaved:")

    print(
        "results/XAUUSDm_walkforward_equity.csv"
    )

    print(
        "results/XAUUSDm_walkforward_trades.csv"
    )

    print(
        "results/XAUUSDm_walkforward_backtest_data.csv"
    )

    # ========================================================
    # 9. PLOTS
    # ========================================================

    plot_equity(
        equity
    )

    plot_drawdown(
        equity
    )


if __name__ == "__main__":

    main()