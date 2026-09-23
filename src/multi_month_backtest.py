"""
Independent-month walk-forward Kalman backtest.

Each selected month is processed independently.

Therefore:
- Kalman state resets for every month
- Z-score history resets for every month
- Trading position resets for every month
- No trade crosses between selected months
- Parameters remain fixed
- No optimization is performed

Holding time:
- CalendarHoursHeld = actual wall-clock duration
- TradingHoursHeld  = available hourly bars held
- HoursHeld         = alias for TradingHoursHeld
"""

from pathlib import Path

import numpy as np
import pandas as pd

from data import ticks_to_bars
from walk_forward_kalman import (
    walk_forward_kalman,
    calculate_zscore,
)
from backtest import KalmanBacktester


# ============================================================
# PATHS
# ============================================================

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")


# ============================================================
# DATA FILES
# ============================================================

# Automatically discover YEARLY XAUUSDm files.
# Expected names:
#   Exness_XAUUSDm_2019.csv
#   Exness_XAUUSDm_2020.csv
#   ...
#   Exness_XAUUSDm_2026.csv
#
# Monthly files such as Exness_XAUUSDm_2025_01.csv are ignored.
YEAR_FILES = sorted(
    [
        p.name
        for p in DATA_DIR.glob("Exness_XAUUSDm_20??.csv")
        if p.is_file()
    ]
)

if not YEAR_FILES:
    raise FileNotFoundError(
        "No yearly XAUUSDm files found in data/. "
        "Expected files such as Exness_XAUUSDm_2019.csv."
    )


# ============================================================
# KALMAN PARAMETERS
# ============================================================

PHI = 0.995
Q = 0.05
R = 0.25

MEAN_WINDOW = 168
Z_WINDOW = 48


# ============================================================
# TRADING PARAMETERS
# ============================================================

Z_ENTRY = 2.0
Z_EXIT = 0.5
Z_STOP = 3.5

MAX_HOLD_HOURS = 24

INITIAL_CAPITAL = 10_000.0

RISK_PER_TRADE = 0.005

MAX_NOTIONAL_FRACTION = 1.0

ASSUMED_STOP_RETURN = 0.01


# ============================================================
# LOAD ONE MONTH
# ============================================================

def load_year_file(filename):

    path = DATA_DIR / filename

    print("\n" + "=" * 70)
    print(
        f"Loading: {filename}"
    )
    print("=" * 70)

    if not path.exists():

        raise FileNotFoundError(
            f"File not found: {path}\n"
            f"Current working directory: {Path.cwd()}"
        )

    df = pd.read_csv(path)

    print(
        f"Ticks loaded: {len(df):,}"
    )

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    if "Timestamp" not in df.columns:

        raise ValueError(
            "'Timestamp' column not found."
        )

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        utc=True,
        errors="coerce",
    )

    df = df.dropna(
        subset=["Timestamp"]
    )

    df = df.sort_values(
        "Timestamp"
    )

    df = df.drop_duplicates(
        subset=["Timestamp"]
    )

    # --------------------------------------------------------
    # Bid / Ask
    # --------------------------------------------------------

    if "Bid" not in df.columns:

        raise ValueError(
            "'Bid' column not found."
        )

    if "Ask" not in df.columns:

        raise ValueError(
            "'Ask' column not found."
        )

    # --------------------------------------------------------
    # Mid
    # --------------------------------------------------------

    if "Mid" not in df.columns:

        df["Mid"] = (
            df["Bid"]
            + df["Ask"]
        ) / 2.0

    # --------------------------------------------------------
    # Spread
    # --------------------------------------------------------

    if "Spread" not in df.columns:

        df["Spread"] = (
            df["Ask"]
            - df["Bid"]
        )

    # --------------------------------------------------------
    # Hourly bars
    # --------------------------------------------------------

    bars = ticks_to_bars(
        df,
        timeframe="1h",
    )

    bars = bars.sort_index()

    print(
        f"Hourly bars: {len(bars):,}"
    )

    if len(bars) > 0:

        print(
            f"Start: {bars.index.min()}"
        )

        print(
            f"End:   {bars.index.max()}"
        )

    return bars


# ============================================================
# PREPARE KALMAN + Z-SCORE
# ============================================================

def prepare_month(bars):

    bars = bars.copy()

    prices = (
        bars["Close"]
        .to_numpy()
    )

    (
        states,
        variances,
        residuals,
        means,
    ) = walk_forward_kalman(
        prices,
        phi=PHI,
        q=Q,
        r=R,
        mean_window=MEAN_WINDOW,
    )

    bars["KalmanState"] = states

    bars["KalmanVariance"] = (
        variances
    )

    bars["Residual"] = (
        residuals
    )

    bars["RollingMean"] = (
        means
    )

    # IMPORTANT:
    # Keep this as a pandas Series so the
    # DatetimeIndex is preserved.

    bars["ZScore"] = (
        calculate_zscore(
            bars["Residual"],
            window=Z_WINDOW,
        )
    )

    return bars


# ============================================================
# NORMALIZE EQUITY
# ============================================================

def normalize_equity(equity):

    if equity is None:

        return pd.Series(
            dtype=float
        )

    if isinstance(
        equity,
        pd.DataFrame,
    ):

        if equity.empty and len(equity.columns) == 0:
            return pd.Series(dtype=float)

        if "Equity" in equity.columns:

            series = equity[
                "Equity"
            ]

        elif "equity" in equity.columns:

            series = equity[
                "equity"
            ]

        elif len(equity.columns) == 1:

            series = equity.iloc[:, 0]

        else:

            raise ValueError(
                "Cannot identify equity column. "
                f"Columns: {list(equity.columns)}"
            )

    elif isinstance(
        equity,
        pd.Series,
    ):

        series = equity

    else:

        series = pd.Series(
            equity
        )

    series = pd.to_numeric(
        series,
        errors="coerce",
    )

    series = series.dropna()

    return series.astype(float)


# ============================================================
# FLAT EQUITY FOR NO-SIGNAL PERIOD
# ============================================================

def make_flat_equity(index):
    """
    Return a valid flat equity curve when a period has no valid
    Z-score observations. This prevents an empty backtester result
    from being mistaken for a processing failure.
    """
    if len(index) == 0:
        return pd.DataFrame(
            columns=["Cash", "Equity", "Position", "Units"],
            index=pd.DatetimeIndex([]),
        )

    return pd.DataFrame(
        {
            "Cash": INITIAL_CAPITAL,
            "Equity": INITIAL_CAPITAL,
            "Position": 0,
            "Units": 0.0,
        },
        index=index,
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    equity,
    trades,
):

    equity_series = normalize_equity(
        equity
    )

    if len(equity_series) == 0:

        return {
            "ReturnPct": np.nan,
            "MaxDDPct": np.nan,
            "Sharpe": np.nan,
            "Trades": 0,
            "WinRatePct": np.nan,
            "ProfitFactor": np.nan,
            "Expectancy": np.nan,
            "AvgHoldHours": np.nan,
        }

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    initial = float(
        equity_series.iloc[0]
    )

    final = float(
        equity_series.iloc[-1]
    )

    total_return = (
        final / initial - 1.0
    ) * 100.0

    # --------------------------------------------------------
    # Drawdown
    # --------------------------------------------------------

    running_max = (
        equity_series.cummax()
    )

    drawdown = (
        equity_series
        / running_max
        - 1.0
    )

    max_dd = (
        drawdown.min()
        * 100.0
    )

    # --------------------------------------------------------
    # Sharpe
    # --------------------------------------------------------

    returns = (
        equity_series
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
    # No trades
    # --------------------------------------------------------

    if (
        trades is None
        or len(trades) == 0
    ):

        return {
            "ReturnPct": total_return,
            "MaxDDPct": max_dd,
            "Sharpe": sharpe,
            "Trades": 0,
            "WinRatePct": np.nan,
            "ProfitFactor": np.nan,
            "Expectancy": np.nan,
            "AvgHoldHours": np.nan,
        }

    trades = trades.copy()

    # --------------------------------------------------------
    # PnL
    # --------------------------------------------------------

    if "PnL" not in trades.columns:

        raise ValueError(
            "PnL column not found in trades."
        )

    pnl = pd.to_numeric(
        trades["PnL"],
        errors="coerce",
    ).dropna()

    trade_count = len(pnl)

    wins = pnl[
        pnl > 0
    ]

    losses = pnl[
        pnl < 0
    ]

    # --------------------------------------------------------
    # Win rate
    # --------------------------------------------------------

    if trade_count > 0:

        win_rate = (
            len(wins)
            / trade_count
            * 100.0
        )

    else:

        win_rate = np.nan

    # --------------------------------------------------------
    # Profit factor
    # --------------------------------------------------------

    gross_profit = wins.sum()

    gross_loss = abs(
        losses.sum()
    )

    if gross_loss > 0:

        profit_factor = (
            gross_profit
            / gross_loss
        )

    elif gross_profit > 0:

        profit_factor = np.inf

    else:

        profit_factor = np.nan

    # --------------------------------------------------------
    # Expectancy
    # --------------------------------------------------------

    expectancy = pnl.mean()

    # --------------------------------------------------------
    # Trading-hour holding time
    # --------------------------------------------------------

    if (
        "TradingHoursHeld"
        in trades.columns
    ):

        hold_hours = pd.to_numeric(
            trades[
                "TradingHoursHeld"
            ],
            errors="coerce",
        )

    elif "HoursHeld" in trades.columns:

        hold_hours = pd.to_numeric(
            trades[
                "HoursHeld"
            ],
            errors="coerce",
        )

    else:

        hold_hours = pd.Series(
            dtype=float
        )

    avg_hold = (
        hold_hours.mean()
    )

    return {
        "ReturnPct": total_return,
        "MaxDDPct": max_dd,
        "Sharpe": sharpe,
        "Trades": trade_count,
        "WinRatePct": win_rate,
        "ProfitFactor": profit_factor,
        "Expectancy": expectancy,
        "AvgHoldHours": avg_hold,
    }


# ============================================================
# RUN ONE MONTH
# ============================================================

def run_one_period(bars, period):

    # --------------------------------------------------------
    # Validate period data
    # --------------------------------------------------------

    bars = bars.copy()

    if len(bars) == 0:
        raise ValueError(
            f"No hourly bars found for {period}"
        )

    # --------------------------------------------------------
    # Prepare Kalman + Z-score
    # --------------------------------------------------------

    bars = prepare_month(
        bars
    )

    # --------------------------------------------------------
    # Signal statistics
    # --------------------------------------------------------

    valid_z = (
        bars["ZScore"]
        .dropna()
    )

    print("\nSignal statistics")
    print("-" * 70)

    if len(valid_z) > 0:

        print(
            f"Valid Z observations: "
            f"{len(valid_z):,}"
        )

        print(
            f"Z mean: "
            f"{valid_z.mean():.4f}"
        )

        print(
            f"Z std:  "
            f"{valid_z.std():.4f}"
        )

        print(
            f"Z min:  "
            f"{valid_z.min():.4f}"
        )

        print(
            f"Z max:  "
            f"{valid_z.max():.4f}"
        )

        long_signals = (
            valid_z <= -Z_ENTRY
        ).sum()

        short_signals = (
            valid_z >= Z_ENTRY
        ).sum()

        print(
            f"Long signals: "
            f"{long_signals}"
        )

        print(
            f"Short signals: "
            f"{short_signals}"
        )

    else:

        print(
            "No valid Z-score observations."
        )

    # ========================================================
    # BACKTEST
    # ========================================================

    # A valid Z-score requires enough warm-up history.
    # With MEAN_WINDOW=168 and Z_WINDOW=48, the first useful
    # observations occur only after the warm-up period.
    if len(valid_z) == 0:
        print("\nWARNING: No valid Z-score observations for this period.")
        print(
            f"Bars available: {len(bars)} | "
            f"Required warm-up is approximately "
            f"{MEAN_WINDOW + Z_WINDOW + 1} bars."
        )
        print(
            "This period will be recorded as a flat/no-signal period "
            "instead of crashing the full historical run."
        )

        equity = make_flat_equity(bars.index)
        trades = pd.DataFrame()
        metrics = calculate_metrics(
            equity,
            trades,
        )
        metrics["Status"] = "NO_VALID_Z"
        metrics["ValidZ"] = 0

    else:
        backtester = KalmanBacktester(
            entry_z=Z_ENTRY,
            exit_z=Z_EXIT,
            stop_z=Z_STOP,
            max_hold_hours=MAX_HOLD_HOURS,
            initial_capital=INITIAL_CAPITAL,
            risk_per_trade=RISK_PER_TRADE,
            max_notional_fraction=(
                MAX_NOTIONAL_FRACTION
            ),
            assumed_stop_return=(
                ASSUMED_STOP_RETURN
            ),
        )

        equity, trades = (
            backtester.run(
                bars
            )
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        metrics = calculate_metrics(
            equity,
            trades,
        )
        metrics["Status"] = "OK"
        metrics["ValidZ"] = len(valid_z)

    # --------------------------------------------------------
    # Period
    # --------------------------------------------------------

    metrics["Period"] = str(period)

    metrics["Start"] = bars.index.min()

    metrics["End"] = bars.index.max()

    metrics = {
        "Period": metrics["Period"],
        "Start": metrics["Start"],
        "End": metrics["End"],
        "ReturnPct": metrics["ReturnPct"],
        "MaxDDPct": metrics["MaxDDPct"],
        "Sharpe": metrics["Sharpe"],
        "Trades": metrics["Trades"],
        "WinRatePct": metrics["WinRatePct"],
        "ProfitFactor": metrics["ProfitFactor"],
        "Expectancy": metrics["Expectancy"],
        "AvgHoldHours": metrics["AvgHoldHours"],
        "ValidZ": metrics.get("ValidZ", len(valid_z)),
        "Status": metrics.get("Status", "OK"),
    }

    # ========================================================
    # PRINT RESULT
    # ========================================================

    print("\nBacktest result")
    print("-" * 70)

    print(
        f"Initial capital: "
        f"${INITIAL_CAPITAL:,.2f}"
    )

    equity_series = normalize_equity(
        equity
    )

    if len(equity_series) > 0:

        print(
            f"Final equity:    "
            f"${equity_series.iloc[-1]:,.2f}"
        )

    print(
        f"Total return:    "
        f"{metrics['ReturnPct']:.2f}%"
    )

    print(
        f"Max drawdown:    "
        f"{metrics['MaxDDPct']:.2f}%"
    )

    if np.isfinite(
        metrics["Sharpe"]
    ):

        print(
            f"Sharpe:          "
            f"{metrics['Sharpe']:.3f}"
        )

    else:

        print(
            "Sharpe:          NaN"
        )

    print(
        f"Trades:          "
        f"{metrics['Trades']}"
    )

    if np.isfinite(
        metrics["WinRatePct"]
    ):

        print(
            f"Win rate:        "
            f"{metrics['WinRatePct']:.2f}%"
        )

    else:

        print(
            "Win rate:        NaN"
        )

    if np.isfinite(
        metrics["ProfitFactor"]
    ):

        print(
            f"Profit factor:   "
            f"{metrics['ProfitFactor']:.3f}"
        )

    elif metrics["ProfitFactor"] == np.inf:

        print(
            "Profit factor:   inf"
        )

    else:

        print(
            "Profit factor:   NaN"
        )

    if np.isfinite(
        metrics["Expectancy"]
    ):

        print(
            f"Expectancy:      "
            f"${metrics['Expectancy']:.2f}"
        )

    else:

        print(
            "Expectancy:      NaN"
        )

    if np.isfinite(
        metrics["AvgHoldHours"]
    ):

        print(
            f"Avg hold:        "
            f"{metrics['AvgHoldHours']:.2f} "
            f"trading hours"
        )

    else:

        print(
            "Avg hold:        NaN"
        )

    # ========================================================
    # TIME-STOP VALIDATION
    # ========================================================

    if (
        trades is not None
        and not trades.empty
    ):

        # ----------------------------------------------------
        # Trading-hour holding time
        # ----------------------------------------------------

        if (
            "TradingHoursHeld"
            in trades.columns
        ):

            trading_hold = pd.to_numeric(
                trades[
                    "TradingHoursHeld"
                ],
                errors="coerce",
            )

        else:

            trading_hold = pd.to_numeric(
                trades[
                    "HoursHeld"
                ],
                errors="coerce",
            )

        max_trading_hold = (
            trading_hold.max()
        )

        print(
            f"Maximum trading-hour hold: "
            f"{max_trading_hold:.1f}h"
        )

        # ----------------------------------------------------
        # Calendar holding time
        # ----------------------------------------------------

        if (
            "CalendarHoursHeld"
            in trades.columns
        ):

            calendar_hold = pd.to_numeric(
                trades[
                    "CalendarHoursHeld"
                ],
                errors="coerce",
            )

            max_calendar_hold = (
                calendar_hold.max()
            )

            print(
                f"Maximum calendar hold: "
                f"{max_calendar_hold:.1f}h"
            )

        # ----------------------------------------------------
        # TIME STOP trades
        # ----------------------------------------------------

        time_stop_trades = trades[
            trades["ExitReason"]
            == "TIME_STOP"
        ]

        if len(time_stop_trades) > 0:

            time_stop_hold = pd.to_numeric(
                time_stop_trades[
                    "TradingHoursHeld"
                ],
                errors="coerce",
            )

            max_time_stop_hold = (
                time_stop_hold.max()
            )

            print(
                f"Max TIME_STOP trading hold: "
                f"{max_time_stop_hold:.1f}h"
            )

            if (
                max_time_stop_hold
                > MAX_HOLD_HOURS
            ):

                print(
                    "\nWARNING:"
                )

                print(
                    "TIME_STOP exceeded "
                    "MAX_HOLD_HOURS."
                )

    # --------------------------------------------------------
    # Trade count
    # --------------------------------------------------------

    if (
        trades is not None
        and not trades.empty
    ):

        print(
            f"\nTrade count: "
            f"{len(trades)}"
        )

    return (
        bars,
        equity,
        trades,
        metrics,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print(
        "XAUUSDm 2019-2026 "
        "INDEPENDENT-MONTH WALK-FORWARD BACKTEST"
    )
    print("=" * 70)

    # ========================================================
    # PARAMETERS
    # ========================================================

    print("\nFixed parameters:")

    print(
        f"PHI                = {PHI}"
    )

    print(
        f"Q                  = {Q}"
    )

    print(
        f"R                  = {R}"
    )

    print(
        f"MEAN_WINDOW        = {MEAN_WINDOW}"
    )

    print(
        f"Z_WINDOW           = {Z_WINDOW}"
    )

    print(
        f"Z_ENTRY            = {Z_ENTRY}"
    )

    print(
        f"Z_EXIT             = {Z_EXIT}"
    )

    print(
        f"Z_STOP             = {Z_STOP}"
    )

    print(
        f"MAX_HOLD_HOURS     = "
        f"{MAX_HOLD_HOURS}"
    )

    print(
        f"INITIAL_CAPITAL    = "
        f"${INITIAL_CAPITAL:,.2f}"
    )

    print(
        f"RISK_PER_TRADE     = "
        f"{RISK_PER_TRADE}"
    )

    print(
        f"MAX_NOTIONAL       = "
        f"{MAX_NOTIONAL_FRACTION}"
    )

    print(
        f"ASSUMED_STOP_RETURN= "
        f"{ASSUMED_STOP_RETURN}"
    )

    print("\nIMPORTANT:")

    print(
        "Every month is processed independently."
    )

    print(
        "No Kalman state, position, or Z-score "
        "history is carried across months."
    )

    print(
        "MAX_HOLD_HOURS is measured in "
        "available hourly trading bars."
    )

    print(
        "Calendar gaps such as weekends do not "
        "consume the trading-hour holding limit."
    )

    print("\nYearly files discovered:")
    for filename in YEAR_FILES:
        print(f"  - {filename}")

    # ========================================================
    # RESULT CONTAINERS
    # ========================================================

    all_bars = []

    all_equity = []

    all_trades = []

    all_metrics = []

    # ========================================================
    # RUN EACH YEAR -> SPLIT INTO MONTHS
    # ========================================================

    for filename in YEAR_FILES:

        try:

            # ------------------------------------------------
            # Load one year at a time.
            # This avoids loading all 2019-2026 ticks at once.
            # ------------------------------------------------

            year_bars = load_year_file(filename)

            if len(year_bars) == 0:
                raise ValueError(
                    f"No hourly bars found for {filename}"
                )

            # ------------------------------------------------
            # Split the hourly bars into calendar months.
            # Each month is then processed independently.
            # ------------------------------------------------

            month_periods = (
                year_bars.index
                .to_period("M")
                .unique()
                .sort_values()
            )

            print(
                f"\n{filename}: "
                f"{len(month_periods)} month(s) found."
            )

            for period in month_periods:

                period_str = str(period)

                month_mask = (
                    year_bars.index.to_period("M")
                    == period
                )

                month_bars = (
                    year_bars.loc[month_mask]
                    .copy()
                    .sort_index()
                )

                if len(month_bars) == 0:
                    continue

                print("\n")
                print("#" * 70)
                print(
                    f"PROCESSING PERIOD: {period_str}"
                )
                print("#" * 70)

                (
                    bars,
                    equity,
                    trades,
                    metrics,
                ) = run_one_period(
                    month_bars,
                    period_str,
                )

                # ------------------------------------------------
                # Bars
                # ------------------------------------------------

                temp_bars = bars.copy()

                temp_bars["Period"] = (
                    metrics["Period"]
                )

                all_bars.append(
                    temp_bars
                )

                # ------------------------------------------------
                # Equity
                # ------------------------------------------------

                if isinstance(
                    equity,
                    pd.DataFrame,
                ):

                    equity_df = (
                        equity.copy()
                    )

                else:

                    equity_df = (
                        equity.to_frame(
                            name="Equity"
                        )
                    )

                equity_df["Period"] = (
                    metrics["Period"]
                )

                all_equity.append(
                    equity_df
                )

                # ------------------------------------------------
                # Trades
                # ------------------------------------------------

                if (
                    trades is not None
                    and len(trades) > 0
                ):

                    temp_trades = (
                        trades.copy()
                    )

                    temp_trades["Period"] = (
                        metrics["Period"]
                    )

                    all_trades.append(
                        temp_trades
                    )

                # ------------------------------------------------
                # Metrics
                # ------------------------------------------------

                all_metrics.append(
                    metrics
                )

        except Exception as e:

            print("\nERROR")
            print("-" * 70)

            print(
                f"File: {filename}"
            )

            print(
                f"Error: {e}"
            )

            raise

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = pd.DataFrame(
        all_metrics
    )

    print("\n\n")
    print("=" * 110)
    print(
        "INDEPENDENT PERIOD RESULTS"
    )
    print("=" * 110)

    display_columns = [
        "Period",
        "ReturnPct",
        "MaxDDPct",
        "Sharpe",
        "Trades",
        "WinRatePct",
        "ProfitFactor",
        "Expectancy",
        "AvgHoldHours",
        "ValidZ",
        "Status",
    ]

    print(
        summary[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.3f}",
        )
    )

    # ========================================================
    # ROBUSTNESS SUMMARY
    # ========================================================

    returns = (
        summary[
            "ReturnPct"
        ]
        .dropna()
    )

    drawdowns = (
        summary[
            "MaxDDPct"
        ]
        .dropna()
    )

    profitable_periods = (
        returns > 0
    ).sum()

    losing_periods = (
        returns < 0
    ).sum()

    print("\n")
    print("=" * 70)
    print(
        "ROBUSTNESS SUMMARY"
    )
    print("=" * 70)

    print(
        f"Periods tested:       "
        f"{len(summary)}"
    )

    print(
        f"Profitable periods:   "
        f"{profitable_periods}"
    )

    print(
        f"Losing periods:       "
        f"{losing_periods}"
    )

    if len(returns) > 0:

        print(
            f"Average return:       "
            f"{returns.mean():.2f}%"
        )

        print(
            f"Median return:        "
            f"{returns.median():.2f}%"
        )

        print(
            f"Best return:          "
            f"{returns.max():.2f}%"
        )

        print(
            f"Worst return:         "
            f"{returns.min():.2f}%"
        )

    if len(drawdowns) > 0:

        print(
            f"Worst max drawdown:   "
            f"{drawdowns.min():.2f}%"
        )

    print(
        f"Total trades:         "
        f"{summary['Trades'].sum()}"
    )

    # ========================================================
    # ANNUAL SUMMARY
    # ========================================================

    if len(summary) > 0:

        summary["Year"] = (
            pd.to_datetime(
                summary["Period"].astype(str) + "-01",
                errors="coerce",
            )
            .dt.year
        )

        annual_rows = []

        for year, group in summary.groupby(
            "Year",
            dropna=True,
        ):

            monthly_returns = (
                group["ReturnPct"]
                .dropna()
                / 100.0
            )

            compounded_return = (
                (1.0 + monthly_returns).prod()
                - 1.0
            ) * 100.0

            annual_rows.append(
                {
                    "Year": int(year),
                    "MonthsTested": len(group),
                    "CompoundedReturnPct": compounded_return,
                    "ProfitableMonths": int(
                        (group["ReturnPct"] > 0).sum()
                    ),
                    "LosingMonths": int(
                        (group["ReturnPct"] < 0).sum()
                    ),
                    "WorstMonthlyDDPct": group[
                        "MaxDDPct"
                    ].min(),
                    "TotalTrades": int(
                        group["Trades"].sum()
                    ),
                }
            )

        annual_summary = pd.DataFrame(
            annual_rows
        ).sort_values("Year")

        print("\n")
        print("=" * 110)
        print(
            "ANNUAL SUMMARY"
        )
        print("=" * 110)

        print(
            annual_summary.to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.3f}",
            )
        )

    else:

        annual_summary = pd.DataFrame()

    # ========================================================
    # COMPOUNDED MONTHLY BASELINE
    # ========================================================

    if len(summary) > 0:

        monthly_returns = (
            summary["ReturnPct"]
            .dropna()
            / 100.0
        )

        compounded_all = (
            (1.0 + monthly_returns).prod()
            - 1.0
        ) * 100.0

        print("\n")
        print("=" * 70)
        print(
            "FULL-PERIOD COMPOUNDED MONTHLY BASELINE"
        )
        print("=" * 70)

        print(
            f"Months tested:       {len(monthly_returns)}"
        )

        print(
            f"Compounded return:   {compounded_all:.2f}%"
        )

        print(
            "Note: each month is independently reset "
            "to the same initial capital, so this is a "
            "compounded research statistic, not a continuous "
            "portfolio equity curve."
        )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    if len(all_bars) > 0:

        combined_bars = pd.concat(
            all_bars,
            axis=0,
        )

        combined_bars.to_csv(
            RESULTS_DIR
            / "XAUUSDm_independent_month_data.csv"
        )

    # --------------------------------------------------------
    # Equity
    # --------------------------------------------------------

    if len(all_equity) > 0:

        combined_equity = pd.concat(
            all_equity,
            axis=0,
        )

        combined_equity.to_csv(
            RESULTS_DIR
            / "XAUUSDm_independent_month_equity.csv"
        )

    # --------------------------------------------------------
    # Trades
    # --------------------------------------------------------

    if len(all_trades) > 0:

        combined_trades = pd.concat(
            all_trades,
            axis=0,
            ignore_index=True,
        )

        combined_trades.to_csv(
            RESULTS_DIR
            / "XAUUSDm_independent_month_trades.csv",
            index=False,
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary.to_csv(
        RESULTS_DIR
        / "XAUUSDm_independent_month_summary.csv",
        index=False,
    )

    # Explicitly named 2019-2026 summary for the historical run.
    summary.to_csv(
        RESULTS_DIR
        / "XAUUSDm_2019_2026_monthly_summary.csv",
        index=False,
    )

    if "annual_summary" in locals() and len(annual_summary) > 0:
        annual_summary.to_csv(
            RESULTS_DIR
            / "XAUUSDm_annual_summary.csv",
            index=False,
        )

    # ========================================================
    # FILES SAVED
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "FILES SAVED"
    )
    print("=" * 70)

    print(
        RESULTS_DIR
        / "XAUUSDm_independent_month_data.csv"
    )

    print(
        RESULTS_DIR
        / "XAUUSDm_independent_month_equity.csv"
    )

    print(
        RESULTS_DIR
        / "XAUUSDm_independent_month_trades.csv"
    )

    print(
        RESULTS_DIR
        / "XAUUSDm_independent_month_summary.csv"
    )

    print(
        RESULTS_DIR
        / "XAUUSDm_2019_2026_monthly_summary.csv"
    )

    if "annual_summary" in locals() and len(annual_summary) > 0:
        print(
            RESULTS_DIR
            / "XAUUSDm_annual_summary.csv"
        )

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "BACKTEST COMPLETE"
    )
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()