from pathlib import Path
import sys

import numpy as np
import pandas as pd


# ============================================================
# PATH SETUP
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
MT5_DIR = DATA_DIR / "mt5_h1"
RESULTS_DIR = ROOT / "results" / "mt5_h1"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# MT5 H1 FILE
# ============================================================

MT5_FILE = (
    MT5_DIR
    / "XAUUSDm_H1_2019_2026.csv"
)


# ============================================================
# BACKTEST PERIOD
# ============================================================

START_DATE = pd.Timestamp(
    "2025-01-01 00:00:00",
    tz="UTC",
)

END_DATE = pd.Timestamp(
    "2025-07-20 23:00:00",
    tz="UTC",
)


# ============================================================
# LOCKED KALMAN PARAMETERS
# ============================================================
#
# IMPORTANT:
#
# These parameters are NOT being optimized.
#
# They are the parameters already selected during the
# previous research/validation stage.
# ============================================================

PHI = 0.999
Q = 0.25
R = 0.25

MEAN_WINDOW = 168
Z_WINDOW = 36

ENTRY_Z = 2.25
EXIT_Z = 0.25
STOP_Z = 3.0

MAX_HOLD_HOURS = 24


# ============================================================
# PORTFOLIO PARAMETERS
# ============================================================

INITIAL_CAPITAL = 10_000.0

RISK_PER_TRADE = 0.005

MAX_NOTIONAL_FRACTION = 1.0

ASSUMED_STOP_RETURN = 0.01


# ============================================================
# XAUUSDm POINT SIZE
# ============================================================

POINT_SIZE = 0.001


# ============================================================
# IMPORT ORIGINAL KALMAN IMPLEMENTATION
# ============================================================

sys.path.insert(
    0,
    str(ROOT / "src"),
)

from walk_forward_kalman import (
    walk_forward_kalman,
    calculate_zscore,
)


# ============================================================
# LOAD MT5 H1
# ============================================================

def load_mt5_h1():

    print()
    print("=" * 70)
    print("LOADING MT5 H1 DATA")
    print("=" * 70)

    if not MT5_FILE.exists():

        raise FileNotFoundError(
            f"\nMT5 H1 file not found:\n"
            f"{MT5_FILE}\n"
        )

    df = pd.read_csv(
        MT5_FILE
    )

    print(
        f"Rows loaded: "
        f"{len(df):,}"
    )

    print(
        f"Columns: "
        f"{list(df.columns)}"
    )

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    if "Timestamp" in df.columns:

        time_col = "Timestamp"

    elif "Time" in df.columns:

        time_col = "Time"

    else:

        raise ValueError(
            "Could not find Timestamp/Time column."
        )

    df[time_col] = pd.to_datetime(
        df[time_col],
        utc=True,
        errors="coerce",
    )

    df = df.dropna(
        subset=[time_col]
    )

    df = (
        df
        .set_index(time_col)
        .sort_index()
    )

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required = {
        "Open",
        "High",
        "Low",
        "Close",
        "SpreadPoints",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            f"Missing columns: {missing}"
        )

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    for column in [
        "Open",
        "High",
        "Low",
        "Close",
        "SpreadPoints",
    ]:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Remove invalid data
    # --------------------------------------------------------

    df = df.dropna(
        subset=[
            "Open",
            "High",
            "Low",
            "Close",
            "SpreadPoints",
        ]
    )

    df = df[
        (df["Open"] > 0)
        &
        (df["High"] > 0)
        &
        (df["Low"] > 0)
        &
        (df["Close"] > 0)
        &
        (df["SpreadPoints"] >= 0)
    ]

    # --------------------------------------------------------
    # Restrict period
    # --------------------------------------------------------

    df = df.loc[
        (df.index >= START_DATE)
        &
        (df.index <= END_DATE)
    ].copy()

    if df.empty:

        raise ValueError(
            "No MT5 H1 data in selected period."
        )

    print(
        f"Bars in backtest period: "
        f"{len(df):,}"
    )

    print(
        f"Start: "
        f"{df.index.min()}"
    )

    print(
        f"End: "
        f"{df.index.max()}"
    )

    return df


# ============================================================
# ADD BID / ASK APPROXIMATION
# ============================================================

def add_bid_ask(
    df,
):

    """
    MT5 H1 export gives us OHLC + spread.

    We therefore reconstruct an approximate bid/ask
    around the MT5 OHLC midpoint:

        Bid = Mid - spread / 2
        Ask = Mid + spread / 2

    This is an H1 research approximation.

    Final verification must still use MT5 Strategy Tester
    with Every Tick Based on Real Tick.
    """

    spread_price = (
        df["SpreadPoints"]
        * POINT_SIZE
    )

    df["SpreadPrice"] = (
        spread_price
    )

    # --------------------------------------------------------
    # Open
    # --------------------------------------------------------

    df["Bid_Open"] = (
        df["Open"]
        - spread_price / 2.0
    )

    df["Ask_Open"] = (
        df["Open"]
        + spread_price / 2.0
    )

    # --------------------------------------------------------
    # Close
    # --------------------------------------------------------

    df["Bid_Close"] = (
        df["Close"]
        - spread_price / 2.0
    )

    df["Ask_Close"] = (
        df["Close"]
        + spread_price / 2.0
    )

    return df


# ============================================================
# ORIGINAL KALMAN IMPLEMENTATION
# ============================================================

def calculate_kalman_features(
    df,
):

    print()
    print("=" * 70)
    print("CALCULATING KALMAN FEATURES")
    print("=" * 70)

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # This is the exact same implementation used by the
    # original research code.
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Preserve index
    # --------------------------------------------------------

    df["KalmanState"] = pd.Series(
        states,
        index=df.index,
    )

    df["KalmanVariance"] = pd.Series(
        variances,
        index=df.index,
    )

    df["Residual"] = pd.Series(
        residuals,
        index=df.index,
    )

    df["RollingMean"] = pd.Series(
        means,
        index=df.index,
    )

    # --------------------------------------------------------
    # EXACT original Z-score implementation
    # --------------------------------------------------------

    df["Z"] = calculate_zscore(
        df["Residual"],
        window=Z_WINDOW,
    )

    valid_z = (
        df["Z"].notna()
    )

    print(
        f"Valid Z observations: "
        f"{valid_z.sum():,}"
    )

    if valid_z.any():

        print(
            f"Z mean: "
            f"{df.loc[valid_z, 'Z'].mean():.6f}"
        )

        print(
            f"Z std: "
            f"{df.loc[valid_z, 'Z'].std():.6f}"
        )

        print(
            f"Z min: "
            f"{df.loc[valid_z, 'Z'].min():.6f}"
        )

        print(
            f"Z max: "
            f"{df.loc[valid_z, 'Z'].max():.6f}"
        )

    return df


# ============================================================
# POSITION SIZING
# ============================================================

def calculate_units(
    equity,
    price,
):
    """
    Research position sizing.

    Risk budget:
        equity * RISK_PER_TRADE

    Assumed stop:
        ASSUMED_STOP_RETURN

    Position notional is also capped by
    MAX_NOTIONAL_FRACTION * equity.

    This is still a research approximation.
    Exact MT5 contract sizing comes later.
    """

    if equity <= 0:
        return 0.0

    if price <= 0:
        return 0.0

    risk_budget = (
        equity
        * RISK_PER_TRADE
    )

    stop_loss_per_unit = (
        price
        * ASSUMED_STOP_RETURN
    )

    if stop_loss_per_unit <= 0:
        return 0.0

    units_from_risk = (
        risk_budget
        /
        stop_loss_per_unit
    )

    max_notional = (
        equity
        * MAX_NOTIONAL_FRACTION
    )

    units_from_notional = (
        max_notional
        /
        price
    )

    units = min(
        units_from_risk,
        units_from_notional,
    )

    return max(
        units,
        0.0,
    )


# ============================================================
# BACKTEST
# ============================================================

def run_backtest(
    df,
):

    print()
    print("=" * 70)
    print("RUNNING MT5 H1 BACKTEST")
    print("=" * 70)

    equity = (
        INITIAL_CAPITAL
    )

    cash_equity = (
        INITIAL_CAPITAL
    )

    position = 0

    units = 0.0

    entry_price = np.nan

    entry_time = None

    entry_z = np.nan

    entry_index = None

    entry_equity = np.nan

    trades = []

    equity_curve = []

    # --------------------------------------------------------
    # Event loop
    # --------------------------------------------------------

    for i in range(
        len(df)
    ):

        timestamp = df.index[i]

        row = df.iloc[i]

        z = row["Z"]

        close_mid = row["Close"]

        bid_close = row["Bid_Close"]

        ask_close = row["Ask_Close"]

        # ----------------------------------------------------
        # Mark-to-market
        # ----------------------------------------------------

        if position == 1:

            equity = (
                cash_equity
                +
                units
                * (
                    bid_close
                    - entry_price
                )
            )

        elif position == -1:

            equity = (
                cash_equity
                +
                units
                * (
                    entry_price
                    - ask_close
                )
            )

        else:

            equity = cash_equity

        # ----------------------------------------------------
        # Record equity
        # ----------------------------------------------------

        equity_curve.append(
            {
                "Timestamp": timestamp,
                "Equity": equity,
                "CashEquity": cash_equity,
                "Position": position,
                "Units": units,
                "Mid": close_mid,
                "Z": z,
            }
        )

        # ----------------------------------------------------
        # Need next bar for execution.
        # ----------------------------------------------------

        if i >= len(df) - 1:

            continue

        next_row = df.iloc[i + 1]

        next_timestamp = df.index[i + 1]

        next_bid_open = (
            next_row["Bid_Open"]
        )

        next_ask_open = (
            next_row["Ask_Open"]
        )

        # ====================================================
        # POSITION MANAGEMENT
        # ====================================================

        if position != 0:

            # ------------------------------------------------
            # Hours held
            # ------------------------------------------------

            trading_hours_held = (
                i
                -
                entry_index
            )

            calendar_hours_held = (
                (
                    timestamp
                    - entry_time
                )
                .total_seconds()
                / 3600.0
            )

            # ------------------------------------------------
            # Exit conditions
            # ------------------------------------------------

            exit_reason = None

            if pd.notna(z):

                # Mean-reversion exit
                if abs(z) <= EXIT_Z:

                    exit_reason = "Z_EXIT"

                # Stop
                elif abs(z) >= STOP_Z:

                    exit_reason = "Z_STOP"

            # Maximum holding period
            if (
                exit_reason is None
                and
                trading_hours_held
                >= MAX_HOLD_HOURS
            ):

                exit_reason = "MAX_HOLD"

            # ------------------------------------------------
            # Execute exit on NEXT bar
            # ------------------------------------------------

            if exit_reason is not None:

                if position == 1:

                    exit_price = (
                        next_bid_open
                    )

                    pnl = (
                        units
                        * (
                            exit_price
                            - entry_price
                        )
                    )

                else:

                    exit_price = (
                        next_ask_open
                    )

                    pnl = (
                        units
                        * (
                            entry_price
                            - exit_price
                        )
                    )

                cash_equity += pnl

                equity = (
                    cash_equity
                )

                trade_return = (
                    pnl
                    /
                    entry_equity
                )

                trades.append(
                    {
                        "EntryTime": entry_time,
                        "ExitTime": next_timestamp,
                        "Side": (
                            "LONG"
                            if position == 1
                            else "SHORT"
                        ),
                        "Units": units,
                        "EntryPrice": entry_price,
                        "ExitPrice": exit_price,
                        "EntryZ": entry_z,
                        "ExitZ": z,
                        "TradingHoursHeld": (
                            trading_hours_held
                        ),
                        "CalendarHoursHeld": (
                            calendar_hours_held
                        ),
                        "HoursHeld": (
                            trading_hours_held
                        ),
                        "Return": (
                            trade_return
                        ),
                        "PnL": pnl,
                        "ExitReason": exit_reason,
                    }
                )

                # ------------------------------------------------
                # Reset position
                # ------------------------------------------------

                position = 0

                units = 0.0

                entry_price = np.nan

                entry_time = None

                entry_z = np.nan

                entry_index = None

                entry_equity = np.nan

                # ------------------------------------------------
                # Don't enter another position on the same
                # signal bar.
                # ------------------------------------------------

                continue

        # ====================================================
        # ENTRY
        # ====================================================

        if position == 0:

            if pd.isna(z):

                continue

            # ------------------------------------------------
            # LONG
            # ------------------------------------------------

            if z <= -ENTRY_Z:

                entry_price = (
                    next_ask_open
                )

                position = 1

                units = calculate_units(
                    equity,
                    entry_price,
                )

                if units <= 0:

                    position = 0
                    continue

                entry_time = (
                    next_timestamp
                )

                entry_z = z

                entry_index = (
                    i + 1
                )

                entry_equity = (
                    equity
                )

                continue

            # ------------------------------------------------
            # SHORT
            # ------------------------------------------------

            if z >= ENTRY_Z:

                entry_price = (
                    next_bid_open
                )

                position = -1

                units = calculate_units(
                    equity,
                    entry_price,
                )

                if units <= 0:

                    position = 0
                    continue

                entry_time = (
                    next_timestamp
                )

                entry_z = z

                entry_index = (
                    i + 1
                )

                entry_equity = (
                    equity
                )

                continue

    # ========================================================
    # FORCE CLOSE REMAINING POSITION
    # ========================================================

    if position != 0:

        last_row = df.iloc[-1]

        last_timestamp = df.index[-1]

        if position == 1:

            exit_price = (
                last_row["Bid_Close"]
            )

            pnl = (
                units
                * (
                    exit_price
                    - entry_price
                )
            )

        else:

            exit_price = (
                last_row["Ask_Close"]
            )

            pnl = (
                units
                * (
                    entry_price
                    - exit_price
                )
            )

        cash_equity += pnl

        equity = cash_equity

        trading_hours_held = (
            len(df)
            - 1
            - entry_index
        )

        calendar_hours_held = (
            (
                last_timestamp
                - entry_time
            )
            .total_seconds()
            / 3600.0
        )

        trade_return = (
            pnl
            /
            entry_equity
        )

        trades.append(
            {
                "EntryTime": entry_time,
                "ExitTime": last_timestamp,
                "Side": (
                    "LONG"
                    if position == 1
                    else "SHORT"
                ),
                "Units": units,
                "EntryPrice": entry_price,
                "ExitPrice": exit_price,
                "EntryZ": entry_z,
                "ExitZ": last_row["Z"],
                "TradingHoursHeld": (
                    trading_hours_held
                ),
                "CalendarHoursHeld": (
                    calendar_hours_held
                ),
                "HoursHeld": (
                    trading_hours_held
                ),
                "Return": (
                    trade_return
                ),
                "PnL": pnl,
                "ExitReason": "END_OF_DATA",
            }
        )

    # ========================================================
    # CREATE DATAFRAMES
    # ========================================================

    equity_df = pd.DataFrame(
        equity_curve
    )

    trades_df = pd.DataFrame(
        trades
    )

    # --------------------------------------------------------
    # Make sure equity is sorted
    # --------------------------------------------------------

    if not equity_df.empty:

        equity_df = (
            equity_df
            .sort_values("Timestamp")
            .reset_index(drop=True)
        )

    # ========================================================
    # METRICS
    # ========================================================

    metrics = calculate_metrics(
        equity_df,
        trades_df,
    )

    return (
        equity_df,
        trades_df,
        metrics,
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    equity_df,
    trades_df,
):

    metrics = {}

    # --------------------------------------------------------
    # Basic equity
    # --------------------------------------------------------

    if equity_df.empty:

        return {
            "InitialEquity": INITIAL_CAPITAL,
            "FinalEquity": INITIAL_CAPITAL,
            "TotalReturn": 0.0,
            "MaxDrawdown": 0.0,
            "Sharpe": np.nan,
            "Trades": 0,
            "WinRate": np.nan,
            "ProfitFactor": np.nan,
            "Expectancy": 0.0,
        }

    equity_series = (
        equity_df["Equity"]
        .astype(float)
    )

    initial_equity = (
        INITIAL_CAPITAL
    )

    final_equity = (
        equity_series.iloc[-1]
    )

    total_return = (
        final_equity
        /
        initial_equity
        - 1.0
    )

    # --------------------------------------------------------
    # Drawdown
    # --------------------------------------------------------

    running_max = (
        equity_series
        .cummax()
    )

    drawdown = (
        equity_series
        /
        running_max
        - 1.0
    )

    max_drawdown = (
        drawdown.min()
    )

    # --------------------------------------------------------
    # Hourly returns
    # --------------------------------------------------------

    returns = (
        equity_series
        .pct_change()
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
    )

    if (
        len(returns) > 1
        and returns.std() > 0
    ):

        sharpe = (
            returns.mean()
            /
            returns.std()
        ) * np.sqrt(24 * 365)

    else:

        sharpe = np.nan

    # --------------------------------------------------------
    # Trade metrics
    # --------------------------------------------------------

    trade_count = (
        len(trades_df)
    )

    if trade_count > 0:

        wins = (
            trades_df["PnL"] > 0
        )

        losses = (
            trades_df["PnL"] < 0
        )

        win_rate = (
            wins.mean()
        )

        gross_profit = (
            trades_df.loc[
                wins,
                "PnL",
            ].sum()
        )

        gross_loss = (
            trades_df.loc[
                losses,
                "PnL",
            ].sum()
        )

        if gross_loss < 0:

            profit_factor = (
                gross_profit
                /
                abs(gross_loss)
            )

        else:

            profit_factor = np.inf

        expectancy = (
            trades_df["PnL"]
            .mean()
        )

    else:

        win_rate = np.nan

        profit_factor = np.nan

        expectancy = 0.0

    metrics = {
        "InitialEquity": initial_equity,
        "FinalEquity": final_equity,
        "TotalReturn": total_return,
        "MaxDrawdown": max_drawdown,
        "Sharpe": sharpe,
        "Trades": trade_count,
        "WinRate": win_rate,
        "ProfitFactor": profit_factor,
        "Expectancy": expectancy,
    }

    return metrics


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
    metrics,
    trades_df,
):

    print()
    print("=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)

    print(
        f"Initial equity: "
        f"${metrics['InitialEquity']:,.2f}"
    )

    print(
        f"Final equity: "
        f"${metrics['FinalEquity']:,.2f}"
    )

    print(
        f"Total return: "
        f"{metrics['TotalReturn'] * 100:.3f}%"
    )

    print(
        f"Maximum DD: "
        f"{metrics['MaxDrawdown'] * 100:.3f}%"
    )

    print(
        f"Sharpe: "
        f"{metrics['Sharpe']:.3f}"
    )

    print(
        f"Trades: "
        f"{metrics['Trades']}"
    )

    print(
        f"Win rate: "
        f"{metrics['WinRate'] * 100:.2f}%"
        if pd.notna(metrics["WinRate"])
        else "Win rate: N/A"
    )

    print(
        f"Profit factor: "
        f"{metrics['ProfitFactor']:.3f}"
        if np.isfinite(metrics["ProfitFactor"])
        else "Profit factor: inf"
    )

    print(
        f"Expectancy: "
        f"${metrics['Expectancy']:.4f}"
    )

    # --------------------------------------------------------
    # Exit reasons
    # --------------------------------------------------------

    if not trades_df.empty:

        print()
        print(
            "Exit reasons:"
        )

        print(
            trades_df[
                "ExitReason"
            ]
            .value_counts()
            .to_string()
        )

        # ----------------------------------------------------
        # Average holding time
        # ----------------------------------------------------

        print()

        print(
            f"Average trading hours held: "
            f"{trades_df['TradingHoursHeld'].mean():.2f}"
        )

        print(
            f"Maximum trading hours held: "
            f"{trades_df['TradingHoursHeld'].max():.2f}"
        )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    df,
    equity_df,
    trades_df,
    metrics,
):

    print()
    print("=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    # --------------------------------------------------------
    # Save processed data
    # --------------------------------------------------------

    processed_file = (
        RESULTS_DIR
        / "XAUUSDm_processed_H1.csv"
    )

    df.to_csv(
        processed_file
    )

    print(
        f"Processed data:\n"
        f"{processed_file}"
    )

    # --------------------------------------------------------
    # Save equity
    # --------------------------------------------------------

    equity_file = (
        RESULTS_DIR
        / "XAUUSDm_equity.csv"
    )

    equity_df.to_csv(
        equity_file,
        index=False,
    )

    print(
        f"Equity:\n"
        f"{equity_file}"
    )

    # --------------------------------------------------------
    # Save trades
    # --------------------------------------------------------

    trades_file = (
        RESULTS_DIR
        / "XAUUSDm_trades.csv"
    )

    trades_df.to_csv(
        trades_file,
        index=False,
    )

    print(
        f"Trades:\n"
        f"{trades_file}"
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary_file = (
        RESULTS_DIR
        / "XAUUSDm_summary.csv"
    )

    summary_df = pd.DataFrame(
        [
            metrics
        ]
    )

    summary_df.to_csv(
        summary_file,
        index=False,
    )

    print(
        f"Summary:\n"
        f"{summary_file}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("XAUUSDm MT5 H1 BACKTEST")
    print("=" * 70)

    print()
    print("This version uses the ORIGINAL:")
    print(
        "  walk_forward_kalman()"
    )
    print(
        "  calculate_zscore()"
    )

    print()
    print("Locked parameters:")
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
        f"MAX_HOLD     = {MAX_HOLD_HOURS} trading hours"
    )

    print()
    print("Portfolio parameters:")
    print(
        f"Initial capital = ${INITIAL_CAPITAL:,.2f}"
    )
    print(
        f"Risk/trade     = {RISK_PER_TRADE * 100:.2f}%"
    )
    print(
        f"Max notional   = {MAX_NOTIONAL_FRACTION * 100:.0f}%"
    )

    # ========================================================
    # LOAD DATA
    # ========================================================

    df = load_mt5_h1()

    # ========================================================
    # BID / ASK
    # ========================================================

    df = add_bid_ask(
        df
    )

    # ========================================================
    # KALMAN + Z-SCORE
    # ========================================================

    df = calculate_kalman_features(
        df
    )

    # ========================================================
    # BACKTEST
    # ========================================================

    (
        equity_df,
        trades_df,
        metrics,
    ) = run_backtest(
        df
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print_results(
        metrics,
        trades_df,
    )

    # ========================================================
    # SAVE
    # ========================================================

    save_results(
        df,
        equity_df,
        trades_df,
        metrics,
    )

    # ========================================================
    # FINISHED
    # ========================================================

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":

    main()