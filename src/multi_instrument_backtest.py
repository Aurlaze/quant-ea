from pathlib import Path
import sys
import numpy as np
import pandas as pd

# ---------------------------------------------------------
# Make src imports work when running:
# python src/multi_instrument_backtest.py
# ---------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from walk_forward_kalman import (
    walk_forward_kalman,
    calculate_zscore,
)


# =========================================================
# CONFIGURATION
# =========================================================

DATA_DIR = PROJECT_ROOT / "data" / "mt5_h1"
RESULTS_DIR = PROJECT_ROOT / "results" / "multi_instrument"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Universe
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Kalman parameters
# LOCKED -- DO NOT OPTIMIZE YET
# ---------------------------------------------------------

PHI = 0.999
Q = 0.25
R = 0.25

MEAN_WINDOW = 168
Z_WINDOW = 36

ENTRY_Z = 2.25
EXIT_Z = 0.25
STOP_Z = 3.0

MAX_HOLD_HOURS = 24


# ---------------------------------------------------------
# Portfolio / sizing
# ---------------------------------------------------------

INITIAL_CAPITAL = 10_000.0

RISK_PER_TRADE = 0.005

# No leverage.
MAX_NOTIONAL_FRACTION = 1.0

# Temporary research assumption.
# Final MT5 tester will use actual contract/tick value.
ASSUMED_STOP_RETURN = 0.01


# ---------------------------------------------------------
# Backtest period
# ---------------------------------------------------------

START_DATE = pd.Timestamp("2019-01-01", tz="UTC")
END_DATE = pd.Timestamp("2026-09-19", tz="UTC")


# ---------------------------------------------------------
# MT5 point sizes
# Used only to reconstruct approximate bid/ask from
# H1 mid-price + spread.
# ---------------------------------------------------------

POINT_SIZE = {
    "EURUSDm": 0.00001,
    "GBPUSDm": 0.00001,
    "USDJPYm": 0.001,

    "XAUUSDm": 0.001,
    "XAGUSDm": 0.001,

    "USTECm": 0.01,
    "JP225m": 0.1,

    "BTCUSDm": 0.01,
    "ETHUSDm": 0.01,

    "USOILm": 0.01,
}


# =========================================================
# DATA LOADING
# =========================================================

def find_column(df, candidates):
    """
    Find a dataframe column using case-insensitive matching.
    """

    lower_map = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    for candidate in candidates:
        candidate_lower = candidate.lower()

        if candidate_lower in lower_map:
            return lower_map[candidate_lower]

    return None


def load_mt5_h1(symbol):
    """
    Load MT5 H1 CSV and normalize column names.
    """

    path = DATA_DIR / f"{symbol}_H1_2019_2026.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing data file:\n{path}"
        )

    print(f"\nLoading {symbol}")
    print(f"File: {path}")

    df = pd.read_csv(path)

    print(f"Raw rows: {len(df):,}")

    # -----------------------------------------------------
    # Identify columns
    # -----------------------------------------------------

    time_col = find_column(
        df,
        [
            "time",
            "timestamp",
            "datetime",
            "date",
        ],
    )

    open_col = find_column(df, ["open"])
    high_col = find_column(df, ["high"])
    low_col = find_column(df, ["low"])
    close_col = find_column(df, ["close"])

    spread_col = find_column(
        df,
        [
            "spreadpoints",
            "spread_points",
            "spread",
        ],
    )

    required = {
        "time": time_col,
        "open": open_col,
        "high": high_col,
        "low": low_col,
        "close": close_col,
        "spread": spread_col,
    }

    missing = [
        name
        for name, col in required.items()
        if col is None
    ]

    if missing:
        raise ValueError(
            f"{symbol}: Missing columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    # -----------------------------------------------------
    # Normalize
    # -----------------------------------------------------

    out = pd.DataFrame()

    out["Time"] = pd.to_datetime(
        df[time_col],
        utc=True,
        errors="coerce",
    )

    out["Open"] = pd.to_numeric(
        df[open_col],
        errors="coerce",
    )

    out["High"] = pd.to_numeric(
        df[high_col],
        errors="coerce",
    )

    out["Low"] = pd.to_numeric(
        df[low_col],
        errors="coerce",
    )

    out["Close"] = pd.to_numeric(
        df[close_col],
        errors="coerce",
    )

    out["SpreadPoints"] = pd.to_numeric(
        df[spread_col],
        errors="coerce",
    )

    # -----------------------------------------------------
    # Clean
    # -----------------------------------------------------

    out = out.dropna()

    out = out[
        (out["Open"] > 0)
        & (out["High"] > 0)
        & (out["Low"] > 0)
        & (out["Close"] > 0)
    ]

    out = out[
        (out["High"] >= out["Low"])
        & (out["High"] >= out["Open"])
        & (out["High"] >= out["Close"])
        & (out["Low"] <= out["Open"])
        & (out["Low"] <= out["Close"])
    ]

    out = out.sort_values("Time")

    out = out.drop_duplicates(
        subset=["Time"],
        keep="first",
    )

    # -----------------------------------------------------
    # Restrict period
    # -----------------------------------------------------

    out = out[
        (out["Time"] >= START_DATE)
        & (out["Time"] <= END_DATE)
    ].copy()

    out = out.reset_index(drop=True)

    if len(out) == 0:
        raise ValueError(
            f"{symbol}: no data in requested period."
        )

    # -----------------------------------------------------
    # Reconstruct approximate bid/ask
    #
    # MT5 H1 OHLC is treated as mid-price OHLC.
    # SpreadPoints is converted using symbol point size.
    # -----------------------------------------------------

    point_size = POINT_SIZE[symbol]

    out["SpreadPrice"] = (
        out["SpreadPoints"] * point_size
    )

    out["Bid"] = (
        out["Close"]
        - out["SpreadPrice"] / 2.0
    )

    out["Ask"] = (
        out["Close"]
        + out["SpreadPrice"] / 2.0
    )

    # Approximate next-bar executable prices
    out["Bid_Open"] = (
        out["Open"]
        - out["SpreadPrice"] / 2.0
    )

    out["Ask_Open"] = (
        out["Open"]
        + out["SpreadPrice"] / 2.0
    )

    # Mid-price used by Kalman
    out["Mid"] = out["Close"]

    return out


# =========================================================
# KALMAN + Z SCORE
# =========================================================

def calculate_features(df):
    """
    Apply the EXACT Kalman implementation used in the
    validated XAU comparison.
    """

    prices = df["Mid"]

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

    residual_series = pd.Series(
        residuals,
        index=df.index,
        dtype=float,
    )

    zscore = calculate_zscore(
        residual_series,
        window=Z_WINDOW,
    )

    df = df.copy()

    df["KalmanState"] = states
    df["KalmanVariance"] = variances
    df["Residual"] = residuals
    df["RollingMean"] = means
    df["ZScore"] = zscore

    return df


# =========================================================
# POSITION SIZING
# =========================================================

def calculate_units(
    equity,
    entry_price,
):
    """
    Research position sizing.

    Risk budget:
        equity * RISK_PER_TRADE

    Assumed loss at stop:
        units * entry_price * ASSUMED_STOP_RETURN

    Therefore:
        units =
            risk_budget /
            (entry_price * assumed_stop_return)

    Then cap notional at MAX_NOTIONAL_FRACTION.

    This is deliberately conservative and does NOT model
    actual MT5 contract/tick value yet.
    """

    if entry_price <= 0:
        return 0.0

    risk_budget = (
        equity * RISK_PER_TRADE
    )

    units = (
        risk_budget
        / (
            entry_price
            * ASSUMED_STOP_RETURN
        )
    )

    max_notional = (
        equity
        * MAX_NOTIONAL_FRACTION
    )

    max_units = (
        max_notional
        / entry_price
    )

    units = min(
        units,
        max_units,
    )

    return max(units, 0.0)


# =========================================================
# BACKTEST
# =========================================================

def run_backtest(
    symbol,
    df,
):
    """
    Event-driven research backtest.

    Signal is generated using bar t.

    Execution occurs at bar t+1 open:

        LONG  -> Ask_Open
        SHORT -> Bid_Open

    Exit:

        LONG  -> Bid_Open
        SHORT -> Ask_Open

    Equity is marked to market every bar.
    """

    df = calculate_features(df)

    equity = INITIAL_CAPITAL

    equity_curve = []

    trades = []

    position = 0
    # +1 = long
    # -1 = short
    #  0 = flat

    units = 0.0
    entry_price = np.nan
    entry_time = None
    entry_z = np.nan

    trading_hours_held = 0
    calendar_hours_held = 0.0

    pending_action = None

    pending_reason = None

    # -----------------------------------------------------
    # Iterate through bars
    # -----------------------------------------------------

    for i in range(len(df)):

        row = df.iloc[i]

        timestamp = row["Time"]

        z = row["ZScore"]

        # -------------------------------------------------
        # Execute action generated on previous bar
        # -------------------------------------------------

        if pending_action is not None:

            action = pending_action
            reason = pending_reason

            # ---------------------------------------------
            # ENTRY
            # ---------------------------------------------

            if action == "ENTER_LONG":

                execution_price = row["Ask_Open"]

                units = calculate_units(
                    equity,
                    execution_price,
                )

                if units > 0:

                    position = 1

                    entry_price = execution_price

                    entry_time = timestamp

                    entry_z = previous_z

                    trading_hours_held = 0

                    calendar_hours_held = 0.0

            elif action == "ENTER_SHORT":

                execution_price = row["Bid_Open"]

                units = calculate_units(
                    equity,
                    execution_price,
                )

                if units > 0:

                    position = -1

                    entry_price = execution_price

                    entry_time = timestamp

                    entry_z = previous_z

                    trading_hours_held = 0

                    calendar_hours_held = 0.0

            # ---------------------------------------------
            # EXIT
            # ---------------------------------------------

            elif action == "EXIT":

                if position != 0:

                    if position == 1:
                        execution_price = row["Bid_Open"]

                        pnl = (
                            execution_price
                            - entry_price
                        ) * units

                    else:
                        execution_price = row["Ask_Open"]

                        pnl = (
                            entry_price
                            - execution_price
                        ) * units

                    equity_before = equity

                    equity += pnl

                    exit_time = timestamp

                    calendar_hours = (
                        exit_time
                        - entry_time
                    ).total_seconds() / 3600.0

                    trade_return = (
                        pnl / equity_before
                    )

                    trades.append(
                        {
                            "Symbol": symbol,
                            "EntryTime": entry_time,
                            "ExitTime": exit_time,
                            "Side": (
                                "LONG"
                                if position == 1
                                else "SHORT"
                            ),
                            "Units": units,
                            "EntryPrice": entry_price,
                            "ExitPrice": execution_price,
                            "EntryZ": entry_z,
                            "ExitZ": previous_z,
                            "TradingHoursHeld": trading_hours_held,
                            "CalendarHoursHeld": calendar_hours,
                            "HoursHeld": trading_hours_held,
                            "Return": trade_return,
                            "PnL": pnl,
                            "ExitReason": reason,
                        }
                    )

                    position = 0

                    units = 0.0

                    entry_price = np.nan

                    entry_time = None

                    entry_z = np.nan

                    trading_hours_held = 0

                    calendar_hours_held = 0.0

            pending_action = None
            pending_reason = None

        # -------------------------------------------------
        # Mark to market
        # -------------------------------------------------

        if position == 0:

            current_equity = equity

        elif position == 1:

            current_equity = (
                equity
                + (
                    row["Bid"]
                    - entry_price
                ) * units
            )

        else:

            current_equity = (
                equity
                + (
                    entry_price
                    - row["Ask"]
                ) * units
            )

        equity_curve.append(
            {
                "Time": timestamp,
                "Equity": current_equity,
                "Position": position,
                "ZScore": z,
            }
        )

        # -------------------------------------------------
        # Need previous Z
        # -------------------------------------------------

        if i == 0:

            previous_z = z

            continue

        # -------------------------------------------------
        # Update holding time
        # -------------------------------------------------

        if position != 0:

            trading_hours_held += 1

            calendar_hours_held = (
                timestamp
                - entry_time
            ).total_seconds() / 3600.0

        # -------------------------------------------------
        # Ignore until Z is valid
        # -------------------------------------------------

        if pd.isna(z):

            previous_z = z

            continue

        # -------------------------------------------------
        # Generate signals
        # -------------------------------------------------

        if position == 0:

            if z <= -ENTRY_Z:

                pending_action = "ENTER_LONG"

                pending_reason = "Z_ENTRY_LONG"

            elif z >= ENTRY_Z:

                pending_action = "ENTER_SHORT"

                pending_reason = "Z_ENTRY_SHORT"

        elif position == 1:

            # Mean-reversion exit
            if abs(z) <= EXIT_Z:

                pending_action = "EXIT"

                pending_reason = "Z_EXIT"

            # Stop
            elif abs(z) >= STOP_Z:

                pending_action = "EXIT"

                pending_reason = "Z_STOP"

            # Maximum holding period
            elif (
                trading_hours_held
                >= MAX_HOLD_HOURS
            ):

                pending_action = "EXIT"

                pending_reason = "MAX_HOLD"

        elif position == -1:

            # Mean-reversion exit
            if abs(z) <= EXIT_Z:

                pending_action = "EXIT"

                pending_reason = "Z_EXIT"

            # Stop
            elif abs(z) >= STOP_Z:

                pending_action = "EXIT"

                pending_reason = "Z_STOP"

            # Maximum holding period
            elif (
                trading_hours_held
                >= MAX_HOLD_HOURS
            ):

                pending_action = "EXIT"

                pending_reason = "MAX_HOLD"

        previous_z = z

    # -----------------------------------------------------
    # Force close at end if still open
    # -----------------------------------------------------

    if position != 0:

        last = df.iloc[-1]

        if position == 1:

            execution_price = last["Bid"]

            pnl = (
                execution_price
                - entry_price
            ) * units

        else:

            execution_price = last["Ask"]

            pnl = (
                entry_price
                - execution_price
            ) * units

        equity_before = equity

        equity += pnl

        exit_time = last["Time"]

        calendar_hours = (
            exit_time
            - entry_time
        ).total_seconds() / 3600.0

        trade_return = (
            pnl / equity_before
        )

        trades.append(
            {
                "Symbol": symbol,
                "EntryTime": entry_time,
                "ExitTime": exit_time,
                "Side": (
                    "LONG"
                    if position == 1
                    else "SHORT"
                ),
                "Units": units,
                "EntryPrice": entry_price,
                "ExitPrice": execution_price,
                "EntryZ": entry_z,
                "ExitZ": previous_z,
                "TradingHoursHeld": trading_hours_held,
                "CalendarHoursHeld": calendar_hours,
                "HoursHeld": trading_hours_held,
                "Return": trade_return,
                "PnL": pnl,
                "ExitReason": "END_OF_DATA",
            }
        )

    # -----------------------------------------------------
    # Convert results
    # -----------------------------------------------------

    equity_df = pd.DataFrame(equity_curve)

    trades_df = pd.DataFrame(trades)

    # -----------------------------------------------------
    # Equity statistics
    # -----------------------------------------------------

    if len(equity_df) > 0:

        equity_df["Peak"] = (
            equity_df["Equity"]
            .cummax()
        )

        equity_df["Drawdown"] = (
            equity_df["Equity"]
            / equity_df["Peak"]
            - 1.0
        )

        max_dd = (
            equity_df["Drawdown"]
            .min()
        )

        final_equity = (
            equity_df["Equity"].iloc[-1]
        )

    else:

        max_dd = 0.0

        final_equity = INITIAL_CAPITAL

    total_return = (
        final_equity
        / INITIAL_CAPITAL
        - 1.0
    )

    # -----------------------------------------------------
    # Hourly Sharpe
    # -----------------------------------------------------

    if len(equity_df) > 1:

        returns = (
            equity_df["Equity"]
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
                / returns.std()
                * np.sqrt(24 * 365)
            )

        else:

            sharpe = np.nan

    else:

        sharpe = np.nan

    # -----------------------------------------------------
    # Trade statistics
    # -----------------------------------------------------

    trade_count = len(trades_df)

    if trade_count > 0:

        winning = (
            trades_df["PnL"] > 0
        )

        win_rate = (
            winning.mean()
        )

        gross_profit = (
            trades_df.loc[
                trades_df["PnL"] > 0,
                "PnL",
            ].sum()
        )

        gross_loss = abs(
            trades_df.loc[
                trades_df["PnL"] < 0,
                "PnL",
            ].sum()
        )

        if gross_loss > 0:

            profit_factor = (
                gross_profit
                / gross_loss
            )

        else:

            profit_factor = np.inf

        expectancy = (
            trades_df["PnL"].mean()
        )

        avg_hold = (
            trades_df[
                "TradingHoursHeld"
            ].mean()
        )

        max_hold = (
            trades_df[
                "TradingHoursHeld"
            ].max()
        )

    else:

        win_rate = np.nan
        profit_factor = np.nan
        expectancy = np.nan
        avg_hold = np.nan
        max_hold = np.nan

    # -----------------------------------------------------
    # Exit reasons
    # -----------------------------------------------------

    if trade_count > 0:

        exit_counts = (
            trades_df[
                "ExitReason"
            ]
            .value_counts()
            .to_dict()
        )

    else:

        exit_counts = {}

    # -----------------------------------------------------
    # Monthly returns
    # -----------------------------------------------------

    if len(equity_df) > 0:

        monthly_equity = (
            equity_df
            .set_index("Time")["Equity"]
            .resample("ME")
            .last()
        )

        monthly_returns = (
            monthly_equity
            .pct_change()
        )

        # First month relative to initial capital
        if len(monthly_equity) > 0:

            first_return = (
                monthly_equity.iloc[0]
                / INITIAL_CAPITAL
                - 1.0
            )

            monthly_returns.iloc[0] = first_return

        profitable_months = int(
            (monthly_returns > 0).sum()
        )

        losing_months = int(
            (monthly_returns < 0).sum()
        )

        flat_months = int(
            (monthly_returns == 0).sum()
        )

        worst_month = (
            monthly_returns.min()
            if len(monthly_returns) > 0
            else np.nan
        )

        best_month = (
            monthly_returns.max()
            if len(monthly_returns) > 0
            else np.nan
        )

    else:

        monthly_returns = pd.Series(
            dtype=float
        )

        profitable_months = 0
        losing_months = 0
        flat_months = 0
        worst_month = np.nan
        best_month = np.nan

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    summary = {
        "Symbol": symbol,
        "Start": df["Time"].iloc[0],
        "End": df["Time"].iloc[-1],
        "Bars": len(df),

        "InitialEquity": INITIAL_CAPITAL,
        "FinalEquity": final_equity,

        "TotalReturn": total_return,
        "MaxDrawdown": max_dd,
        "Sharpe": sharpe,

        "Trades": trade_count,
        "WinRate": win_rate,
        "ProfitFactor": profit_factor,
        "Expectancy": expectancy,

        "AvgTradingHoursHeld": avg_hold,
        "MaxTradingHoursHeld": max_hold,

        "ProfitableMonths": profitable_months,
        "LosingMonths": losing_months,
        "FlatMonths": flat_months,

        "BestMonth": best_month,
        "WorstMonth": worst_month,

        "ZValidObservations": int(
            df["ZScore"].notna().sum()
        ),

        "Exit_Z_EXIT": exit_counts.get(
            "Z_EXIT",
            0,
        ),

        "Exit_Z_STOP": exit_counts.get(
            "Z_STOP",
            0,
        ),

        "Exit_MAX_HOLD": exit_counts.get(
            "MAX_HOLD",
            0,
        ),

        "Exit_END_OF_DATA": exit_counts.get(
            "END_OF_DATA",
            0,
        ),
    }

    return (
        df,
        equity_df,
        trades_df,
        summary,
        monthly_returns,
    )


# =========================================================
# SAVE ONE INSTRUMENT
# =========================================================

def save_instrument_results(
    symbol,
    processed,
    equity,
    trades,
    summary,
    monthly_returns,
):

    symbol_dir = RESULTS_DIR / symbol

    symbol_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    processed.to_csv(
        symbol_dir / "processed_H1.csv",
        index=False,
    )

    equity.to_csv(
        symbol_dir / "equity.csv",
        index=False,
    )

    trades.to_csv(
        symbol_dir / "trades.csv",
        index=False,
    )

    pd.DataFrame(
        [summary]
    ).to_csv(
        symbol_dir / "summary.csv",
        index=False,
    )

    monthly_df = (
        monthly_returns
        .rename("MonthlyReturn")
        .to_frame()
    )

    monthly_df.index.name = "Month"

    monthly_df.to_csv(
        symbol_dir / "monthly_returns.csv"
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 80)
    print("MULTI-INSTRUMENT KALMAN MEAN REVERSION BACKTEST")
    print("=" * 80)

    print()
    print("LOCKED PARAMETERS")
    print("-" * 80)

    print(f"PHI             : {PHI}")
    print(f"Q               : {Q}")
    print(f"R               : {R}")
    print(f"MEAN_WINDOW     : {MEAN_WINDOW}")
    print(f"Z_WINDOW        : {Z_WINDOW}")
    print(f"ENTRY_Z         : {ENTRY_Z}")
    print(f"EXIT_Z          : {EXIT_Z}")
    print(f"STOP_Z          : {STOP_Z}")
    print(f"MAX_HOLD_HOURS  : {MAX_HOLD_HOURS}")

    print()
    print(f"Initial capital : ${INITIAL_CAPITAL:,.2f}")
    print(
        f"Risk/trade      : "
        f"{RISK_PER_TRADE * 100:.2f}%"
    )
    print(
        f"Max notional    : "
        f"{MAX_NOTIONAL_FRACTION * 100:.0f}%"
    )

    print()
    print(
        f"Period          : "
        f"{START_DATE} -> {END_DATE}"
    )

    print()
    print("=" * 80)

    all_summaries = []

    all_monthly = {}

    for symbol in INSTRUMENTS:

        print()
        print("#" * 80)
        print(f"BACKTESTING {symbol}")
        print("#" * 80)

        try:

            df = load_mt5_h1(symbol)

            (
                processed,
                equity,
                trades,
                summary,
                monthly_returns,
            ) = run_backtest(
                symbol,
                df,
            )

            save_instrument_results(
                symbol,
                processed,
                equity,
                trades,
                summary,
                monthly_returns,
            )

            all_summaries.append(
                summary
            )

            all_monthly[symbol] = (
                monthly_returns
            )

            print()
            print(
                f"{symbol} RESULT"
            )
            print("-" * 60)

            print(
                f"Bars              : "
                f"{summary['Bars']:,}"
            )

            print(
                f"Valid Z           : "
                f"{summary['ZValidObservations']:,}"
            )

            print(
                f"Final equity      : "
                f"${summary['FinalEquity']:,.2f}"
            )

            print(
                f"Total return      : "
                f"{summary['TotalReturn'] * 100:.3f}%"
            )

            print(
                f"Maximum DD        : "
                f"{summary['MaxDrawdown'] * 100:.3f}%"
            )

            print(
                f"Sharpe            : "
                f"{summary['Sharpe']:.3f}"
            )

            print(
                f"Trades            : "
                f"{summary['Trades']:,}"
            )

            print(
                f"Win rate          : "
                f"{summary['WinRate'] * 100:.2f}%"
            )

            print(
                f"Profit factor     : "
                f"{summary['ProfitFactor']:.3f}"
            )

            print(
                f"Expectancy        : "
                f"${summary['Expectancy']:.4f}"
            )

            print(
                f"Profitable months : "
                f"{summary['ProfitableMonths']}"
            )

            print(
                f"Losing months     : "
                f"{summary['LosingMonths']}"
            )

            print(
                f"Best month        : "
                f"{summary['BestMonth'] * 100:.3f}%"
            )

            print(
                f"Worst month       : "
                f"{summary['WorstMonth'] * 100:.3f}%"
            )

        except Exception as e:

            print()
            print(
                f"ERROR processing {symbol}:"
            )

            print(
                repr(e)
            )

    # =====================================================
    # COMBINED SUMMARY
    # =====================================================

    if not all_summaries:

        print(
            "\nNo instruments completed."
        )

        return

    summary_df = pd.DataFrame(
        all_summaries
    )

    summary_df.to_csv(
        RESULTS_DIR
        / "all_instruments_summary.csv",
        index=False,
    )

    # -----------------------------------------------------
    # Monthly cross-sectional portfolio
    #
    # Equal-weight average of instrument monthly returns.
    #
    # This is a research diagnostic, NOT a final portfolio
    # implementation.
    # -----------------------------------------------------

    if all_monthly:

        monthly_matrix = pd.DataFrame(
            all_monthly
        )

        equal_weight_monthly = (
            monthly_matrix.mean(
                axis=1,
                skipna=True,
            )
        )

        equal_weight_monthly.name = (
            "EqualWeightMonthlyReturn"
        )

        monthly_matrix.to_csv(
            RESULTS_DIR
            / "instrument_monthly_returns.csv"
        )

        equal_weight_monthly.to_csv(
            RESULTS_DIR
            / "equal_weight_monthly_returns.csv"
        )

        # -------------------------------------------------
        # Compound equal-weight monthly curve
        # -------------------------------------------------

        portfolio_equity = (
            INITIAL_CAPITAL
            * (
                1.0
                + equal_weight_monthly.fillna(0.0)
            ).cumprod()
        )

        portfolio_curve = pd.DataFrame(
            {
                "Equity": portfolio_equity,
                "MonthlyReturn": (
                    equal_weight_monthly
                ),
            }
        )

        portfolio_curve.to_csv(
            RESULTS_DIR
            / "equal_weight_portfolio.csv"
        )

        portfolio_dd = (
            portfolio_equity
            / portfolio_equity.cummax()
            - 1.0
        )

        portfolio_return = (
            portfolio_equity.iloc[-1]
            / INITIAL_CAPITAL
            - 1.0
        )

        portfolio_max_dd = (
            portfolio_dd.min()
        )

    else:

        portfolio_return = np.nan
        portfolio_max_dd = np.nan

    # =====================================================
    # FINAL CONSOLE TABLE
    # =====================================================

    display_columns = [
        "Symbol",
        "Bars",
        "TotalReturn",
        "MaxDrawdown",
        "Sharpe",
        "Trades",
        "WinRate",
        "ProfitFactor",
        "ProfitableMonths",
        "LosingMonths",
        "WorstMonth",
    ]

    display_df = summary_df[
        display_columns
    ].copy()

    display_df["TotalReturn"] *= 100
    display_df["MaxDrawdown"] *= 100
    display_df["WinRate"] *= 100
    display_df["WorstMonth"] *= 100

    print()
    print()
    print("=" * 120)
    print("FINAL 10-INSTRUMENT SUMMARY")
    print("=" * 120)

    print(
        display_df.to_string(
            index=False,
            formatters={
                "TotalReturn": "{:.3f}%".format,
                "MaxDrawdown": "{:.3f}%".format,
                "Sharpe": "{:.3f}".format,
                "WinRate": "{:.2f}%".format,
                "ProfitFactor": "{:.3f}".format,
                "WorstMonth": "{:.3f}%".format,
            },
        )
    )

    print()
    print("=" * 80)
    print("EQUAL-WEIGHT CROSS-ASSET DIAGNOSTIC")
    print("=" * 80)

    print(
        f"Compound return : "
        f"{portfolio_return * 100:.3f}%"
    )

    print(
        f"Maximum DD      : "
        f"{portfolio_max_dd * 100:.3f}%"
    )

    print()
    print(
        f"Results saved to:"
    )

    print(
        RESULTS_DIR
    )

    print()
    print("=" * 80)
    print("BACKTEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()