from pathlib import Path
import sys
import argparse

import numpy as np
import pandas as pd


# ============================================================
# PATH SETUP
# ============================================================

SRC_DIR = Path(__file__).resolve().parent
ROOT = SRC_DIR.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ============================================================
# IMPORT ORIGINAL KALMAN IMPLEMENTATION
# ============================================================

from walk_forward_kalman import (
    walk_forward_kalman,
    calculate_zscore,
)

from markov_regime import add_markov_regime


# ============================================================
# PATHS
# ============================================================

DATA_DIR = ROOT / "data" / "mt5_h1"

RESULTS_DIR = (
    ROOT
    / "results"
    / "actual_trade_regimes"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# INSTRUMENTS
# ============================================================

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


# ============================================================
# BACKTEST PERIOD
#
# IMPORTANT:
# This now matches the canonical XAUUSDm backtest_h1.py
# validation period.
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
# HMM PARAMETERS
# ============================================================

HMM_WINDOW = 1000
HMM_MIN_OBSERVATIONS = 300
HMM_REFIT_EVERY = 24

MR_THRESHOLD = 0.60


# ============================================================
# POINT SIZES
# ============================================================

POINT_SIZES = {
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


# ============================================================
# LOAD MT5 H1 DATA
# ============================================================

def load_mt5_h1(symbol):

    path = (
        DATA_DIR
        / f"{symbol}_H1_2019_2026.csv"
    )

    print()
    print("=" * 80)
    print(f"LOADING {symbol}")
    print("=" * 80)

    if not path.exists():

        raise FileNotFoundError(
            f"\nMT5 H1 file not found:\n{path}"
        )

    df = pd.read_csv(path)

    print(
        f"Rows loaded: {len(df):,}"
    )

    print(
        f"Columns: {list(df.columns)}"
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
            f"{symbol}: could not find Timestamp/Time column."
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
            f"{symbol}: missing columns: {missing}"
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
    # Restrict to canonical validation period
    # --------------------------------------------------------

    df = df.loc[
        (df.index >= START_DATE)
        &
        (df.index <= END_DATE)
    ].copy()

    if df.empty:

        raise ValueError(
            f"{symbol}: no data in selected period."
        )

    print(
        f"Bars in analysis period: {len(df):,}"
    )

    print(
        f"Start: {df.index.min()}"
    )

    print(
        f"End:   {df.index.max()}"
    )

    return df


# ============================================================
# ADD BID / ASK APPROXIMATION
# ============================================================

def add_bid_ask(
    df,
    symbol,
):

    if symbol not in POINT_SIZES:

        raise ValueError(
            f"No point size configured for {symbol}"
        )

    point_size = POINT_SIZES[
        symbol
    ]

    spread_price = (
        df["SpreadPoints"]
        * point_size
    )

    spread_price = (
        spread_price
        .clip(lower=0)
        .fillna(0)
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
# KALMAN FEATURES
# ============================================================

def calculate_kalman_features(
    df,
):

    print()
    print("=" * 80)
    print("CALCULATING KALMAN FEATURES")
    print("=" * 80)

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
    # Preserve exact dataframe index
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
# HMM / MARKOV REGIME
# ============================================================

def calculate_markov_features(
    df,
):

    print()
    print("=" * 80)
    print("CALCULATING CAUSAL MARKOV / HMM REGIME")
    print("=" * 80)

    df = add_markov_regime(
        df,
        hmm_window=HMM_WINDOW,
        min_observations=HMM_MIN_OBSERVATIONS,
        refit_every=HMM_REFIT_EVERY,
    )

    valid_hmm = (
        df["P_MeanReverting"]
        .notna()
    )

    print(
        f"Valid HMM observations: "
        f"{valid_hmm.sum():,}"
    )

    if valid_hmm.any():

        print(
            f"Mean P(MR): "
            f"{df.loc[valid_hmm, 'P_MeanReverting'].mean():.4f}"
        )

        print(
            f"Mean P(Trend): "
            f"{df.loc[valid_hmm, 'P_Trending'].mean():.4f}"
        )

        if (
            "MR_Autocorrelation"
            in df.columns
        ):

            print(
                f"MR autocorrelation: "
                f"{df.loc[valid_hmm, 'MR_Autocorrelation'].mean():.5f}"
            )

        if (
            "Trend_Autocorrelation"
            in df.columns
        ):

            print(
                f"Trend autocorrelation: "
                f"{df.loc[valid_hmm, 'Trend_Autocorrelation'].mean():.5f}"
            )

    return df


# ============================================================
# REGIME CLASSIFICATION
# ============================================================

def classify_regime(
    p_mr,
):

    if pd.isna(p_mr):

        return "UNKNOWN"

    if p_mr >= MR_THRESHOLD:

        return "MR"

    return "TREND"


# ============================================================
# POSITION SIZING
#
# Matches canonical backtest_h1.py.
# ============================================================

def calculate_units(
    equity,
    price,
):

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
        / stop_loss_per_unit
    )

    max_notional = (
        equity
        * MAX_NOTIONAL_FRACTION
    )

    units_from_notional = (
        max_notional
        / price
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
#
# IMPORTANT:
# This follows the canonical backtest_h1.py event loop:
#
# 1. Mark-to-market current bar
# 2. Determine exit on current Z
# 3. Execute exit on next bar
# 4. If flat, determine entry on current Z
# 5. Execute entry on next bar
# 6. No same-bar re-entry after an exit
#
# Markov is an ENTRY GATE only.
# ============================================================

def run_backtest(
    df,
    symbol,
    use_markov=False,
):

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

    entry_regime = "UNKNOWN"

    entry_p_mr = np.nan

    entry_p_trend = np.nan

    trades = []

    equity_curve = []

    blocked_entries = []

    # --------------------------------------------------------
    # Event loop
    # --------------------------------------------------------

    for i in range(
        len(df)
    ):

        timestamp = (
            df.index[i]
        )

        row = df.iloc[i]

        z = row["Z"]

        close_mid = row["Close"]

        bid_close = row["Bid_Close"]

        ask_close = row["Ask_Close"]

        p_mr = row.get(
            "P_MeanReverting",
            np.nan,
        )

        p_trend = row.get(
            "P_Trending",
            np.nan,
        )

        regime = classify_regime(
            p_mr
        )

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
                "Timestamp":
                    timestamp,

                "Equity":
                    equity,

                "CashEquity":
                    cash_equity,

                "Position":
                    position,

                "Units":
                    units,

                "Mid":
                    close_mid,

                "Z":
                    z,

                "P_MeanReverting":
                    p_mr,

                "P_Trending":
                    p_trend,

                "Regime":
                    regime,

                "MarkovEnabled":
                    use_markov,
            }
        )

        # ----------------------------------------------------
        # Need next bar for execution.
        # ----------------------------------------------------

        if i >= len(df) - 1:

            continue

        next_row = df.iloc[
            i + 1
        ]

        next_timestamp = (
            df.index[i + 1]
        )

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
            # Trading hours held
            #
            # Same definition as canonical engine:
            # current index - entry index
            # ------------------------------------------------

            trading_hours_held = (
                i
                - entry_index
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

                    exit_reason = (
                        "Z_EXIT"
                    )

                # Stop
                elif abs(z) >= STOP_Z:

                    exit_reason = (
                        "Z_STOP"
                    )

            # Maximum holding period
            if (
                exit_reason is None
                and
                trading_hours_held
                >= MAX_HOLD_HOURS
            ):

                exit_reason = (
                    "MAX_HOLD"
                )

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
                    / entry_equity
                )

                trades.append(
                    {
                        "Symbol":
                            symbol,

                        "EntryTime":
                            entry_time,

                        "ExitTime":
                            next_timestamp,

                        "Side":
                            (
                                "LONG"
                                if position == 1
                                else "SHORT"
                            ),

                        "Units":
                            units,

                        "EntryPrice":
                            entry_price,

                        "ExitPrice":
                            exit_price,

                        "EntryZ":
                            entry_z,

                        "ExitZ":
                            z,

                        "EntryRegime":
                            entry_regime,

                        "EntryPMR":
                            entry_p_mr,

                        "EntryPTrend":
                            entry_p_trend,

                        "ExitRegime":
                            regime,

                        "ExitPMR":
                            p_mr,

                        "ExitPTrend":
                            p_trend,

                        "TradingHoursHeld":
                            trading_hours_held,

                        "CalendarHoursHeld":
                            calendar_hours_held,

                        "HoursHeld":
                            trading_hours_held,

                        "Return":
                            trade_return,

                        "PnL":
                            pnl,

                        "ExitReason":
                            exit_reason,

                        "MarkovEnabled":
                            use_markov,
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

                entry_regime = "UNKNOWN"

                entry_p_mr = np.nan

                entry_p_trend = np.nan

                # ------------------------------------------------
                # IMPORTANT:
                #
                # Match canonical engine:
                # don't enter another position on this
                # same signal bar after an exit.
                # ------------------------------------------------

                continue

        # ====================================================
        # ENTRY
        # ====================================================

        if position == 0:

            if pd.isna(z):

                continue

            desired_side = None

            # ------------------------------------------------
            # LONG signal
            # ------------------------------------------------

            if z <= -ENTRY_Z:

                desired_side = "LONG"

            # ------------------------------------------------
            # SHORT signal
            # ------------------------------------------------

            elif z >= ENTRY_Z:

                desired_side = "SHORT"

            # ------------------------------------------------
            # If no signal
            # ------------------------------------------------

            if desired_side is None:

                continue

            # =================================================
            # MARKOV ENTRY GATE
            # =================================================

            if use_markov:

                # No HMM probability = cannot trade
                if pd.isna(p_mr):

                    blocked_entries.append(
                        {
                            "Symbol":
                                symbol,

                            "Time":
                                timestamp,

                            "Side":
                                desired_side,

                            "ZScore":
                                z,

                            "PMR":
                                p_mr,

                            "PTrend":
                                p_trend,

                            "Regime":
                                "UNKNOWN",

                            "BlockReason":
                                "NO_HMM",
                        }
                    )

                    continue

                # Require MR probability >= threshold
                if p_mr < MR_THRESHOLD:

                    blocked_entries.append(
                        {
                            "Symbol":
                                symbol,

                            "Time":
                                timestamp,

                            "Side":
                                desired_side,

                            "ZScore":
                                z,

                            "PMR":
                                p_mr,

                            "PTrend":
                                p_trend,

                            "Regime":
                                regime,

                            "BlockReason":
                                "TREND_REGIME",
                        }
                    )

                    continue

            # =================================================
            # ENTRY EXECUTION
            # =================================================

            if desired_side == "LONG":

                entry_price = (
                    next_ask_open
                )

                position = 1

            else:

                entry_price = (
                    next_bid_open
                )

                position = -1

            # ------------------------------------------------
            # Position sizing
            # ------------------------------------------------

            units = calculate_units(
                equity,
                entry_price,
            )

            if units <= 0:

                position = 0

                continue

            # ------------------------------------------------
            # Store entry state
            # ------------------------------------------------

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

            entry_regime = (
                regime
            )

            entry_p_mr = (
                p_mr
            )

            entry_p_trend = (
                p_trend
            )

    # ========================================================
    # FORCE CLOSE REMAINING POSITION
    # ========================================================

    if position != 0:

        last_row = df.iloc[-1]

        last_timestamp = (
            df.index[-1]
        )

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

        equity = (
            cash_equity
        )

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
            / entry_equity
        )

        last_p_mr = last_row.get(
            "P_MeanReverting",
            np.nan,
        )

        last_p_trend = last_row.get(
            "P_Trending",
            np.nan,
        )

        trades.append(
            {
                "Symbol":
                    symbol,

                "EntryTime":
                    entry_time,

                "ExitTime":
                    last_timestamp,

                "Side":
                    (
                        "LONG"
                        if position == 1
                        else "SHORT"
                    ),

                "Units":
                    units,

                "EntryPrice":
                    entry_price,

                "ExitPrice":
                    exit_price,

                "EntryZ":
                    entry_z,

                "ExitZ":
                    last_row["Z"],

                "EntryRegime":
                    entry_regime,

                "EntryPMR":
                    entry_p_mr,

                "EntryPTrend":
                    entry_p_trend,

                "ExitRegime":
                    classify_regime(
                        last_p_mr
                    ),

                "ExitPMR":
                    last_p_mr,

                "ExitPTrend":
                    last_p_trend,

                "TradingHoursHeld":
                    trading_hours_held,

                "CalendarHoursHeld":
                    calendar_hours_held,

                "HoursHeld":
                    trading_hours_held,

                "Return":
                    trade_return,

                "PnL":
                    pnl,

                "ExitReason":
                    "END_OF_DATA",

                "MarkovEnabled":
                    use_markov,
            }
        )

    # ========================================================
    # DATAFRAMES
    # ========================================================

    equity_df = pd.DataFrame(
        equity_curve
    )

    trades_df = pd.DataFrame(
        trades
    )

    blocked_df = pd.DataFrame(
        blocked_entries
    )

    if not equity_df.empty:

        equity_df = (
            equity_df
            .sort_values("Timestamp")
            .reset_index(drop=True)
        )

    return (
        equity_df,
        trades_df,
        blocked_df,
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    equity_df,
    trades_df,
):

    if equity_df.empty:

        return {
            "InitialEquity":
                INITIAL_CAPITAL,

            "FinalEquity":
                INITIAL_CAPITAL,

            "TotalReturn":
                0.0,

            "MaxDrawdown":
                0.0,

            "Sharpe":
                np.nan,

            "Trades":
                0,

            "WinRate":
                np.nan,

            "ProfitFactor":
                np.nan,

            "Expectancy":
                0.0,
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
        / initial_equity
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
        / running_max
        - 1.0
    )

    max_drawdown = (
        drawdown.min()
    )

    # --------------------------------------------------------
    # Hourly Sharpe
    #
    # Same annualization as canonical engine.
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
            / returns.std()
        ) * np.sqrt(
            24 * 365
        )

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
                / abs(gross_loss)
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

    return {
        "InitialEquity":
            initial_equity,

        "FinalEquity":
            final_equity,

        "TotalReturn":
            total_return,

        "MaxDrawdown":
            max_drawdown,

        "Sharpe":
            sharpe,

        "Trades":
            trade_count,

        "WinRate":
            win_rate,

        "ProfitFactor":
            profit_factor,

        "Expectancy":
            expectancy,
    }


# ============================================================
# REGIME BREAKDOWN
# ============================================================

def calculate_regime_breakdown(
    trades_df,
):

    if trades_df.empty:

        return pd.DataFrame()

    records = []

    for regime in [
        "MR",
        "TREND",
    ]:

        for side in [
            "LONG",
            "SHORT",
        ]:

            subset = trades_df[
                (
                    trades_df[
                        "EntryRegime"
                    ]
                    == regime
                )
                &
                (
                    trades_df[
                        "Side"
                    ]
                    == side
                )
            ]

            if subset.empty:

                continue

            metrics = calculate_metrics(
                pd.DataFrame(
                    {
                        "Equity":
                            (
                                INITIAL_CAPITAL
                                +
                                subset["PnL"].cumsum()
                            )
                    }
                ),
                subset,
            )

            records.append(
                {
                    "EntryRegime":
                        regime,

                    "Side":
                        side,

                    "Trades":
                        metrics["Trades"],

                    "WinRate":
                        metrics["WinRate"],

                    "ProfitFactor":
                        metrics["ProfitFactor"],

                    "Expectancy":
                        metrics["Expectancy"],

                    "TotalPnL":
                        subset["PnL"].sum(),
                }
            )

    return pd.DataFrame(
        records
    )


# ============================================================
# EXIT BREAKDOWN
# ============================================================

def calculate_exit_breakdown(
    trades_df,
):

    if trades_df.empty:

        return pd.DataFrame()

    records = []

    for regime in [
        "MR",
        "TREND",
    ]:

        for side in [
            "LONG",
            "SHORT",
        ]:

            for reason in [
                "Z_EXIT",
                "Z_STOP",
                "MAX_HOLD",
                "END_OF_DATA",
            ]:

                subset = trades_df[
                    (
                        trades_df[
                            "EntryRegime"
                        ]
                        == regime
                    )
                    &
                    (
                        trades_df[
                            "Side"
                        ]
                        == side
                    )
                    &
                    (
                        trades_df[
                            "ExitReason"
                        ]
                        == reason
                    )
                ]

                if subset.empty:

                    continue

                metrics = calculate_metrics(
                    pd.DataFrame(
                        {
                            "Equity":
                                (
                                    INITIAL_CAPITAL
                                    +
                                    subset["PnL"].cumsum()
                                )
                        }
                    ),
                    subset,
                )

                records.append(
                    {
                        "EntryRegime":
                            regime,

                        "Side":
                            side,

                        "ExitReason":
                            reason,

                        "Trades":
                            metrics["Trades"],

                        "WinRate":
                            metrics["WinRate"],

                        "ProfitFactor":
                            metrics["ProfitFactor"],

                        "Expectancy":
                            metrics["Expectancy"],

                        "TotalPnL":
                            subset["PnL"].sum(),
                    }
                )

    return pd.DataFrame(
        records
    )


# ============================================================
# BLOCKED SIGNAL FORWARD RETURNS
#
# This is diagnostic only.
# It does NOT affect the backtest.
# ============================================================

def analyze_blocked_signals(
    df,
    blocked_df,
):

    if blocked_df.empty:

        return pd.DataFrame()

    records = []

    time_to_index = {
        timestamp: i
        for i, timestamp
        in enumerate(df.index)
    }

    for _, signal in (
        blocked_df.iterrows()
    ):

        if signal["Time"] not in time_to_index:

            continue

        i = time_to_index[
            signal["Time"]
        ]

        record = {
            "Symbol":
                signal["Symbol"],

            "Time":
                signal["Time"],

            "Side":
                signal["Side"],

            "ZScore":
                signal["ZScore"],

            "PMR":
                signal["PMR"],

            "PTrend":
                signal["PTrend"],

            "Regime":
                signal["Regime"],

            "BlockReason":
                signal["BlockReason"],
        }

        for horizon in [
            1,
            3,
            6,
            12,
            24,
        ]:

            future_index = (
                i + horizon
            )

            if future_index >= len(df):

                record[
                    f"ForwardReturn_{horizon}h"
                ] = np.nan

                continue

            current_price = (
                df.iloc[i]["Close"]
            )

            future_price = (
                df.iloc[
                    future_index
                ]["Close"]
            )

            forward_return = (
                future_price
                / current_price
                - 1.0
            )

            # Direction-adjusted.
            if signal["Side"] == "SHORT":

                forward_return *= -1.0

            record[
                f"ForwardReturn_{horizon}h"
            ] = forward_return

        records.append(
            record
        )

    return pd.DataFrame(
        records
    )


# ============================================================
# PRINT BASIC METRICS
# ============================================================

def print_metrics_table(
    kalman_metrics,
    markov_metrics,
):

    print()
    print("=" * 90)
    print("KALMAN vs KALMAN + MARKOV")
    print("=" * 90)

    print()

    print(
        f"{'Metric':<25}"
        f"{'Kalman':>20}"
        f"{'Kalman+Markov':>25}"
    )

    print("-" * 75)

    print(
        f"{'Final Equity':<25}"
        f"${kalman_metrics['FinalEquity']:>18,.2f}"
        f"${markov_metrics['FinalEquity']:>23,.2f}"
    )

    print(
        f"{'Total Return':<25}"
        f"{kalman_metrics['TotalReturn'] * 100:>19.3f}%"
        f"{markov_metrics['TotalReturn'] * 100:>24.3f}%"
    )

    print(
        f"{'Maximum DD':<25}"
        f"{kalman_metrics['MaxDrawdown'] * 100:>19.3f}%"
        f"{markov_metrics['MaxDrawdown'] * 100:>24.3f}%"
    )

    print(
        f"{'Sharpe':<25}"
        f"{kalman_metrics['Sharpe']:>20.3f}"
        f"{markov_metrics['Sharpe']:>25.3f}"
    )

    print(
        f"{'Trades':<25}"
        f"{kalman_metrics['Trades']:>20,}"
        f"{markov_metrics['Trades']:>25,}"
    )

    print(
        f"{'Win Rate':<25}"
        f"{kalman_metrics['WinRate'] * 100:>19.2f}%"
        f"{markov_metrics['WinRate'] * 100:>24.2f}%"
    )

    print(
        f"{'Profit Factor':<25}"
        f"{kalman_metrics['ProfitFactor']:>20.3f}"
        f"{markov_metrics['ProfitFactor']:>25.3f}"
    )

    print(
        f"{'Expectancy':<25}"
        f"${kalman_metrics['Expectancy']:>18.4f}"
        f"${markov_metrics['Expectancy']:>23.4f}"
    )


# ============================================================
# PRINT REGIME BREAKDOWN
# ============================================================

def print_regime_breakdown(
    title,
    breakdown,
):

    print()
    print("=" * 90)
    print(title)
    print("=" * 90)

    if breakdown.empty:

        print(
            "No trades."
        )

        return

    display = (
        breakdown.copy()
    )

    display["WinRate"] = (
        display["WinRate"]
        * 100
    )

    print(
        display.to_string(
            index=False,
            formatters={
                "WinRate":
                    lambda x:
                    f"{x:.2f}%",

                "ProfitFactor":
                    lambda x:
                    (
                        f"{x:.3f}"
                        if np.isfinite(x)
                        else "inf"
                    ),

                "Expectancy":
                    lambda x:
                    f"{x:.4f}",

                "TotalPnL":
                    lambda x:
                    f"{x:.2f}",
            },
        )
    )


# ============================================================
# PRINT EXIT BREAKDOWN
# ============================================================

def print_exit_breakdown(
    breakdown,
):

    print()
    print("=" * 90)
    print("MARKOV EXIT BREAKDOWN")
    print("=" * 90)

    if breakdown.empty:

        print(
            "No trades."
        )

        return

    display = (
        breakdown.copy()
    )

    display["WinRate"] = (
        display["WinRate"]
        * 100
    )

    print(
        display.to_string(
            index=False,
            formatters={
                "WinRate":
                    lambda x:
                    f"{x:.2f}%",

                "ProfitFactor":
                    lambda x:
                    (
                        f"{x:.3f}"
                        if np.isfinite(x)
                        else "inf"
                    ),

                "Expectancy":
                    lambda x:
                    f"{x:.4f}",

                "TotalPnL":
                    lambda x:
                    f"{x:.2f}",
            },
        )
    )


# ============================================================
# PRINT BLOCKED SIGNAL ANALYSIS
# ============================================================

def print_blocked_analysis(
    blocked_analysis,
):

    print()
    print("=" * 90)
    print("BLOCKED SIGNAL FORWARD RETURNS")
    print("=" * 90)

    if blocked_analysis.empty:

        print(
            "No blocked signals."
        )

        return

    print(
        f"Total blocked signals: "
        f"{len(blocked_analysis):,}"
    )

    for side in [
        "LONG",
        "SHORT",
    ]:

        subset = blocked_analysis[
            blocked_analysis["Side"]
            == side
        ]

        print()
        print(
            f"Blocked {side}: "
            f"{len(subset):,}"
        )

        for horizon in [
            1,
            3,
            6,
            12,
            24,
        ]:

            values = (
                subset[
                    f"ForwardReturn_{horizon}h"
                ]
                .dropna()
            )

            if values.empty:

                continue

            print(
                f"  {horizon:>2}h: "
                f"mean="
                f"{values.mean() * 100:.4f}% "
                f"| median="
                f"{values.median() * 100:.4f}% "
                f"| win="
                f"{(values > 0).mean() * 100:.2f}% "
                f"| N="
                f"{len(values)}"
            )


# ============================================================
# ANALYZE ONE SYMBOL
# ============================================================

def analyze_symbol(
    symbol,
):

    print()
    print("#" * 100)
    print(
        f"ANALYZING {symbol}"
    )
    print("#" * 100)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_mt5_h1(
        symbol
    )

    # --------------------------------------------------------
    # Bid / Ask
    # --------------------------------------------------------

    df = add_bid_ask(
        df,
        symbol,
    )

    # --------------------------------------------------------
    # Kalman
    # --------------------------------------------------------

    df = calculate_kalman_features(
        df
    )

    # --------------------------------------------------------
    # Markov
    # --------------------------------------------------------

    df = calculate_markov_features(
        df
    )

    # ========================================================
    # KALMAN ONLY
    # ========================================================

    print()
    print(
        "Running canonical-style Kalman-only backtest..."
    )

    (
        kalman_equity,
        kalman_trades,
        kalman_blocked,
    ) = run_backtest(
        df,
        symbol,
        use_markov=False,
    )

    # ========================================================
    # KALMAN + MARKOV
    # ========================================================

    print()
    print(
        "Running canonical-style Kalman + Markov backtest..."
    )

    (
        markov_equity,
        markov_trades,
        blocked,
    ) = run_backtest(
        df,
        symbol,
        use_markov=True,
    )

    # ========================================================
    # METRICS
    # ========================================================

    kalman_metrics = calculate_metrics(
        kalman_equity,
        kalman_trades,
    )

    markov_metrics = calculate_metrics(
        markov_equity,
        markov_trades,
    )

    # ========================================================
    # COMPARISON
    # ========================================================

    print_metrics_table(
        kalman_metrics,
        markov_metrics,
    )

    # ========================================================
    # REGIME BREAKDOWN
    # ========================================================

    kalman_regime = (
        calculate_regime_breakdown(
            kalman_trades
        )
    )

    markov_regime = (
        calculate_regime_breakdown(
            markov_trades
        )
    )

    print_regime_breakdown(
        "KALMAN-ONLY TRADES BY ENTRY REGIME",
        kalman_regime,
    )

    print_regime_breakdown(
        "MARKOV-ACTIVE TRADES BY ENTRY REGIME",
        markov_regime,
    )

    # ========================================================
    # EXIT BREAKDOWN
    # ========================================================

    markov_exits = (
        calculate_exit_breakdown(
            markov_trades
        )
    )

    print_exit_breakdown(
        markov_exits
    )

    # ========================================================
    # BLOCKED SIGNALS
    # ========================================================

    blocked_analysis = (
        analyze_blocked_signals(
            df,
            blocked,
        )
    )

    print_blocked_analysis(
        blocked_analysis
    )

    # ========================================================
    # SAVE
    # ========================================================

    symbol_dir = (
        RESULTS_DIR
        / symbol
    )

    symbol_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Trades
    # --------------------------------------------------------

    kalman_trades.to_csv(
        symbol_dir
        / "kalman_trades.csv",
        index=False,
    )

    markov_trades.to_csv(
        symbol_dir
        / "markov_trades.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Equity
    # --------------------------------------------------------

    kalman_equity.to_csv(
        symbol_dir
        / "kalman_equity.csv",
        index=False,
    )

    markov_equity.to_csv(
        symbol_dir
        / "markov_equity.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Blocked signals
    # --------------------------------------------------------

    blocked.to_csv(
        symbol_dir
        / "blocked_signals.csv",
        index=False,
    )

    blocked_analysis.to_csv(
        symbol_dir
        / "blocked_signal_forward_returns.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Breakdowns
    # --------------------------------------------------------

    kalman_regime.to_csv(
        symbol_dir
        / "kalman_regime_breakdown.csv",
        index=False,
    )

    markov_regime.to_csv(
        symbol_dir
        / "markov_regime_breakdown.csv",
        index=False,
    )

    markov_exits.to_csv(
        symbol_dir
        / "markov_exit_breakdown.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = {
        "Symbol":
            symbol,

        "StartDate":
            START_DATE,

        "EndDate":
            END_DATE,

        "KalmanFinalEquity":
            kalman_metrics[
                "FinalEquity"
            ],

        "KalmanReturn":
            kalman_metrics[
                "TotalReturn"
            ],

        "KalmanMaxDD":
            kalman_metrics[
                "MaxDrawdown"
            ],

        "KalmanSharpe":
            kalman_metrics[
                "Sharpe"
            ],

        "KalmanTrades":
            kalman_metrics[
                "Trades"
            ],

        "KalmanWinRate":
            kalman_metrics[
                "WinRate"
            ],

        "KalmanProfitFactor":
            kalman_metrics[
                "ProfitFactor"
            ],

        "KalmanExpectancy":
            kalman_metrics[
                "Expectancy"
            ],

        "MarkovFinalEquity":
            markov_metrics[
                "FinalEquity"
            ],

        "MarkovReturn":
            markov_metrics[
                "TotalReturn"
            ],

        "MarkovMaxDD":
            markov_metrics[
                "MaxDrawdown"
            ],

        "MarkovSharpe":
            markov_metrics[
                "Sharpe"
            ],

        "MarkovTrades":
            markov_metrics[
                "Trades"
            ],

        "MarkovWinRate":
            markov_metrics[
                "WinRate"
            ],

        "MarkovProfitFactor":
            markov_metrics[
                "ProfitFactor"
            ],

        "MarkovExpectancy":
            markov_metrics[
                "Expectancy"
            ],

        "BlockedSignals":
            len(blocked),
    }

    summary_df = pd.DataFrame(
        [summary]
    )

    summary_df.to_csv(
        symbol_dir
        / "summary.csv",
        index=False,
    )

    print()
    print("=" * 90)
    print(
        f"{symbol} RESULTS SAVED"
    )
    print("=" * 90)

    print(
        symbol_dir
    )

    return summary


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Analyze actual Kalman trades "
            "by causal Markov regime."
        )
    )

    parser.add_argument(
        "--symbol",
        default="XAUUSDm",
        help=(
            "Symbol to analyze. "
            "Use ALL for all instruments."
        ),
    )

    args = parser.parse_args()

    if (
        args.symbol.upper()
        == "ALL"
    ):

        symbols = (
            INSTRUMENTS
        )

    else:

        if (
            args.symbol
            not in INSTRUMENTS
        ):

            raise ValueError(
                f"Unknown symbol: "
                f"{args.symbol}"
            )

        symbols = [
            args.symbol
        ]

    # ========================================================
    # HEADER
    # ========================================================

    print()
    print("=" * 100)

    print(
        "ACTUAL KALMAN TRADE × MARKOV REGIME ANALYSIS"
    )

    print("=" * 100)

    print()

    print(
        "Research period:"
    )

    print(
        f"  Start = {START_DATE}"
    )

    print(
        f"  End   = {END_DATE}"
    )

    print()

    print(
        "Locked Kalman parameters:"
    )

    print(
        f"  PHI          = {PHI}"
    )

    print(
        f"  Q            = {Q}"
    )

    print(
        f"  R            = {R}"
    )

    print(
        f"  MEAN_WINDOW  = {MEAN_WINDOW}"
    )

    print(
        f"  Z_WINDOW     = {Z_WINDOW}"
    )

    print(
        f"  ENTRY_Z      = {ENTRY_Z}"
    )

    print(
        f"  EXIT_Z       = {EXIT_Z}"
    )

    print(
        f"  STOP_Z       = {STOP_Z}"
    )

    print(
        f"  MAX_HOLD     = "
        f"{MAX_HOLD_HOURS} trading hours"
    )

    print()

    print(
        "Markov parameters:"
    )

    print(
        f"  HMM_WINDOW   = {HMM_WINDOW}"
    )

    print(
        f"  MIN_OBS      = {HMM_MIN_OBSERVATIONS}"
    )

    print(
        f"  REFIT_EVERY  = {HMM_REFIT_EVERY}h"
    )

    print(
        f"  MR_THRESHOLD = {MR_THRESHOLD}"
    )

    print()

    print(
        "No optimization."
    )

    print(
        "No parameter changes."
    )

    print()

    # ========================================================
    # RUN
    # ========================================================

    summaries = []

    for symbol in symbols:

        try:

            summary = analyze_symbol(
                symbol
            )

            summaries.append(
                summary
            )

        except Exception as error:

            print()
            print(
                "=" * 90
            )

            print(
                f"ERROR: {symbol}"
            )

            print(
                repr(error)
            )

            print(
                "=" * 90
            )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    if summaries:

        summary_df = pd.DataFrame(
            summaries
        )

        summary_file = (
            RESULTS_DIR
            / "summary.csv"
        )

        summary_df.to_csv(
            summary_file,
            index=False,
        )

        print()
        print("=" * 100)

        print(
            "FINAL SUMMARY"
        )

        print("=" * 100)

        display_columns = [
            "Symbol",

            "KalmanFinalEquity",
            "KalmanReturn",
            "KalmanMaxDD",
            "KalmanTrades",
            "KalmanProfitFactor",

            "MarkovFinalEquity",
            "MarkovReturn",
            "MarkovMaxDD",
            "MarkovTrades",
            "MarkovProfitFactor",

            "BlockedSignals",
        ]

        display_columns = [
            c
            for c in display_columns
            if c in summary_df.columns
        ]

        print(
            summary_df[
                display_columns
            ].to_string(
                index=False
            )
        )

        print()
        print(
            f"Summary saved to:\n"
            f"{summary_file}"
        )

    print()
    print("=" * 100)

    print(
        "DONE"
    )

    print("=" * 100)


if __name__ == "__main__":

    main()