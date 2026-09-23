from pathlib import Path
import sys

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
# IMPORT EXISTING COMPONENTS
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
    / "walk_forward_directional"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# SYMBOL
# ============================================================

SYMBOL = "XAUUSDm"


# ============================================================
# WALK-FORWARD TEST WINDOWS
#
# Each calendar period is a TEST period.
#
# The model is processed chronologically.
#
# 2019 is included as an initial historical test period.
# From 2020 onward, every test period has prior history
# available to the causal Kalman/HMM calculations.
# ============================================================

TEST_PERIODS = {
    2019: (
        pd.Timestamp(
            "2019-01-01 00:00:00",
            tz="UTC",
        ),
        pd.Timestamp(
            "2019-12-31 23:00:00",
            tz="UTC",
        ),
    ),

    2020: (
        pd.Timestamp(
            "2020-01-01 00:00:00",
            tz="UTC",
        ),
        pd.Timestamp(
            "2020-12-31 23:00:00",
            tz="UTC",
        ),
    ),

    2021: (
        pd.Timestamp(
            "2021-01-01 00:00:00",
            tz="UTC",
        ),
        pd.Timestamp(
            "2021-12-31 23:00:00",
            tz="UTC",
        ),
    ),

    2022: (
        pd.Timestamp(
            "2022-01-01 00:00:00",
            tz="UTC",
        ),
        pd.Timestamp(
            "2022-12-31 23:00:00",
            tz="UTC",
        ),
    ),

    2023: (
        pd.Timestamp(
            "2023-01-01 00:00:00",
            tz="UTC",
        ),
        pd.Timestamp(
            "2023-12-31 23:00:00",
            tz="UTC",
        ),
    ),

    2024: (
        pd.Timestamp(
            "2024-01-01 00:00:00",
            tz="UTC",
        ),
        pd.Timestamp(
            "2024-12-31 23:00:00",
            tz="UTC",
        ),
    ),

    2025: (
        pd.Timestamp(
            "2025-01-01 00:00:00",
            tz="UTC",
        ),
        pd.Timestamp(
            "2025-07-20 23:00:00",
            tz="UTC",
        ),
    ),
}


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
# MARKOV PARAMETERS
# ============================================================

HMM_WINDOW = 1000
HMM_MIN_OBSERVATIONS = 300
HMM_REFIT_EVERY = 24

MR_THRESHOLD = 0.60


# ============================================================
# XAU POINT SIZE
# ============================================================

POINT_SIZE = 0.001


# ============================================================
# FOUR FIXED VARIANTS
# ============================================================

VARIANTS = {

    "BASELINE": {
        "long_enabled": True,
        "short_enabled": True,
        "short_requires_mr": False,
    },

    "LONG_ONLY": {
        "long_enabled": True,
        "short_enabled": False,
        "short_requires_mr": False,
    },

    "SHORT_ONLY": {
        "long_enabled": False,
        "short_enabled": True,
        "short_requires_mr": False,
    },

    "MARKOV_SHORT_FILTER": {
        "long_enabled": True,
        "short_enabled": True,
        "short_requires_mr": True,
    },
}


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    path = (
        DATA_DIR
        / f"{SYMBOL}_H1_2019_2026.csv"
    )

    print("=" * 100)
    print("LOADING DATA")
    print("=" * 100)

    print(
        f"Path: {path}"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"File not found:\n{path}"
        )

    df = pd.read_csv(path)

    print(
        f"Rows loaded: {len(df):,}"
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
            "Timestamp/Time column not found."
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
    # Required fields
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
    # Remove invalid rows
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

    print(
        f"Clean rows: {len(df):,}"
    )

    print(
        f"Start: {df.index.min()}"
    )

    print(
        f"End:   {df.index.max()}"
    )

    return df


# ============================================================
# BID / ASK RECONSTRUCTION
# ============================================================

def add_bid_ask(
    df,
):

    spread_price = (
        df["SpreadPoints"]
        * POINT_SIZE
    )

    spread_price = (
        spread_price
        .clip(lower=0)
        .fillna(0)
    )

    df["SpreadPrice"] = (
        spread_price
    )

    df["Bid_Open"] = (
        df["Open"]
        - spread_price / 2.0
    )

    df["Ask_Open"] = (
        df["Open"]
        + spread_price / 2.0
    )

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
#
# These are calculated once over the chronological history.
#
# walk_forward_kalman() is causal:
# each state only uses information available up to that bar.
# ============================================================

def add_kalman_features(
    df,
):

    print()
    print(
        "=" * 80
    )

    print(
        "CALCULATING CAUSAL KALMAN FEATURES"
    )

    print(
        "=" * 80
    )

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

    df["Z"] = calculate_zscore(
        df["Residual"],
        window=Z_WINDOW,
    )

    print(
        f"Valid Z observations: "
        f"{df['Z'].notna().sum():,}"
    )

    return df


# ============================================================
# MARKOV FEATURES
#
# add_markov_regime() performs causal rolling HMM fitting.
# ============================================================

def add_markov_features(
    df,
):

    print()
    print(
        "=" * 80
    )

    print(
        "CALCULATING CAUSAL MARKOV / HMM"
    )

    print(
        "=" * 80
    )

    df = add_markov_regime(
        df,
        hmm_window=HMM_WINDOW,
        min_observations=HMM_MIN_OBSERVATIONS,
        refit_every=HMM_REFIT_EVERY,
    )

    valid = (
        df["P_MeanReverting"]
        .notna()
    )

    print(
        f"Valid HMM observations: "
        f"{valid.sum():,}"
    )

    if valid.any():

        print(
            f"Mean P(MR): "
            f"{df.loc[valid, 'P_MeanReverting'].mean():.4f}"
        )

        print(
            f"Mean P(Trend): "
            f"{df.loc[valid, 'P_Trending'].mean():.4f}"
        )

    return df


# ============================================================
# REGIME
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

    return max(
        min(
            units_from_risk,
            units_from_notional,
        ),
        0.0,
    )


# ============================================================
# BACKTEST ONE VARIANT
#
# IMPORTANT:
# This runs CONTINUOUSLY across the complete historical
# dataset.
#
# We do NOT reset capital at each year.
#
# The yearly numbers are extracted from the same chronological
# equity curve.
# ============================================================

def run_variant(
    df,
    variant_name,
    config,
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

    # ========================================================
    # FULL CHRONOLOGICAL LOOP
    # ========================================================

    for i in range(
        len(df)
    ):

        timestamp = (
            df.index[i]
        )

        row = df.iloc[i]

        z = row["Z"]

        bid_close = (
            row["Bid_Close"]
        )

        ask_close = (
            row["Ask_Close"]
        )

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

        # ====================================================
        # MARK TO MARKET
        # ====================================================

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

            equity = (
                cash_equity
            )

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

                "Z":
                    z,

                "P_MeanReverting":
                    p_mr,

                "P_Trending":
                    p_trend,

                "Regime":
                    regime,

                "Variant":
                    variant_name,
            }
        )

        # ----------------------------------------------------
        # Need next bar
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

            exit_reason = None

            if pd.notna(z):

                if abs(z) <= EXIT_Z:

                    exit_reason = (
                        "Z_EXIT"
                    )

                elif abs(z) >= STOP_Z:

                    exit_reason = (
                        "Z_STOP"
                    )

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
            # EXIT NEXT BAR
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

                trades.append(
                    {
                        "Variant":
                            variant_name,

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

                        "Return":
                            (
                                pnl
                                / entry_equity
                            ),

                        "PnL":
                            pnl,

                        "ExitReason":
                            exit_reason,
                    }
                )

                # ------------------------------------------------
                # Reset
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
                # No same-bar re-entry
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
            # LONG
            # ------------------------------------------------

            if (
                z <= -ENTRY_Z
                and
                config["long_enabled"]
            ):

                desired_side = "LONG"

            # ------------------------------------------------
            # SHORT
            # ------------------------------------------------

            elif (
                z >= ENTRY_Z
                and
                config["short_enabled"]
            ):

                desired_side = "SHORT"

            if desired_side is None:

                continue

            # ------------------------------------------------
            # Markov short filter
            # ------------------------------------------------

            if (
                desired_side == "SHORT"
                and
                config[
                    "short_requires_mr"
                ]
            ):

                if pd.isna(p_mr):

                    continue

                if (
                    p_mr
                    < MR_THRESHOLD
                ):

                    continue

            # ------------------------------------------------
            # Entry next bar
            # ------------------------------------------------

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
            # Store entry
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
    # FORCE CLOSE AT END
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

        trades.append(
            {
                "Variant":
                    variant_name,

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
                        last_row.get(
                            "P_MeanReverting",
                            np.nan,
                        )
                    ),

                "ExitPMR":
                    last_row.get(
                        "P_MeanReverting",
                        np.nan,
                    ),

                "ExitPTrend":
                    last_row.get(
                        "P_Trending",
                        np.nan,
                    ),

                "TradingHoursHeld":
                    trading_hours_held,

                "CalendarHoursHeld":
                    calendar_hours_held,

                "Return":
                    (
                        pnl
                        / entry_equity
                    ),

                "PnL":
                    pnl,

                "ExitReason":
                    "END_OF_DATA",
            }
        )

    return (
        pd.DataFrame(equity_curve),
        pd.DataFrame(trades),
    )


# ============================================================
# EQUITY METRICS
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

            "Return":
                0.0,

            "MaxDD":
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

            "TotalPnL":
                0.0,
        }

    equity = (
        equity_df["Equity"]
        .astype(float)
    )

    final_equity = (
        equity.iloc[-1]
    )

    total_return = (
        final_equity
        / INITIAL_CAPITAL
        - 1.0
    )

    # --------------------------------------------------------
    # Drawdown
    # --------------------------------------------------------

    running_max = (
        equity
        .cummax()
    )

    drawdown = (
        equity
        / running_max
        - 1.0
    )

    max_dd = (
        drawdown.min()
    )

    # --------------------------------------------------------
    # Sharpe
    # --------------------------------------------------------

    returns = (
        equity
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

    # --------------------------------------------------------
    # Trades
    # --------------------------------------------------------

    if trades_df.empty:

        return {
            "InitialEquity":
                INITIAL_CAPITAL,

            "FinalEquity":
                final_equity,

            "Return":
                total_return,

            "MaxDD":
                max_dd,

            "Sharpe":
                sharpe,

            "Trades":
                0,

            "WinRate":
                np.nan,

            "ProfitFactor":
                np.nan,

            "Expectancy":
                0.0,

            "TotalPnL":
                0.0,
        }

    pnl = (
        trades_df["PnL"]
        .astype(float)
    )

    wins = (
        pnl > 0
    )

    losses = (
        pnl < 0
    )

    gross_profit = (
        pnl[wins].sum()
    )

    gross_loss = (
        pnl[losses].sum()
    )

    if gross_loss < 0:

        profit_factor = (
            gross_profit
            / abs(gross_loss)
        )

    else:

        profit_factor = np.inf

    return {
        "InitialEquity":
            INITIAL_CAPITAL,

        "FinalEquity":
            final_equity,

        "Return":
            total_return,

        "MaxDD":
            max_dd,

        "Sharpe":
            sharpe,

        "Trades":
            len(trades_df),

        "WinRate":
            wins.mean(),

        "ProfitFactor":
            profit_factor,

        "Expectancy":
            pnl.mean(),

        "TotalPnL":
            pnl.sum(),
    }


# ============================================================
# YEARLY PERFORMANCE FROM CONTINUOUS EQUITY CURVE
# ============================================================

def calculate_period_metrics(
    equity_df,
    trades_df,
    year,
    start,
    end,
):

    period_equity = equity_df.loc[
        (equity_df["Timestamp"] >= start)
        &
        (equity_df["Timestamp"] <= end)
    ].copy()

    if period_equity.empty:

        return None

    # --------------------------------------------------------
    # Important:
    #
    # Return is calculated relative to equity at the beginning
    # of the test period, NOT from a reset $10,000.
    # --------------------------------------------------------

    start_equity = (
        period_equity["Equity"]
        .iloc[0]
    )

    end_equity = (
        period_equity["Equity"]
        .iloc[-1]
    )

    period_return = (
        end_equity
        / start_equity
        - 1.0
    )

    # --------------------------------------------------------
    # Period drawdown
    # --------------------------------------------------------

    running_max = (
        period_equity["Equity"]
        .cummax()
    )

    period_dd = (
        period_equity["Equity"]
        / running_max
        - 1.0
    )

    max_dd = (
        period_dd.min()
    )

    # --------------------------------------------------------
    # Trades whose exits occur in this period
    # --------------------------------------------------------

    period_trades = trades_df.loc[
        (trades_df["ExitTime"] >= start)
        &
        (trades_df["ExitTime"] <= end)
    ].copy()

    if period_trades.empty:

        return {
            "Year":
                year,

            "StartEquity":
                start_equity,

            "EndEquity":
                end_equity,

            "Return":
                period_return,

            "MaxDD":
                max_dd,

            "Trades":
                0,

            "WinRate":
                np.nan,

            "ProfitFactor":
                np.nan,

            "Expectancy":
                0.0,
        }

    pnl = (
        period_trades["PnL"]
        .astype(float)
    )

    wins = (
        pnl > 0
    )

    losses = (
        pnl < 0
    )

    gross_profit = (
        pnl[wins].sum()
    )

    gross_loss = (
        pnl[losses].sum()
    )

    if gross_loss < 0:

        pf = (
            gross_profit
            / abs(gross_loss)
        )

    else:

        pf = np.inf

    return {
        "Year":
            year,

        "StartEquity":
            start_equity,

        "EndEquity":
            end_equity,

        "Return":
            period_return,

        "MaxDD":
            max_dd,

        "Trades":
            len(period_trades),

        "WinRate":
            wins.mean(),

        "ProfitFactor":
            pf,

        "Expectancy":
            pnl.mean(),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 110)

    print(
        "WALK-FORWARD DIRECTIONAL VALIDATION"
    )

    print("=" * 110)

    print()

    print(
        f"Symbol: {SYMBOL}"
    )

    print(
        "Capital is continuous across the complete history."
    )

    print(
        "No yearly capital reset."
    )

    print(
        "No parameter optimization."
    )

    print(
        "No parameter changes between periods."
    )

    print()

    print(
        "Locked Kalman:"
    )

    print(
        f"  PHI={PHI}"
        f"  Q={Q}"
        f"  R={R}"
    )

    print(
        f"  MeanWindow={MEAN_WINDOW}"
        f"  ZWindow={Z_WINDOW}"
    )

    print(
        f"  Entry={ENTRY_Z}"
        f"  Exit={EXIT_Z}"
        f"  Stop={STOP_Z}"
        f"  MaxHold={MAX_HOLD_HOURS}h"
    )

    print()

    print(
        "Locked Markov:"
    )

    print(
        f"  HMMWindow={HMM_WINDOW}"
        f"  MinObs={HMM_MIN_OBSERVATIONS}"
        f"  Refit={HMM_REFIT_EVERY}h"
        f"  MRThreshold={MR_THRESHOLD}"
    )

    # ========================================================
    # LOAD
    # ========================================================

    df = load_data()

    # ========================================================
    # BID / ASK
    # ========================================================

    df = add_bid_ask(
        df
    )

    # ========================================================
    # CAUSAL FEATURES
    # ========================================================

    df = add_kalman_features(
        df
    )

    df = add_markov_features(
        df
    )

    # ========================================================
    # RUN EACH VARIANT
    # ========================================================

    all_yearly_results = []

    for variant_name, config in (
        VARIANTS.items()
    ):

        print()
        print("#" * 110)

        print(
            f"RUNNING CONTINUOUS VARIANT: "
            f"{variant_name}"
        )

        print(
            "#" * 110
        )

        (
            equity_df,
            trades_df,
        ) = run_variant(
            df,
            variant_name,
            config,
        )

        # ----------------------------------------------------
        # Full-history metrics
        # ----------------------------------------------------

        full_metrics = calculate_metrics(
            equity_df,
            trades_df,
        )

        print()
        print(
            "FULL HISTORY"
        )

        print(
            f"Final equity: "
            f"${full_metrics['FinalEquity']:,.2f}"
        )

        print(
            f"Return: "
            f"{full_metrics['Return'] * 100:.3f}%"
        )

        print(
            f"Max DD: "
            f"{full_metrics['MaxDD'] * 100:.3f}%"
        )

        print(
            f"Sharpe: "
            f"{full_metrics['Sharpe']:.3f}"
        )

        print(
            f"Trades: "
            f"{full_metrics['Trades']:,}"
        )

        print(
            f"Win rate: "
            f"{full_metrics['WinRate'] * 100:.2f}%"
        )

        print(
            f"PF: "
            f"{full_metrics['ProfitFactor']:.3f}"
        )

        # ----------------------------------------------------
        # Save full curve
        # ----------------------------------------------------

        equity_path = (
            RESULTS_DIR
            / f"{variant_name}_equity.csv"
        )

        trades_path = (
            RESULTS_DIR
            / f"{variant_name}_trades.csv"
        )

        equity_df.to_csv(
            equity_path,
            index=False,
        )

        trades_df.to_csv(
            trades_path,
            index=False,
        )

        # ----------------------------------------------------
        # Extract yearly performance
        # ----------------------------------------------------

        for year, (
            start,
            end,
        ) in TEST_PERIODS.items():

            result = calculate_period_metrics(
                equity_df,
                trades_df,
                year,
                start,
                end,
            )

            if result is None:

                continue

            result["Variant"] = (
                variant_name
            )

            all_yearly_results.append(
                result
            )

    # ========================================================
    # RESULTS DATAFRAME
    # ========================================================

    results_df = pd.DataFrame(
        all_yearly_results
    )

    results_df = results_df[
        [
            "Year",
            "Variant",
            "StartEquity",
            "EndEquity",
            "Return",
            "MaxDD",
            "Trades",
            "WinRate",
            "ProfitFactor",
            "Expectancy",
        ]
    ]

    results_df = results_df.sort_values(
        [
            "Year",
            "Variant",
        ]
    )

    results_path = (
        RESULTS_DIR
        / "walk_forward_yearly_results.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    # ========================================================
    # RETURN MATRIX
    # ========================================================

    return_matrix = (
        results_df
        .pivot(
            index="Year",
            columns="Variant",
            values="Return",
        )
        * 100
    )

    return_matrix_path = (
        RESULTS_DIR
        / "walk_forward_return_matrix.csv"
    )

    return_matrix.to_csv(
        return_matrix_path
    )

    # ========================================================
    # PRINT YEARLY RETURNS
    # ========================================================

    print()
    print("=" * 120)

    print(
        "WALK-FORWARD YEARLY RETURNS"
    )

    print("=" * 120)

    print(
        return_matrix.to_string(
            float_format=lambda x:
            f"{x:.3f}%"
        )
    )

    # ========================================================
    # CONTINUOUS FULL-HISTORY SUMMARY
    # ========================================================

    print()
    print("=" * 120)

    print(
        "CONTINUOUS FULL-HISTORY SUMMARY"
    )

    print("=" * 120)

    for variant_name in VARIANTS:

        subset = results_df[
            results_df["Variant"]
            == variant_name
        ]

        if subset.empty:

            continue

        returns = (
            subset["Return"]
            .astype(float)
        )

        profitable = (
            returns > 0
        ).sum()

        losing = (
            returns < 0
        ).sum()

        compounded = (
            np.prod(
                1.0 + returns
            )
            - 1.0
        )

        print()
        print(
            f"{variant_name}"
        )

        print(
            f"  Profitable periods: "
            f"{profitable}/{len(returns)}"
        )

        print(
            f"  Losing periods: "
            f"{losing}/{len(returns)}"
        )

        print(
            f"  Mean period return: "
            f"{returns.mean() * 100:.3f}%"
        )

        print(
            f"  Median period return: "
            f"{returns.median() * 100:.3f}%"
        )

        print(
            f"  Compounded period return: "
            f"{compounded * 100:.3f}%"
        )

        print(
            f"  Best period: "
            f"{returns.max() * 100:.3f}%"
        )

        print(
            f"  Worst period: "
            f"{returns.min() * 100:.3f}%"
        )

        print(
            f"  Worst period DD: "
            f"{subset['MaxDD'].min() * 100:.3f}%"
        )

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    summary_records = []

    for variant_name in VARIANTS:

        subset = results_df[
            results_df["Variant"]
            == variant_name
        ]

        if subset.empty:

            continue

        returns = (
            subset["Return"]
            .astype(float)
        )

        summary_records.append(
            {
                "Variant":
                    variant_name,

                "Periods":
                    len(subset),

                "ProfitablePeriods":
                    int(
                        (returns > 0).sum()
                    ),

                "LosingPeriods":
                    int(
                        (returns < 0).sum()
                    ),

                "AverageReturn":
                    returns.mean(),

                "MedianReturn":
                    returns.median(),

                "CompoundedPeriodReturn":
                    (
                        np.prod(
                            1.0 + returns
                        )
                        - 1.0
                    ),

                "BestPeriod":
                    returns.max(),

                "WorstPeriod":
                    returns.min(),

                "WorstPeriodDD":
                    subset["MaxDD"].min(),

                "TotalTrades":
                    int(
                        subset["Trades"].sum()
                    ),
            }
        )

    summary_df = pd.DataFrame(
        summary_records
    )

    summary_path = (
        RESULTS_DIR
        / "walk_forward_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    # ========================================================
    # FILES
    # ========================================================

    print()
    print("=" * 110)

    print(
        "FILES SAVED"
    )

    print("=" * 110)

    print(
        results_path
    )

    print(
        return_matrix_path
    )

    print(
        summary_path
    )

    print()

    print(
        RESULTS_DIR
    )

    print()
    print("=" * 110)

    print(
        "DONE"
    )

    print("=" * 110)


if __name__ == "__main__":

    main()