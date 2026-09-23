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
# IMPORT EXISTING RESEARCH COMPONENTS
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
    / "directional_variants"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# SYMBOLS
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
# VALIDATION PERIOD
#
# EXACTLY THE SAME PERIOD AS THE CANONICAL XAU BACKTEST
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
# MARKOV PARAMETERS
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
# LOAD DATA
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
            f"MT5 H1 file not found:\n{path}"
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
            "Could not find Timestamp or Time column."
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

    # --------------------------------------------------------
    # Validation period
    # --------------------------------------------------------

    df = df.loc[
        (df.index >= START_DATE)
        &
        (df.index <= END_DATE)
    ].copy()

    if df.empty:

        raise ValueError(
            f"{symbol}: no data in validation period."
        )

    print(
        f"Bars in period: {len(df):,}"
    )

    print(
        f"Start: {df.index.min()}"
    )

    print(
        f"End:   {df.index.max()}"
    )

    return df


# ============================================================
# RECONSTRUCT APPROXIMATE BID / ASK
# ============================================================

def add_bid_ask(
    df,
    symbol,
):

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
# ============================================================

def add_kalman_features(df):

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

    valid_z = df["Z"].notna()

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
# MARKOV FEATURES
# ============================================================

def add_markov_features(df):

    print()
    print("=" * 80)
    print("CALCULATING CAUSAL MARKOV / HMM")
    print("=" * 80)

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

        if "MR_Autocorrelation" in df.columns:

            print(
                f"MR autocorrelation: "
                f"{df.loc[valid, 'MR_Autocorrelation'].mean():.5f}"
            )

        if "Trend_Autocorrelation" in df.columns:

            print(
                f"Trend autocorrelation: "
                f"{df.loc[valid, 'Trend_Autocorrelation'].mean():.5f}"
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
# VARIANT DEFINITIONS
# ============================================================

VARIANTS = {

    # --------------------------------------------------------
    # A
    # --------------------------------------------------------
    "BASELINE": {
        "long_enabled": True,
        "short_enabled": True,
        "short_requires_mr": False,
        "long_requires_mr": False,
    },

    # --------------------------------------------------------
    # B
    # --------------------------------------------------------
    "LONG_ONLY": {
        "long_enabled": True,
        "short_enabled": False,
        "short_requires_mr": False,
        "long_requires_mr": False,
    },

    # --------------------------------------------------------
    # C
    # --------------------------------------------------------
    "SHORT_ONLY": {
        "long_enabled": False,
        "short_enabled": True,
        "short_requires_mr": False,
        "long_requires_mr": False,
    },

    # --------------------------------------------------------
    # D
    #
    # LONG remains unrestricted.
    #
    # SHORT requires P(MR) >= 0.60.
    # --------------------------------------------------------
    "MARKOV_SHORT_FILTER": {
        "long_enabled": True,
        "short_enabled": True,
        "short_requires_mr": True,
        "long_requires_mr": False,
    },
}


# ============================================================
# BACKTEST ONE VARIANT
# ============================================================

def run_variant(
    df,
    symbol,
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

    blocked_signals = []

    # ========================================================
    # EVENT LOOP
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
        # Need next bar for execution
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
        # EXISTING POSITION
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
            # EXIT ON NEXT BAR
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

                        "HoursHeld":
                            trading_hours_held,

                        "Return":
                            trade_return,

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
                # IMPORTANT:
                #
                # Same behavior as canonical engine:
                # no same-bar re-entry.
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

            # =================================================
            # MARKOV SHORT FILTER
            # =================================================

            if (
                desired_side == "SHORT"
                and
                config[
                    "short_requires_mr"
                ]
            ):

                if pd.isna(p_mr):

                    blocked_signals.append(
                        {
                            "Symbol":
                                symbol,

                            "Variant":
                                variant_name,

                            "Time":
                                timestamp,

                            "Side":
                                desired_side,

                            "Z":
                                z,

                            "PMR":
                                p_mr,

                            "PTrend":
                                p_trend,

                            "Reason":
                                "NO_HMM",
                        }
                    )

                    continue

                if (
                    p_mr
                    < MR_THRESHOLD
                ):

                    blocked_signals.append(
                        {
                            "Symbol":
                                symbol,

                            "Variant":
                                variant_name,

                            "Time":
                                timestamp,

                            "Side":
                                desired_side,

                            "Z":
                                z,

                            "PMR":
                                p_mr,

                            "PTrend":
                                p_trend,

                            "Reason":
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
            # Sizing
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
    # FORCE CLOSE
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

        trades.append(
            {
                "Symbol":
                    symbol,

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

                "HoursHeld":
                    trading_hours_held,

                "Return":
                    trade_return,

                "PnL":
                    pnl,

                "ExitReason":
                    "END_OF_DATA",
            }
        )

    return (
        pd.DataFrame(equity_curve),
        pd.DataFrame(trades),
        pd.DataFrame(blocked_signals),
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
    # Trade metrics
    # --------------------------------------------------------

    trades_count = (
        len(trades_df)
    )

    if trades_count > 0:

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

        win_rate = (
            wins.mean()
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

        expectancy = (
            pnl.mean()
        )

        total_pnl = (
            pnl.sum()
        )

    else:

        win_rate = np.nan

        profit_factor = np.nan

        expectancy = 0.0

        total_pnl = 0.0

    return {
        "FinalEquity":
            final_equity,

        "Return":
            total_return,

        "MaxDD":
            max_dd,

        "Sharpe":
            sharpe,

        "Trades":
            trades_count,

        "WinRate":
            win_rate,

        "ProfitFactor":
            profit_factor,

        "Expectancy":
            expectancy,

        "TotalPnL":
            total_pnl,
    }


# ============================================================
# SIDE BREAKDOWN
# ============================================================

def side_breakdown(
    trades_df,
):

    records = []

    if trades_df.empty:

        return pd.DataFrame()

    for side in [
        "LONG",
        "SHORT",
    ]:

        subset = trades_df[
            trades_df["Side"]
            == side
        ]

        if subset.empty:

            continue

        pnl = (
            subset["PnL"]
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

        records.append(
            {
                "Side":
                    side,

                "Trades":
                    len(subset),

                "WinRate":
                    wins.mean(),

                "ProfitFactor":
                    pf,

                "Expectancy":
                    pnl.mean(),

                "TotalPnL":
                    pnl.sum(),
            }
        )

    return pd.DataFrame(
        records
    )


# ============================================================
# PRINT VARIANT RESULT
# ============================================================

def print_variant_result(
    name,
    metrics,
):

    pf = metrics[
        "ProfitFactor"
    ]

    pf_text = (
        "inf"
        if np.isinf(pf)
        else f"{pf:.3f}"
    )

    print(
        f"{name:<25}"
        f"{metrics['FinalEquity']:>14,.2f}"
        f"{metrics['Return'] * 100:>12.3f}%"
        f"{metrics['MaxDD'] * 100:>12.3f}%"
        f"{metrics['Sharpe']:>10.3f}"
        f"{metrics['Trades']:>9,}"
        f"{metrics['WinRate'] * 100:>11.2f}%"
        f"{pf_text:>10}"
        f"{metrics['Expectancy']:>12.4f}"
    )


# ============================================================
# ANALYZE SYMBOL
# ============================================================

def analyze_symbol(
    symbol,
):

    print()
    print("#" * 100)
    print(
        f"DIRECTIONAL VARIANTS: {symbol}"
    )
    print("#" * 100)

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    df = load_mt5_h1(
        symbol
    )

    df = add_bid_ask(
        df,
        symbol,
    )

    df = add_kalman_features(
        df
    )

    df = add_markov_features(
        df
    )

    # --------------------------------------------------------
    # Run all variants
    # --------------------------------------------------------

    all_results = []

    symbol_dir = (
        RESULTS_DIR
        / symbol
    )

    symbol_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for variant_name, config in (
        VARIANTS.items()
    ):

        print()
        print(
            "-" * 80
        )

        print(
            f"RUNNING VARIANT: "
            f"{variant_name}"
        )

        print(
            "-" * 80
        )

        (
            equity_df,
            trades_df,
            blocked_df,
        ) = run_variant(
            df,
            symbol,
            variant_name,
            config,
        )

        metrics = calculate_metrics(
            equity_df,
            trades_df,
        )

        metrics["Symbol"] = (
            symbol
        )

        metrics["Variant"] = (
            variant_name
        )

        metrics["BlockedSignals"] = (
            len(blocked_df)
        )

        all_results.append(
            metrics
        )

        # ----------------------------------------------------
        # Save variant outputs
        # ----------------------------------------------------

        equity_df.to_csv(
            symbol_dir
            / f"{variant_name}_equity.csv",
            index=False,
        )

        trades_df.to_csv(
            symbol_dir
            / f"{variant_name}_trades.csv",
            index=False,
        )

        blocked_df.to_csv(
            symbol_dir
            / f"{variant_name}_blocked.csv",
            index=False,
        )

        # ----------------------------------------------------
        # Side breakdown
        # ----------------------------------------------------

        side_df = side_breakdown(
            trades_df
        )

        side_df.to_csv(
            symbol_dir
            / f"{variant_name}_side_breakdown.csv",
            index=False,
        )

        print()
        print(
            f"{variant_name}"
        )

        print(
            f"  Final equity: "
            f"${metrics['FinalEquity']:,.2f}"
        )

        print(
            f"  Return: "
            f"{metrics['Return'] * 100:.3f}%"
        )

        print(
            f"  Max DD: "
            f"{metrics['MaxDD'] * 100:.3f}%"
        )

        print(
            f"  Sharpe: "
            f"{metrics['Sharpe']:.3f}"
        )

        print(
            f"  Trades: "
            f"{metrics['Trades']:,}"
        )

        print(
            f"  Win rate: "
            f"{metrics['WinRate'] * 100:.2f}%"
        )

        print(
            f"  Profit factor: "
            f"{metrics['ProfitFactor']:.3f}"
        )

        print(
            f"  Expectancy: "
            f"${metrics['Expectancy']:.4f}"
        )

        print(
            f"  Blocked signals: "
            f"{len(blocked_df):,}"
        )

        if not side_df.empty:

            print()
            print(
                "  Side breakdown:"
            )

            print(
                side_df.to_string(
                    index=False
                )
            )

    # ========================================================
    # FINAL COMPARISON
    # ========================================================

    summary_df = pd.DataFrame(
        all_results
    )

    summary_df.to_csv(
        symbol_dir
        / "summary.csv",
        index=False,
    )

    print()
    print("=" * 115)
    print(
        f"{symbol} — DIRECTIONAL VARIANT COMPARISON"
    )
    print("=" * 115)

    print()

    print(
        f"{'Variant':<25}"
        f"{'Equity':>14}"
        f"{'Return':>12}"
        f"{'MaxDD':>12}"
        f"{'Sharpe':>10}"
        f"{'Trades':>9}"
        f"{'WinRate':>11}"
        f"{'PF':>10}"
        f"{'Expectancy':>12}"
    )

    print("-" * 115)

    for _, row in summary_df.iterrows():

        print_variant_result(
            row["Variant"],
            row,
        )

    print()
    print(
        f"Results saved to:\n"
        f"{symbol_dir}"
    )

    return summary_df


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Test directional asymmetry "
            "in the canonical Kalman strategy."
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
    print("=" * 115)

    print(
        "DIRECTIONAL ASYMMETRY EXPERIMENT"
    )

    print("=" * 115)

    print()

    print(
        f"Period:"
    )

    print(
        f"  {START_DATE}"
    )

    print(
        f"  → {END_DATE}"
    )

    print()

    print(
        "Locked Kalman:"
    )

    print(
        f"  PHI={PHI}"
        f"  Q={Q}"
        f"  R={R}"
        f"  Mean={MEAN_WINDOW}"
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
        "Variants:"
    )

    print(
        "  A = BASELINE"
    )

    print(
        "  B = LONG_ONLY"
    )

    print(
        "  C = SHORT_ONLY"
    )

    print(
        "  D = MARKOV_SHORT_FILTER"
    )

    print()

    print(
        "No optimization."
    )

    print(
        "No parameter search."
    )

    print(
        "No GARCH."
    )

    print()

    # ========================================================
    # RUN
    # ========================================================

    summaries = []

    for symbol in symbols:

        try:

            result = analyze_symbol(
                symbol
            )

            summaries.append(
                result
            )

        except Exception as error:

            print()
            print(
                "=" * 80
            )

            print(
                f"ERROR: {symbol}"
            )

            print(
                repr(error)
            )

            print(
                "=" * 80
            )

    # ========================================================
    # COMBINE RESULTS
    # ========================================================

    if summaries:

        combined = pd.concat(
            summaries,
            ignore_index=True,
        )

        combined_file = (
            RESULTS_DIR
            / "all_symbols_summary.csv"
        )

        combined.to_csv(
            combined_file,
            index=False,
        )

        print()
        print("=" * 115)

        print(
            "ALL RESULTS"
        )

        print("=" * 115)

        print(
            combined[
                [
                    "Symbol",
                    "Variant",
                    "FinalEquity",
                    "Return",
                    "MaxDD",
                    "Sharpe",
                    "Trades",
                    "WinRate",
                    "ProfitFactor",
                    "Expectancy",
                    "BlockedSignals",
                ]
            ].to_string(
                index=False
            )
        )

        print()
        print(
            f"Combined results saved to:\n"
            f"{combined_file}"
        )

    print()
    print("=" * 115)

    print(
        "DONE"
    )

    print("=" * 115)


if __name__ == "__main__":

    main()