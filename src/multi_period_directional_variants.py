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

DATA_DIR = (
    ROOT
    / "data"
    / "mt5_h1"
)

RESULTS_DIR = (
    ROOT
    / "results"
    / "multi_period_directional"
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
# PERIODS
#
# 2025 deliberately ends on July 20 because that is the
# canonical validation period used in the previous experiment.
# ============================================================

PERIODS = {
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
# LOAD FULL DATA
# ============================================================

def load_data():

    path = (
        DATA_DIR
        / f"{SYMBOL}_H1_2019_2026.csv"
    )

    print("=" * 100)
    print(
        f"LOADING {SYMBOL}"
    )
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
        f"Rows: {len(df):,}"
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
        f"Data start: {df.index.min()}"
    )

    print(
        f"Data end:   {df.index.max()}"
    )

    return df


# ============================================================
# ADD APPROXIMATE BID / ASK
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
# ADD KALMAN FEATURES
# ============================================================

def add_kalman_features(
    df,
):

    print()
    print(
        "Calculating Kalman features..."
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
        f"Valid Z: "
        f"{df['Z'].notna().sum():,}"
    )

    return df


# ============================================================
# ADD MARKOV FEATURES
# ============================================================

def add_markov_features(
    df,
):

    print(
        "Calculating causal Markov/HMM..."
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
# POSITION SIZE
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
# BACKTEST ONE VARIANT ON ONE PERIOD
# ============================================================

def run_variant(
    df,
    period_start,
    period_end,
    variant_name,
    config,
):

    # --------------------------------------------------------
    # Slice period
    # --------------------------------------------------------

    period_df = df.loc[
        (df.index >= period_start)
        &
        (df.index <= period_end)
    ].copy()

    if period_df.empty:

        return (
            pd.DataFrame(),
            pd.DataFrame(),
        )

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

    # ========================================================
    # EVENT LOOP
    # ========================================================

    for i in range(
        len(period_df)
    ):

        timestamp = (
            period_df.index[i]
        )

        row = period_df.iloc[i]

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

        # ----------------------------------------------------
        # MARK TO MARKET
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

        if i >= len(period_df) - 1:

            continue

        next_row = period_df.iloc[
            i + 1
        ]

        next_timestamp = (
            period_df.index[i + 1]
        )

        next_bid_open = (
            next_row["Bid_Open"]
        )

        next_ask_open = (
            next_row["Ask_Open"]
        )

        # ====================================================
        # MANAGE POSITION
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
            # Execute exit next bar
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
                        "PeriodStart":
                            period_start,

                        "PeriodEnd":
                            period_end,

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
                            regime,

                        "EntryPMR":
                            p_mr,

                        "EntryPTrend":
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

                # ------------------------------------------------
                # IMPORTANT:
                # no same-bar re-entry
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
            # Size
            # ------------------------------------------------

            units = calculate_units(
                equity,
                entry_price,
            )

            if units <= 0:

                position = 0

                continue

            # ------------------------------------------------
            # Store
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

    # ========================================================
    # FORCE CLOSE
    # ========================================================

    if position != 0:

        last_row = period_df.iloc[-1]

        last_timestamp = (
            period_df.index[-1]
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

        trades.append(
            {
                "PeriodStart":
                    period_start,

                "PeriodEnd":
                    period_end,

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
                    classify_regime(
                        last_row.get(
                            "P_MeanReverting",
                            np.nan,
                        )
                    ),

                "EntryPMR":
                    last_row.get(
                        "P_MeanReverting",
                        np.nan,
                    ),

                "EntryPTrend":
                    last_row.get(
                        "P_Trending",
                        np.nan,
                    ),

                "TradingHoursHeld":
                    (
                        len(period_df)
                        - 1
                        - entry_index
                    ),

                "CalendarHoursHeld":
                    (
                        (
                            last_timestamp
                            - entry_time
                        )
                        .total_seconds()
                        / 3600.0
                    ),

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
        pd.DataFrame(
            equity_curve
        ),
        pd.DataFrame(
            trades
        ),
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

    if trades_df.empty:

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
# YEARLY RESULT
# ============================================================

def run_period(
    df,
    year,
    period_start,
    period_end,
):

    print()
    print("#" * 100)
    print(
        f"PERIOD {year}"
    )
    print("#" * 100)

    period_data = df.loc[
        (df.index >= period_start)
        &
        (df.index <= period_end)
    ]

    print(
        f"Bars: {len(period_data):,}"
    )

    if period_data.empty:

        return []

    results = []

    period_dir = (
        RESULTS_DIR
        / str(year)
    )

    period_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for variant_name, config in (
        VARIANTS.items()
    ):

        print()
        print(
            f"Running {variant_name}..."
        )

        (
            equity_df,
            trades_df,
        ) = run_variant(
            df,
            period_start,
            period_end,
            variant_name,
            config,
        )

        metrics = calculate_metrics(
            equity_df,
            trades_df,
        )

        result = {
            "Year":
                year,

            "PeriodStart":
                period_start,

            "PeriodEnd":
                period_end,

            "Variant":
                variant_name,

            **metrics,
        }

        results.append(
            result
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        equity_df.to_csv(
            period_dir
            / f"{variant_name}_equity.csv",
            index=False,
        )

        trades_df.to_csv(
            period_dir
            / f"{variant_name}_trades.csv",
            index=False,
        )

        # ----------------------------------------------------
        # Print
        # ----------------------------------------------------

        pf = metrics[
            "ProfitFactor"
        ]

        pf_text = (
            "inf"
            if np.isinf(pf)
            else f"{pf:.3f}"
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
            f"{metrics['Trades']}"
        )

        print(
            f"  Win rate: "
            f"{metrics['WinRate'] * 100:.2f}%"
        )

        print(
            f"  PF: "
            f"{pf_text}"
        )

        print(
            f"  Expectancy: "
            f"${metrics['Expectancy']:.4f}"
        )

    return results


# ============================================================
# ROBUSTNESS SUMMARY
# ============================================================

def build_robustness_summary(
    yearly_results,
):

    df = pd.DataFrame(
        yearly_results
    )

    if df.empty:

        return (
            pd.DataFrame(),
            pd.DataFrame(),
        )

    # ========================================================
    # Variant-level aggregate
    # ========================================================

    aggregate = []

    for variant in VARIANTS:

        subset = df[
            df["Variant"]
            == variant
        ].copy()

        returns = (
            subset["Return"]
            .astype(float)
        )

        dds = (
            subset["MaxDD"]
            .astype(float)
        )

        trades = (
            subset["Trades"]
            .astype(float)
        )

        # ----------------------------------------------------
        # Compound return if each period starts at $10k.
        #
        # This is a research statistic, not a continuous
        # portfolio backtest.
        # ----------------------------------------------------

        compounded_return = (
            np.prod(
                1.0 + returns
            )
            - 1.0
        )

        # ----------------------------------------------------
        # Arithmetic average
        # ----------------------------------------------------

        avg_return = (
            returns.mean()
        )

        median_return = (
            returns.median()
        )

        # ----------------------------------------------------
        # Positive / negative periods
        # ----------------------------------------------------

        profitable_periods = (
            (returns > 0).sum()
        )

        losing_periods = (
            (returns < 0).sum()
        )

        flat_periods = (
            (returns == 0).sum()
        )

        # ----------------------------------------------------
        # Worst / best
        # ----------------------------------------------------

        best_return = (
            returns.max()
        )

        worst_return = (
            returns.min()
        )

        worst_dd = (
            dds.min()
        )

        avg_dd = (
            dds.mean()
        )

        # ----------------------------------------------------
        # Trade count
        # ----------------------------------------------------

        total_trades = (
            trades.sum()
        )

        # ----------------------------------------------------
        # Pooled trade metrics
        # ----------------------------------------------------

        aggregate.append(
            {
                "Variant":
                    variant,

                "Periods":
                    len(subset),

                "ProfitablePeriods":
                    profitable_periods,

                "LosingPeriods":
                    losing_periods,

                "FlatPeriods":
                    flat_periods,

                "AverageReturn":
                    avg_return,

                "MedianReturn":
                    median_return,

                "CompoundedReturn":
                    compounded_return,

                "BestPeriod":
                    best_return,

                "WorstPeriod":
                    worst_return,

                "AverageDD":
                    avg_dd,

                "WorstDD":
                    worst_dd,

                "TotalTrades":
                    int(total_trades),
            }
        )

    aggregate_df = pd.DataFrame(
        aggregate
    )

    # ========================================================
    # Year x variant matrix
    # ========================================================

    return_df = df.pivot(
        index="Year",
        columns="Variant",
        values="Return",
    )

    return_df = (
        return_df
        .reset_index()
    )

    return (
        aggregate_df,
        return_df,
    )


# ============================================================
# PRINT YEARLY MATRIX
# ============================================================

def print_yearly_matrix(
    results_df,
):

    print()
    print("=" * 110)
    print(
        "YEAR-BY-YEAR RETURNS"
    )
    print("=" * 110)

    matrix = (
        results_df
        .pivot(
            index="Year",
            columns="Variant",
            values="Return",
        )
        * 100
    )

    print(
        matrix.to_string(
            float_format=lambda x:
            f"{x:.3f}%"
        )
    )


# ============================================================
# PRINT AGGREGATE
# ============================================================

def print_aggregate(
    aggregate_df,
):

    print()
    print("=" * 120)
    print(
        "ROBUSTNESS SUMMARY"
    )
    print("=" * 120)

    display = (
        aggregate_df.copy()
    )

    for column in [
        "AverageReturn",
        "MedianReturn",
        "CompoundedReturn",
        "BestPeriod",
        "WorstPeriod",
        "AverageDD",
        "WorstDD",
    ]:

        display[column] = (
            display[column]
            * 100
        )

    print(
        display.to_string(
            index=False,
            formatters={
                "AverageReturn":
                    lambda x:
                    f"{x:.3f}%",

                "MedianReturn":
                    lambda x:
                    f"{x:.3f}%",

                "CompoundedReturn":
                    lambda x:
                    f"{x:.3f}%",

                "BestPeriod":
                    lambda x:
                    f"{x:.3f}%",

                "WorstPeriod":
                    lambda x:
                    f"{x:.3f}%",

                "AverageDD":
                    lambda x:
                    f"{x:.3f}%",

                "WorstDD":
                    lambda x:
                    f"{x:.3f}%",
            },
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 110)

    print(
        "MULTI-PERIOD DIRECTIONAL ROBUSTNESS TEST"
    )

    print("=" * 110)

    print()

    print(
        f"Symbol: {SYMBOL}"
    )

    print(
        "No optimization."
    )

    print(
        "No parameter changes between periods."
    )

    print(
        "Same four variants throughout."
    )

    print()

    print(
        "Locked parameters:"
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
        "Markov:"
    )

    print(
        f"  Window={HMM_WINDOW}"
        f"  MinObs={HMM_MIN_OBSERVATIONS}"
        f"  Refit={HMM_REFIT_EVERY}h"
        f"  Threshold={MR_THRESHOLD}"
    )

    print()

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
    # IMPORTANT
    #
    # Features are calculated on the FULL historical series
    # before period slicing.
    #
    # This preserves the causal rolling structure of the
    # Kalman/HMM models instead of resetting them at every
    # calendar year.
    # ========================================================

    df = add_kalman_features(
        df
    )

    df = add_markov_features(
        df
    )

    # ========================================================
    # RUN ALL PERIODS
    # ========================================================

    yearly_results = []

    for year, (
        period_start,
        period_end,
    ) in PERIODS.items():

        results = run_period(
            df,
            year,
            period_start,
            period_end,
        )

        yearly_results.extend(
            results
        )

    # ========================================================
    # SAVE YEARLY RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        yearly_results
    )

    results_file = (
        RESULTS_DIR
        / "yearly_results.csv"
    )

    results_df.to_csv(
        results_file,
        index=False,
    )

    # ========================================================
    # ROBUSTNESS SUMMARY
    # ========================================================

    (
        aggregate_df,
        return_matrix,
    ) = build_robustness_summary(
        yearly_results
    )

    aggregate_file = (
        RESULTS_DIR
        / "robustness_summary.csv"
    )

    matrix_file = (
        RESULTS_DIR
        / "yearly_return_matrix.csv"
    )

    aggregate_df.to_csv(
        aggregate_file,
        index=False,
    )

    return_matrix.to_csv(
        matrix_file,
        index=False,
    )

    # ========================================================
    # PRINT
    # ========================================================

    print_yearly_matrix(
        results_df
    )

    print_aggregate(
        aggregate_df
    )

    print()
    print("=" * 110)

    print(
        "FILES SAVED"
    )

    print("=" * 110)

    print(
        results_file
    )

    print(
        aggregate_file
    )

    print(
        matrix_file
    )

    print()
    print("=" * 110)

    print(
        "DONE"
    )

    print("=" * 110)


if __name__ == "__main__":

    main()