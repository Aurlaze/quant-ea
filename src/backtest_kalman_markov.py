from pathlib import Path
import sys
import argparse

import numpy as np
import pandas as pd


# =========================================================
# PATHS
# =========================================================

SRC_DIR = Path(
    __file__
).resolve().parent

PROJECT_ROOT = (
    SRC_DIR.parent
)

if str(SRC_DIR) not in sys.path:

    sys.path.insert(
        0,
        str(SRC_DIR),
    )


from walk_forward_kalman import (
    walk_forward_kalman,
    calculate_zscore,
)

from markov_regime import (
    add_markov_regime,
)


# =========================================================
# PATHS
# =========================================================

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "mt5_h1"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "kalman_markov_v2"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# INSTRUMENTS
# =========================================================

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


# =========================================================
# KALMAN PARAMETERS
# =========================================================

PHI = 0.999

Q = 0.25

R = 0.25

MEAN_WINDOW = 168

Z_WINDOW = 36

ENTRY_Z = 2.25

EXIT_Z = 0.25

STOP_Z = 3.0

MAX_HOLD_HOURS = 24


# =========================================================
# MARKOV PARAMETERS
# =========================================================

MR_PROBABILITY_THRESHOLD = 0.60

HMM_WINDOW = 1000

HMM_MIN_OBSERVATIONS = 300

HMM_REFIT_EVERY = 24


# =========================================================
# RISK
# =========================================================

INITIAL_CAPITAL = 10_000.0

RISK_PER_TRADE = 0.005

MAX_NOTIONAL_FRACTION = 1.0

ASSUMED_STOP_RETURN = 0.01


# =========================================================
# PERIOD
# =========================================================

START_DATE = pd.Timestamp(
    "2019-01-01",
    tz="UTC",
)

END_DATE = pd.Timestamp(
    "2026-09-19",
    tz="UTC",
)


# =========================================================
# POINT SIZE
# =========================================================

POINT_SIZE = {

    "EURUSDm":
        0.00001,

    "GBPUSDm":
        0.00001,

    "USDJPYm":
        0.001,

    "XAUUSDm":
        0.001,

    "XAGUSDm":
        0.001,

    "USTECm":
        0.01,

    "JP225m":
        0.1,

    "BTCUSDm":
        0.01,

    "ETHUSDm":
        0.01,

    "USOILm":
        0.01,
}


# =========================================================
# DATA LOADER
# =========================================================

def find_column(
    df,
    candidates,
):

    lower_map = {
        str(column)
        .strip()
        .lower():
            column
        for column in df.columns
    }

    for candidate in candidates:

        key = candidate.lower()

        if key in lower_map:

            return lower_map[key]

    return None


def load_mt5_h1(
    symbol,
):

    path = (
        DATA_DIR
        / f"{symbol}_H1_2019_2026.csv"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Missing:\n{path}"
        )

    raw = pd.read_csv(
        path
    )

    time_col = find_column(
        raw,
        [
            "time",
            "timestamp",
            "datetime",
            "date",
        ],
    )

    open_col = find_column(
        raw,
        ["open"],
    )

    high_col = find_column(
        raw,
        ["high"],
    )

    low_col = find_column(
        raw,
        ["low"],
    )

    close_col = find_column(
        raw,
        ["close"],
    )

    spread_col = find_column(
        raw,
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
        for name, value
        in required.items()
        if value is None
    ]

    if missing:

        raise ValueError(
            f"{symbol}: missing "
            f"{missing}"
        )

    df = pd.DataFrame()

    df["Time"] = pd.to_datetime(
        raw[time_col],
        utc=True,
        errors="coerce",
    )

    df["Open"] = pd.to_numeric(
        raw[open_col],
        errors="coerce",
    )

    df["High"] = pd.to_numeric(
        raw[high_col],
        errors="coerce",
    )

    df["Low"] = pd.to_numeric(
        raw[low_col],
        errors="coerce",
    )

    df["Close"] = pd.to_numeric(
        raw[close_col],
        errors="coerce",
    )

    df["SpreadPoints"] = pd.to_numeric(
        raw[spread_col],
        errors="coerce",
    )

    df = (
        df
        .dropna()
        .sort_values("Time")
        .drop_duplicates(
            subset=["Time"]
        )
    )

    df = df[
        (df["Time"] >= START_DATE)
        &
        (df["Time"] <= END_DATE)
    ].copy()

    df = df.reset_index(
        drop=True
    )

    # -----------------------------------------------------
    # Validate OHLC
    # -----------------------------------------------------

    df = df[
        (df["Open"] > 0)
        &
        (df["High"] > 0)
        &
        (df["Low"] > 0)
        &
        (df["Close"] > 0)
    ]

    df = df[
        (df["High"] >= df["Low"])
        &
        (df["High"] >= df["Open"])
        &
        (df["High"] >= df["Close"])
        &
        (df["Low"] <= df["Open"])
        &
        (df["Low"] <= df["Close"])
    ]

    # -----------------------------------------------------
    # Bid / Ask approximation
    # -----------------------------------------------------

    point_size = POINT_SIZE[
        symbol
    ]

    df["SpreadPrice"] = (
        df["SpreadPoints"]
        * point_size
    )

    df["Bid"] = (
        df["Close"]
        - df["SpreadPrice"] / 2
    )

    df["Ask"] = (
        df["Close"]
        + df["SpreadPrice"] / 2
    )

    df["Bid_Open"] = (
        df["Open"]
        - df["SpreadPrice"] / 2
    )

    df["Ask_Open"] = (
        df["Open"]
        + df["SpreadPrice"] / 2
    )

    return df.reset_index(
        drop=True
    )


# =========================================================
# KALMAN
# =========================================================

def add_kalman_features(
    df,
):

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

    residual_series = pd.Series(
        residuals,
        index=df.index,
        dtype=float,
    )

    zscore = calculate_zscore(
        residual_series,
        window=Z_WINDOW,
    )

    out = df.copy()

    out[
        "KalmanState"
    ] = states

    out[
        "KalmanVariance"
    ] = variances

    out[
        "Residual"
    ] = residuals

    out[
        "RollingMean"
    ] = means

    out[
        "ZScore"
    ] = zscore

    return out


# =========================================================
# POSITION SIZE
# =========================================================

def calculate_units(
    equity,
    entry_price,
):

    if (
        equity <= 0
        or entry_price <= 0
    ):

        return 0.0

    risk_budget = (
        equity
        * RISK_PER_TRADE
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

    return max(
        0.0,
        min(
            units,
            max_units,
        ),
    )


# =========================================================
# BACKTEST
# =========================================================

def run_backtest(
    symbol,
    df,
    use_markov,
):

    equity = (
        INITIAL_CAPITAL
    )

    position = 0

    units = 0.0

    entry_price = np.nan

    entry_time = None

    entry_z = np.nan

    trading_hours_held = 0

    previous_z = np.nan

    pending_action = None

    pending_reason = None

    trades = []

    equity_records = []

    blocked_entries = 0

    # =====================================================
    # LOOP
    # =====================================================

    for i in range(
        len(df)
    ):

        row = df.iloc[i]

        timestamp = row[
            "Time"
        ]

        z = row[
            "ZScore"
        ]

        # -------------------------------------------------
        # Execute pending action
        # -------------------------------------------------

        if pending_action is not None:

            action = (
                pending_action
            )

            reason = (
                pending_reason
            )

            # =============================================
            # LONG ENTRY
            # =============================================

            if action == "ENTER_LONG":

                execution_price = (
                    row["Ask_Open"]
                )

                new_units = (
                    calculate_units(
                        equity,
                        execution_price,
                    )
                )

                if new_units > 0:

                    position = 1

                    units = new_units

                    entry_price = (
                        execution_price
                    )

                    entry_time = (
                        timestamp
                    )

                    entry_z = (
                        previous_z
                    )

                    trading_hours_held = 0

            # =============================================
            # SHORT ENTRY
            # =============================================

            elif action == "ENTER_SHORT":

                execution_price = (
                    row["Bid_Open"]
                )

                new_units = (
                    calculate_units(
                        equity,
                        execution_price,
                    )
                )

                if new_units > 0:

                    position = -1

                    units = new_units

                    entry_price = (
                        execution_price
                    )

                    entry_time = (
                        timestamp
                    )

                    entry_z = (
                        previous_z
                    )

                    trading_hours_held = 0

            # =============================================
            # EXIT
            # =============================================

            elif action == "EXIT":

                if position != 0:

                    if position == 1:

                        execution_price = (
                            row["Bid_Open"]
                        )

                        pnl = (
                            execution_price
                            - entry_price
                        ) * units

                    else:

                        execution_price = (
                            row["Ask_Open"]
                        )

                        pnl = (
                            entry_price
                            - execution_price
                        ) * units

                    equity_before = (
                        equity
                    )

                    equity += pnl

                    calendar_hours = (
                        timestamp
                        - entry_time
                    ).total_seconds()/3600

                    trade_return = (
                        pnl
                        / equity_before
                    )

                    trades.append(
                        {
                            "Symbol":
                                symbol,

                            "EntryTime":
                                entry_time,

                            "ExitTime":
                                timestamp,

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
                                execution_price,

                            "EntryZ":
                                entry_z,

                            "ExitZ":
                                previous_z,

                            "TradingHoursHeld":
                                trading_hours_held,

                            "CalendarHoursHeld":
                                calendar_hours,

                            "Return":
                                trade_return,

                            "PnL":
                                pnl,

                            "ExitReason":
                                reason,
                        }
                    )

                    position = 0

                    units = 0.0

                    entry_price = np.nan

                    entry_time = None

                    entry_z = np.nan

                    trading_hours_held = 0

            pending_action = None

            pending_reason = None

        # -------------------------------------------------
        # Mark-to-market
        # -------------------------------------------------

        if position == 0:

            current_equity = (
                equity
            )

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

        equity_records.append(
            {
                "Time":
                    timestamp,

                "Equity":
                    current_equity,

                "Position":
                    position,

                "ZScore":
                    z,

                "P_MeanReverting":
                    row[
                        "P_MeanReverting"
                    ]
                    if use_markov
                    else np.nan,

                "P_Trending":
                    row[
                        "P_Trending"
                    ]
                    if use_markov
                    else np.nan,

                "HMMState":
                    row[
                        "HMMState"
                    ]
                    if use_markov
                    else np.nan,
            }
        )

        # -------------------------------------------------
        # First observation
        # -------------------------------------------------

        if i == 0:

            previous_z = z

            continue

        # -------------------------------------------------
        # Holding period
        # -------------------------------------------------

        if position != 0:

            trading_hours_held += 1

        # -------------------------------------------------
        # No signal if Z unavailable
        # -------------------------------------------------

        if pd.isna(z):

            previous_z = z

            continue

        # =================================================
        # REGIME GATE
        # =================================================

        if use_markov:

            p_mr = row[
                "P_MeanReverting"
            ]

            regime_allows = (
                pd.notna(p_mr)
                and
                p_mr
                >= MR_PROBABILITY_THRESHOLD
            )

        else:

            regime_allows = True

        # =================================================
        # ENTRY
        # =================================================

        if position == 0:

            # ---------------------------------------------
            # LONG
            # ---------------------------------------------

            if z <= -ENTRY_Z:

                if regime_allows:

                    pending_action = (
                        "ENTER_LONG"
                    )

                    pending_reason = (
                        "Z_ENTRY_LONG"
                    )

                else:

                    blocked_entries += 1

            # ---------------------------------------------
            # SHORT
            # ---------------------------------------------

            elif z >= ENTRY_Z:

                if regime_allows:

                    pending_action = (
                        "ENTER_SHORT"
                    )

                    pending_reason = (
                        "Z_ENTRY_SHORT"
                    )

                else:

                    blocked_entries += 1

        # =================================================
        # EXIT
        # =================================================

        else:

            # ---------------------------------------------
            # MEAN REVERSION EXIT
            # ---------------------------------------------

            if abs(z) <= EXIT_Z:

                pending_action = (
                    "EXIT"
                )

                pending_reason = (
                    "Z_EXIT"
                )

            # ---------------------------------------------
            # STOP
            # ---------------------------------------------

            elif abs(z) >= STOP_Z:

                pending_action = (
                    "EXIT"
                )

                pending_reason = (
                    "Z_STOP"
                )

            # ---------------------------------------------
            # MAX HOLD
            # ---------------------------------------------

            elif (
                trading_hours_held
                >= MAX_HOLD_HOURS
            ):

                pending_action = (
                    "EXIT"
                )

                pending_reason = (
                    "MAX_HOLD"
                )

        previous_z = z

    # =====================================================
    # FORCE CLOSE
    # =====================================================

    if position != 0:

        last = df.iloc[-1]

        if position == 1:

            execution_price = (
                last["Bid"]
            )

            pnl = (
                execution_price
                - entry_price
            ) * units

        else:

            execution_price = (
                last["Ask"]
            )

            pnl = (
                entry_price
                - execution_price
            ) * units

        equity_before = (
            equity
        )

        equity += pnl

        calendar_hours = (
            last["Time"]
            - entry_time
        ).total_seconds()/3600

        trades.append(
            {
                "Symbol":
                    symbol,

                "EntryTime":
                    entry_time,

                "ExitTime":
                    last["Time"],

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
                    execution_price,

                "EntryZ":
                    entry_z,

                "ExitZ":
                    previous_z,

                "TradingHoursHeld":
                    trading_hours_held,

                "CalendarHoursHeld":
                    calendar_hours,

                "Return":
                    pnl
                    / equity_before,

                "PnL":
                    pnl,

                "ExitReason":
                    "END_OF_DATA",
            }
        )

    # =====================================================
    # EQUITY DATAFRAME
    # =====================================================

    equity_df = pd.DataFrame(
        equity_records
    )

    trades_df = pd.DataFrame(
        trades
    )

    equity_df["Peak"] = (
        equity_df["Equity"]
        .cummax()
    )

    equity_df["Drawdown"] = (
        equity_df["Equity"]
        / equity_df["Peak"]
        - 1
    )

    # =====================================================
    # PERFORMANCE
    # =====================================================

    final_equity = (
        equity_df["Equity"]
        .iloc[-1]
    )

    total_return = (
        final_equity
        / INITIAL_CAPITAL
        - 1
    )

    max_dd = (
        equity_df["Drawdown"]
        .min()
    )

    # -----------------------------------------------------
    # Sharpe
    # -----------------------------------------------------

    returns = (
        equity_df["Equity"]
        .pct_change()
        .replace(
            [
                np.inf,
                -np.inf,
            ],
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
            * np.sqrt(
                24 * 365
            )
        )

    else:

        sharpe = np.nan

    # -----------------------------------------------------
    # Trade statistics
    # -----------------------------------------------------

    trades_count = len(
        trades_df
    )

    if trades_count > 0:

        wins = (
            trades_df["PnL"]
            > 0
        )

        win_rate = (
            wins.mean()
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
            trades_df["PnL"]
            .mean()
        )

    else:

        win_rate = np.nan

        profit_factor = np.nan

        expectancy = np.nan

    # -----------------------------------------------------
    # Monthly returns
    # -----------------------------------------------------

    monthly_equity = (
        equity_df
        .set_index("Time")
        ["Equity"]
        .resample("ME")
        .last()
    )

    monthly_returns = (
        monthly_equity
        .pct_change()
    )

    if len(monthly_returns) > 0:

        monthly_returns.iloc[0] = (
            monthly_equity.iloc[0]
            / INITIAL_CAPITAL
            - 1
        )

    profitable_months = int(
        (
            monthly_returns
            > 0
        ).sum()
    )

    losing_months = int(
        (
            monthly_returns
            < 0
        ).sum()
    )

    # -----------------------------------------------------
    # HMM diagnostics
    # -----------------------------------------------------

    if use_markov:

        p_mr = (
            df[
                "P_MeanReverting"
            ]
            .dropna()
        )

        if len(p_mr) > 0:

            mean_p_mr = (
                p_mr.mean()
            )

            fraction_allowed = (
                p_mr
                >= MR_PROBABILITY_THRESHOLD
            ).mean()

        else:

            mean_p_mr = np.nan

            fraction_allowed = np.nan

        hmm_valid = len(
            p_mr
        )

    else:

        mean_p_mr = np.nan

        fraction_allowed = np.nan

        hmm_valid = 0

    summary = {

        "Symbol":
            symbol,

        "Mode":
            (
                "Kalman+Markov"
                if use_markov
                else "KalmanOnly"
            ),

        "InitialEquity":
            INITIAL_CAPITAL,

        "FinalEquity":
            final_equity,

        "TotalReturn":
            total_return,

        "MaxDrawdown":
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

        "ProfitableMonths":
            profitable_months,

        "LosingMonths":
            losing_months,

        "BlockedEntries":
            blocked_entries,

        "HMMValidBars":
            hmm_valid,

        "MeanMRProbability":
            mean_p_mr,

        "FractionMRAllowed":
            fraction_allowed,
    }

    return (
        df,
        equity_df,
        trades_df,
        summary,
    )


# =========================================================
# RUN ONE SYMBOL
# =========================================================

def process_symbol(
    symbol,
):

    print()
    print(
        "=" * 110
    )

    print(
        f"PROCESSING {symbol}"
    )

    print(
        "=" * 110
    )

    df = load_mt5_h1(
        symbol
    )

    print(
        f"Bars: {len(df):,}"
    )

    # -----------------------------------------------------
    # Kalman first
    # -----------------------------------------------------

    print(
        "Calculating Kalman..."
    )

    df = add_kalman_features(
        df
    )

    # -----------------------------------------------------
    # Markov
    # -----------------------------------------------------

    print(
        "Calculating causal HMM..."
    )

    df = add_markov_regime(
        df,
        hmm_window=HMM_WINDOW,
        min_observations=HMM_MIN_OBSERVATIONS,
        refit_every=HMM_REFIT_EVERY,
    )

    valid_hmm = (
        df[
            "P_MeanReverting"
        ]
        .notna()
        .sum()
    )

    print(
        f"HMM valid bars: "
        f"{valid_hmm:,}"
    )

    if valid_hmm > 0:

        print(
            "Mean P(MR): "
            f"{df['P_MeanReverting'].mean():.3f}"
        )

        print(
            "Mean P(Trend): "
            f"{df['P_Trending'].mean():.3f}"
        )

        print(
            "MR autocorrelation: "
            f"{df['MR_Autocorrelation'].dropna().mean():.5f}"
        )

        print(
            "Trend autocorrelation: "
            f"{df['Trend_Autocorrelation'].dropna().mean():.5f}"
        )

    # =====================================================
    # BASELINE
    # =====================================================

    print()
    print(
        "Running KALMAN ONLY..."
    )

    (
        baseline_df,
        baseline_equity,
        baseline_trades,
        baseline_summary,
    ) = run_backtest(
        symbol,
        df,
        use_markov=False,
    )

    # =====================================================
    # MARKOV
    # =====================================================

    print()
    print(
        "Running KALMAN + MARKOV..."
    )

    (
        markov_df,
        markov_equity,
        markov_trades,
        markov_summary,
    ) = run_backtest(
        symbol,
        df,
        use_markov=True,
    )

    # =====================================================
    # SAVE
    # =====================================================

    symbol_dir = (
        RESULTS_DIR
        / symbol
    )

    symbol_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # Processed
    # -----------------------------------------------------

    markov_df.to_csv(
        symbol_dir
        / "processed.csv",
        index=False,
    )

    # -----------------------------------------------------
    # Baseline
    # -----------------------------------------------------

    baseline_equity.to_csv(
        symbol_dir
        / "kalman_only_equity.csv",
        index=False,
    )

    baseline_trades.to_csv(
        symbol_dir
        / "kalman_only_trades.csv",
        index=False,
    )

    # -----------------------------------------------------
    # Markov
    # -----------------------------------------------------

    markov_equity.to_csv(
        symbol_dir
        / "kalman_markov_equity.csv",
        index=False,
    )

    markov_trades.to_csv(
        symbol_dir
        / "kalman_markov_trades.csv",
        index=False,
    )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    comparison = pd.DataFrame(
        [
            baseline_summary,
            markov_summary,
        ]
    )

    comparison.to_csv(
        symbol_dir
        / "comparison.csv",
        index=False,
    )

    # =====================================================
    # PRINT
    # =====================================================

    print()
    print(
        f"{symbol} RESULT"
    )

    print(
        "-" * 100
    )

    print(
        f"{'Metric':<25}"
        f"{'Kalman':>20}"
        f"{'Kalman+Markov':>25}"
    )

    print(
        "-" * 100
    )

    print(
        f"{'Return':<25}"
        f"{baseline_summary['TotalReturn'] * 100:>19.3f}%"
        f"{markov_summary['TotalReturn'] * 100:>24.3f}%"
    )

    print(
        f"{'Max DD':<25}"
        f"{baseline_summary['MaxDrawdown'] * 100:>19.3f}%"
        f"{markov_summary['MaxDrawdown'] * 100:>24.3f}%"
    )

    print(
        f"{'Sharpe':<25}"
        f"{baseline_summary['Sharpe']:>20.3f}"
        f"{markov_summary['Sharpe']:>25.3f}"
    )

    print(
        f"{'Trades':<25}"
        f"{baseline_summary['Trades']:>20}"
        f"{markov_summary['Trades']:>25}"
    )

    print(
        f"{'Win rate':<25}"
        f"{baseline_summary['WinRate'] * 100:>19.2f}%"
        f"{markov_summary['WinRate'] * 100:>24.2f}%"
    )

    print(
        f"{'Profit factor':<25}"
        f"{baseline_summary['ProfitFactor']:>20.3f}"
        f"{markov_summary['ProfitFactor']:>25.3f}"
    )

    print(
        f"{'Profitable months':<25}"
        f"{baseline_summary['ProfitableMonths']:>20}"
        f"{markov_summary['ProfitableMonths']:>25}"
    )

    print(
        f"{'Losing months':<25}"
        f"{baseline_summary['LosingMonths']:>20}"
        f"{markov_summary['LosingMonths']:>25}"
    )

    print(
        f"{'Blocked entries':<25}"
        f"{'N/A':>20}"
        f"{markov_summary['BlockedEntries']:>25}"
    )

    return (
        baseline_summary,
        markov_summary,
    )


# =========================================================
# MAIN
# =========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--symbol",
        type=str,
        default="ALL",
        help=(
            "Instrument to test. "
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
                f"{args.symbol}\n"
                f"Available: "
                f"{INSTRUMENTS}"
            )

        symbols = [
            args.symbol
        ]

    print()
    print(
        "=" * 110
    )

    print(
        "KALMAN + CAUSAL MARKOV REGIME FILTER V2"
    )

    print(
        "=" * 110
    )

    print()
    print(
        "Kalman parameters:"
    )

    print(
        f"  PHI       = {PHI}"
    )

    print(
        f"  Q         = {Q}"
    )

    print(
        f"  R         = {R}"
    )

    print(
        f"  Mean      = {MEAN_WINDOW}"
    )

    print(
        f"  Z window  = {Z_WINDOW}"
    )

    print(
        f"  Entry     = {ENTRY_Z}"
    )

    print(
        f"  Exit      = {EXIT_Z}"
    )

    print(
        f"  Stop      = {STOP_Z}"
    )

    print(
        f"  Max hold  = {MAX_HOLD_HOURS}h"
    )

    print()
    print(
        "Markov parameters:"
    )

    print(
        f"  HMM window = {HMM_WINDOW}"
    )

    print(
        f"  Refit      = {HMM_REFIT_EVERY}h"
    )

    print(
        f"  MR threshold = "
        f"{MR_PROBABILITY_THRESHOLD}"
    )

    print()
    print(
        "No optimization."
    )

    all_results = []

    # =====================================================
    # RUN
    # =====================================================

    for symbol in symbols:

        try:

            (
                baseline,
                markov,
            ) = process_symbol(
                symbol
            )

            all_results.extend(
                [
                    baseline,
                    markov,
                ]
            )

        except Exception as error:

            print()
            print(
                f"ERROR: {symbol}"
            )

            print(
                repr(error)
            )

    # =====================================================
    # SAVE MASTER RESULTS
    # =====================================================

    if len(all_results) == 0:

        print(
            "No results generated."
        )

        return

    results_df = pd.DataFrame(
        all_results
    )

    results_df.to_csv(
        RESULTS_DIR
        / "all_results.csv",
        index=False,
    )

    # =====================================================
    # FINAL TABLE
    # =====================================================

    print()
    print()
    print(
        "=" * 130
    )

    print(
        "FINAL RESULTS"
    )

    print(
        "=" * 130
    )

    for symbol in symbols:

        rows = (
            results_df[
                results_df["Symbol"]
                == symbol
            ]
        )

        if len(rows) != 2:

            continue

        kalman = (
            rows[
                rows["Mode"]
                == "KalmanOnly"
            ]
            .iloc[0]
        )

        markov = (
            rows[
                rows["Mode"]
                == "Kalman+Markov"
            ]
            .iloc[0]
        )

        print()
        print(
            symbol
        )

        print(
            f"  Return     : "
            f"{kalman['TotalReturn'] * 100:>8.3f}%"
            f" -> "
            f"{markov['TotalReturn'] * 100:>8.3f}%"
        )

        print(
            f"  Max DD     : "
            f"{kalman['MaxDrawdown'] * 100:>8.3f}%"
            f" -> "
            f"{markov['MaxDrawdown'] * 100:>8.3f}%"
        )

        print(
            f"  Sharpe     : "
            f"{kalman['Sharpe']:>8.3f}"
            f" -> "
            f"{markov['Sharpe']:>8.3f}"
        )

        print(
            f"  Trades     : "
            f"{kalman['Trades']:>8}"
            f" -> "
            f"{markov['Trades']:>8}"
        )

        print(
            f"  PF         : "
            f"{kalman['ProfitFactor']:>8.3f}"
            f" -> "
            f"{markov['ProfitFactor']:>8.3f}"
        )

        print(
            f"  Blocked    : "
            f"{markov['BlockedEntries']:>8}"
        )

    print()
    print(
        "=" * 130
    )

    print(
        "DONE"
    )

    print(
        f"Results:"
        f"\n{RESULTS_DIR}"
    )


if __name__ == "__main__":

    main()