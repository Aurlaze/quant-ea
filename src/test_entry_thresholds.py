from pathlib import Path
import sys
import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
ROOT = SRC_DIR.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from walk_forward_kalman import (
    walk_forward_kalman,
    calculate_zscore,
)

from markov_regime import add_markov_regime


# ============================================================
# CONFIG
# ============================================================

SYMBOL = "XAUUSDm"

DATA_FILE = (
    ROOT
    / "data"
    / "mt5_h1"
    / f"{SYMBOL}_H1_2019_2026.csv"
)

RESULT_DIR = (
    ROOT
    / "results"
    / "entry_threshold_test"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LOCKED PARAMETERS
# ============================================================

PHI = 0.999
Q = 0.25
R = 0.25

MEAN_WINDOW = 168
Z_WINDOW = 36

EXIT_Z = 0.25
STOP_Z = 3.0
MAX_HOLD_HOURS = 24

HMM_WINDOW = 1000
HMM_MIN_OBSERVATIONS = 300
HMM_REFIT_EVERY = 24

MR_THRESHOLD = 0.60

INITIAL_CAPITAL = 10_000.0
RISK_PER_TRADE = 0.005
MAX_NOTIONAL_FRACTION = 1.0
ASSUMED_STOP_RETURN = 0.01

POINT_SIZE = 0.001

ENTRY_THRESHOLDS = [
    2.25,
    2.50,
    3.00,
    3.50,
]


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    df = pd.read_csv(
        DATA_FILE
    )

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        utc=True,
    )

    df = (
        df
        .set_index("Timestamp")
        .sort_index()
    )

    numeric = [
        "Open",
        "High",
        "Low",
        "Close",
        "SpreadPoints",
    ]

    for col in numeric:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df = df.dropna(
        subset=numeric
    )

    return df


# ============================================================
# BID / ASK
# ============================================================

def add_bid_ask(df):

    spread = (
        df["SpreadPoints"]
        * POINT_SIZE
    )

    df["Bid_Open"] = (
        df["Open"]
        - spread / 2
    )

    df["Ask_Open"] = (
        df["Open"]
        + spread / 2
    )

    df["Bid_Close"] = (
        df["Close"]
        - spread / 2
    )

    df["Ask_Close"] = (
        df["Close"]
        + spread / 2
    )

    return df


# ============================================================
# FEATURES
# ============================================================

def add_features(df):

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

    df = add_markov_regime(
        df,
        hmm_window=HMM_WINDOW,
        min_observations=HMM_MIN_OBSERVATIONS,
        refit_every=HMM_REFIT_EVERY,
    )

    return df


# ============================================================
# SIZING
# ============================================================

def calculate_units(
    equity,
    price,
):

    risk_budget = (
        equity
        * RISK_PER_TRADE
    )

    stop_loss_per_unit = (
        price
        * ASSUMED_STOP_RETURN
    )

    if (
        equity <= 0
        or price <= 0
        or stop_loss_per_unit <= 0
    ):

        return 0.0

    risk_units = (
        risk_budget
        / stop_loss_per_unit
    )

    notional_units = (
        equity
        * MAX_NOTIONAL_FRACTION
        / price
    )

    return min(
        risk_units,
        notional_units,
    )


# ============================================================
# EXACT CANONICAL ENGINE
# ============================================================

def run_variant(
    df,
    entry_z,
):

    equity = INITIAL_CAPITAL
    cash_equity = INITIAL_CAPITAL

    position = 0
    units = 0.0

    entry_price = np.nan
    entry_time = None
    entry_index = None
    entry_equity = np.nan
    entry_z_value = np.nan

    trades = []
    equity_curve = []

    for i in range(len(df)):

        timestamp = df.index[i]
        row = df.iloc[i]

        z = row["Z"]

        p_mr = row[
            "P_MeanReverting"
        ]

        # ====================================================
        # MARK TO MARKET
        # ====================================================

        if position == 1:

            equity = (
                cash_equity
                + units
                * (
                    row["Bid_Close"]
                    - entry_price
                )
            )

        elif position == -1:

            equity = (
                cash_equity
                + units
                * (
                    entry_price
                    - row["Ask_Close"]
                )
            )

        else:

            equity = cash_equity

        equity_curve.append(
            {
                "Timestamp":
                    timestamp,

                "Equity":
                    equity,
            }
        )

        # ----------------------------------------------------
        # No next bar
        # ----------------------------------------------------

        if i >= len(df) - 1:

            continue

        next_row = df.iloc[
            i + 1
        ]

        next_timestamp = df.index[
            i + 1
        ]

        # ====================================================
        # EXIT
        # ====================================================

        if position != 0:

            trading_hours_held = (
                i - entry_index
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

            if exit_reason is not None:

                if position == 1:

                    exit_price = (
                        next_row[
                            "Bid_Open"
                        ]
                    )

                    pnl = (
                        units
                        * (
                            exit_price
                            - entry_price
                        )
                    )

                    side = "LONG"

                else:

                    exit_price = (
                        next_row[
                            "Ask_Open"
                        ]
                    )

                    pnl = (
                        units
                        * (
                            entry_price
                            - exit_price
                        )
                    )

                    side = "SHORT"

                cash_equity += pnl

                trades.append(
                    {
                        "EntryTime":
                            entry_time,

                        "ExitTime":
                            next_timestamp,

                        "Side":
                            side,

                        "EntryZ":
                            entry_z_value,

                        "EntryPMR":
                            df.iloc[
                                entry_index
                            ][
                                "P_MeanReverting"
                            ],

                        "ExitReason":
                            exit_reason,

                        "TradingHoursHeld":
                            trading_hours_held,

                        "PnL":
                            pnl,

                        "Return":
                            pnl
                            / entry_equity,
                    }
                )

                # Reset

                position = 0
                units = 0.0

                entry_price = np.nan
                entry_time = None
                entry_index = None
                entry_equity = np.nan
                entry_z_value = np.nan

                # IMPORTANT:
                # no same-bar re-entry

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

            if z <= -entry_z:

                entry_price = (
                    next_row[
                        "Ask_Open"
                    ]
                )

                units = calculate_units(
                    equity,
                    entry_price,
                )

                if units <= 0:

                    continue

                position = 1

                entry_time = (
                    next_timestamp
                )

                entry_index = (
                    i + 1
                )

                entry_equity = (
                    equity
                )

                entry_z_value = z

                continue

            # ------------------------------------------------
            # SHORT
            #
            # EXACT SAME MARKOV FILTER
            # ------------------------------------------------

            if z >= entry_z:

                if pd.isna(p_mr):

                    continue

                if (
                    p_mr
                    < MR_THRESHOLD
                ):

                    continue

                entry_price = (
                    next_row[
                        "Bid_Open"
                    ]
                )

                units = calculate_units(
                    equity,
                    entry_price,
                )

                if units <= 0:

                    continue

                position = -1

                entry_time = (
                    next_timestamp
                )

                entry_index = (
                    i + 1
                )

                entry_equity = (
                    equity
                )

                entry_z_value = z

    # ========================================================
    # FORCE CLOSE
    # ========================================================

    if position != 0:

        row = df.iloc[-1]

        timestamp = df.index[-1]

        if position == 1:

            exit_price = (
                row["Bid_Close"]
            )

            pnl = (
                units
                * (
                    exit_price
                    - entry_price
                )
            )

            side = "LONG"

        else:

            exit_price = (
                row["Ask_Close"]
            )

            pnl = (
                units
                * (
                    entry_price
                    - exit_price
                )
            )

            side = "SHORT"

        cash_equity += pnl

        trades.append(
            {
                "EntryTime":
                    entry_time,

                "ExitTime":
                    timestamp,

                "Side":
                    side,

                "EntryZ":
                    entry_z_value,

                "EntryPMR":
                    df.iloc[
                        entry_index
                    ][
                        "P_MeanReverting"
                    ],

                "ExitReason":
                    "END_OF_DATA",

                "TradingHoursHeld":
                    len(df)
                    - 1
                    - entry_index,

                "PnL":
                    pnl,

                "Return":
                    pnl
                    / entry_equity,
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

def metrics(
    equity_df,
    trades_df,
):

    equity = (
        equity_df["Equity"]
    )

    final_equity = (
        equity.iloc[-1]
    )

    total_return = (
        final_equity
        / INITIAL_CAPITAL
        - 1
    )

    peak = equity.cummax()

    dd = (
        equity
        / peak
        - 1
    )

    returns = (
        equity
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
            * np.sqrt(24 * 365)
        )

    else:

        sharpe = np.nan

    if trades_df.empty:

        return {
            "FinalEquity":
                final_equity,

            "Return":
                total_return,

            "MaxDD":
                dd.min(),

            "Sharpe":
                sharpe,

            "Trades":
                0,

            "WinRate":
                np.nan,

            "PF":
                np.nan,

            "Expectancy":
                0.0,
        }

    pnl = (
        trades_df["PnL"]
    )

    wins = (
        pnl > 0
    )

    gross_profit = (
        pnl[wins].sum()
    )

    gross_loss = (
        pnl[~wins].sum()
    )

    if gross_loss < 0:

        pf = (
            gross_profit
            / abs(gross_loss)
        )

    else:

        pf = np.inf

    return {
        "FinalEquity":
            final_equity,

        "Return":
            total_return,

        "MaxDD":
            dd.min(),

        "Sharpe":
            sharpe,

        "Trades":
            len(trades_df),

        "WinRate":
            wins.mean(),

        "PF":
            pf,

        "Expectancy":
            pnl.mean(),
    }


# ============================================================
# YEARLY
# ============================================================

def yearly_results(
    equity_df,
    trades_df,
    threshold,
):

    equity_df = equity_df.copy()

    equity_df["Year"] = (
        equity_df["Timestamp"].dt.year
    )

    trades_df = trades_df.copy()

    if not trades_df.empty:

        trades_df["Year"] = (
            trades_df["ExitTime"].dt.year
        )

    results = []

    for year, group in equity_df.groupby(
        "Year"
    ):

        start = (
            group["Equity"].iloc[0]
        )

        end = (
            group["Equity"].iloc[-1]
        )

        ret = (
            end / start - 1
        )

        peak = (
            group["Equity"]
            .cummax()
        )

        dd = (
            group["Equity"]
            / peak
            - 1
        )

        year_trades = (
            trades_df[
                trades_df["Year"]
                == year
            ]
        )

        if year_trades.empty:

            pf = np.nan
            win_rate = np.nan
            expectancy = 0.0
            count = 0

        else:

            pnl = (
                year_trades["PnL"]
            )

            wins = (
                pnl > 0
            )

            gp = pnl[wins].sum()
            gl = pnl[~wins].sum()

            pf = (
                gp / abs(gl)
                if gl < 0
                else np.inf
            )

            win_rate = (
                wins.mean()
            )

            expectancy = (
                pnl.mean()
            )

            count = len(
                year_trades
            )

        results.append(
            {
                "EntryZ":
                    threshold,

                "Year":
                    year,

                "Return":
                    ret,

                "MaxDD":
                    dd.min(),

                "Trades":
                    count,

                "WinRate":
                    win_rate,

                "PF":
                    pf,

                "Expectancy":
                    expectancy,
            }
        )

    return pd.DataFrame(
        results
    )


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_data()

    df = add_bid_ask(
        df
    )

    print()
    print(
        "Calculating causal features..."
    )

    df = add_features(
        df
    )

    # --------------------------------------------------------
    # Sanity check BEFORE backtesting
    # --------------------------------------------------------

    print()
    print("=" * 110)
    print("SIGNAL SANITY CHECK")
    print("=" * 110)

    print(
        f"Z <= -2.25: "
        f"{(df['Z'] <= -2.25).sum():,}"
    )

    print(
        f"Z >= +2.25: "
        f"{(df['Z'] >= 2.25).sum():,}"
    )

    print(
        f"Z <= -3.50: "
        f"{(df['Z'] <= -3.50).sum():,}"
    )

    print(
        f"Z >= +3.50: "
        f"{(df['Z'] >= 3.50).sum():,}"
    )

    print(
        f"Valid P(MR): "
        f"{df['P_MeanReverting'].notna().sum():,}"
    )

    all_summary = []
    all_yearly = []

    for threshold in ENTRY_THRESHOLDS:

        print()
        print("=" * 110)

        print(
            f"ENTRY Z = {threshold:.2f}"
        )

        print("=" * 110)

        equity_df, trades_df = (
            run_variant(
                df,
                threshold,
            )
        )

        result = metrics(
            equity_df,
            trades_df,
        )

        print(
            f"Final equity: "
            f"${result['FinalEquity']:,.2f}"
        )

        print(
            f"Return: "
            f"{result['Return'] * 100:.3f}%"
        )

        print(
            f"Max DD: "
            f"{result['MaxDD'] * 100:.3f}%"
        )

        print(
            f"Sharpe: "
            f"{result['Sharpe']:.3f}"
        )

        print(
            f"Trades: "
            f"{result['Trades']:,}"
        )

        print(
            f"Win rate: "
            f"{result['WinRate'] * 100:.2f}%"
        )

        print(
            f"PF: "
            f"{result['PF']:.3f}"
        )

        print(
            f"Expectancy: "
            f"${result['Expectancy']:.4f}"
        )

        all_summary.append(
            {
                "EntryZ":
                    threshold,

                **result,
            }
        )

        yearly = yearly_results(
            equity_df,
            trades_df,
            threshold,
        )

        all_yearly.append(
            yearly
        )

        trades_df.to_csv(
            RESULT_DIR
            / (
                f"trades_entry_{threshold:.2f}.csv"
            ),
            index=False,
        )

    summary_df = pd.DataFrame(
        all_summary
    )

    yearly_df = pd.concat(
        all_yearly,
        ignore_index=True,
    )

    summary_df.to_csv(
        RESULT_DIR
        / "threshold_summary.csv",
        index=False,
    )

    yearly_df.to_csv(
        RESULT_DIR
        / "threshold_yearly_results.csv",
        index=False,
    )

    matrix = (
        yearly_df
        .pivot(
            index="Year",
            columns="EntryZ",
            values="Return",
        )
        * 100
    )

    matrix.to_csv(
        RESULT_DIR
        / "threshold_return_matrix.csv"
    )

    print()
    print("=" * 120)
    print("ENTRY THRESHOLD SUMMARY")
    print("=" * 120)

    print(
        summary_df.to_string(
            index=False,
            float_format=lambda x:
            f"{x:.6f}",
        )
    )

    print()
    print("=" * 120)
    print("YEARLY RETURN MATRIX")
    print("=" * 120)

    print(
        matrix.to_string(
            float_format=lambda x:
            f"{x:.3f}%"
        )
    )

    print()
    print(
        f"Results saved to:\n{RESULT_DIR}"
    )


if __name__ == "__main__":
    main()