from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from arch import arch_model

from walk_forward_kalman import (
    walk_forward_kalman,
    calculate_zscore,
)
from markov_regime import add_markov_regime


# ============================================================
# CONFIG
# ============================================================

SYMBOL = "XAUUSDm"

DATA_PATH = Path(
    "data/mt5_h1/XAUUSDm_H1_2019_2026.csv"
)

CANONICAL_TRADES_PATH = Path(
    "results/walk_forward_directional/"
    "MARKOV_SHORT_FILTER_trades.csv"
)

START_DATE = pd.Timestamp(
    "2019-01-01",
    tz="UTC",
)

END_DATE = pd.Timestamp(
    "2026-09-18 20:00",
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


# ============================================================
# LOCKED STRATEGY PARAMETERS
# ============================================================

ENTRY_Z = 2.25
EXIT_Z = 0.25
STOP_Z = 3.0
MAX_HOLD_HOURS = 24


# ============================================================
# LOCKED MARKOV PARAMETER
# ============================================================

MR_PROBABILITY_THRESHOLD = 0.60


# ============================================================
# XAU SETTINGS
# ============================================================

POINT_SIZE = 0.001


# ============================================================
# CANONICAL TRADE COUNT
# ============================================================

EXPECTED_TRADES = 1351


# ============================================================
# GARCH CONFIGURATION
# ============================================================

GARCH_P = 1
GARCH_Q = 1

# Rolling GARCH training window
GARCH_WINDOW = 1000

# Refit every 24 H1 bars
GARCH_REFIT_EVERY = 24

# Minimum observations required
GARCH_MIN_OBSERVATIONS = 300

# Percentage scaling for ARCH package
GARCH_SCALE = 100.0


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = Path(
    "results/garch_diagnostic"
)


# ============================================================
# GENERAL HELPERS
# ============================================================

def print_section(title):

    print()
    print("=" * 110)
    print(title)
    print("=" * 110)


def safe_pf(returns):

    returns = (
        pd.Series(returns)
        .dropna()
    )

    gross_profit = returns[
        returns > 0
    ].sum()

    gross_loss = abs(
        returns[
            returns <= 0
        ].sum()
    )

    if gross_loss <= 0:

        return np.inf

    return (
        gross_profit
        / gross_loss
    )


# ============================================================
# LOAD H1 DATA
# ============================================================

def load_h1():

    print(
        "Loading H1 data..."
    )

    df = pd.read_csv(
        DATA_PATH
    )

    required = {
        "Timestamp",
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

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        utc=True,
    )

    df = (
        df
        .sort_values("Timestamp")
        .reset_index(drop=True)
    )

    df = df[
        (df["Timestamp"] >= START_DATE)
        & (df["Timestamp"] <= END_DATE)
    ].copy()

    df = (
        df
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Approximate bid / ask
    # --------------------------------------------------------

    spread_price = (
        df["SpreadPoints"]
        * POINT_SIZE
    )

    df["Bid"] = (
        df["Close"]
        - spread_price / 2.0
    )

    df["Ask"] = (
        df["Close"]
        + spread_price / 2.0
    )

    return df


# ============================================================
# KALMAN FEATURES
# ============================================================

def add_kalman_features(df):

    print(
        "Calculating causal Kalman features..."
    )

    prices = df["Close"]

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
    )

    z = calculate_zscore(
        residual_series,
        window=Z_WINDOW,
    )

    df["KalmanState"] = states

    df["KalmanVariance"] = variances

    df["Residual"] = residuals

    df["RollingMean"] = means

    df["Z"] = z.values

    return df


# ============================================================
# BASIC VOLATILITY FEATURES
# ============================================================

def add_basic_volatility(df):

    print(
        "Calculating basic volatility features..."
    )

    log_return = np.log(
        df["Close"]
        / df["Close"].shift(1)
    )

    df["LogReturn"] = log_return

    # --------------------------------------------------------
    # Realized volatility
    # --------------------------------------------------------

    df["RV_24"] = (
        log_return
        .rolling(24)
        .std()
    )

    df["RV_168"] = (
        log_return
        .rolling(168)
        .std()
    )

    df["VolRatio_24_168"] = (
        df["RV_24"]
        / df["RV_168"]
    )

    # --------------------------------------------------------
    # True range
    # --------------------------------------------------------

    previous_close = (
        df["Close"]
        .shift(1)
    )

    tr1 = (
        df["High"]
        - df["Low"]
    )

    tr2 = (
        df["High"]
        - previous_close
    ).abs()

    tr3 = (
        df["Low"]
        - previous_close
    ).abs()

    df["TrueRange"] = pd.concat(
        [
            tr1,
            tr2,
            tr3,
        ],
        axis=1,
    ).max(axis=1)

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    df["ATR_24"] = (
        df["TrueRange"]
        .rolling(24)
        .mean()
    )

    df["ATR_168"] = (
        df["TrueRange"]
        .rolling(168)
        .mean()
    )

    df["ATR_Ratio"] = (
        df["ATR_24"]
        / df["ATR_168"]
    )

    return df


# ============================================================
# CAUSAL GARCH(1,1)
# ============================================================

def calculate_causal_garch(df):

    """
    Causal rolling GARCH(1,1).

    At each refit:

        training data =
        observations strictly before current bar

    The fitted model produces a one-step-ahead forecast.

    The forecast is then carried forward until the next
    scheduled refit.

    No future observations are used.
    """

    print()
    print(
        "Calculating causal GARCH(1,1)..."
    )

    returns = (
        df["LogReturn"]
        * GARCH_SCALE
    )

    n = len(df)

    forecast_variance = np.full(
        n,
        np.nan,
        dtype=float,
    )

    forecast_volatility = np.full(
        n,
        np.nan,
        dtype=float,
    )

    last_refit = (
        -GARCH_REFIT_EVERY
    )

    refit_count = 0
    failed_count = 0

    with warnings.catch_warnings():

        warnings.filterwarnings(
            "ignore"
        )

        for i in range(n):

            # ------------------------------------------------
            # Need enough history
            # ------------------------------------------------

            if (
                i
                < GARCH_MIN_OBSERVATIONS
            ):

                continue

            # ------------------------------------------------
            # Refit every N bars
            # ------------------------------------------------

            if (
                i - last_refit
                >= GARCH_REFIT_EVERY
            ):

                start = max(
                    0,
                    i - GARCH_WINDOW,
                )

                train = (
                    returns
                    .iloc[start:i]
                    .dropna()
                )

                if (
                    len(train)
                    < GARCH_MIN_OBSERVATIONS
                ):

                    continue

                try:

                    model = arch_model(
                        train,
                        mean="Zero",
                        vol="GARCH",
                        p=GARCH_P,
                        q=GARCH_Q,
                        dist="normal",
                        rescale=False,
                    )

                    result = model.fit(
                        disp="off",
                        show_warning=False,
                    )

                    forecast = (
                        result
                        .forecast(
                            horizon=1,
                            reindex=False,
                        )
                    )

                    variance = float(
                        forecast
                        .variance
                        .iloc[-1, 0]
                    )

                    if (
                        np.isfinite(
                            variance
                        )
                        and variance > 0
                    ):

                        forecast_variance[
                            i
                        ] = variance

                        forecast_volatility[
                            i
                        ] = np.sqrt(
                            variance
                        )

                        last_refit = i

                        refit_count += 1

                    else:

                        failed_count += 1

                except Exception:

                    failed_count += 1

            # ------------------------------------------------
            # Carry previous forecast
            # ------------------------------------------------

            else:

                if i > 0:

                    forecast_variance[
                        i
                    ] = forecast_variance[
                        i - 1
                    ]

                    forecast_volatility[
                        i
                    ] = forecast_volatility[
                        i - 1
                    ]

    # --------------------------------------------------------
    # Save GARCH features
    # --------------------------------------------------------

    df["GARCHVariancePct"] = (
        forecast_variance
    )

    df["GARCHVolPct"] = (
        forecast_volatility
    )

    # Decimal volatility
    df["GARCHVol"] = (
        forecast_volatility
        / GARCH_SCALE
    )

    print(
        f"GARCH refits: "
        f"{refit_count:,}"
    )

    print(
        f"GARCH failed fits: "
        f"{failed_count:,}"
    )

    valid = (
        df["GARCHVol"]
        .notna()
        .sum()
    )

    print(
        f"Valid GARCH forecasts: "
        f"{valid:,}"
    )

    return df


# ============================================================
# MARKOV FEATURES
# ============================================================

def add_markov_features(df):

    print(
        "Calculating causal Markov regime..."
    )

    df = add_markov_regime(
        df
    )

    return df


# ============================================================
# LOAD CANONICAL TRADES
# ============================================================

def load_canonical_trades():

    print()
    print(
        "Loading canonical "
        "MARKOV_SHORT_FILTER trades..."
    )

    if not CANONICAL_TRADES_PATH.exists():

        raise FileNotFoundError(
            "\nCanonical trade file not found:\n"
            f"{CANONICAL_TRADES_PATH}"
        )

    trades = pd.read_csv(
        CANONICAL_TRADES_PATH
    )

    print(
        f"Canonical trades loaded: "
        f"{len(trades):,}"
    )

    return trades


# ============================================================
# NORMALIZE TRADE DATA
# ============================================================

def normalize_trades(trades):

    trades = trades.copy()

    # --------------------------------------------------------
    # Datetime columns
    # --------------------------------------------------------

    for col in [
        "EntryTime",
        "ExitTime",
        "EntrySignalTime",
    ]:

        if col in trades.columns:

            trades[col] = pd.to_datetime(
                trades[col],
                utc=True,
            )

    # --------------------------------------------------------
    # Numeric columns
    # --------------------------------------------------------

    for col in [
        "EntryZ",
        "ExitZ",
        "PnL",
        "Pnl",
        "Return",
        "GrossReturn",
        "EntryPrice",
        "ExitPrice",
        "TradingHoursHeld",
        "CalendarHoursHeld",
        "HoursHeld",
    ]:

        if col in trades.columns:

            trades[col] = pd.to_numeric(
                trades[col],
                errors="coerce",
            )

    # --------------------------------------------------------
    # PnL
    # --------------------------------------------------------

    if (
        "PnL" not in trades.columns
        and "Pnl" in trades.columns
    ):

        trades["PnL"] = trades[
            "Pnl"
        ]

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    if (
        "GrossReturn"
        not in trades.columns
        and "Return" in trades.columns
    ):

        trades["GrossReturn"] = trades[
            "Return"
        ]

    # --------------------------------------------------------
    # Holding period
    # --------------------------------------------------------

    if (
        "TradingHoursHeld"
        not in trades.columns
    ):

        if (
            "HoursHeld"
            in trades.columns
        ):

            trades["TradingHoursHeld"] = (
                trades["HoursHeld"]
            )

        elif (
            "CalendarHoursHeld"
            in trades.columns
        ):

            trades["TradingHoursHeld"] = (
                trades["CalendarHoursHeld"]
            )

    return trades


# ============================================================
# FIND INDEX
# ============================================================

def find_index(
    df,
    timestamp,
):

    if pd.isna(timestamp):

        return None

    matches = np.flatnonzero(
        df["Timestamp"].values
        == timestamp.to_datetime64()
    )

    if len(matches) == 0:

        return None

    return int(
        matches[0]
    )


# ============================================================
# ATTACH ENTRY FEATURES
# ============================================================

def attach_entry_features(
    trades,
    df,
):

    print()
    print(
        "Attaching entry-time "
        "GARCH / volatility features..."
    )

    records = []

    missing = 0

    for _, trade in trades.iterrows():

        record = trade.to_dict()

        idx = find_index(
            df,
            trade["EntryTime"],
        )

        if idx is None:

            missing += 1

            records.append(
                record
            )

            continue

        row = df.iloc[idx]

        # ----------------------------------------------------
        # GARCH
        # ----------------------------------------------------

        record[
            "EntryGARCHVol"
        ] = row[
            "GARCHVol"
        ]

        record[
            "EntryGARCHVolPct"
        ] = row[
            "GARCHVolPct"
        ]

        record[
            "EntryGARCHVariancePct"
        ] = row[
            "GARCHVariancePct"
        ]

        # ----------------------------------------------------
        # Realized volatility
        # ----------------------------------------------------

        record[
            "EntryRV24"
        ] = row[
            "RV_24"
        ]

        record[
            "EntryRV168"
        ] = row[
            "RV_168"
        ]

        record[
            "EntryVolRatio"
        ] = row[
            "VolRatio_24_168"
        ]

        # ----------------------------------------------------
        # ATR
        # ----------------------------------------------------

        record[
            "EntryATR24"
        ] = row[
            "ATR_24"
        ]

        record[
            "EntryATR168"
        ] = row[
            "ATR_168"
        ]

        record[
            "EntryATRRatio"
        ] = row[
            "ATR_Ratio"
        ]

        # ----------------------------------------------------
        # Existing model features
        # ----------------------------------------------------

        record[
            "EntryZ_Recheck"
        ] = row[
            "Z"
        ]

        record[
            "EntryP_MR"
        ] = row[
            "P_MeanReverting"
        ]

        record[
            "EntryP_Trend"
        ] = row[
            "P_Trending"
        ]

        record[
            "EntryKalmanVariance"
        ] = row[
            "KalmanVariance"
        ]

        records.append(
            record
        )

    result = pd.DataFrame(
        records
    )

    print(
        f"Missing entry timestamps: "
        f"{missing}"
    )

    return result


# ============================================================
# MAE / MFE
# ============================================================

def calculate_mae_mfe(
    trades,
    df,
):

    print()
    print(
        "Calculating MAE/MFE..."
    )

    maes = []
    mfes = []

    for _, trade in trades.iterrows():

        entry_idx = find_index(
            df,
            trade["EntryTime"],
        )

        exit_idx = find_index(
            df,
            trade["ExitTime"],
        )

        if (
            entry_idx is None
            or exit_idx is None
            or exit_idx < entry_idx
        ):

            maes.append(
                np.nan
            )

            mfes.append(
                np.nan
            )

            continue

        path = df.iloc[
            entry_idx:
            exit_idx + 1
        ]

        entry_price = float(
            trade["EntryPrice"]
        )

        side = str(
            trade["Side"]
        ).upper()

        # ----------------------------------------------------
        # LONG
        # ----------------------------------------------------

        if side == "LONG":

            mfe = (
                path["High"].max()
                / entry_price
                - 1.0
            )

            mae = (
                path["Low"].min()
                / entry_price
                - 1.0
            )

        # ----------------------------------------------------
        # SHORT
        # ----------------------------------------------------

        else:

            mfe = (
                entry_price
                / path["Low"].min()
                - 1.0
            )

            mae = (
                entry_price
                / path["High"].max()
                - 1.0
            )

        mfes.append(
            mfe
        )

        maes.append(
            mae
        )

    trades = trades.copy()

    trades["MFE"] = mfes
    trades["MAE"] = maes

    return trades


# ============================================================
# DERIVED COLUMNS
# ============================================================

def add_derived_columns(
    trades,
):

    trades = trades.copy()

    trades["Stopped"] = (
        trades[
            "ExitReason"
        ]
        .astype(str)
        .eq("Z_STOP")
    )

    trades["Winner"] = (
        trades[
            "GrossReturn"
        ]
        > 0
    )

    trades["Outcome"] = np.where(
        trades["Winner"],
        "WIN",
        "LOSS",
    )

    trades["AbsEntryZ"] = (
        trades[
            "EntryZ"
        ]
        .abs()
    )

    trades["EntryRegime"] = np.where(
        trades[
            "EntryP_MR"
        ]
        >= MR_PROBABILITY_THRESHOLD,
        "MR",
        "TREND",
    )

    trades["RegimeStop"] = (
        trades[
            "EntryRegime"
        ]
        .astype(str)
        + "_"
        + trades[
            "Stopped"
        ]
        .astype(str)
    )

    return trades


# ============================================================
# QUANTILE BUCKET
# ============================================================

def add_quantile_bucket(
    trades,
    source,
    destination,
):

    trades = trades.copy()

    valid = trades[
        source
    ].notna()

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Explicit object dtype prevents pandas from creating a
    # float64 column and then rejecting string categories.
    # --------------------------------------------------------

    trades[
        destination
    ] = pd.Series(
        pd.NA,
        index=trades.index,
        dtype="object",
    )

    if valid.sum() < 20:

        return trades

    try:

        buckets = pd.qcut(
            trades.loc[
                valid,
                source
            ],
            q=4,
            labels=[
                "Q1 Low",
                "Q2",
                "Q3",
                "Q4 High",
            ],
            duplicates="drop",
        )

        trades.loc[
            valid,
            destination
        ] = (
            buckets
            .astype(str)
            .to_numpy()
        )

    except ValueError:

        pass

    return trades


# ============================================================
# GROUP SUMMARY
# ============================================================

def make_summary(
    trades,
    group_column,
):

    rows = []

    for (
        value,
        group,
    ) in trades.groupby(
        group_column,
        dropna=False,
        observed=False,
    ):

        returns = (
            group[
                "GrossReturn"
            ]
            .dropna()
        )

        row = {
            group_column: value,

            "Trades": len(
                group
            ),

            "WinRate": (
                (returns > 0).mean()
                if len(returns)
                else np.nan
            ),

            "PF": (
                safe_pf(returns)
                if len(returns)
                else np.nan
            ),

            "MeanReturn": (
                returns.mean()
                if len(returns)
                else np.nan
            ),

            "MedianReturn": (
                returns.median()
                if len(returns)
                else np.nan
            ),

            "TotalPnL": (
                group[
                    "PnL"
                ].sum()
                if "PnL" in group
                else np.nan
            ),

            "MeanPnL": (
                group[
                    "PnL"
                ].mean()
                if "PnL" in group
                else np.nan
            ),

            "MeanMAE": (
                group[
                    "MAE"
                ].mean()
            ),

            "MedianMAE": (
                group[
                    "MAE"
                ].median()
            ),

            "MeanMFE": (
                group[
                    "MFE"
                ].mean()
            ),

            "MedianMFE": (
                group[
                    "MFE"
                ].median()
            ),

            "StopRate": (
                group[
                    "Stopped"
                ].mean()
            ),
        }

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# PRINT + SAVE
# ============================================================

def print_save(
    trades,
    column,
    filename,
    title,
):

    print_section(
        title
    )

    result = make_summary(
        trades,
        column,
    )

    print(
        result.to_string(
            index=False
        )
    )

    result.to_csv(
        OUTPUT_DIR
        / filename,
        index=False,
    )

    return result


# ============================================================
# GARCH VS OTHER VOLATILITY
# ============================================================

def compare_volatility_metrics(
    trades
):

    print_section(
        "GARCH VS REALIZED VOLATILITY"
    )

    comparison = []

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # These are TRADE dataframe columns.
    #
    # GARCH feature is called EntryGARCHVol, not GARCHVol.
    # --------------------------------------------------------

    metrics = [
        (
            "EntryGARCHVol",
            "GARCH",
        ),

        (
            "EntryRV24",
            "RV24",
        ),

        (
            "EntryRV168",
            "RV168",
        ),

        (
            "EntryVolRatio",
            "VolRatio24_168",
        ),

        (
            "EntryATRRatio",
            "ATRRatio",
        ),
    ]

    for (
        column,
        label,
    ) in metrics:

        if column not in trades.columns:

            print(
                f"Skipping {label}: "
                f"column '{column}' not found."
            )

            continue

        valid = trades[
            column
        ].notna()

        subset = trades[
            valid
        ].copy()

        if len(subset) < 20:

            print(
                f"Skipping {label}: "
                f"only {len(subset)} valid rows."
            )

            continue

        try:

            subset[
                "_bucket"
            ] = pd.qcut(
                subset[
                    column
                ],
                q=4,
                labels=[
                    "Q1",
                    "Q2",
                    "Q3",
                    "Q4",
                ],
                duplicates="drop",
            )

        except ValueError:

            print(
                f"Could not create quartiles "
                f"for {label}."
            )

            continue

        for (
            bucket,
            group,
        ) in subset.groupby(
            "_bucket",
            observed=False,
        ):

            comparison.append(
                {
                    "Metric": label,

                    "Bucket": str(
                        bucket
                    ),

                    "Trades": len(
                        group
                    ),

                    "StopRate": (
                        group[
                            "Stopped"
                        ].mean()
                    ),

                    "WinRate": (
                        group[
                            "Winner"
                        ].mean()
                    ),

                    "MeanReturn": (
                        group[
                            "GrossReturn"
                        ].mean()
                    ),

                    "MeanMAE": (
                        group[
                            "MAE"
                        ].mean()
                    ),

                    "MeanMFE": (
                        group[
                            "MFE"
                        ].mean()
                    ),

                    "TotalPnL": (
                        group[
                            "PnL"
                        ].sum()
                    ),

                    "MeanPnL": (
                        group[
                            "PnL"
                        ].mean()
                    ),
                }
            )

    comparison = pd.DataFrame(
        comparison
    )

    if len(comparison) > 0:

        print(
            comparison.to_string(
                index=False
            )
        )

        comparison.to_csv(
            OUTPUT_DIR
            / "garch_vs_realized_comparison.csv",
            index=False,
        )

    return comparison


# ============================================================
# CORRELATION ANALYSIS
# ============================================================

def calculate_correlations(
    trades
):

    print_section(
        "ENTRY VOLATILITY CORRELATIONS"
    )

    columns = [
        "EntryGARCHVol",
        "EntryRV24",
        "EntryRV168",
        "EntryVolRatio",
        "EntryATRRatio",
        "AbsEntryZ",
        "MAE",
        "MFE",
        "GrossReturn",
    ]

    columns = [
        column
        for column in columns
        if column in trades.columns
    ]

    correlations = (
        trades[
            columns
        ]
        .corr()
    )

    print(
        correlations.to_string()
    )

    correlations.to_csv(
        OUTPUT_DIR
        / "entry_volatility_correlations.csv"
    )

    return correlations


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # 1. LOAD DATA
    # ========================================================

    df = load_h1()

    print(
        f"Rows: "
        f"{len(df):,}"
    )

    print(
        f"Start: "
        f"{df['Timestamp'].min()}"
    )

    print(
        f"End: "
        f"{df['Timestamp'].max()}"
    )

    # ========================================================
    # 2. KALMAN
    # ========================================================

    df = add_kalman_features(
        df
    )

    # ========================================================
    # 3. BASIC VOLATILITY
    # ========================================================

    df = add_basic_volatility(
        df
    )

    # ========================================================
    # 4. GARCH
    # ========================================================

    df = calculate_causal_garch(
        df
    )

    # ========================================================
    # 5. MARKOV
    # ========================================================

    df = add_markov_features(
        df
    )

    # ========================================================
    # 6. LOAD CANONICAL TRADES
    # ========================================================

    trades = load_canonical_trades()

    trades = normalize_trades(
        trades
    )

    if len(trades) != EXPECTED_TRADES:

        raise ValueError(
            f"Expected exactly "
            f"{EXPECTED_TRADES} canonical "
            f"trades, got "
            f"{len(trades)}."
        )

    print(
        "Canonical trade count verified."
    )

    # ========================================================
    # 7. ATTACH FEATURES
    # ========================================================

    trades = attach_entry_features(
        trades,
        df,
    )

    # ========================================================
    # 8. MAE / MFE
    # ========================================================

    trades = calculate_mae_mfe(
        trades,
        df,
    )

    # ========================================================
    # 9. DERIVED COLUMNS
    # ========================================================

    trades = add_derived_columns(
        trades
    )

    # ========================================================
    # 10. OVERALL
    # ========================================================

    print_section(
        "OVERALL GARCH DIAGNOSTIC"
    )

    print(
        f"Trades: "
        f"{len(trades):,}"
    )

    print(
        f"Win rate: "
        f"{trades['Winner'].mean() * 100:.2f}%"
    )

    print(
        f"Mean return: "
        f"{trades['GrossReturn'].mean() * 100:.4f}%"
    )

    print(
        f"Mean MFE: "
        f"{trades['MFE'].mean() * 100:.4f}%"
    )

    print(
        f"Mean MAE: "
        f"{trades['MAE'].mean() * 100:.4f}%"
    )

    print(
        f"Stop rate: "
        f"{trades['Stopped'].mean() * 100:.2f}%"
    )

    print(
        f"Valid GARCH entry forecasts: "
        f"{trades['EntryGARCHVol'].notna().sum():,}"
    )

    # ========================================================
    # 11. EXIT REASON
    # ========================================================

    print_save(
        trades,
        "ExitReason",
        "exit_summary.csv",
        "EXIT REASON",
    )

    # ========================================================
    # 12. STOPPED VS NON-STOPPED
    # ========================================================

    print_save(
        trades,
        "Stopped",
        "stop_summary.csv",
        "STOPPED VS NON-STOPPED",
    )

    # ========================================================
    # 13. WINNERS VS LOSERS
    # ========================================================

    print_save(
        trades,
        "Outcome",
        "outcome_summary.csv",
        "WINNERS VS LOSERS",
    )

    # ========================================================
    # 14. GARCH QUARTILES
    # ========================================================

    trades = add_quantile_bucket(
        trades,
        "EntryGARCHVol",
        "GARCHVolBucket",
    )

    print_save(
        trades,
        "GARCHVolBucket",
        "garch_vol_summary.csv",
        "GARCH FORECAST VOLATILITY",
    )

    # ========================================================
    # 15. GARCH × STOP
    # ========================================================

    trades["GARCHStop"] = (
        trades[
            "GARCHVolBucket"
        ].astype(str)
        + "_"
        + trades[
            "Stopped"
        ].astype(str)
    )

    print_save(
        trades,
        "GARCHStop",
        "garch_stop_summary.csv",
        "GARCH VOLATILITY × STOP STATUS",
    )

    # ========================================================
    # 16. GARCH × OUTCOME
    # ========================================================

    trades["GARCHOutcome"] = (
        trades[
            "GARCHVolBucket"
        ].astype(str)
        + "_"
        + trades[
            "Outcome"
        ].astype(str)
    )

    print_save(
        trades,
        "GARCHOutcome",
        "garch_outcome_summary.csv",
        "GARCH VOLATILITY × OUTCOME",
    )

    # ========================================================
    # 17. ENTRY Z QUARTILES
    # ========================================================

    trades = add_quantile_bucket(
        trades,
        "AbsEntryZ",
        "EntryZQuantile",
    )

    # ========================================================
    # 18. GARCH × ENTRY Z
    # ========================================================

    trades["GARCHZ"] = (
        trades[
            "GARCHVolBucket"
        ].astype(str)
        + "_"
        + trades[
            "EntryZQuantile"
        ].astype(str)
    )

    print_save(
        trades,
        "GARCHZ",
        "garch_z_summary.csv",
        "GARCH VOLATILITY × ENTRY Z",
    )

    # ========================================================
    # 19. GARCH × MARKOV REGIME
    # ========================================================

    trades["GARCHRegime"] = (
        trades[
            "GARCHVolBucket"
        ].astype(str)
        + "_"
        + trades[
            "EntryRegime"
        ].astype(str)
    )

    print_save(
        trades,
        "GARCHRegime",
        "garch_regime_summary.csv",
        "GARCH VOLATILITY × MARKOV REGIME",
    )

    # ========================================================
    # 20. GARCH × DIRECTION
    # ========================================================

    trades["GARCHDirection"] = (
        trades[
            "GARCHVolBucket"
        ].astype(str)
        + "_"
        + trades[
            "Side"
        ].astype(str)
    )

    print_save(
        trades,
        "GARCHDirection",
        "garch_direction_summary.csv",
        "GARCH VOLATILITY × DIRECTION",
    )

    # ========================================================
    # 21. SAVE INTERMEDIATE RESULTS
    #
    # This is intentionally BEFORE the comparison section.
    # If a later analysis crashes, we still have the expensive
    # GARCH results.
    # ========================================================

    trades.to_csv(
        OUTPUT_DIR
        / "canonical_trades_garch_diagnostic.csv",
        index=False,
    )

    df.to_csv(
        OUTPUT_DIR
        / "processed_garch_features.csv",
        index=False,
    )

    print()
    print(
        "Intermediate GARCH results saved."
    )

    # ========================================================
    # 22. GARCH VS REALIZED VOLATILITY
    # ========================================================

    compare_volatility_metrics(
        trades
    )

    # ========================================================
    # 23. GARCH × EXIT REASON
    # ========================================================

    print_section(
        "GARCH VOLATILITY × EXIT REASON"
    )

    temp = trades.copy()

    temp["GARCHExit"] = (
        temp[
            "GARCHVolBucket"
        ].astype(str)
        + "_"
        + temp[
            "ExitReason"
        ].astype(str)
    )

    print_save(
        temp,
        "GARCHExit",
        "garch_exit_summary.csv",
        "GARCH VOLATILITY × EXIT REASON",
    )

    # ========================================================
    # 24. MAE / MFE BY GARCH BUCKET
    # ========================================================

    print_section(
        "MAE / MFE BY GARCH BUCKET"
    )

    mae_mfe = (
        trades
        .groupby(
            "GARCHVolBucket",
            observed=False,
        )
        .agg(
            Trades=(
                "GrossReturn",
                "size",
            ),

            MeanMAE=(
                "MAE",
                "mean",
            ),

            MedianMAE=(
                "MAE",
                "median",
            ),

            MeanMFE=(
                "MFE",
                "mean",
            ),

            MedianMFE=(
                "MFE",
                "median",
            ),

            MeanReturn=(
                "GrossReturn",
                "mean",
            ),

            StopRate=(
                "Stopped",
                "mean",
            ),
        )
        .reset_index()
    )

    print(
        mae_mfe.to_string(
            index=False
        )
    )

    mae_mfe.to_csv(
        OUTPUT_DIR
        / "garch_mae_mfe.csv",
        index=False,
    )

    # ========================================================
    # 25. YEARLY
    # ========================================================

    trades["Year"] = (
        trades[
            "EntryTime"
        ].dt.year
    )

    yearly = (
        trades
        .groupby("Year")
        .agg(
            Trades=(
                "GrossReturn",
                "size",
            ),

            WinRate=(
                "Winner",
                "mean",
            ),

            MeanReturn=(
                "GrossReturn",
                "mean",
            ),

            TotalPnL=(
                "PnL",
                "sum",
            ),

            StopRate=(
                "Stopped",
                "mean",
            ),

            MeanGARCHVol=(
                "EntryGARCHVol",
                "mean",
            ),

            MeanMAE=(
                "MAE",
                "mean",
            ),

            MeanMFE=(
                "MFE",
                "mean",
            ),
        )
        .reset_index()
    )

    print_section(
        "YEARLY GARCH DIAGNOSTIC"
    )

    print(
        yearly.to_string(
            index=False
        )
    )

    yearly.to_csv(
        OUTPUT_DIR
        / "yearly_garch_summary.csv",
        index=False,
    )

    # ========================================================
    # 26. CORRELATIONS
    # ========================================================

    calculate_correlations(
        trades
    )

    # ========================================================
    # 27. FINAL SAVE
    # ========================================================

    trades.to_csv(
        OUTPUT_DIR
        / "canonical_trades_garch_diagnostic.csv",
        index=False,
    )

    df.to_csv(
        OUTPUT_DIR
        / "processed_garch_features.csv",
        index=False,
    )

    # ========================================================
    # FINAL
    # ========================================================

    print_section(
        "FILES SAVED"
    )

    print(
        OUTPUT_DIR.resolve()
    )

    print()

    print(
        f"Canonical trades analyzed: "
        f"{len(trades):,}"
    )

    print(
        "GARCH diagnostic complete."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()