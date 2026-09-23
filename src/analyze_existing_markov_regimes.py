from pathlib import Path
import warnings
import os
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


# ============================================================
# WARNINGS
# ============================================================

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = Path("data/mt5_h1")

# These are the EXISTING Markov trade files.
# We are NOT generating a new trade universe.
TRADE_DIR = Path("results/kalman_markov_v2")

OUTPUT_DIR = Path(
    "results/instrument_regime_diagnostic"
)

SYMBOLS = [
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
# MARKOV V2 SETTINGS
# ============================================================

HMM_WINDOW = 1000
MIN_OBSERVATIONS = 300

N_COMPONENTS = 2

# Same general HMM architecture used previously.
N_ITER = 100
TOL = 1e-4
RANDOM_STATE = 42

# Existing Markov classification threshold.
MR_THRESHOLD = 0.60


# ============================================================
# LOAD H1 DATA
# ============================================================

def load_h1(symbol: str) -> pd.DataFrame:

    path = DATA_DIR / f"{symbol}_H1_2019_2026.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing H1 file: {path}"
        )

    df = pd.read_csv(path)

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    timestamp_candidates = [
        "Timestamp",
        "Time",
        "Datetime",
        "DateTime",
        "Date",
    ]

    time_col = None

    for candidate in timestamp_candidates:

        if candidate in df.columns:
            time_col = candidate
            break

    if time_col is None:

        raise ValueError(
            f"{symbol}: no timestamp column found.\n"
            f"Columns: {list(df.columns)}"
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
        df.sort_values(time_col)
        .drop_duplicates(subset=[time_col])
        .set_index(time_col)
    )

    # --------------------------------------------------------
    # OHLC
    # --------------------------------------------------------

    required = [
        "Open",
        "High",
        "Low",
        "Close",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"{symbol}: missing OHLC columns: {missing}"
        )

    for column in required:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=required
    )

    return df


# ============================================================
# BUILD HMM FEATURES
#
# Same V2 feature concept:
#
#   1. LogReturn
#   2. Volatility24
#   3. Autocorr24
# ============================================================

def build_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    close = pd.to_numeric(
        df["Close"],
        errors="coerce",
    )

    # --------------------------------------------------------
    # Log return
    # --------------------------------------------------------

    log_return = np.log(
        close / close.shift(1)
    )

    # --------------------------------------------------------
    # 24-hour volatility
    # --------------------------------------------------------

    volatility24 = (
        log_return
        .rolling(24)
        .std()
    )

    # --------------------------------------------------------
    # Rolling lag-1 autocorrelation
    # --------------------------------------------------------

    autocorr24 = (
        log_return
        .rolling(24)
        .corr(
            log_return.shift(1)
        )
    )

    features = pd.DataFrame(
        {
            "LogReturn": log_return,
            "Volatility24": volatility24,
            "Autocorr24": autocorr24,
        },
        index=df.index,
    )

    return features


# ============================================================
# CLASSIFY HMM STATES
#
# Primary:
#     lower autocorrelation = MR
#
# Tie-break:
#     lower volatility = MR
# ============================================================

def classify_states(
    model,
    scaler,
    train_features: pd.DataFrame,
):

    feature_columns = [
        "LogReturn",
        "Volatility24",
        "Autocorr24",
    ]

    X = train_features[
        feature_columns
    ].values

    X_scaled = scaler.transform(X)

    states = model.predict(X_scaled)

    state_stats = []

    for state in range(
        N_COMPONENTS
    ):

        mask = (
            states == state
        )

        if mask.sum() == 0:

            state_stats.append(
                {
                    "state": state,
                    "autocorr": np.nan,
                    "volatility": np.nan,
                }
            )

            continue

        state_data = train_features.iloc[
            mask
        ]

        state_stats.append(
            {
                "state": state,
                "autocorr": state_data[
                    "Autocorr24"
                ].mean(),
                "volatility": state_data[
                    "Volatility24"
                ].mean(),
            }
        )

    stats_df = pd.DataFrame(
        state_stats
    )

    # --------------------------------------------------------
    # Determine MR state
    # --------------------------------------------------------

    autocorr_values = (
        stats_df["autocorr"]
        .dropna()
    )

    if len(autocorr_values) == 0:

        raise RuntimeError(
            "Unable to classify HMM states."
        )

    autocorr_diff = (
        autocorr_values.max()
        - autocorr_values.min()
    )

    if autocorr_diff >= 0.005:

        mr_state = int(
            stats_df.loc[
                stats_df["autocorr"].idxmin(),
                "state",
            ]
        )

    else:

        # Tie-break:
        # lower volatility = MR
        mr_state = int(
            stats_df.loc[
                stats_df["volatility"].idxmin(),
                "state",
            ]
        )

    trend_state = (
        1 - mr_state
    )

    return (
        mr_state,
        trend_state,
        stats_df,
    )


# ============================================================
# FIT HMM AT ONE ENTRY
#
# IMPORTANT:
# We only use data available UP TO the entry timestamp.
#
# Therefore this remains causal.
# ============================================================

def regime_at_entry(
    features: pd.DataFrame,
    entry_time,
):

    # --------------------------------------------------------
    # Only information available at entry.
    # --------------------------------------------------------

    available = (
        features
        .loc[:entry_time]
        .dropna()
    )

    if len(available) < MIN_OBSERVATIONS:

        return {
            "EntryPMR": np.nan,
            "EntryPTrend": np.nan,
            "EntryRegime": "UNKNOWN",
        }

    # --------------------------------------------------------
    # Rolling training window
    # --------------------------------------------------------

    train = available.iloc[
        -HMM_WINDOW:
    ]

    if len(train) < MIN_OBSERVATIONS:

        return {
            "EntryPMR": np.nan,
            "EntryPTrend": np.nan,
            "EntryRegime": "UNKNOWN",
        }

    feature_columns = [
        "LogReturn",
        "Volatility24",
        "Autocorr24",
    ]

    X = train[
        feature_columns
    ].values

    # --------------------------------------------------------
    # Standardize using ONLY training data.
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X
    )

    # --------------------------------------------------------
    # HMM
    # --------------------------------------------------------

    model = GaussianHMM(
        n_components=N_COMPONENTS,
        covariance_type="diag",
        n_iter=N_ITER,
        tol=TOL,
        random_state=RANDOM_STATE,
        min_covar=1e-6,
    )

    # hmmlearn prints convergence messages directly.
    # Suppress them because tiny convergence deltas are
    # expected in this diagnostic.
    with open(
        os.devnull,
        "w",
    ) as devnull:

        with redirect_stdout(
            devnull
        ):

            model.fit(
                X_scaled
            )

    # --------------------------------------------------------
    # Determine which state is MR.
    # --------------------------------------------------------

    (
        mr_state,
        trend_state,
        _,
    ) = classify_states(
        model,
        scaler,
        train,
    )

    # --------------------------------------------------------
    # Entry observation
    # --------------------------------------------------------

    entry_row = features.loc[
        [entry_time]
    ][feature_columns]

    if entry_row.isna().any().any():

        return {
            "EntryPMR": np.nan,
            "EntryPTrend": np.nan,
            "EntryRegime": "UNKNOWN",
        }

    entry_X = entry_row.values

    entry_scaled = scaler.transform(
        entry_X
    )

    # --------------------------------------------------------
    # Posterior probability at ENTRY
    # --------------------------------------------------------

    probabilities = model.predict_proba(
        entry_scaled
    )[0]

    p_mr = float(
        probabilities[mr_state]
    )

    p_trend = float(
        probabilities[trend_state]
    )

    # --------------------------------------------------------
    # Regime classification
    # --------------------------------------------------------

    if p_mr >= MR_THRESHOLD:

        regime = "MR"

    else:

        regime = "TREND"

    return {
        "EntryPMR": p_mr,
        "EntryPTrend": p_trend,
        "EntryRegime": regime,
    }


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    group: pd.DataFrame,
):

    pnl = pd.to_numeric(
        group["PnL"],
        errors="coerce",
    ).dropna()

    returns = pd.to_numeric(
        group["Return"],
        errors="coerce",
    ).dropna()

    if len(pnl) == 0:

        return {
            "Trades": 0,
            "WinRate": np.nan,
            "ProfitFactor": np.nan,
            "ExpectancyPnL": np.nan,
            "TotalPnL": np.nan,
            "AvgReturn": np.nan,
            "MedianReturn": np.nan,
            "StopRate": np.nan,
        }

    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    gross_profit = wins.sum()

    gross_loss = abs(
        losses.sum()
    )

    if gross_loss > 0:

        profit_factor = (
            gross_profit
            / gross_loss
        )

    else:

        profit_factor = np.inf

    stop_rate = np.nan

    if "ExitReason" in group.columns:

        stop_rate = (
            group["ExitReason"]
            .eq("Z_STOP")
            .mean()
        )

    return {
        "Trades": len(pnl),
        "WinRate": (
            pnl > 0
        ).mean(),
        "ProfitFactor": profit_factor,
        "ExpectancyPnL": pnl.mean(),
        "TotalPnL": pnl.sum(),
        "AvgReturn": (
            returns.mean()
            if len(returns)
            else np.nan
        ),
        "MedianReturn": (
            returns.median()
            if len(returns)
            else np.nan
        ),
        "StopRate": stop_rate,
    }


# ============================================================
# PROCESS ONE SYMBOL
# ============================================================

def process_symbol(
    symbol: str,
):

    print()
    print("=" * 80)
    print(symbol)
    print("=" * 80)

    # --------------------------------------------------------
    # Existing trade file
    # --------------------------------------------------------

    trade_path = (
        TRADE_DIR
        / symbol
        / "kalman_markov_trades.csv"
    )

    if not trade_path.exists():

        print(
            f"Missing trade file:\n"
            f"{trade_path}"
        )

        return None

    trades = pd.read_csv(
        trade_path
    )

    if trades.empty:

        print("No trades.")

        return None

    # --------------------------------------------------------
    # Entry timestamp
    # --------------------------------------------------------

    if "EntryTime" not in trades.columns:

        raise ValueError(
            f"{symbol}: trade file has no EntryTime."
        )

    trades["EntryTime"] = pd.to_datetime(
        trades["EntryTime"],
        utc=True,
        errors="coerce",
    )

    trades = trades.dropna(
        subset=["EntryTime"]
    ).copy()

    trades = trades.sort_values(
        "EntryTime"
    ).reset_index(
        drop=True
    )

    print(
        f"Existing trades: "
        f"{len(trades):,}"
    )

    # --------------------------------------------------------
    # H1
    # --------------------------------------------------------

    h1 = load_h1(
        symbol
    )

    print(
        f"H1 bars: "
        f"{len(h1):,}"
    )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    features = build_features(
        h1
    )

    # --------------------------------------------------------
    # Unique entry timestamps
    # --------------------------------------------------------

    entry_times = (
        trades["EntryTime"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    print(
        f"Unique entry times: "
        f"{len(entry_times):,}"
    )

    print(
        "Calculating causal regime "
        "at entries..."
    )

    # --------------------------------------------------------
    # Calculate regime
    # --------------------------------------------------------

    regime_records = []

    for i, entry_time in enumerate(
        entry_times,
        start=1,
    ):

        result = regime_at_entry(
            features,
            entry_time,
        )

        result[
            "EntryTime"
        ] = entry_time

        regime_records.append(
            result
        )

        if (
            i % 100 == 0
            or i == len(entry_times)
        ):

            print(
                f"  {i:,}/"
                f"{len(entry_times):,}"
            )

    regime_df = pd.DataFrame(
        regime_records
    )

    # --------------------------------------------------------
    # Merge with EXISTING trades
    #
    # IMPORTANT:
    # The original trade files do not have
    # EntryPMR / EntryPTrend / EntryRegime,
    # so no "_New" suffix exists.
    # --------------------------------------------------------

    trades = trades.merge(
        regime_df,
        on="EntryTime",
        how="left",
    )

    # Rename the diagnostic columns
    trades.rename(
        columns={
            "EntryPMR":
                "EntryPMR_Diagnostic",

            "EntryPTrend":
                "EntryPTrend_Diagnostic",

            "EntryRegime":
                "EntryRegime_Diagnostic",
        },
        inplace=True,
    )

    # --------------------------------------------------------
    # Save enriched trade file
    # --------------------------------------------------------

    symbol_dir = (
        OUTPUT_DIR
        / symbol
    )

    symbol_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    trade_output = (
        symbol_dir
        / "markov_trades_with_regime.csv"
    )

    trades.to_csv(
        trade_output,
        index=False,
    )

    # --------------------------------------------------------
    # Valid regime observations
    # --------------------------------------------------------

    valid = trades[
        trades[
            "EntryRegime_Diagnostic"
        ].isin(
            ["MR", "TREND"]
        )
    ].copy()

    # --------------------------------------------------------
    # Direction × Regime
    # --------------------------------------------------------

    rows = []

    for (
        regime,
        direction,
    ), group in valid.groupby(
        [
            "EntryRegime_Diagnostic",
            "Side",
        ],
        dropna=False,
    ):

        metrics = calculate_metrics(
            group
        )

        row = {
            "Symbol": symbol,
            "Regime": regime,
            "Direction": direction,
        }

        row.update(
            metrics
        )

        rows.append(
            row
        )

    breakdown = pd.DataFrame(
        rows
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    if not breakdown.empty:

        regime_order = {
            "MR": 0,
            "TREND": 1,
        }

        direction_order = {
            "LONG": 0,
            "SHORT": 1,
        }

        breakdown[
            "_RegimeOrder"
        ] = breakdown[
            "Regime"
        ].map(
            regime_order
        )

        breakdown[
            "_DirectionOrder"
        ] = breakdown[
            "Direction"
        ].map(
            direction_order
        )

        breakdown = (
            breakdown
            .sort_values(
                [
                    "_RegimeOrder",
                    "_DirectionOrder",
                ]
            )
            .drop(
                columns=[
                    "_RegimeOrder",
                    "_DirectionOrder",
                ]
            )
            .reset_index(
                drop=True
            )
        )

    # --------------------------------------------------------
    # Save breakdown
    # --------------------------------------------------------

    breakdown_path = (
        symbol_dir
        / "direction_regime_breakdown.csv"
    )

    breakdown.to_csv(
        breakdown_path,
        index=False,
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print()
    print(
        "Direction × Regime"
    )
    print("-" * 80)

    if breakdown.empty:

        print(
            "No valid regime "
            "classifications."
        )

    else:

        display_columns = [
            "Regime",
            "Direction",
            "Trades",
            "WinRate",
            "ProfitFactor",
            "ExpectancyPnL",
            "TotalPnL",
            "AvgReturn",
            "StopRate",
        ]

        display = breakdown[
            display_columns
        ].copy()

        # Make percentages easier to read
        display["WinRate"] = (
            display["WinRate"] * 100
        )

        display["AvgReturn"] = (
            display["AvgReturn"] * 100
        )

        display["StopRate"] = (
            display["StopRate"] * 100
        )

        print(
            display.to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.4f}",
            )
        )

    print()
    print(
        f"Saved:\n"
        f"  {trade_output}\n"
        f"  {breakdown_path}"
    )

    return breakdown


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_results = []

    print("=" * 80)
    print(
        "ENTRY-ONLY MARKOV REGIME DIAGNOSTIC"
    )
    print("=" * 80)

    print()
    print(
        "Existing trades are used as the "
        "authoritative trade universe."
    )

    print(
        "The HMM is evaluated only at "
        "existing entry timestamps."
    )

    print(
        "No new trades are generated."
    )

    print()

    # --------------------------------------------------------
    # All instruments
    # --------------------------------------------------------

    for symbol in SYMBOLS:

        try:

            result = process_symbol(
                symbol
            )

            if result is not None:

                all_results.append(
                    result
                )

        except Exception as exc:

            print()
            print(
                f"ERROR processing "
                f"{symbol}:"
            )

            print(
                repr(exc)
            )

    # --------------------------------------------------------
    # Combined result
    # --------------------------------------------------------

    if all_results:

        combined = pd.concat(
            all_results,
            ignore_index=True,
        )

        combined_path = (
            OUTPUT_DIR
            / "ALL_INSTRUMENTS_DIRECTION_REGIME.csv"
        )

        combined.to_csv(
            combined_path,
            index=False,
        )

        print()
        print("=" * 80)
        print(
            "ALL INSTRUMENTS"
        )
        print("=" * 80)

        display = combined.copy()

        display["WinRate"] = (
            display["WinRate"] * 100
        )

        display["AvgReturn"] = (
            display["AvgReturn"] * 100
        )

        display["StopRate"] = (
            display["StopRate"] * 100
        )

        print(
            display.to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.4f}",
            )
        )

        print()
        print(
            f"Combined result saved to:\n"
            f"{combined_path}"
        )

    else:

        print()
        print(
            "No instrument results "
            "were generated."
        )

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()