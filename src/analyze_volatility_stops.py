from pathlib import Path

import numpy as np
import pandas as pd

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


# ------------------------------------------------------------
# Locked Kalman parameters
# ------------------------------------------------------------

PHI = 0.999
Q = 0.25
R = 0.25

MEAN_WINDOW = 168
Z_WINDOW = 36

ENTRY_Z = 2.25
EXIT_Z = 0.25
STOP_Z = 3.0
MAX_HOLD_HOURS = 24


# ------------------------------------------------------------
# Markov parameters
# ------------------------------------------------------------

MR_PROBABILITY_THRESHOLD = 0.60


# ------------------------------------------------------------
# XAUUSDm point size
# ------------------------------------------------------------

POINT_SIZE = 0.001


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = Path(
    "results/volatility_stop_diagnostic"
)


# ============================================================
# HELPERS
# ============================================================

def print_section(title):

    print()
    print("=" * 110)
    print(title)
    print("=" * 110)


def safe_pf(returns):

    gross_profit = returns[returns > 0].sum()

    gross_loss = abs(
        returns[returns <= 0].sum()
    )

    if gross_loss <= 0:
        return np.inf

    return gross_profit / gross_loss


# ============================================================
# LOAD H1 DATA
# ============================================================

def load_h1():

    print("Loading H1 data...")

    df = pd.read_csv(DATA_PATH)

    required = {
        "Timestamp",
        "Open",
        "High",
        "Low",
        "Close",
        "SpreadPoints",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns in H1 data: {missing}"
        )

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        utc=True,
    )

    df = df.sort_values(
        "Timestamp"
    ).reset_index(drop=True)

    df = df[
        (df["Timestamp"] >= START_DATE)
        & (df["Timestamp"] <= END_DATE)
    ].copy()

    df = df.reset_index(drop=True)

    # --------------------------------------------------------
    # Reconstruct approximate Bid / Ask
    # --------------------------------------------------------

    spread_price = (
        df["SpreadPoints"] * POINT_SIZE
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
# VOLATILITY FEATURES
# ============================================================

def add_volatility_features(df):

    print(
        "Calculating volatility features..."
    )

    log_return = np.log(
        df["Close"]
        / df["Close"].shift(1)
    )

    df["LogReturn"] = log_return

    # --------------------------------------------------------
    # Realized volatility
    # --------------------------------------------------------

    df["RV_6"] = (
        log_return
        .rolling(6)
        .std()
    )

    df["RV_12"] = (
        log_return
        .rolling(12)
        .std()
    )

    df["RV_24"] = (
        log_return
        .rolling(24)
        .std()
    )

    df["RV_48"] = (
        log_return
        .rolling(48)
        .std()
    )

    df["RV_168"] = (
        log_return
        .rolling(168)
        .std()
    )

    # --------------------------------------------------------
    # Volatility ratios
    # --------------------------------------------------------

    df["VolRatio_24_168"] = (
        df["RV_24"]
        / df["RV_168"]
    )

    df["VolRatio_6_24"] = (
        df["RV_6"]
        / df["RV_24"]
    )

    # --------------------------------------------------------
    # Absolute return / shock
    # --------------------------------------------------------

    df["AbsReturn"] = (
        log_return.abs()
    )

    # --------------------------------------------------------
    # True Range
    # --------------------------------------------------------

    previous_close = (
        df["Close"].shift(1)
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
        [tr1, tr2, tr3],
        axis=1,
    ).max(axis=1)

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
# MARKOV FEATURES
# ============================================================

def add_markov_features(df):

    print(
        "Calculating causal Markov regime..."
    )

    df = add_markov_regime(df)

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
            f"{CANONICAL_TRADES_PATH}\n\n"
            "Run the canonical walk-forward "
            "directional backtest first."
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
# NORMALIZE TRADE COLUMNS
# ============================================================

def normalize_trade_columns(trades):

    # --------------------------------------------------------
    # Timestamp columns
    # --------------------------------------------------------

    timestamp_columns = [
        "EntryTime",
        "ExitTime",
        "EntrySignalTime",
    ]

    for col in timestamp_columns:

        if col in trades.columns:

            trades[col] = pd.to_datetime(
                trades[col],
                utc=True,
            )

    # --------------------------------------------------------
    # Basic numeric columns
    # --------------------------------------------------------

    numeric_columns = [
        "EntryPrice",
        "ExitPrice",
        "EntryZ",
        "ExitZ",
        "Pnl",
        "PnL",
        "Return",
        "GrossReturn",
        "TradingHoursHeld",
        "CalendarHoursHeld",
        "HoursHeld",
        "EntryP_MR",
        "EntryP_Trend",
        "P_MeanReverting",
        "P_Trending",
    ]

    for col in numeric_columns:

        if col in trades.columns:

            trades[col] = pd.to_numeric(
                trades[col],
                errors="coerce",
            )

    # --------------------------------------------------------
    # Normalize PnL
    # --------------------------------------------------------

    if (
        "PnL" not in trades.columns
        and "Pnl" in trades.columns
    ):

        trades["PnL"] = trades["Pnl"]

    # --------------------------------------------------------
    # Normalize return
    # --------------------------------------------------------

    if (
        "GrossReturn" not in trades.columns
        and "Return" in trades.columns
    ):

        trades["GrossReturn"] = trades[
            "Return"
        ]

    # --------------------------------------------------------
    # Normalize holding hours
    # --------------------------------------------------------

    if (
        "TradingHoursHeld"
        not in trades.columns
    ):

        if "HoursHeld" in trades.columns:

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

    # --------------------------------------------------------
    # Exit reason
    # --------------------------------------------------------

    if "ExitReason" not in trades.columns:

        raise ValueError(
            "Canonical trade file does not "
            "contain ExitReason."
        )

    # --------------------------------------------------------
    # Side
    # --------------------------------------------------------

    if "Side" not in trades.columns:

        raise ValueError(
            "Canonical trade file does not "
            "contain Side."
        )

    return trades


# ============================================================
# MAP TIMESTAMP -> H1 INDEX
# ============================================================

def make_timestamp_index(df):

    return pd.Series(
        df.index,
        index=df["Timestamp"],
    )


# ============================================================
# GET EXACT ENTRY ROW
# ============================================================

def find_entry_index(
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

    return int(matches[0])


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
        "volatility/regime features..."
    )

    records = []

    missing_entries = 0

    for _, trade in trades.iterrows():

        entry_time = trade[
            "EntryTime"
        ]

        idx = find_entry_index(
            df,
            entry_time,
        )

        record = trade.to_dict()

        if idx is None:

            missing_entries += 1

            for col in [
                "EntryRV_6",
                "EntryRV_12",
                "EntryRV_24",
                "EntryRV_48",
                "EntryRV_168",
                "EntryVolRatio_24_168",
                "EntryVolRatio_6_24",
                "EntryATR24",
                "EntryATR168",
                "EntryATRRatio",
                "EntryAbsReturn",
                "EntryResidual",
                "EntryKalmanVariance",
                "EntryZ_Recheck",
                "EntryP_MR_Recheck",
                "EntryP_Trend_Recheck",
            ]:

                record[col] = np.nan

            records.append(record)

            continue

        row = df.iloc[idx]

        # ----------------------------------------------------
        # Entry features
        # ----------------------------------------------------

        record[
            "EntryRV_6"
        ] = row["RV_6"]

        record[
            "EntryRV_12"
        ] = row["RV_12"]

        record[
            "EntryRV_24"
        ] = row["RV_24"]

        record[
            "EntryRV_48"
        ] = row["RV_48"]

        record[
            "EntryRV_168"
        ] = row["RV_168"]

        record[
            "EntryVolRatio_24_168"
        ] = row["VolRatio_24_168"]

        record[
            "EntryVolRatio_6_24"
        ] = row["VolRatio_6_24"]

        record[
            "EntryATR24"
        ] = row["ATR_24"]

        record[
            "EntryATR168"
        ] = row["ATR_168"]

        record[
            "EntryATRRatio"
        ] = row["ATR_Ratio"]

        record[
            "EntryAbsReturn"
        ] = row["AbsReturn"]

        record[
            "EntryResidual"
        ] = row["Residual"]

        record[
            "EntryKalmanVariance"
        ] = row["KalmanVariance"]

        record[
            "EntryZ_Recheck"
        ] = row["Z"]

        record[
            "EntryP_MR_Recheck"
        ] = row["P_MeanReverting"]

        record[
            "EntryP_Trend_Recheck"
        ] = row["P_Trending"]

        records.append(record)

    result = pd.DataFrame(
        records
    )

    print(
        f"Missing entry timestamps: "
        f"{missing_entries}"
    )

    if missing_entries > 0:

        print(
            "WARNING: Some canonical "
            "entries could not be mapped "
            "to H1 bars."
        )

    return result


# ============================================================
# CALCULATE MAE / MFE
# ============================================================

def calculate_mae_mfe(
    trades,
    df,
):

    print()
    print(
        "Calculating MAE/MFE from "
        "canonical trade windows..."
    )

    mae_values = []
    mfe_values = []

    max_adverse_price_values = []
    max_favorable_price_values = []

    path_missing = 0

    for _, trade in trades.iterrows():

        entry_time = trade[
            "EntryTime"
        ]

        exit_time = trade[
            "ExitTime"
        ]

        side = str(
            trade["Side"]
        ).upper()

        entry_price = float(
            trade["EntryPrice"]
        )

        entry_idx = find_entry_index(
            df,
            entry_time,
        )

        exit_idx = find_entry_index(
            df,
            exit_time,
        )

        if (
            entry_idx is None
            or exit_idx is None
            or exit_idx < entry_idx
        ):

            mae_values.append(np.nan)
            mfe_values.append(np.nan)

            max_adverse_price_values.append(
                np.nan
            )

            max_favorable_price_values.append(
                np.nan
            )

            path_missing += 1

            continue

        path = df.iloc[
            entry_idx:exit_idx + 1
        ]

        if side == "LONG":

            favorable_prices = (
                path["High"]
            )

            adverse_prices = (
                path["Low"]
            )

            mfe = (
                favorable_prices.max()
                / entry_price
                - 1.0
            )

            mae = (
                adverse_prices.min()
                / entry_price
                - 1.0
            )

            max_favorable_price = (
                favorable_prices.max()
            )

            max_adverse_price = (
                adverse_prices.min()
            )

        elif side == "SHORT":

            favorable_prices = (
                path["Low"]
            )

            adverse_prices = (
                path["High"]
            )

            mfe = (
                entry_price
                / favorable_prices.min()
                - 1.0
            )

            mae = (
                entry_price
                / adverse_prices.max()
                - 1.0
            )

            max_favorable_price = (
                favorable_prices.min()
            )

            max_adverse_price = (
                adverse_prices.max()
            )

        else:

            mfe = np.nan
            mae = np.nan

            max_favorable_price = np.nan
            max_adverse_price = np.nan

        mae_values.append(mae)
        mfe_values.append(mfe)

        max_adverse_price_values.append(
            max_adverse_price
        )

        max_favorable_price_values.append(
            max_favorable_price
        )

    trades = trades.copy()

    trades["MAE"] = mae_values

    trades["MFE"] = mfe_values

    trades[
        "MaxAdversePrice"
    ] = max_adverse_price_values

    trades[
        "MaxFavorablePrice"
    ] = max_favorable_price_values

    print(
        f"Trades with missing paths: "
        f"{path_missing}"
    )

    return trades


# ============================================================
# CREATE DERIVED COLUMNS
# ============================================================

def add_derived_trade_columns(
    trades,
):

    trades = trades.copy()

    # --------------------------------------------------------
    # Winner / loser
    # --------------------------------------------------------

    if "GrossReturn" in trades.columns:

        trades["Winner"] = (
            trades["GrossReturn"] > 0
        )

    elif "PnL" in trades.columns:

        trades["Winner"] = (
            trades["PnL"] > 0
        )

    else:

        raise ValueError(
            "No GrossReturn or PnL "
            "column found."
        )

    trades["Outcome"] = np.where(
        trades["Winner"],
        "WIN",
        "LOSS",
    )

    # --------------------------------------------------------
    # Stop flag
    # --------------------------------------------------------

    trades["Stopped"] = (
        trades["ExitReason"]
        .astype(str)
        .eq("Z_STOP")
    )

    # --------------------------------------------------------
    # Absolute entry Z
    # --------------------------------------------------------

    trades["AbsEntryZ"] = (
        trades["EntryZ"]
        .abs()
    )

    # --------------------------------------------------------
    # Entry Z buckets
    # --------------------------------------------------------

    trades["ZBucket"] = pd.cut(
        trades["AbsEntryZ"],
        bins=[
            2.25,
            2.50,
            3.00,
            3.50,
            np.inf,
        ],
        labels=[
            "2.25-2.50",
            "2.50-3.00",
            "3.00-3.50",
            ">3.50",
        ],
        include_lowest=True,
    )

    # --------------------------------------------------------
    # Regime
    # --------------------------------------------------------

    p_mr = trades[
        "EntryP_MR_Recheck"
    ]

    trades["EntryRegime"] = np.where(
        p_mr >= MR_PROBABILITY_THRESHOLD,
        "MR",
        "TREND",
    )

    # --------------------------------------------------------
    # Regime + stop
    # --------------------------------------------------------

    trades["RegimeStop"] = (
        trades["EntryRegime"]
        + "_"
        + trades["Stopped"].astype(str)
    )

    # --------------------------------------------------------
    # Direction + stop
    # --------------------------------------------------------

    trades["DirectionStop"] = (
        trades["Side"].astype(str)
        + "_"
        + trades["Stopped"].astype(str)
    )

    return trades


# ============================================================
# GENERIC GROUP SUMMARY
# ============================================================

def group_summary(
    trades,
    column,
):

    rows = []

    for value, group in trades.groupby(
        column,
        dropna=False,
        observed=False,
    ):

        if len(group) == 0:
            continue

        if "GrossReturn" in group.columns:

            returns = (
                group["GrossReturn"]
                .dropna()
            )

            mean_return = (
                returns.mean()
                if len(returns) > 0
                else np.nan
            )

            median_return = (
                returns.median()
                if len(returns) > 0
                else np.nan
            )

            win_rate = (
                (returns > 0).mean()
                if len(returns) > 0
                else np.nan
            )

            pf = (
                safe_pf(returns)
                if len(returns) > 0
                else np.nan
            )

        else:

            mean_return = np.nan
            median_return = np.nan
            win_rate = np.nan
            pf = np.nan

        row = {
            column: value,
            "Trades": len(group),
            "WinRate": win_rate,
            "PF": pf,
            "MeanReturn": mean_return,
            "MedianReturn": median_return,
        }

        # ----------------------------------------------------
        # PnL
        # ----------------------------------------------------

        if "PnL" in group.columns:

            row["TotalPnL"] = (
                group["PnL"]
                .sum()
            )

            row["MeanPnL"] = (
                group["PnL"]
                .mean()
            )

        # ----------------------------------------------------
        # MAE
        # ----------------------------------------------------

        if "MAE" in group.columns:

            row["MeanMAE"] = (
                group["MAE"]
                .mean()
            )

            row["MedianMAE"] = (
                group["MAE"]
                .median()
            )

        # ----------------------------------------------------
        # MFE
        # ----------------------------------------------------

        if "MFE" in group.columns:

            row["MeanMFE"] = (
                group["MFE"]
                .mean()
            )

            row["MedianMFE"] = (
                group["MFE"]
                .median()
            )

        # ----------------------------------------------------
        # Holding
        # ----------------------------------------------------

        if (
            "TradingHoursHeld"
            in group.columns
        ):

            row["MeanHoldHours"] = (
                group[
                    "TradingHoursHeld"
                ].mean()
            )

        # ----------------------------------------------------
        # Stop rate
        # ----------------------------------------------------

        if "Stopped" in group.columns:

            row["StopRate"] = (
                group["Stopped"]
                .mean()
            )

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# VOLATILITY BUCKETS
# ============================================================

def add_quantile_bucket(
    trades,
    source_column,
    output_column,
):

    valid = trades[
        source_column
    ].notna()

    if valid.sum() < 10:

        trades[
            output_column
        ] = np.nan

        return trades

    try:

        trades.loc[
            valid,
            output_column
        ] = pd.qcut(
            trades.loc[
                valid,
                source_column
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

    except ValueError:

        trades[
            output_column
        ] = np.nan

    return trades


# ============================================================
# PRINT + SAVE SUMMARY
# ============================================================

def print_and_save_summary(
    trades,
    column,
    filename,
    title,
):

    print_section(title)

    result = group_summary(
        trades,
        column,
    )

    if len(result) == 0:

        print(
            "No valid groups."
        )

    else:

        print(
            result.to_string(
                index=False
            )
        )

        result.to_csv(
            OUTPUT_DIR / filename,
            index=False,
        )

    return result


# ============================================================
# YEARLY SUMMARY
# ============================================================

def yearly_summary(trades):

    temp = trades.copy()

    temp["Year"] = (
        temp["EntryTime"]
        .dt.year
    )

    result = group_summary(
        temp,
        "Year",
    )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    df = load_h1()

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Start: "
        f"{df['Timestamp'].min()}"
    )

    print(
        f"End: "
        f"{df['Timestamp'].max()}"
    )

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    df = add_kalman_features(df)

    df = add_volatility_features(df)

    df = add_markov_features(df)

    # --------------------------------------------------------
    # CANONICAL TRADES
    # --------------------------------------------------------

    trades = load_canonical_trades()

    trades = normalize_trade_columns(
        trades
    )

    # --------------------------------------------------------
    # VERIFY CANONICAL COUNT
    # --------------------------------------------------------

    expected_trades = 1351

    print()

    print(
        f"Expected canonical trades: "
        f"{expected_trades:,}"
    )

    print(
        f"Loaded canonical trades: "
        f"{len(trades):,}"
    )

    if len(trades) != expected_trades:

        print(
            "\nWARNING:"
            "\nThe canonical trade file does "
            "not contain the expected 1,351 "
            "trades."
            "\nContinue only if this is "
            "intentional."
        )

    else:

        print(
            "Canonical trade count verified."
        )

    # --------------------------------------------------------
    # ATTACH ENTRY FEATURES
    # --------------------------------------------------------

    trades = attach_entry_features(
        trades,
        df,
    )

    # --------------------------------------------------------
    # MAE / MFE
    # --------------------------------------------------------

    trades = calculate_mae_mfe(
        trades,
        df,
    )

    # --------------------------------------------------------
    # DERIVED COLUMNS
    # --------------------------------------------------------

    trades = add_derived_trade_columns(
        trades
    )

    # --------------------------------------------------------
    # OVERALL
    # --------------------------------------------------------

    print_section(
        "OVERALL CANONICAL TRADE DIAGNOSTICS"
    )

    print(
        f"Trades: "
        f"{len(trades):,}"
    )

    if "GrossReturn" in trades.columns:

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

    # --------------------------------------------------------
    # EXIT REASON
    # --------------------------------------------------------

    print_and_save_summary(
        trades,
        "ExitReason",
        "exit_summary.csv",
        "EXIT REASON",
    )

    # --------------------------------------------------------
    # WIN / LOSS
    # --------------------------------------------------------

    print_and_save_summary(
        trades,
        "Outcome",
        "outcome_summary.csv",
        "WINNERS VS LOSERS",
    )

    # --------------------------------------------------------
    # STOP STATUS
    # --------------------------------------------------------

    print_and_save_summary(
        trades,
        "Stopped",
        "stop_summary.csv",
        "STOPPED VS NON-STOPPED",
    )

    # --------------------------------------------------------
    # ENTRY Z
    # --------------------------------------------------------

    print_and_save_summary(
        trades,
        "ZBucket",
        "z_summary.csv",
        "ENTRY Z BUCKET",
    )

    # --------------------------------------------------------
    # DIRECTION
    # --------------------------------------------------------

    print_and_save_summary(
        trades,
        "Side",
        "direction_summary.csv",
        "DIRECTION",
    )

    # --------------------------------------------------------
    # DIRECTION + STOP
    # --------------------------------------------------------

    print_and_save_summary(
        trades,
        "DirectionStop",
        "direction_stop_summary.csv",
        "DIRECTION + STOP STATUS",
    )

    # --------------------------------------------------------
    # REGIME
    # --------------------------------------------------------

    print_and_save_summary(
        trades,
        "EntryRegime",
        "regime_summary.csv",
        "ENTRY REGIME",
    )

    # --------------------------------------------------------
    # REGIME + STOP
    # --------------------------------------------------------

    print_and_save_summary(
        trades,
        "RegimeStop",
        "regime_stop_summary.csv",
        "REGIME + STOP STATUS",
    )

    # ========================================================
    # VOLATILITY BUCKETS
    # ========================================================

    trades = add_quantile_bucket(
        trades,
        "EntryVolRatio_24_168",
        "VolRatioBucket",
    )

    print_and_save_summary(
        trades,
        "VolRatioBucket",
        "volatility_ratio_summary.csv",
        "VOLATILITY RATIO 24H / 168H",
    )

    trades = add_quantile_bucket(
        trades,
        "EntryATRRatio",
        "ATRRatioBucket",
    )

    print_and_save_summary(
        trades,
        "ATRRatioBucket",
        "atr_ratio_summary.csv",
        "ATR RATIO 24H / 168H",
    )

    trades = add_quantile_bucket(
        trades,
        "EntryRV_24",
        "RV24Bucket",
    )

    print_and_save_summary(
        trades,
        "RV24Bucket",
        "rv24_summary.csv",
        "24H REALIZED VOLATILITY",
    )

    trades = add_quantile_bucket(
        trades,
        "EntryRV_168",
        "RV168Bucket",
    )

    print_and_save_summary(
        trades,
        "RV168Bucket",
        "rv168_summary.csv",
        "168H REALIZED VOLATILITY",
    )

    # --------------------------------------------------------
    # VOLATILITY × STOP
    # --------------------------------------------------------

    for bucket_column, filename, title in [
        (
            "VolRatioBucket",
            "volratio_stop_summary.csv",
            "VOLATILITY RATIO × STOP STATUS",
        ),
        (
            "ATRRatioBucket",
            "atr_ratio_stop_summary.csv",
            "ATR RATIO × STOP STATUS",
        ),
        (
            "RV24Bucket",
            "rv24_stop_summary.csv",
            "24H REALIZED VOLATILITY × STOP STATUS",
        ),
        (
            "RV168Bucket",
            "rv168_stop_summary.csv",
            "168H REALIZED VOLATILITY × STOP STATUS",
        ),
    ]:

        temp = trades.copy()

        temp["VolStop"] = (
            temp[bucket_column]
            .astype(str)
            + "_"
            + temp["Stopped"].astype(str)
        )

        print_and_save_summary(
            temp,
            "VolStop",
            filename,
            title,
        )

    # --------------------------------------------------------
    # VOLATILITY × OUTCOME
    # --------------------------------------------------------

    for bucket_column, filename, title in [
        (
            "VolRatioBucket",
            "volratio_outcome_summary.csv",
            "VOLATILITY RATIO × OUTCOME",
        ),
        (
            "ATRRatioBucket",
            "atr_ratio_outcome_summary.csv",
            "ATR RATIO × OUTCOME",
        ),
        (
            "RV24Bucket",
            "rv24_outcome_summary.csv",
            "24H REALIZED VOLATILITY × OUTCOME",
        ),
        (
            "RV168Bucket",
            "rv168_outcome_summary.csv",
            "168H REALIZED VOLATILITY × OUTCOME",
        ),
    ]:

        temp = trades.copy()

        temp["VolOutcome"] = (
            temp[bucket_column]
            .astype(str)
            + "_"
            + temp["Outcome"].astype(str)
        )

        print_and_save_summary(
            temp,
            "VolOutcome",
            filename,
            title,
        )

    # --------------------------------------------------------
    # MAE / MFE BY EXIT
    # --------------------------------------------------------

    print_section(
        "MAE / MFE BY EXIT REASON"
    )

    mae_mfe_exit = (
        trades
        .groupby(
            "ExitReason",
            dropna=False,
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
            MedianReturn=(
                "GrossReturn",
                "median",
            ),
            MeanHoldHours=(
                "TradingHoursHeld",
                "mean",
            ),
        )
        .reset_index()
    )

    print(
        mae_mfe_exit.to_string(
            index=False
        )
    )

    mae_mfe_exit.to_csv(
        OUTPUT_DIR
        / "mae_mfe_exit.csv",
        index=False,
    )

    # --------------------------------------------------------
    # MAE / MFE BY WINNER / LOSER
    # --------------------------------------------------------

    print_section(
        "MAE / MFE BY OUTCOME"
    )

    mae_mfe_outcome = (
        trades
        .groupby(
            "Outcome",
            dropna=False,
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
            MedianReturn=(
                "GrossReturn",
                "median",
            ),
            MeanHoldHours=(
                "TradingHoursHeld",
                "mean",
            ),
        )
        .reset_index()
    )

    print(
        mae_mfe_outcome.to_string(
            index=False
        )
    )

    mae_mfe_outcome.to_csv(
        OUTPUT_DIR
        / "mae_mfe_outcome.csv",
        index=False,
    )

    # --------------------------------------------------------
    # YEARLY
    # --------------------------------------------------------

    print_section(
        "YEARLY TRADE DIAGNOSTICS"
    )

    yearly = yearly_summary(
        trades
    )

    print(
        yearly.to_string(
            index=False
        )
    )

    yearly.to_csv(
        OUTPUT_DIR
        / "yearly_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # SAVE COMPLETE TRADE FILE
    # --------------------------------------------------------

    trades.to_csv(
        OUTPUT_DIR
        / "canonical_trade_volatility_diagnostics.csv",
        index=False,
    )

    # --------------------------------------------------------
    # SAVE PROCESSED FEATURES
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_DIR
        / "processed_features.csv",
        index=False,
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print_section(
        "FILES SAVED"
    )

    print(
        OUTPUT_DIR.resolve()
    )

    print()

    print(
        "Canonical trade count:"
        f" {len(trades):,}"
    )

    print(
        "Diagnostic complete."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()