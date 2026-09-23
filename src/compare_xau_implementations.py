from pathlib import Path
import sys

import numpy as np
import pandas as pd


# ============================================================
# PATH SETUP
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
MT5_DIR = DATA_DIR / "mt5_h1"
RESULTS_DIR = ROOT / "results" / "xau_comparison"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# COMPARISON PERIOD
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
# INPUT FILES
# ============================================================

RAW_FILE = (
    DATA_DIR
    / "Exness_XAUUSDm_2025.csv"
)

MT5_FILE = (
    MT5_DIR
    / "XAUUSDm_H1_2019_2026.csv"
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


# ============================================================
# IMPORT ORIGINAL IMPLEMENTATION
# ============================================================

sys.path.insert(
    0,
    str(ROOT / "src"),
)

from data import (
    load_exness_ticks,
    ticks_to_bars,
)

from walk_forward_kalman import (
    walk_forward_kalman,
    calculate_zscore,
)


# ============================================================
# LOAD ORIGINAL TICK DATA
# ============================================================

def load_original_xau_h1():

    print()
    print("=" * 70)
    print("LOADING ORIGINAL XAU TICK DATA")
    print("=" * 70)

    if not RAW_FILE.exists():

        raise FileNotFoundError(
            f"\nRaw XAU tick file not found:\n"
            f"{RAW_FILE}\n"
        )

    ticks = load_exness_ticks(
        str(RAW_FILE)
    )

    print(
        f"Raw ticks loaded: "
        f"{len(ticks):,}"
    )

    # --------------------------------------------------------
    # Restrict raw ticks to comparison period
    # --------------------------------------------------------

    ticks = ticks[
        (ticks["Timestamp"] >= START_DATE)
        &
        (ticks["Timestamp"] <= END_DATE)
    ].copy()

    print(
        f"Ticks in comparison period: "
        f"{len(ticks):,}"
    )

    if ticks.empty:

        raise ValueError(
            "No raw XAU ticks found "
            "inside the comparison period."
        )

    # --------------------------------------------------------
    # Original tick -> H1 conversion
    # --------------------------------------------------------

    bars = ticks_to_bars(
        ticks,
        timeframe="1h",
    )

    bars = bars.loc[
        (bars.index >= START_DATE)
        &
        (bars.index <= END_DATE)
    ].copy()

    print(
        f"Original tick-derived H1 bars: "
        f"{len(bars):,}"
    )

    if bars.empty:

        raise ValueError(
            "No tick-derived H1 bars were created."
        )

    return bars


# ============================================================
# LOAD MT5 H1 DATA
# ============================================================

def load_mt5_xau_h1():

    print()
    print("=" * 70)
    print("LOADING MT5 XAU H1 DATA")
    print("=" * 70)

    if not MT5_FILE.exists():

        raise FileNotFoundError(
            f"\nMT5 XAU H1 file not found:\n"
            f"{MT5_FILE}\n"
        )

    mt5 = pd.read_csv(
        MT5_FILE
    )

    print(
        f"MT5 rows loaded: "
        f"{len(mt5):,}"
    )

    print(
        f"MT5 columns: "
        f"{list(mt5.columns)}"
    )

    # --------------------------------------------------------
    # Find timestamp column
    # --------------------------------------------------------

    possible_time_columns = [
        "Time",
        "Timestamp",
        "time",
        "timestamp",
        "Datetime",
        "Date",
    ]

    time_col = None

    for candidate in possible_time_columns:

        if candidate in mt5.columns:

            time_col = candidate
            break

    if time_col is None:

        raise ValueError(
            "\nCould not find a timestamp column.\n"
            f"Available columns: {list(mt5.columns)}"
        )

    print(
        f"Using timestamp column: "
        f"{time_col}"
    )

    # --------------------------------------------------------
    # Convert timestamp to UTC
    # --------------------------------------------------------

    mt5[time_col] = pd.to_datetime(
        mt5[time_col],
        utc=True,
        errors="coerce",
    )

    invalid_time = (
        mt5[time_col]
        .isna()
        .sum()
    )

    if invalid_time > 0:

        print(
            f"WARNING: dropping "
            f"{invalid_time:,} invalid timestamps."
        )

        mt5 = mt5.dropna(
            subset=[time_col]
        )

    # --------------------------------------------------------
    # Set timestamp index
    # --------------------------------------------------------

    mt5 = (
        mt5
        .set_index(time_col)
        .sort_index()
    )

    # --------------------------------------------------------
    # Required OHLC
    # --------------------------------------------------------

    required_columns = {
        "Open",
        "High",
        "Low",
        "Close",
    }

    missing = (
        required_columns
        - set(mt5.columns)
    )

    if missing:

        raise ValueError(
            "\nMissing required MT5 columns:\n"
            f"{missing}\n"
            f"Available columns: {list(mt5.columns)}"
        )

    # --------------------------------------------------------
    # Convert OHLC to numeric
    # --------------------------------------------------------

    for column in [
        "Open",
        "High",
        "Low",
        "Close",
    ]:

        mt5[column] = pd.to_numeric(
            mt5[column],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Drop invalid OHLC
    # --------------------------------------------------------

    before = len(mt5)

    mt5 = mt5.dropna(
        subset=[
            "Open",
            "High",
            "Low",
            "Close",
        ]
    )

    dropped = before - len(mt5)

    if dropped > 0:

        print(
            f"Dropped invalid OHLC rows: "
            f"{dropped:,}"
        )

    # --------------------------------------------------------
    # Restrict comparison period
    # --------------------------------------------------------

    mt5 = mt5.loc[
        (mt5.index >= START_DATE)
        &
        (mt5.index <= END_DATE)
    ].copy()

    print(
        f"MT5 H1 bars in comparison period: "
        f"{len(mt5):,}"
    )

    if mt5.empty:

        raise ValueError(
            "No MT5 H1 bars found "
            "inside the comparison period."
        )

    return mt5


# ============================================================
# ALIGN BOTH DATASETS
# ============================================================

def prepare_comparison(
    original,
    mt5,
):

    print()
    print("=" * 70)
    print("ALIGNING H1 DATA")
    print("=" * 70)

    # --------------------------------------------------------
    # Rename original
    # --------------------------------------------------------

    original = original.rename(
        columns={
            "Open": "Original_Open",
            "High": "Original_High",
            "Low": "Original_Low",
            "Close": "Original_Close",
        }
    )

    # --------------------------------------------------------
    # Rename MT5
    # --------------------------------------------------------

    mt5 = mt5.rename(
        columns={
            "Open": "MT5_Open",
            "High": "MT5_High",
            "Low": "MT5_Low",
            "Close": "MT5_Close",
        }
    )

    # --------------------------------------------------------
    # Keep OHLC only
    # --------------------------------------------------------

    original = original[
        [
            "Original_Open",
            "Original_High",
            "Original_Low",
            "Original_Close",
        ]
    ]

    mt5 = mt5[
        [
            "MT5_Open",
            "MT5_High",
            "MT5_Low",
            "MT5_Close",
        ]
    ]

    # --------------------------------------------------------
    # Inner join
    # --------------------------------------------------------

    comparison = original.join(
        mt5,
        how="inner",
    )

    comparison = comparison.sort_index()

    print(
        f"Aligned bars: "
        f"{len(comparison):,}"
    )

    if comparison.empty:

        raise ValueError(
            "No overlapping H1 timestamps found."
        )

    print(
        f"First aligned bar: "
        f"{comparison.index.min()}"
    )

    print(
        f"Last aligned bar: "
        f"{comparison.index.max()}"
    )

    return comparison


# ============================================================
# PRICE DIFFERENCES
# ============================================================

def calculate_price_differences(
    df,
):

    print()
    print("=" * 70)
    print("CALCULATING PRICE DIFFERENCES")
    print("=" * 70)

    for field in [
        "Open",
        "High",
        "Low",
        "Close",
    ]:

        original_col = (
            f"Original_{field}"
        )

        mt5_col = (
            f"MT5_{field}"
        )

        abs_diff_col = (
            f"{field}_AbsDiff"
        )

        pct_diff_col = (
            f"{field}_PctDiff"
        )

        # Absolute difference
        df[abs_diff_col] = (
            df[original_col]
            - df[mt5_col]
        ).abs()

        # Percentage difference
        df[pct_diff_col] = (
            (
                df[original_col]
                /
                df[mt5_col]
            )
            - 1.0
        ) * 100.0

    # Close difference in basis points
    df["Close_Diff_Bps"] = (
        (
            df["Original_Close"]
            /
            df["MT5_Close"]
        )
        - 1.0
    ) * 10_000.0

    return df


# ============================================================
# KALMAN FEATURES
# ============================================================

def calculate_kalman_features(
    price_series,
):

    """
    Uses the ORIGINAL Kalman implementation.

    The exact same Kalman implementation is applied to
    both datasets so that we isolate differences caused
    by the underlying price data.
    """

    (
        states,
        variances,
        residuals,
        means,
    ) = walk_forward_kalman(
        price_series,
        phi=PHI,
        q=Q,
        r=R,
        mean_window=MEAN_WINDOW,
    )

    # --------------------------------------------------------
    # Preserve index
    # --------------------------------------------------------

    states = pd.Series(
        states,
        index=price_series.index,
    )

    variances = pd.Series(
        variances,
        index=price_series.index,
    )

    residuals = pd.Series(
        residuals,
        index=price_series.index,
    )

    means = pd.Series(
        means,
        index=price_series.index,
    )

    # --------------------------------------------------------
    # Z-score
    # --------------------------------------------------------

    zscore = calculate_zscore(
        residuals,
        window=Z_WINDOW,
    )

    return (
        states,
        variances,
        residuals,
        means,
        zscore,
    )


# ============================================================
# KALMAN COMPARISON
# ============================================================

def calculate_kalman_comparison(
    df,
):

    print()
    print("=" * 70)
    print("CALCULATING KALMAN FEATURES")
    print("=" * 70)

    # --------------------------------------------------------
    # Original tick-derived H1
    # --------------------------------------------------------

    (
        original_state,
        original_variance,
        original_residual,
        original_mean,
        original_z,
    ) = calculate_kalman_features(
        df["Original_Close"]
    )

    # --------------------------------------------------------
    # MT5 H1
    # --------------------------------------------------------

    (
        mt5_state,
        mt5_variance,
        mt5_residual,
        mt5_mean,
        mt5_z,
    ) = calculate_kalman_features(
        df["MT5_Close"]
    )

    # --------------------------------------------------------
    # Store original values
    # --------------------------------------------------------

    df["Original_KalmanState"] = (
        original_state
    )

    df["MT5_KalmanState"] = (
        mt5_state
    )

    df["Original_KalmanVariance"] = (
        original_variance
    )

    df["MT5_KalmanVariance"] = (
        mt5_variance
    )

    df["Original_Residual"] = (
        original_residual
    )

    df["MT5_Residual"] = (
        mt5_residual
    )

    df["Original_Mean"] = (
        original_mean
    )

    df["MT5_Mean"] = (
        mt5_mean
    )

    df["Original_Z"] = (
        original_z
    )

    df["MT5_Z"] = (
        mt5_z
    )

    # --------------------------------------------------------
    # Differences
    # --------------------------------------------------------

    df["KalmanState_AbsDiff"] = (
        df["Original_KalmanState"]
        - df["MT5_KalmanState"]
    ).abs()

    df["KalmanVariance_AbsDiff"] = (
        df["Original_KalmanVariance"]
        - df["MT5_KalmanVariance"]
    ).abs()

    df["Residual_AbsDiff"] = (
        df["Original_Residual"]
        - df["MT5_Residual"]
    ).abs()

    df["Mean_AbsDiff"] = (
        df["Original_Mean"]
        - df["MT5_Mean"]
    ).abs()

    df["Z_AbsDiff"] = (
        df["Original_Z"]
        - df["MT5_Z"]
    ).abs()

    return df


# ============================================================
# SIGNAL CLASSIFICATION
# ============================================================

def classify_signal(
    z,
):

    """
    Entry signal:

        +1 = LONG
        -1 = SHORT
         0 = no entry
    """

    if pd.isna(z):

        return np.nan

    if z <= -ENTRY_Z:

        return 1

    if z >= ENTRY_Z:

        return -1

    return 0


def classify_zone(
    z,
):

    """
    Diagnostic Z-score zone:

         1 = long entry zone
        -1 = short entry zone
         2 = exit zone
         3 = stop zone
         0 = neutral
    """

    if pd.isna(z):

        return np.nan

    if abs(z) >= STOP_Z:

        return 3

    if z <= -ENTRY_Z:

        return 1

    if z >= ENTRY_Z:

        return -1

    if abs(z) <= EXIT_Z:

        return 2

    return 0


# ============================================================
# SIGNAL COMPARISON
# ============================================================

def calculate_signal_comparison(
    df,
):

    print()
    print("=" * 70)
    print("CALCULATING SIGNAL COMPARISON")
    print("=" * 70)

    # --------------------------------------------------------
    # Calculate entry signals
    # --------------------------------------------------------

    df["Original_Signal"] = (
        df["Original_Z"]
        .apply(classify_signal)
    )

    df["MT5_Signal"] = (
        df["MT5_Z"]
        .apply(classify_signal)
    )

    # --------------------------------------------------------
    # Calculate diagnostic zones
    # --------------------------------------------------------

    df["Original_Zone"] = (
        df["Original_Z"]
        .apply(classify_zone)
    )

    df["MT5_Zone"] = (
        df["MT5_Z"]
        .apply(classify_zone)
    )

    # --------------------------------------------------------
    # Valid Z-score observations
    # --------------------------------------------------------

    valid = (
        df["Original_Z"].notna()
        &
        df["MT5_Z"].notna()
    )

    # ========================================================
    # FIX FOR PANDAS DTYPE ERROR
    # ========================================================
    #
    # Previous version created a float64 column containing
    # NaN and then attempted to assign a boolean array.
    #
    # np.where() creates a consistent numeric result:
    #
    #     1.0 = True
    #     0.0 = False
    #     NaN = invalid observation
    #
    # This avoids the LossySetitemError.
    # ========================================================

    df["Signal_Agree"] = np.where(
        valid,
        (
            df["Original_Signal"]
            ==
            df["MT5_Signal"]
        ).astype(float),
        np.nan,
    )

    df["Zone_Agree"] = np.where(
        valid,
        (
            df["Original_Zone"]
            ==
            df["MT5_Zone"]
        ).astype(float),
        np.nan,
    )

    return df


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(
    df,
):

    print()
    print("=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    print(
        f"Period: "
        f"{df.index.min()} "
        f"-> "
        f"{df.index.max()}"
    )

    print(
        f"Aligned bars: "
        f"{len(df):,}"
    )

    # ========================================================
    # PRICE
    # ========================================================

    print()
    print("-" * 70)
    print("PRICE DIFFERENCES")
    print("-" * 70)

    for field in [
        "Open",
        "High",
        "Low",
        "Close",
    ]:

        abs_col = (
            f"{field}_AbsDiff"
        )

        pct_col = (
            f"{field}_PctDiff"
        )

        print(
            f"{field:5s} | "
            f"mean abs = "
            f"{df[abs_col].mean():.8f} | "
            f"median abs = "
            f"{df[abs_col].median():.8f} | "
            f"max abs = "
            f"{df[abs_col].max():.8f}"
        )

        print(
            f"      | "
            f"mean pct = "
            f"{df[pct_col].mean():.8f}% | "
            f"median pct = "
            f"{df[pct_col].median():.8f}%"
        )

    print()

    print(
        f"Close mean difference: "
        f"{df['Close_Diff_Bps'].mean():.6f} bps"
    )

    print(
        f"Close median difference: "
        f"{df['Close_Diff_Bps'].median():.6f} bps"
    )

    print(
        f"Close max absolute difference: "
        f"{df['Close_Diff_Bps'].abs().max():.6f} bps"
    )

    # ========================================================
    # KALMAN
    # ========================================================

    print()
    print("-" * 70)
    print("KALMAN DIFFERENCES")
    print("-" * 70)

    print(
        f"Kalman state mean abs diff: "
        f"{df['KalmanState_AbsDiff'].mean():.8f}"
    )

    print(
        f"Kalman state median abs diff: "
        f"{df['KalmanState_AbsDiff'].median():.8f}"
    )

    print(
        f"Kalman state max abs diff: "
        f"{df['KalmanState_AbsDiff'].max():.8f}"
    )

    print(
        f"Kalman variance mean abs diff: "
        f"{df['KalmanVariance_AbsDiff'].mean():.8f}"
    )

    print(
        f"Residual mean abs diff: "
        f"{df['Residual_AbsDiff'].mean():.8f}"
    )

    print(
        f"Residual median abs diff: "
        f"{df['Residual_AbsDiff'].median():.8f}"
    )

    print(
        f"Residual max abs diff: "
        f"{df['Residual_AbsDiff'].max():.8f}"
    )

    # ========================================================
    # Z-SCORE
    # ========================================================

    valid_z = (
        df["Original_Z"].notna()
        &
        df["MT5_Z"].notna()
    )

    print()
    print("-" * 70)
    print("Z-SCORE COMPARISON")
    print("-" * 70)

    if not valid_z.any():

        print(
            "No valid Z-score observations."
        )

    else:

        original_z = df.loc[
            valid_z,
            "Original_Z",
        ]

        mt5_z = df.loc[
            valid_z,
            "MT5_Z",
        ]

        z_corr = (
            original_z.corr(
                mt5_z
            )
        )

        z_mean_abs_diff = (
            df.loc[
                valid_z,
                "Z_AbsDiff",
            ].mean()
        )

        z_median_abs_diff = (
            df.loc[
                valid_z,
                "Z_AbsDiff",
            ].median()
        )

        z_max_abs_diff = (
            df.loc[
                valid_z,
                "Z_AbsDiff",
            ].max()
        )

        print(
            f"Valid Z observations: "
            f"{valid_z.sum():,}"
        )

        print(
            f"Z-score correlation: "
            f"{z_corr:.6f}"
        )

        print(
            f"Mean |Z difference|: "
            f"{z_mean_abs_diff:.6f}"
        )

        print(
            f"Median |Z difference|: "
            f"{z_median_abs_diff:.6f}"
        )

        print(
            f"Max |Z difference|: "
            f"{z_max_abs_diff:.6f}"
        )

        print()

        print(
            f"Original Z mean: "
            f"{original_z.mean():.6f}"
        )

        print(
            f"MT5 Z mean: "
            f"{mt5_z.mean():.6f}"
        )

        print(
            f"Original Z std: "
            f"{original_z.std():.6f}"
        )

        print(
            f"MT5 Z std: "
            f"{mt5_z.std():.6f}"
        )

    # ========================================================
    # SIGNAL AGREEMENT
    # ========================================================

    valid_signal = (
        df["Original_Signal"].notna()
        &
        df["MT5_Signal"].notna()
    )

    print()
    print("-" * 70)
    print("SIGNAL AGREEMENT")
    print("-" * 70)

    if not valid_signal.any():

        print(
            "No valid signal observations."
        )

    else:

        original_signal = df.loc[
            valid_signal,
            "Original_Signal",
        ]

        mt5_signal = df.loc[
            valid_signal,
            "MT5_Signal",
        ]

        agreement = (
            original_signal.to_numpy()
            ==
            mt5_signal.to_numpy()
        )

        agreement_rate = (
            agreement.mean()
            * 100.0
        )

        print(
            f"Signal observations: "
            f"{valid_signal.sum():,}"
        )

        print(
            f"Signal agreement: "
            f"{agreement_rate:.2f}%"
        )

        print(
            f"Signal disagreements: "
            f"{(~agreement).sum():,}"
        )

        # ----------------------------------------------------
        # Original signal counts
        # ----------------------------------------------------

        print()
        print(
            "Original signal counts:"
        )

        print(
            original_signal
            .value_counts()
            .sort_index()
            .to_string()
        )

        # ----------------------------------------------------
        # MT5 signal counts
        # ----------------------------------------------------

        print()
        print(
            "MT5 signal counts:"
        )

        print(
            mt5_signal
            .value_counts()
            .sort_index()
            .to_string()
        )

        # ----------------------------------------------------
        # Disagreements
        # ----------------------------------------------------

        disagreements = df.loc[
            valid_signal
            &
            (
                df["Original_Signal"]
                !=
                df["MT5_Signal"]
            )
        ]

        print()
        print(
            "First 20 signal disagreements:"
        )

        if disagreements.empty:

            print(
                "None."
            )

        else:

            print(
                disagreements[
                    [
                        "Original_Close",
                        "MT5_Close",
                        "Original_Z",
                        "MT5_Z",
                        "Original_Signal",
                        "MT5_Signal",
                    ]
                ]
                .head(20)
                .to_string()
            )

    # ========================================================
    # ZONE AGREEMENT
    # ========================================================

    valid_zone = (
        df["Original_Zone"].notna()
        &
        df["MT5_Zone"].notna()
    )

    print()
    print("-" * 70)
    print("Z-SCORE ZONE AGREEMENT")
    print("-" * 70)

    if valid_zone.any():

        zone_agreement = (
            df.loc[
                valid_zone,
                "Original_Zone",
            ].to_numpy()
            ==
            df.loc[
                valid_zone,
                "MT5_Zone",
            ].to_numpy()
        )

        print(
            f"Zone observations: "
            f"{valid_zone.sum():,}"
        )

        print(
            f"Zone agreement: "
            f"{zone_agreement.mean() * 100:.2f}%"
        )

        print(
            f"Zone disagreements: "
            f"{(~zone_agreement).sum():,}"
        )

    # ========================================================
    # DATASET COVERAGE
    # ========================================================

    print()
    print("-" * 70)
    print("DATASET COVERAGE")
    print("-" * 70)

    print(
        f"First aligned timestamp: "
        f"{df.index.min()}"
    )

    print(
        f"Last aligned timestamp: "
        f"{df.index.max()}"
    )

    print(
        f"Aligned H1 bars: "
        f"{len(df):,}"
    )

    # --------------------------------------------------------
    # Expected hourly timestamps.
    # --------------------------------------------------------

    expected_hours = pd.date_range(
        start=START_DATE,
        end=END_DATE,
        freq="1h",
    )

    missing_aligned = (
        len(expected_hours)
        - len(df)
    )

    print(
        f"Expected hourly timestamps: "
        f"{len(expected_hours):,}"
    )

    print(
        f"Missing from overlap: "
        f"{missing_aligned:,}"
    )


# ============================================================
# SAVE SUMMARY
# ============================================================

def save_summary(
    df,
):

    summary_file = (
        RESULTS_DIR
        / "xau_h1_comparison_summary.txt"
    )

    valid_z = (
        df["Original_Z"].notna()
        &
        df["MT5_Z"].notna()
    )

    valid_signal = (
        df["Original_Signal"].notna()
        &
        df["MT5_Signal"].notna()
    )

    lines = []

    lines.append(
        "XAUUSDm IMPLEMENTATION COMPARISON"
    )

    lines.append(
        "=" * 70
    )

    lines.append(
        f"Start: {df.index.min()}"
    )

    lines.append(
        f"End: {df.index.max()}"
    )

    lines.append(
        f"Aligned bars: {len(df):,}"
    )

    lines.append("")

    # ========================================================
    # PRICE
    # ========================================================

    lines.append(
        "PRICE DIFFERENCES"
    )

    lines.append(
        f"Close mean abs diff: "
        f"{df['Close_AbsDiff'].mean():.10f}"
    )

    lines.append(
        f"Close median abs diff: "
        f"{df['Close_AbsDiff'].median():.10f}"
    )

    lines.append(
        f"Close max abs diff: "
        f"{df['Close_AbsDiff'].max():.10f}"
    )

    lines.append(
        f"Close mean diff bps: "
        f"{df['Close_Diff_Bps'].mean():.10f}"
    )

    lines.append("")

    # ========================================================
    # KALMAN
    # ========================================================

    lines.append(
        "KALMAN DIFFERENCES"
    )

    lines.append(
        f"State mean abs diff: "
        f"{df['KalmanState_AbsDiff'].mean():.10f}"
    )

    lines.append(
        f"State max abs diff: "
        f"{df['KalmanState_AbsDiff'].max():.10f}"
    )

    lines.append(
        f"Residual mean abs diff: "
        f"{df['Residual_AbsDiff'].mean():.10f}"
    )

    lines.append(
        f"Residual max abs diff: "
        f"{df['Residual_AbsDiff'].max():.10f}"
    )

    lines.append("")

    # ========================================================
    # Z-SCORE
    # ========================================================

    lines.append(
        "Z-SCORE COMPARISON"
    )

    lines.append(
        f"Valid Z observations: "
        f"{valid_z.sum():,}"
    )

    if valid_z.any():

        z_corr = (
            df.loc[
                valid_z,
                "Original_Z",
            ].corr(
                df.loc[
                    valid_z,
                    "MT5_Z",
                ]
            )
        )

        lines.append(
            f"Z correlation: "
            f"{z_corr:.10f}"
        )

        lines.append(
            f"Mean abs Z difference: "
            f"{df.loc[
                valid_z,
                'Z_AbsDiff'
            ].mean():.10f}"
        )

        lines.append(
            f"Max abs Z difference: "
            f"{df.loc[
                valid_z,
                'Z_AbsDiff'
            ].max():.10f}"
        )

    lines.append("")

    # ========================================================
    # SIGNALS
    # ========================================================

    lines.append(
        "SIGNAL AGREEMENT"
    )

    lines.append(
        f"Valid signal observations: "
        f"{valid_signal.sum():,}"
    )

    if valid_signal.any():

        agreement = (
            df.loc[
                valid_signal,
                "Original_Signal",
            ].to_numpy()
            ==
            df.loc[
                valid_signal,
                "MT5_Signal",
            ].to_numpy()
        )

        lines.append(
            f"Signal agreement: "
            f"{agreement.mean() * 100:.6f}%"
        )

        lines.append(
            f"Signal disagreements: "
            f"{(~agreement).sum():,}"
        )

    summary_file.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return summary_file


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("XAUUSDm IMPLEMENTATION COMPARISON")
    print("=" * 70)

    print()
    print("Comparison period:")

    print(
        f"{START_DATE} -> {END_DATE}"
    )

    print()
    print("Locked parameters:")

    print(
        f"PHI          = {PHI}"
    )

    print(
        f"Q            = {Q}"
    )

    print(
        f"R            = {R}"
    )

    print(
        f"MEAN_WINDOW  = {MEAN_WINDOW}"
    )

    print(
        f"Z_WINDOW     = {Z_WINDOW}"
    )

    print(
        f"ENTRY_Z      = {ENTRY_Z}"
    )

    print(
        f"EXIT_Z       = {EXIT_Z}"
    )

    print(
        f"STOP_Z       = {STOP_Z}"
    )

    print()
    print("Original tick file:")

    print(
        RAW_FILE
    )

    print()
    print("MT5 H1 file:")

    print(
        MT5_FILE
    )

    # ========================================================
    # LOAD ORIGINAL
    # ========================================================

    original = (
        load_original_xau_h1()
    )

    # ========================================================
    # LOAD MT5
    # ========================================================

    mt5 = (
        load_mt5_xau_h1()
    )

    # ========================================================
    # ALIGN
    # ========================================================

    comparison = (
        prepare_comparison(
            original,
            mt5,
        )
    )

    # ========================================================
    # PRICE COMPARISON
    # ========================================================

    comparison = (
        calculate_price_differences(
            comparison
        )
    )

    # ========================================================
    # KALMAN COMPARISON
    # ========================================================

    comparison = (
        calculate_kalman_comparison(
            comparison
        )
    )

    # ========================================================
    # SIGNAL COMPARISON
    # ========================================================

    comparison = (
        calculate_signal_comparison(
            comparison
        )
    )

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print_summary(
        comparison
    )

    # ========================================================
    # SAVE FULL CSV
    # ========================================================

    output_file = (
        RESULTS_DIR
        / "xau_h1_comparison_2025.csv"
    )

    comparison.to_csv(
        output_file
    )

    # ========================================================
    # SAVE TEXT SUMMARY
    # ========================================================

    summary_file = (
        save_summary(
            comparison
        )
    )

    # ========================================================
    # DONE
    # ========================================================

    print()
    print("=" * 70)
    print("SAVED")
    print("=" * 70)

    print(
        "Full comparison:"
    )

    print(
        output_file
    )

    print()

    print(
        "Summary:"
    )

    print(
        summary_file
    )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()