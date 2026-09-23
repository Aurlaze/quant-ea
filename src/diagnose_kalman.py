from pathlib import Path
import sys
import numpy as np
import pandas as pd


# =========================================================
# PATHS
# =========================================================

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from walk_forward_kalman import (
    walk_forward_kalman,
    calculate_zscore,
)


DATA_DIR = PROJECT_ROOT / "data" / "mt5_h1"
RESULTS_DIR = PROJECT_ROOT / "results" / "kalman_diagnostics"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# CONFIGURATION
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


# Locked parameters from current strategy
PHI = 0.999
Q = 0.25
R = 0.25

MEAN_WINDOW = 168
Z_WINDOW = 36

ENTRY_Z = 2.25
EXIT_Z = 0.25
STOP_Z = 3.0

MAX_HOLD_HOURS = 24


START_DATE = pd.Timestamp(
    "2019-01-01",
    tz="UTC",
)

END_DATE = pd.Timestamp(
    "2026-09-19",
    tz="UTC",
)


# =========================================================
# DATA LOADING
# =========================================================

def load_data(symbol):

    path = (
        DATA_DIR
        / f"{symbol}_H1_2019_2026.csv"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Missing file:\n{path}"
        )

    df = pd.read_csv(path)

    # -----------------------------------------------------
    # Normalize time
    # -----------------------------------------------------

    time_candidates = [
        "time",
        "Time",
        "timestamp",
        "Timestamp",
        "datetime",
        "Datetime",
        "date",
        "Date",
    ]

    time_col = None

    for col in time_candidates:

        if col in df.columns:

            time_col = col
            break

    if time_col is None:

        raise ValueError(
            f"{symbol}: cannot find time column.\n"
            f"Columns: {list(df.columns)}"
        )

    # -----------------------------------------------------
    # Normalize OHLC
    # -----------------------------------------------------

    def find_col(name):

        for col in df.columns:

            if str(col).lower() == name.lower():

                return col

        return None

    open_col = find_col("open")
    high_col = find_col("high")
    low_col = find_col("low")
    close_col = find_col("close")

    if None in [
        open_col,
        high_col,
        low_col,
        close_col,
    ]:

        raise ValueError(
            f"{symbol}: missing OHLC columns."
        )

    out = pd.DataFrame()

    out["Time"] = pd.to_datetime(
        df[time_col],
        utc=True,
        errors="coerce",
    )

    out["Open"] = pd.to_numeric(
        df[open_col],
        errors="coerce",
    )

    out["High"] = pd.to_numeric(
        df[high_col],
        errors="coerce",
    )

    out["Low"] = pd.to_numeric(
        df[low_col],
        errors="coerce",
    )

    out["Close"] = pd.to_numeric(
        df[close_col],
        errors="coerce",
    )

    # -----------------------------------------------------
    # Clean
    # -----------------------------------------------------

    out = out.dropna()

    out = out[
        (out["Open"] > 0)
        & (out["High"] > 0)
        & (out["Low"] > 0)
        & (out["Close"] > 0)
    ]

    out = out.sort_values("Time")

    out = out.drop_duplicates(
        subset=["Time"]
    )

    # -----------------------------------------------------
    # Period
    # -----------------------------------------------------

    out = out[
        (out["Time"] >= START_DATE)
        & (out["Time"] <= END_DATE)
    ].copy()

    out = out.reset_index(drop=True)

    return out


# =========================================================
# KALMAN FEATURES
# =========================================================

def calculate_features(df):

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
        dtype=float,
    )

    zscore = calculate_zscore(
        residual_series,
        window=Z_WINDOW,
    )

    out = df.copy()

    out["KalmanState"] = states

    out["KalmanVariance"] = variances

    out["Residual"] = residuals

    out["RollingMean"] = means

    out["ZScore"] = zscore

    return out


# =========================================================
# FORWARD RETURN ANALYSIS
# =========================================================

def calculate_forward_returns(
    df,
    horizons=(1, 3, 6, 12, 24),
):

    out = df.copy()

    for h in horizons:

        out[
            f"ForwardReturn_{h}h"
        ] = (
            out["Close"].shift(-h)
            / out["Close"]
            - 1.0
        )

    return out


# =========================================================
# ZONE ANALYSIS
# =========================================================

def zone_label(z):

    if pd.isna(z):

        return "INVALID"

    if z <= -3.0:

        return "<=-3"

    if z <= -2.25:

        return "-3_to_-2.25"

    if z <= -1.5:

        return "-2.25_to_-1.5"

    if z <= -1.0:

        return "-1.5_to_-1"

    if z < 1.0:

        return "-1_to_1"

    if z < 1.5:

        return "1_to_1.5"

    if z < 2.25:

        return "1.5_to_2.25"

    if z < 3.0:

        return "2.25_to_3"

    return ">=3"


# =========================================================
# EXTREME Z ANALYSIS
# =========================================================

def analyze_zones(df):

    valid = df[
        df["ZScore"].notna()
    ].copy()

    valid["Zone"] = (
        valid["ZScore"]
        .apply(zone_label)
    )

    horizons = [
        1,
        3,
        6,
        12,
        24,
    ]

    rows = []

    for zone, group in valid.groupby(
        "Zone",
        sort=False,
    ):

        row = {
            "Zone": zone,
            "Observations": len(group),
        }

        for h in horizons:

            values = (
                group[
                    f"ForwardReturn_{h}h"
                ]
                .dropna()
            )

            if len(values) > 0:

                row[
                    f"MeanForwardReturn_{h}h"
                ] = values.mean()

                row[
                    f"MedianForwardReturn_{h}h"
                ] = values.median()

                row[
                    f"WinRate_{h}h"
                ] = (
                    values > 0
                ).mean()

            else:

                row[
                    f"MeanForwardReturn_{h}h"
                ] = np.nan

                row[
                    f"MedianForwardReturn_{h}h"
                ] = np.nan

                row[
                    f"WinRate_{h}h"
                ] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


# =========================================================
# EXTREME DEVIATION ANALYSIS
# =========================================================

def analyze_extremes(df):

    valid = df[
        df["ZScore"].notna()
    ].copy()

    horizons = [
        1,
        3,
        6,
        12,
        24,
    ]

    conditions = {

        "LongSignal":
            valid["ZScore"] <= -ENTRY_Z,

        "ShortSignal":
            valid["ZScore"] >= ENTRY_Z,

        "LongExtreme":
            valid["ZScore"] <= -STOP_Z,

        "ShortExtreme":
            valid["ZScore"] >= STOP_Z,

        "ModerateLong":
            (
                valid["ZScore"] <= -1.5
            )
            & (
                valid["ZScore"] > -ENTRY_Z
            ),

        "ModerateShort":
            (
                valid["ZScore"] >= 1.5
            )
            & (
                valid["ZScore"] < ENTRY_Z
            ),
    }

    rows = []

    for name, mask in conditions.items():

        group = valid.loc[mask]

        row = {
            "Condition": name,
            "Observations": len(group),
        }

        for h in horizons:

            values = (
                group[
                    f"ForwardReturn_{h}h"
                ]
                .dropna()
            )

            if len(values) > 0:

                row[
                    f"MeanForwardReturn_{h}h"
                ] = values.mean()

                row[
                    f"MedianForwardReturn_{h}h"
                ] = values.median()

                row[
                    f"WinRate_{h}h"
                ] = (
                    values > 0
                ).mean()

            else:

                row[
                    f"MeanForwardReturn_{h}h"
                ] = np.nan

                row[
                    f"MedianForwardReturn_{h}h"
                ] = np.nan

                row[
                    f"WinRate_{h}h"
                ] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


# =========================================================
# LONG / SHORT EXPECTED REVERSION
# =========================================================

def analyze_signal_direction(df):

    valid = df[
        df["ZScore"].notna()
    ].copy()

    horizons = [
        1,
        3,
        6,
        12,
        24,
    ]

    rows = []

    # -----------------------------------------------------
    # LONG
    #
    # If Z is negative and mean reversion works,
    # future return should be positive.
    # -----------------------------------------------------

    long_group = valid[
        valid["ZScore"] <= -ENTRY_Z
    ]

    row = {
        "Signal": "LONG",
        "Observations": len(long_group),
    }

    for h in horizons:

        values = (
            long_group[
                f"ForwardReturn_{h}h"
            ]
            .dropna()
        )

        row[
            f"MeanForwardReturn_{h}h"
        ] = (
            values.mean()
            if len(values)
            else np.nan
        )

        row[
            f"MedianForwardReturn_{h}h"
        ] = (
            values.median()
            if len(values)
            else np.nan
        )

    rows.append(row)

    # -----------------------------------------------------
    # SHORT
    #
    # If mean reversion works,
    # future return should be negative.
    # -----------------------------------------------------

    short_group = valid[
        valid["ZScore"] >= ENTRY_Z
    ]

    row = {
        "Signal": "SHORT",
        "Observations": len(short_group),
    }

    for h in horizons:

        values = (
            short_group[
                f"ForwardReturn_{h}h"
            ]
            .dropna()
        )

        row[
            f"MeanForwardReturn_{h}h"
        ] = (
            values.mean()
            if len(values)
            else np.nan
        )

        row[
            f"MedianForwardReturn_{h}h"
        ] = (
            values.median()
            if len(values)
            else np.nan
        )

    rows.append(row)

    return pd.DataFrame(rows)


# =========================================================
# TREND / MOMENTUM DIAGNOSTIC
# =========================================================

def analyze_persistence(df):

    valid = df[
        df["ZScore"].notna()
    ].copy()

    horizons = [
        1,
        3,
        6,
        12,
        24,
    ]

    rows = []

    conditions = {

        "StrongNegative":
            valid["ZScore"] <= -ENTRY_Z,

        "StrongPositive":
            valid["ZScore"] >= ENTRY_Z,

        "VeryNegative":
            valid["ZScore"] <= -STOP_Z,

        "VeryPositive":
            valid["ZScore"] >= STOP_Z,
    }

    for name, mask in conditions.items():

        group = valid.loc[mask]

        row = {
            "Condition": name,
            "Observations": len(group),
        }

        for h in horizons:

            returns = (
                group[
                    f"ForwardReturn_{h}h"
                ]
                .dropna()
            )

            row[
                f"Mean_{h}h"
            ] = (
                returns.mean()
                if len(returns)
                else np.nan
            )

        rows.append(row)

    return pd.DataFrame(rows)


# =========================================================
# ANNUAL ANALYSIS
# =========================================================

def analyze_years(df):

    valid = df[
        df["ZScore"].notna()
    ].copy()

    valid["Year"] = (
        valid["Time"]
        .dt.year
    )

    rows = []

    for year, group in valid.groupby(
        "Year"
    ):

        row = {
            "Year": year,
            "Observations": len(group),
        }

        # ---------------------------------------------
        # Z distribution
        # ---------------------------------------------

        row["MeanZ"] = (
            group["ZScore"].mean()
        )

        row["StdZ"] = (
            group["ZScore"].std()
        )

        row["MinZ"] = (
            group["ZScore"].min()
        )

        row["MaxZ"] = (
            group["ZScore"].max()
        )

        # ---------------------------------------------
        # Signal counts
        # ---------------------------------------------

        row["LongSignals"] = int(
            (
                group["ZScore"]
                <= -ENTRY_Z
            ).sum()
        )

        row["ShortSignals"] = int(
            (
                group["ZScore"]
                >= ENTRY_Z
            ).sum()
        )

        # ---------------------------------------------
        # Forward returns
        # ---------------------------------------------

        for h in [1, 3, 6, 12, 24]:

            returns = (
                group[
                    f"ForwardReturn_{h}h"
                ]
                .dropna()
            )

            row[
                f"ForwardMean_{h}h"
            ] = (
                returns.mean()
                if len(returns)
                else np.nan
            )

        rows.append(row)

    return pd.DataFrame(rows)


# =========================================================
# Z-SCORE DISTRIBUTION
# =========================================================

def calculate_distribution(df):

    z = (
        df["ZScore"]
        .dropna()
    )

    if len(z) == 0:

        return {
            "ValidZ": 0,
        }

    result = {
        "ValidZ": len(z),

        "MeanZ": z.mean(),
        "StdZ": z.std(),

        "MinZ": z.min(),
        "MaxZ": z.max(),

        "PctAbsZ_gt_1":
            (z.abs() > 1).mean(),

        "PctAbsZ_gt_1.5":
            (z.abs() > 1.5).mean(),

        "PctAbsZ_gt_2":
            (z.abs() > 2).mean(),

        "PctAbsZ_gt_2.25":
            (z.abs() > 2.25).mean(),

        "PctAbsZ_gt_2.5":
            (z.abs() > 2.5).mean(),

        "PctAbsZ_gt_3":
            (z.abs() > 3).mean(),

        "PctZ_lt_-2.25":
            (z < -2.25).mean(),

        "PctZ_gt_2.25":
            (z > 2.25).mean(),

        "PctZ_lt_-3":
            (z < -3).mean(),

        "PctZ_gt_3":
            (z > 3).mean(),
    }

    return result


# =========================================================
# VOLATILITY ANALYSIS
# =========================================================

def analyze_volatility(df):

    out = df.copy()

    out["HourlyReturn"] = (
        out["Close"]
        .pct_change()
    )

    out["Volatility24h"] = (
        out["HourlyReturn"]
        .rolling(24)
        .std()
    )

    valid = out[
        out["ZScore"].notna()
        & out["Volatility24h"].notna()
    ].copy()

    if len(valid) == 0:

        return pd.DataFrame()

    # -----------------------------------------------------
    # Split into volatility regimes
    # -----------------------------------------------------

    low_threshold = (
        valid["Volatility24h"]
        .quantile(0.33)
    )

    high_threshold = (
        valid["Volatility24h"]
        .quantile(0.67)
    )

    valid["VolatilityRegime"] = np.select(

        [
            valid["Volatility24h"]
            <= low_threshold,

            valid["Volatility24h"]
            >= high_threshold,
        ],

        [
            "LOW",
            "HIGH",
        ],

        default="MEDIUM",
    )

    rows = []

    for regime, group in valid.groupby(
        "VolatilityRegime"
    ):

        row = {
            "VolatilityRegime": regime,
            "Observations": len(group),
            "MeanVolatility24h":
                group[
                    "Volatility24h"
                ].mean(),

            "SignalCount": int(
                (
                    group["ZScore"].abs()
                    >= ENTRY_Z
                ).sum()
            ),
        }

        for h in [1, 3, 6, 12, 24]:

            values = (
                group[
                    f"ForwardReturn_{h}h"
                ]
                .dropna()
            )

            row[
                f"MeanForwardReturn_{h}h"
            ] = (
                values.mean()
                if len(values)
                else np.nan
            )

        rows.append(row)

    return pd.DataFrame(rows)


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 90)
    print("KALMAN MEAN REVERSION DIAGNOSTIC")
    print("=" * 90)

    print()
    print("This script DOES NOT optimize parameters.")
    print("It diagnoses why the current signal succeeds/fails.")
    print()

    print(
        f"PHI={PHI}, Q={Q}, R={R}, "
        f"MEAN={MEAN_WINDOW}, Z={Z_WINDOW}, "
        f"ENTRY={ENTRY_Z}, EXIT={EXIT_Z}, STOP={STOP_Z}"
    )

    all_distribution = []
    all_signal = []
    all_years = []
    all_volatility = []

    for symbol in INSTRUMENTS:

        print()
        print("#" * 90)
        print(f"DIAGNOSING {symbol}")
        print("#" * 90)

        try:

            df = load_data(symbol)

            print(
                f"Bars: {len(df):,}"
            )

            df = calculate_features(df)

            df = calculate_forward_returns(df)

            # -------------------------------------------------
            # Distribution
            # -------------------------------------------------

            distribution = (
                calculate_distribution(df)
            )

            distribution["Symbol"] = symbol

            all_distribution.append(
                distribution
            )

            # -------------------------------------------------
            # Zones
            # -------------------------------------------------

            zone_df = analyze_zones(df)

            zone_df.insert(
                0,
                "Symbol",
                symbol,
            )

            zone_df.to_csv(
                RESULTS_DIR
                / f"{symbol}_zones.csv",
                index=False,
            )

            # -------------------------------------------------
            # Signal direction
            # -------------------------------------------------

            signal_df = (
                analyze_signal_direction(df)
            )

            signal_df.insert(
                0,
                "Symbol",
                symbol,
            )

            signal_df.to_csv(
                RESULTS_DIR
                / f"{symbol}_signal_direction.csv",
                index=False,
            )

            all_signal.append(
                signal_df
            )

            # -------------------------------------------------
            # Extreme analysis
            # -------------------------------------------------

            extreme_df = (
                analyze_extremes(df)
            )

            extreme_df.insert(
                0,
                "Symbol",
                symbol,
            )

            extreme_df.to_csv(
                RESULTS_DIR
                / f"{symbol}_extremes.csv",
                index=False,
            )

            # -------------------------------------------------
            # Persistence
            # -------------------------------------------------

            persistence_df = (
                analyze_persistence(df)
            )

            persistence_df.insert(
                0,
                "Symbol",
                symbol,
            )

            persistence_df.to_csv(
                RESULTS_DIR
                / f"{symbol}_persistence.csv",
                index=False,
            )

            # -------------------------------------------------
            # Yearly
            # -------------------------------------------------

            yearly_df = analyze_years(df)

            yearly_df.insert(
                0,
                "Symbol",
                symbol,
            )

            yearly_df.to_csv(
                RESULTS_DIR
                / f"{symbol}_yearly.csv",
                index=False,
            )

            all_years.append(
                yearly_df
            )

            # -------------------------------------------------
            # Volatility
            # -------------------------------------------------

            volatility_df = (
                analyze_volatility(df)
            )

            if len(volatility_df) > 0:

                volatility_df.insert(
                    0,
                    "Symbol",
                    symbol,
                )

                volatility_df.to_csv(
                    RESULTS_DIR
                    / f"{symbol}_volatility.csv",
                    index=False,
                )

                all_volatility.append(
                    volatility_df
                )

            # -------------------------------------------------
            # Save processed data
            # -------------------------------------------------

            df.to_csv(
                RESULTS_DIR
                / f"{symbol}_diagnostic_data.csv",
                index=False,
            )

            # -------------------------------------------------
            # Console summary
            # -------------------------------------------------

            print()
            print("Z DISTRIBUTION")
            print("-" * 60)

            print(
                f"Valid Z             : "
                f"{distribution['ValidZ']:,}"
            )

            print(
                f"Mean Z              : "
                f"{distribution['MeanZ']:.4f}"
            )

            print(
                f"Std Z               : "
                f"{distribution['StdZ']:.4f}"
            )

            print(
                f"|Z| > 2.25          : "
                f"{distribution['PctAbsZ_gt_2.25'] * 100:.2f}%"
            )

            print(
                f"Z < -2.25           : "
                f"{distribution['PctZ_lt_-2.25'] * 100:.2f}%"
            )

            print(
                f"Z > +2.25           : "
                f"{distribution['PctZ_gt_2.25'] * 100:.2f}%"
            )

            print(
                f"|Z| > 3             : "
                f"{(
                    distribution['PctZ_lt_-3']
                    + distribution['PctZ_gt_3']
                ) * 100:.2f}%"
            )

            print()
            print("SIGNAL DIRECTION")
            print("-" * 60)

            print(
                signal_df.to_string(
                    index=False
                )
            )

        except Exception as e:

            print()
            print(
                f"ERROR: {symbol}"
            )

            print(
                repr(e)
            )

    # =====================================================
    # COMBINED OUTPUTS
    # =====================================================

    if all_distribution:

        distribution_df = pd.DataFrame(
            all_distribution
        )

        distribution_df = (
            distribution_df
            .set_index("Symbol")
        )

        distribution_df.to_csv(
            RESULTS_DIR
            / "all_z_distributions.csv"
        )

    if all_signal:

        signal_df = pd.concat(
            all_signal,
            ignore_index=True,
        )

        signal_df.to_csv(
            RESULTS_DIR
            / "all_signal_direction.csv",
            index=False,
        )

    if all_years:

        yearly_df = pd.concat(
            all_years,
            ignore_index=True,
        )

        yearly_df.to_csv(
            RESULTS_DIR
            / "all_yearly_diagnostics.csv",
            index=False,
        )

    if all_volatility:

        volatility_df = pd.concat(
            all_volatility,
            ignore_index=True,
        )

        volatility_df.to_csv(
            RESULTS_DIR
            / "all_volatility_diagnostics.csv",
            index=False,
        )

    # =====================================================
    # FINAL DISTRIBUTION TABLE
    # =====================================================

    if all_distribution:

        final_df = pd.DataFrame(
            all_distribution
        )

        cols = [
            "Symbol",
            "ValidZ",
            "MeanZ",
            "StdZ",
            "PctAbsZ_gt_1.5",
            "PctAbsZ_gt_2",
            "PctAbsZ_gt_2.25",
            "PctAbsZ_gt_3",
            "PctZ_lt_-2.25",
            "PctZ_gt_2.25",
        ]

        final_df = final_df[
            cols
        ].copy()

        for col in [
            "PctAbsZ_gt_1.5",
            "PctAbsZ_gt_2",
            "PctAbsZ_gt_2.25",
            "PctAbsZ_gt_3",
            "PctZ_lt_-2.25",
            "PctZ_gt_2.25",
        ]:

            final_df[col] *= 100

        print()
        print()
        print("=" * 120)
        print("FINAL Z-SCORE DIAGNOSTIC")
        print("=" * 120)

        print(
            final_df.to_string(
                index=False,
                formatters={
                    "MeanZ":
                        "{:.4f}".format,

                    "StdZ":
                        "{:.4f}".format,

                    "PctAbsZ_gt_1.5":
                        "{:.2f}%".format,

                    "PctAbsZ_gt_2":
                        "{:.2f}%".format,

                    "PctAbsZ_gt_2.25":
                        "{:.2f}%".format,

                    "PctAbsZ_gt_3":
                        "{:.2f}%".format,

                    "PctZ_lt_-2.25":
                        "{:.2f}%".format,

                    "PctZ_gt_2.25":
                        "{:.2f}%".format,
                },
            )
        )

    print()
    print("=" * 90)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 90)

    print()
    print(
        f"Results saved to:\n{RESULTS_DIR}"
    )


if __name__ == "__main__":
    main()