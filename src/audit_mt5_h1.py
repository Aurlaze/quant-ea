from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

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

DATA_DIR = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "mt5_h1"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data(symbol: str) -> pd.DataFrame:

    path = DATA_DIR / f"{symbol}_H1_2019_2026.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    df = pd.read_csv(path)

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        utc=True,
    )

    df = df.sort_values("Timestamp").reset_index(drop=True)

    return df


# ============================================================
# AUDIT ONE SYMBOL
# ============================================================

def audit_symbol(symbol: str):

    print()
    print("=" * 75)
    print(f"AUDITING {symbol}")
    print("=" * 75)

    df = load_data(symbol)

    # --------------------------------------------------------
    # Basic information
    # --------------------------------------------------------

    print(f"Rows:       {len(df):,}")
    print(f"Start:      {df['Timestamp'].iloc[0]}")
    print(f"End:        {df['Timestamp'].iloc[-1]}")

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_columns = [
        "Timestamp",
        "Open",
        "High",
        "Low",
        "Close",
        "TickVolume",
        "SpreadPoints",
        "RealVolume",
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        print(
            f"[FAIL] Missing columns: {missing_columns}"
        )
    else:
        print("[OK] Required columns present")

    # --------------------------------------------------------
    # Duplicate timestamps
    # --------------------------------------------------------

    duplicate_count = df["Timestamp"].duplicated().sum()

    print(
        f"Duplicate timestamps: {duplicate_count:,}"
    )

    # --------------------------------------------------------
    # NaN values
    # --------------------------------------------------------

    nan_counts = df[
        [
            "Timestamp",
            "Open",
            "High",
            "Low",
            "Close",
            "TickVolume",
            "SpreadPoints",
        ]
    ].isna().sum()

    total_nan = nan_counts.sum()

    print(f"NaN values:           {total_nan:,}")

    if total_nan > 0:
        print("NaN breakdown:")
        print(nan_counts[nan_counts > 0])

    # --------------------------------------------------------
    # Invalid prices
    # --------------------------------------------------------

    invalid_price_mask = (
        (df["Open"] <= 0)
        | (df["High"] <= 0)
        | (df["Low"] <= 0)
        | (df["Close"] <= 0)
    )

    invalid_prices = invalid_price_mask.sum()

    print(
        f"Invalid prices:       {invalid_prices:,}"
    )

    # --------------------------------------------------------
    # OHLC consistency
    # --------------------------------------------------------

    invalid_ohlc_mask = (
        (df["High"] < df["Open"])
        | (df["High"] < df["Close"])
        | (df["High"] < df["Low"])
        | (df["Low"] > df["Open"])
        | (df["Low"] > df["Close"])
        | (df["Low"] > df["High"])
    )

    invalid_ohlc = invalid_ohlc_mask.sum()

    print(
        f"Invalid OHLC bars:    {invalid_ohlc:,}"
    )

    # --------------------------------------------------------
    # Spread statistics
    # --------------------------------------------------------

    spread = df["SpreadPoints"]

    print()
    print("Spread statistics:")
    print(
        f"  Mean:       {spread.mean():.4f} points"
    )
    print(
        f"  Median:     {spread.median():.4f} points"
    )
    print(
        f"  Std:        {spread.std():.4f} points"
    )
    print(
        f"  Minimum:    {spread.min():.4f} points"
    )
    print(
        f"  Maximum:    {spread.max():.4f} points"
    )
    print(
        f"  95th pct:   {spread.quantile(0.95):.4f} points"
    )
    print(
        f"  99th pct:   {spread.quantile(0.99):.4f} points"
    )

    # --------------------------------------------------------
    # Zero spread
    # --------------------------------------------------------

    zero_spread = (
        df["SpreadPoints"] <= 0
    ).sum()

    print(
        f"Zero/negative spread: {zero_spread:,}"
    )

    # --------------------------------------------------------
    # Time gaps
    # --------------------------------------------------------

    time_diff = df["Timestamp"].diff()

    # A normal H1 sequence has 1 hour between bars.
    one_hour = pd.Timedelta(hours=1)

    gaps = time_diff[
        time_diff > one_hour
    ]

    print()
    print(
        f"Time gaps > 1 hour:   {len(gaps):,}"
    )

    if len(gaps) > 0:

        print(
            f"Largest gap:          {time_diff.max()}"
        )

        largest_gap_index = time_diff.idxmax()

        if pd.notna(largest_gap_index):

            previous_time = df.loc[
                largest_gap_index - 1,
                "Timestamp",
            ]

            next_time = df.loc[
                largest_gap_index,
                "Timestamp",
            ]

            print(
                f"  Before: {previous_time}"
            )
            print(
                f"  After:  {next_time}"
            )

    # --------------------------------------------------------
    # Return statistics
    # --------------------------------------------------------

    returns = np.log(
        df["Close"]
        / df["Close"].shift(1)
    ).dropna()

    print()
    print("Hourly return statistics:")

    print(
        f"  Mean:       {returns.mean():.8f}"
    )
    print(
        f"  Std:        {returns.std():.8f}"
    )
    print(
        f"  Minimum:    {returns.min():.8f}"
    )
    print(
        f"  Maximum:    {returns.max():.8f}"
    )

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    problems = (
        missing_columns
        or duplicate_count > 0
        or total_nan > 0
        or invalid_prices > 0
        or invalid_ohlc > 0
        or zero_spread > 0
    )

    if problems:
        print()
        print("STATUS: REVIEW REQUIRED")
    else:
        print()
        print("STATUS: BASIC DATA CHECK PASSED")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("MT5 H1 DATA AUDIT")
    print("=" * 75)

    print(f"Data directory:")
    print(f"  {DATA_DIR}")

    for symbol in SYMBOLS:

        try:
            audit_symbol(symbol)

        except Exception as e:

            print()
            print(
                f"[ERROR] {symbol}: {e}"
            )

    print()
    print("=" * 75)
    print("AUDIT COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()