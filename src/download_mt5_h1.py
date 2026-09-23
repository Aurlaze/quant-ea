from datetime import datetime, timezone
from pathlib import Path

import MetaTrader5 as mt5
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

# Download broad history first.
# We can later restrict the analysis to the exact 7-year window.
START_DATE = datetime(2019, 1, 1, tzinfo=timezone.utc)
END_DATE = datetime(2026, 9, 20, tzinfo=timezone.utc)

TIMEFRAME = mt5.TIMEFRAME_H1

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "mt5_h1"


# ============================================================
# CONNECTION
# ============================================================

def connect_mt5():
    print("=" * 70)
    print("CONNECTING TO METATRADER 5")
    print("=" * 70)

    if not mt5.initialize():
        print("MT5 initialize() failed")
        print("Error:", mt5.last_error())
        raise SystemExit(1)

    terminal = mt5.terminal_info()

    if terminal is None:
        print("Could not retrieve terminal information.")
        print("Error:", mt5.last_error())
        mt5.shutdown()
        raise SystemExit(1)

    account = mt5.account_info()

    print(f"Terminal: {terminal.name}")
    print(f"Company:  {terminal.company}")

    if account is not None:
        print(f"Account:  {account.login}")
        print(f"Server:   {account.server}")
        print(f"Balance:  {account.balance}")
        print(f"Currency: {account.currency}")

    print("MT5 connection successful.")
    print()


# ============================================================
# DOWNLOAD ONE SYMBOL
# ============================================================

def download_symbol(symbol: str):

    print("=" * 70)
    print(f"DOWNLOADING {symbol}")
    print("=" * 70)

    # Make sure the symbol is available to the terminal.
    selected = mt5.symbol_select(symbol, True)

    if not selected:
        print(f"[ERROR] Could not select {symbol}")
        print("MT5 error:", mt5.last_error())
        return None

    info = mt5.symbol_info(symbol)

    if info is None:
        print(f"[ERROR] Could not retrieve symbol info for {symbol}")
        print("MT5 error:", mt5.last_error())
        return None

    print(f"Description: {info.description}")
    print(f"Digits:     {info.digits}")
    print(f"Point:      {info.point}")
    print()

    # Request H1 history.
    rates = mt5.copy_rates_range(
        symbol,
        TIMEFRAME,
        START_DATE,
        END_DATE,
    )

    if rates is None:
        print(f"[ERROR] No data returned for {symbol}")
        print("MT5 error:", mt5.last_error())
        return None

    if len(rates) == 0:
        print(f"[ERROR] Empty data returned for {symbol}")
        return None

    # Convert numpy structured array to DataFrame.
    df = pd.DataFrame(rates)

    # Convert Unix timestamp to UTC datetime.
    df["time"] = pd.to_datetime(
        df["time"],
        unit="s",
        utc=True,
    )

    # Rename columns to our project naming convention.
    df = df.rename(
        columns={
            "time": "Timestamp",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "tick_volume": "TickVolume",
            "spread": "SpreadPoints",
            "real_volume": "RealVolume",
        }
    )

    # Sort chronologically.
    df = df.sort_values("Timestamp")
    df = df.drop_duplicates(subset="Timestamp")
    df = df.reset_index(drop=True)

    # Basic validation.
    df = df[
        (df["Open"] > 0)
        & (df["High"] > 0)
        & (df["Low"] > 0)
        & (df["Close"] > 0)
    ]

    # Add log return for research.
    df["LogReturn"] = (
        pd.Series(
            pd.NA,
            index=df.index,
            dtype="Float64",
        )
    )

    df.loc[1:, "LogReturn"] = (
        (df.loc[1:, "Close"].astype(float)
         / df.loc[:-1, "Close"].astype(float))
        .apply(lambda x: __import__("math").log(x))
        .values
    )

    # Output path.
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        OUTPUT_DIR
        / f"{symbol}_H1_2019_2026.csv"
    )

    df.to_csv(
        output_file,
        index=False,
    )

    # Summary.
    print(f"Bars:       {len(df):,}")
    print(f"Start:      {df['Timestamp'].iloc[0]}")
    print(f"End:        {df['Timestamp'].iloc[-1]}")
    print(f"File:       {output_file}")
    print(f"File size:  {output_file.stat().st_size / (1024**2):.2f} MB")
    print()

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    connect_mt5()

    successful = []
    failed = []

    try:

        for symbol in SYMBOLS:

            try:
                df = download_symbol(symbol)

                if df is not None and len(df) > 0:
                    successful.append(
                        (symbol, len(df))
                    )
                else:
                    failed.append(symbol)

            except Exception as e:
                print(f"[ERROR] {symbol}: {e}")
                failed.append(symbol)

            print()

    finally:
        mt5.shutdown()

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)

    print("\nSuccessful:")
    for symbol, count in successful:
        print(f"  {symbol:<10} {count:>8,} H1 bars")

    if failed:
        print("\nFailed:")
        for symbol in failed:
            print(f"  {symbol}")

    print()
    print(f"Output directory:")
    print(f"  {OUTPUT_DIR}")

    print()
    print("MT5 connection closed.")


if __name__ == "__main__":
    main()