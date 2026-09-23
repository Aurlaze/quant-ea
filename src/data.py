from pathlib import Path
import numpy as np
import pandas as pd


def load_exness_ticks(csv_path: str) -> pd.DataFrame:
    """
    Load and clean Exness tick data.

    Required columns:
        Timestamp
        Bid
        Ask
    """

    print(f"Loading: {csv_path}")

    df = pd.read_csv(csv_path)

    required = {"Timestamp", "Bid", "Ask"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Timestamp → UTC datetime
    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        utc=True
    )

    # Make sure prices are numeric
    df["Bid"] = pd.to_numeric(
        df["Bid"],
        errors="coerce"
    )

    df["Ask"] = pd.to_numeric(
        df["Ask"],
        errors="coerce"
    )

    # Remove invalid rows
    df = df.dropna(
        subset=["Timestamp", "Bid", "Ask"]
    )

    df = df[
        (df["Bid"] > 0) &
        (df["Ask"] > 0) &
        (df["Ask"] >= df["Bid"])
    ]

    # Sort chronologically
    df = df.sort_values("Timestamp")

    # IMPORTANT:
    # Do NOT remove duplicate timestamps.
    # Multiple ticks can legitimately have
    # the same millisecond timestamp.

    # Mid price
    df["Mid"] = (
        df["Bid"] + df["Ask"]
    ) / 2

    # Spread
    df["Spread"] = (
        df["Ask"] - df["Bid"]
    )

    return df


def ticks_to_bars(
    ticks: pd.DataFrame,
    timeframe: str = "1h"
) -> pd.DataFrame:

    bars = (
        ticks
        .set_index("Timestamp")
        .resample(timeframe)
        .agg(
            Open=("Mid", "first"),
            High=("Mid", "max"),
            Low=("Mid", "min"),
            Close=("Mid", "last"),

            Bid_Open=("Bid", "first"),
            Ask_Open=("Ask", "first"),

            Bid_Close=("Bid", "last"),
            Ask_Close=("Ask", "last"),

            Spread_Mean=("Spread", "mean"),
            Tick_Count=("Mid", "size"),
        )
    )

    bars = bars.dropna(
        subset=[
            "Open",
            "High",
            "Low",
            "Close"
        ]
    )

    bars["LogReturn"] = np.log(
        bars["Close"]
        / bars["Close"].shift(1)
    )

    return bars