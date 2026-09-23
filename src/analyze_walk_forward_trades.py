from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

TRADE_FILE = (
    ROOT
    / "results"
    / "walk_forward_directional"
    / "MARKOV_SHORT_FILTER_trades.csv"
)

RESULT_DIR = (
    ROOT
    / "results"
    / "walk_forward_trade_analysis"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# HELPERS
# ============================================================

def profit_factor(pnl):

    pnl = pd.Series(pnl).dropna()

    gross_profit = pnl[pnl > 0].sum()
    gross_loss = pnl[pnl < 0].sum()

    if gross_loss >= 0:
        return np.inf

    return gross_profit / abs(gross_loss)


def summarize(
    df,
    group_col=None,
):

    if df.empty:

        return pd.DataFrame()

    if group_col is None:

        groups = [
            ("ALL", df)
        ]

    else:

        groups = list(
            df.groupby(
                group_col,
                dropna=False,
            )
        )

    rows = []

    for name, group in groups:

        pnl = group["PnL"]

        wins = pnl > 0

        rows.append(
            {
                "Group": name,
                "Trades": len(group),
                "Wins": int(wins.sum()),
                "Losses": int((~wins).sum()),
                "WinRate": wins.mean(),
                "ProfitFactor": profit_factor(pnl),
                "Expectancy": pnl.mean(),
                "TotalPnL": pnl.sum(),
                "AvgPnL": pnl.mean(),
                "MedianPnL": pnl.median(),
                "AvgReturn": group["Return"].mean(),
                "MedianReturn": group["Return"].median(),
                "AvgHoursHeld": group["TradingHoursHeld"].mean(),
                "MaxHoursHeld": group["TradingHoursHeld"].max(),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# LOAD TRADES
# ============================================================

print("=" * 110)
print("WALK-FORWARD MARKOV SHORT-FILTER TRADE ATTRIBUTION")
print("=" * 110)

print()
print(f"Loading: {TRADE_FILE}")

if not TRADE_FILE.exists():

    raise FileNotFoundError(
        f"Trade file not found:\n{TRADE_FILE}"
    )

trades = pd.read_csv(
    TRADE_FILE
)

print(
    f"Trades loaded: {len(trades):,}"
)


# ============================================================
# DATETIME
# ============================================================

for col in [
    "EntryTime",
    "ExitTime",
]:

    trades[col] = pd.to_datetime(
        trades[col],
        utc=True,
    )


# ============================================================
# BASIC CLEANING
# ============================================================

numeric_columns = [
    "PnL",
    "Return",
    "EntryZ",
    "ExitZ",
    "EntryPMR",
    "EntryPTrend",
    "ExitPMR",
    "ExitPTrend",
    "TradingHoursHeld",
    "CalendarHoursHeld",
    "Units",
    "EntryPrice",
    "ExitPrice",
]

for col in numeric_columns:

    if col in trades.columns:

        trades[col] = pd.to_numeric(
            trades[col],
            errors="coerce",
        )


# ============================================================
# DERIVED FEATURES
# ============================================================

# Direction

trades["Direction"] = (
    trades["Side"]
)


# Entry regime

trades["Regime"] = (
    trades["EntryRegime"]
)


# Entry probability bucket

trades["MRProbabilityBucket"] = pd.cut(
    trades["EntryPMR"],
    bins=[
        -np.inf,
        0.40,
        0.60,
        0.80,
        np.inf,
    ],
    labels=[
        "<0.40",
        "0.40-0.60",
        "0.60-0.80",
        ">=0.80",
    ],
)


# Entry Z magnitude

abs_z = (
    trades["EntryZ"].abs()
)

trades["EntryZBucket"] = pd.cut(
    abs_z,
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


# Holding time

trades["HoldBucket"] = pd.cut(
    trades["TradingHoursHeld"],
    bins=[
        -np.inf,
        2,
        4,
        8,
        12,
        24,
        np.inf,
    ],
    labels=[
        "0-2h",
        "2-4h",
        "4-8h",
        "8-12h",
        "12-24h",
        ">24h",
    ],
    include_lowest=True,
)


# Calendar year

trades["Year"] = (
    trades["ExitTime"]
    .dt.year
)


# Month

trades["Month"] = (
    trades["ExitTime"]
    .dt.to_period("M")
    .astype(str)
)


# Positive / negative

trades["Win"] = (
    trades["PnL"] > 0
)


# ============================================================
# OVERALL
# ============================================================

overall = summarize(
    trades
)

print()
print("=" * 110)
print("OVERALL")
print("=" * 110)

print(
    overall.to_string(
        index=False,
        float_format=lambda x:
        f"{x:.6f}",
    )
)


# ============================================================
# 1. DIRECTION
# ============================================================

direction = summarize(
    trades,
    "Direction",
)

print()
print("=" * 110)
print("1. LONG VS SHORT")
print("=" * 110)

print(
    direction.to_string(
        index=False,
        float_format=lambda x:
        f"{x:.6f}",
    )
)

direction.to_csv(
    RESULT_DIR
    / "direction.csv",
    index=False,
)


# ============================================================
# 2. REGIME
# ============================================================

regime = summarize(
    trades,
    "Regime",
)

print()
print("=" * 110)
print("2. ENTRY REGIME")
print("=" * 110)

print(
    regime.to_string(
        index=False,
        float_format=lambda x:
        f"{x:.6f}",
    )
)

regime.to_csv(
    RESULT_DIR
    / "regime.csv",
    index=False,
)


# ============================================================
# 3. DIRECTION + REGIME
# ============================================================

direction_regime = (
    trades
    .groupby(
        [
            "Direction",
            "Regime",
        ],
        dropna=False,
    )
    .agg(
        Trades=("PnL", "size"),
        Wins=("Win", "sum"),
        TotalPnL=("PnL", "sum"),
        Expectancy=("PnL", "mean"),
        AvgReturn=("Return", "mean"),
        AvgHoursHeld=(
            "TradingHoursHeld",
            "mean",
        ),
    )
    .reset_index()
)

direction_regime["WinRate"] = (
    direction_regime["Wins"]
    / direction_regime["Trades"]
)

pf_rows = []

for _, row in direction_regime.iterrows():

    subset = trades[
        (trades["Direction"] == row["Direction"])
        &
        (trades["Regime"] == row["Regime"])
    ]

    pf_rows.append(
        profit_factor(
            subset["PnL"]
        )
    )

direction_regime["ProfitFactor"] = (
    pf_rows
)

print()
print("=" * 110)
print("3. DIRECTION + REGIME")
print("=" * 110)

print(
    direction_regime.to_string(
        index=False,
        float_format=lambda x:
        f"{x:.6f}",
    )
)

direction_regime.to_csv(
    RESULT_DIR
    / "direction_regime.csv",
    index=False,
)


# ============================================================
# 4. ENTRY Z
# ============================================================

z_analysis = summarize(
    trades,
    "EntryZBucket",
)

print()
print("=" * 110)
print("4. ENTRY Z-SCORE")
print("=" * 110)

print(
    z_analysis.to_string(
        index=False,
        float_format=lambda x:
        f"{x:.6f}",
    )
)

z_analysis.to_csv(
    RESULT_DIR
    / "entry_z.csv",
    index=False,
)


# ============================================================
# 5. ENTRY Z + DIRECTION
# ============================================================

z_direction = (
    trades
    .groupby(
        [
            "Direction",
            "EntryZBucket",
        ],
        observed=False,
        dropna=False,
    )
    .apply(
        lambda x: pd.Series(
            {
                "Trades":
                    len(x),

                "WinRate":
                    x["Win"].mean(),

                "ProfitFactor":
                    profit_factor(
                        x["PnL"]
                    ),

                "Expectancy":
                    x["PnL"].mean(),

                "TotalPnL":
                    x["PnL"].sum(),
            }
        ),
        include_groups=False,
    )
    .reset_index()
)

print()
print("=" * 110)
print("5. ENTRY Z + DIRECTION")
print("=" * 110)

print(
    z_direction.to_string(
        index=False,
        float_format=lambda x:
        f"{x:.6f}",
    )
)

z_direction.to_csv(
    RESULT_DIR
    / "entry_z_direction.csv",
    index=False,
)


# ============================================================
# 6. MARKOV PROBABILITY
# ============================================================

probability = summarize(
    trades,
    "MRProbabilityBucket",
)

print()
print("=" * 110)
print("6. P(MR) BUCKET")
print("=" * 110)

print(
    probability.to_string(
        index=False,
        float_format=lambda x:
        f"{x:.6f}",
    )
)

probability.to_csv(
    RESULT_DIR
    / "mr_probability.csv",
    index=False,
)


# ============================================================
# 7. EXIT REASON
# ============================================================

exit_reason = summarize(
    trades,
    "ExitReason",
)

print()
print("=" * 110)
print("7. EXIT REASON")
print("=" * 110)

print(
    exit_reason.to_string(
        index=False,
        float_format=lambda x:
        f"{x:.6f}",
    )
)

exit_reason.to_csv(
    RESULT_DIR
    / "exit_reason.csv",
    index=False,
)


# ============================================================
# 8. HOLDING TIME
# ============================================================

holding = summarize(
    trades,
    "HoldBucket",
)

print()
print("=" * 110)
print("8. HOLDING TIME")
print("=" * 110)

print(
    holding.to_string(
        index=False,
        float_format=lambda x:
        f"{x:.6f}",
    )
)

holding.to_csv(
    RESULT_DIR
    / "holding_time.csv",
    index=False,
)


# ============================================================
# 9. YEAR
# ============================================================

year = summarize(
    trades,
    "Year",
)

print()
print("=" * 110)
print("9. YEARLY TRADE ATTRIBUTION")
print("=" * 110)

print(
    year.to_string(
        index=False,
        float_format=lambda x:
        f"{x:.6f}",
    )
)

year.to_csv(
    RESULT_DIR
    / "year.csv",
    index=False,
)


# ============================================================
# 10. YEAR + DIRECTION
# ============================================================

year_direction = (
    trades
    .groupby(
        [
            "Year",
            "Direction",
        ],
        dropna=False,
    )
    .agg(
        Trades=("PnL", "size"),
        Wins=("Win", "sum"),
        TotalPnL=("PnL", "sum"),
        Expectancy=("PnL", "mean"),
    )
    .reset_index()
)

year_direction["WinRate"] = (
    year_direction["Wins"]
    / year_direction["Trades"]
)

print()
print("=" * 110)
print("10. YEAR + DIRECTION")
print("=" * 110)

print(
    year_direction.to_string(
        index=False,
        float_format=lambda x:
        f"{x:.6f}",
    )
)

year_direction.to_csv(
    RESULT_DIR
    / "year_direction.csv",
    index=False,
)


# ============================================================
# 11. YEAR + REGIME
# ============================================================

year_regime = (
    trades
    .groupby(
        [
            "Year",
            "Regime",
        ],
        dropna=False,
    )
    .agg(
        Trades=("PnL", "size"),
        Wins=("Win", "sum"),
        TotalPnL=("PnL", "sum"),
        Expectancy=("PnL", "mean"),
    )
    .reset_index()
)

year_regime["WinRate"] = (
    year_regime["Wins"]
    / year_regime["Trades"]
)

print()
print("=" * 110)
print("11. YEAR + REGIME")
print("=" * 110)

print(
    year_regime.to_string(
        index=False,
        float_format=lambda x:
        f"{x:.6f}",
    )
)

year_regime.to_csv(
    RESULT_DIR
    / "year_regime.csv",
    index=False,
)


# ============================================================
# 12. YEAR + DIRECTION + REGIME
# ============================================================

year_direction_regime = (
    trades
    .groupby(
        [
            "Year",
            "Direction",
            "Regime",
        ],
        dropna=False,
    )
    .agg(
        Trades=("PnL", "size"),
        Wins=("Win", "sum"),
        TotalPnL=("PnL", "sum"),
        Expectancy=("PnL", "mean"),
    )
    .reset_index()
)

year_direction_regime["WinRate"] = (
    year_direction_regime["Wins"]
    / year_direction_regime["Trades"]
)

print()
print("=" * 110)
print(
    "12. YEAR + DIRECTION + REGIME"
)
print("=" * 110)

print(
    year_direction_regime.to_string(
        index=False,
        float_format=lambda x:
        f"{x:.6f}",
    )
)

year_direction_regime.to_csv(
    RESULT_DIR
    / "year_direction_regime.csv",
    index=False,
)


# ============================================================
# 13. EXIT REASON BY DIRECTION
# ============================================================

exit_direction = (
    trades
    .groupby(
        [
            "Direction",
            "ExitReason",
        ],
        dropna=False,
    )
    .agg(
        Trades=("PnL", "size"),
        Wins=("Win", "sum"),
        TotalPnL=("PnL", "sum"),
        Expectancy=("PnL", "mean"),
    )
    .reset_index()
)

exit_direction["WinRate"] = (
    exit_direction["Wins"]
    / exit_direction["Trades"]
)

print()
print("=" * 110)
print("13. EXIT REASON + DIRECTION")
print("=" * 110)

print(
    exit_direction.to_string(
        index=False,
        float_format=lambda x:
        f"{x:.6f}",
    )
)

exit_direction.to_csv(
    RESULT_DIR
    / "exit_reason_direction.csv",
    index=False,
)


# ============================================================
# 14. WORST 20 TRADES
# ============================================================

worst = (
    trades
    .sort_values(
        "PnL",
        ascending=True,
    )
    .head(20)
)

print()
print("=" * 110)
print("14. WORST 20 TRADES")
print("=" * 110)

print(
    worst[
        [
            "EntryTime",
            "ExitTime",
            "Side",
            "EntryZ",
            "EntryRegime",
            "EntryPMR",
            "TradingHoursHeld",
            "ExitReason",
            "PnL",
            "Return",
        ]
    ].to_string(
        index=False,
        float_format=lambda x:
        f"{x:.6f}",
    )
)

worst.to_csv(
    RESULT_DIR
    / "worst_20_trades.csv",
    index=False,
)


# ============================================================
# 15. BEST 20 TRADES
# ============================================================

best = (
    trades
    .sort_values(
        "PnL",
        ascending=False,
    )
    .head(20)
)

print()
print("=" * 110)
print("15. BEST 20 TRADES")
print("=" * 110)

print(
    best[
        [
            "EntryTime",
            "ExitTime",
            "Side",
            "EntryZ",
            "EntryRegime",
            "EntryPMR",
            "TradingHoursHeld",
            "ExitReason",
            "PnL",
            "Return",
        ]
    ].to_string(
        index=False,
        float_format=lambda x:
        f"{x:.6f}",
    )
)

best.to_csv(
    RESULT_DIR
    / "best_20_trades.csv",
    index=False,
)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 110)
print("ANALYSIS COMPLETE")
print("=" * 110)

print()
print(
    f"Results saved to:\n{RESULT_DIR}"
)

print()

print(
    "Important: no parameters were changed."
)

print(
    "This is attribution/diagnostic analysis only."
)

print("=" * 110)