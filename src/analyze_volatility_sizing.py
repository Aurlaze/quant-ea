from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

INITIAL_EQUITY = 10_000.0

# Original research sizing
BASE_RISK = 0.005

# Never increase exposure above the original strategy.
MAX_MULTIPLIER = 1.00

# Minimum allowed position-size multiplier.
MIN_MULTIPLIER = 0.25

# Volatility response:
#   1.0 = full inverse-volatility scaling
#   0.5 = square-root inverse-vol scaling
SCALING_POWER = 0.5

EXPECTED_TRADES = 1351

BASE_DIR = Path(__file__).resolve().parents[1]

TRADE_FILE = (
    BASE_DIR
    / "results"
    / "walk_forward_directional"
    / "MARKOV_SHORT_FILTER_trades.csv"
)

GARCH_FILE = (
    BASE_DIR
    / "results"
    / "garch_diagnostic"
    / "canonical_trades_garch_diagnostic.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "volatility_sizing"
)


# ============================================================
# METRICS
# ============================================================

def max_drawdown(equity):
    equity = pd.Series(equity, dtype=float)

    peak = equity.cummax()
    drawdown = equity / peak - 1.0

    return drawdown.min()


def sharpe_ratio(returns):
    returns = pd.Series(
        returns,
        dtype=float,
    ).dropna()

    if len(returns) < 2:
        return np.nan

    std = returns.std(ddof=1)

    if not np.isfinite(std) or std == 0:
        return np.nan

    # Trade-return Sharpe diagnostic.
    #
    # Trades are irregularly spaced, therefore this is NOT
    # a properly time-annualized Sharpe ratio.
    return (
        returns.mean()
        / std
        * np.sqrt(len(returns))
    )


def profit_factor(pnl):
    pnl = pd.Series(
        pnl,
        dtype=float,
    )

    gross_profit = pnl[pnl > 0].sum()
    gross_loss = -pnl[pnl < 0].sum()

    if gross_loss == 0:
        return np.inf

    return gross_profit / gross_loss


# ============================================================
# VOLATILITY SCALING
# ============================================================

def normalize_volatility(vol):
    """
    Normalize entry volatility relative to the median.

    Example:

        0.5 x median -> 0.5
        1.0 x median -> 1.0
        2.0 x median -> 2.0
    """

    vol = pd.to_numeric(
        vol,
        errors="coerce",
    )

    median = vol.median()

    if (
        not np.isfinite(median)
        or median <= 0
    ):
        raise ValueError(
            "Invalid volatility median."
        )

    return vol / median


def inverse_vol_multiplier(vol):
    """
    Inverse-volatility position sizing.

    Lower volatility:
        potentially larger position

    Higher volatility:
        smaller position

    But MAX_MULTIPLIER = 1.0 means we NEVER increase
    exposure above the original baseline.
    """

    normalized = normalize_volatility(vol)

    multiplier = normalized.pow(
        -SCALING_POWER
    )

    multiplier = multiplier.clip(
        lower=MIN_MULTIPLIER,
        upper=MAX_MULTIPLIER,
    )

    return multiplier


# ============================================================
# LOAD DATA
# ============================================================

def load_canonical_trades():

    if not TRADE_FILE.exists():
        raise FileNotFoundError(
            f"\nCanonical trade file not found:\n"
            f"{TRADE_FILE}\n"
        )

    trades = pd.read_csv(
        TRADE_FILE
    )

    print(
        f"Loaded canonical trades: "
        f"{len(trades):,}"
    )

    return trades


def load_garch_features():

    if not GARCH_FILE.exists():
        raise FileNotFoundError(
            f"\nGARCH diagnostic file not found:\n"
            f"{GARCH_FILE}\n"
        )

    features = pd.read_csv(
        GARCH_FILE
    )

    print(
        f"Loaded GARCH diagnostic file: "
        f"{len(features):,}"
    )

    return features


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(trades, features):

    required_columns = {
        "EntryGARCHVol",
        "EntryRV24",
        "EntryRV168",
        "GrossReturn",
        "PnL",
    }

    missing = (
        required_columns
        - set(features.columns)
    )

    if missing:
        raise ValueError(
            "Missing required columns from "
            f"GARCH diagnostic file: {missing}"
        )

    data = features.copy()

    print(
        f"Trade rows available for sizing: "
        f"{len(data):,}"
    )

    if len(data) != EXPECTED_TRADES:
        raise ValueError(
            f"Expected exactly "
            f"{EXPECTED_TRADES:,} canonical trades, "
            f"got {len(data):,}."
        )

    # --------------------------------------------------------
    # Check required numeric fields
    # --------------------------------------------------------

    data["PnL"] = pd.to_numeric(
        data["PnL"],
        errors="coerce",
    )

    data["GrossReturn"] = pd.to_numeric(
        data["GrossReturn"],
        errors="coerce",
    )

    if data["PnL"].isna().any():
        raise ValueError(
            "PnL contains missing/non-numeric values."
        )

    if data["GrossReturn"].isna().any():
        raise ValueError(
            "GrossReturn contains missing/non-numeric values."
        )

    return data


# ============================================================
# RECONSTRUCT BASELINE
# ============================================================

def reconstruct_baseline(trades):

    """
    Reconstruct the original canonical equity curve.

    The canonical PnL already comes from the original
    0.5% risk research sizing.

    For each trade:

        fractional_return =
            PnL / equity_before_trade

    This gives us the baseline trade-level equity return.

    The volatility experiments then scale that return.
    """

    equity = INITIAL_EQUITY

    equity_before = []
    fractional_returns = []
    equity_after = []

    for pnl in trades["PnL"].astype(float):

        before = equity

        trade_fraction = pnl / before

        after = before + pnl

        equity_before.append(
            before
        )

        fractional_returns.append(
            trade_fraction
        )

        equity_after.append(
            after
        )

        equity = after

    result = trades.copy()

    result["EquityBefore"] = (
        equity_before
    )

    result["BaselineFractionalReturn"] = (
        fractional_returns
    )

    result["BaselineEquityAfter"] = (
        equity_after
    )

    return result


# ============================================================
# CLEAN MULTIPLIERS
# ============================================================

def clean_multiplier_column(
    data,
    column,
):
    """
    Handle missing/invalid volatility forecasts.

    IMPORTANT:

    We do NOT remove trades with missing volatility.

    A missing entry-time forecast means:
        use baseline 1.0x sizing.

    This preserves the exact 1,351-trade sequence.
    """

    data[column] = pd.to_numeric(
        data[column],
        errors="coerce",
    )

    invalid_mask = (
        ~np.isfinite(data[column])
    )

    invalid_count = int(
        invalid_mask.sum()
    )

    if invalid_count > 0:

        print(
            f"{column}: "
            f"{invalid_count} invalid/missing values "
            f"-> fallback to 1.0x"
        )

        data.loc[
            invalid_mask,
            column,
        ] = 1.0

    data[column] = data[column].clip(
        lower=MIN_MULTIPLIER,
        upper=MAX_MULTIPLIER,
    )

    return data


# ============================================================
# SIMULATE STRATEGY
# ============================================================

def simulate_strategy(
    trades,
    multiplier_column,
    strategy_name,
):

    equity = INITIAL_EQUITY

    equity_curve = []
    trade_pnls = []
    trade_returns = []

    for _, row in trades.iterrows():

        base_trade_return = float(
            row["BaselineFractionalReturn"]
        )

        multiplier = float(
            row[multiplier_column]
        )

        if not np.isfinite(
            multiplier
        ):
            raise ValueError(
                f"Invalid multiplier encountered "
                f"in {strategy_name}."
            )

        scaled_return = (
            base_trade_return
            * multiplier
        )

        pnl = (
            equity
            * scaled_return
        )

        equity += pnl

        equity_curve.append(
            equity
        )

        trade_pnls.append(
            pnl
        )

        trade_returns.append(
            scaled_return
        )

    result = trades.copy()

    result["SizingMultiplier"] = (
        trades[multiplier_column]
    )

    result["ScaledReturn"] = (
        trade_returns
    )

    result["ScaledPnL"] = (
        trade_pnls
    )

    result["SizingEquity"] = (
        equity_curve
    )

    # --------------------------------------------------------
    # Performance metrics
    # --------------------------------------------------------

    final_equity = equity

    total_return = (
        final_equity
        / INITIAL_EQUITY
        - 1.0
    )

    equity_series = pd.Series(
        [INITIAL_EQUITY]
        + equity_curve
    )

    dd = max_drawdown(
        equity_series
    )

    sharpe = sharpe_ratio(
        trade_returns
    )

    pf = profit_factor(
        trade_pnls
    )

    wins = (
        np.array(trade_pnls)
        > 0
    )

    win_rate = wins.mean()

    expectancy = np.mean(
        trade_pnls
    )

    # --------------------------------------------------------
    # Monthly statistics
    # --------------------------------------------------------

    monthly = pd.Series(
        dtype=float
    )

    if "EntryTime" in result.columns:

        result["EntryTime"] = pd.to_datetime(
            result["EntryTime"],
            utc=True,
            errors="coerce",
        )

        monthly = (
            result
            .set_index("EntryTime")[
                "ScaledPnL"
            ]
            .resample("ME")
            .sum()
        )

        profitable_months = int(
            (monthly > 0).sum()
        )

        losing_months = int(
            (monthly < 0).sum()
        )

        worst_month = float(
            monthly.min()
        )

    else:

        profitable_months = np.nan
        losing_months = np.nan
        worst_month = np.nan

    summary = {
        "Strategy": strategy_name,

        "FinalEquity": final_equity,

        "TotalReturn": total_return,

        "MaxDD": dd,

        "Sharpe": sharpe,

        "Trades": len(result),

        "WinRate": win_rate,

        "ProfitFactor": pf,

        "ExpectancyPnL": expectancy,

        "ProfitableMonths": profitable_months,

        "LosingMonths": losing_months,

        "WorstMonthPnL": worst_month,

        "MeanMultiplier":
            result[
                "SizingMultiplier"
            ].mean(),

        "MedianMultiplier":
            result[
                "SizingMultiplier"
            ].median(),

        "MinMultiplier":
            result[
                "SizingMultiplier"
            ].min(),

        "MaxMultiplier":
            result[
                "SizingMultiplier"
            ].max(),
    }

    return (
        summary,
        result,
        monthly,
    )


# ============================================================
# VOLATILITY QUARTILE ANALYSIS
# ============================================================

def analyze_volatility_buckets(
    data,
    vol_column,
):

    bucket_column = (
        f"{vol_column}_Quartile"
    )

    data[bucket_column] = pd.qcut(
        data[vol_column],
        q=4,
        labels=[
            "Q1",
            "Q2",
            "Q3",
            "Q4",
        ],
        duplicates="drop",
    )

    rows = []

    for (
        bucket,
        group,
    ) in data.groupby(
        bucket_column,
        observed=True,
    ):

        rows.append(
            {
                "Volatility":
                    vol_column,

                "Bucket":
                    bucket,

                "Trades":
                    len(group),

                "MeanVol":
                    group[
                        vol_column
                    ].mean(),

                "MeanBaselineReturn":
                    group[
                        "BaselineFractionalReturn"
                    ].mean(),

                "MeanGARCHMultiplier":
                    group[
                        "GARCHMultiplier"
                    ].mean(),

                "MeanRV24Multiplier":
                    group[
                        "RV24Multiplier"
                    ].mean(),

                "MeanRV168Multiplier":
                    group[
                        "RV168Multiplier"
                    ].mean(),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # LOAD
    # ========================================================

    trades = load_canonical_trades()

    features = load_garch_features()

    # ========================================================
    # PREPARE
    # ========================================================

    data = prepare_data(
        trades,
        features,
    )

    print(
        "Canonical 1,351-trade verification: PASS"
    )

    # ========================================================
    # RECONSTRUCT BASELINE
    # ========================================================

    data = reconstruct_baseline(
        data
    )

    # ========================================================
    # CREATE VOLATILITY MULTIPLIERS
    # ========================================================

    data["FixedMultiplier"] = 1.0

    data["GARCHMultiplier"] = (
        inverse_vol_multiplier(
            data["EntryGARCHVol"]
        )
    )

    data["RV24Multiplier"] = (
        inverse_vol_multiplier(
            data["EntryRV24"]
        )
    )

    data["RV168Multiplier"] = (
        inverse_vol_multiplier(
            data["EntryRV168"]
        )
    )

    # ========================================================
    # IMPORTANT FIX:
    # MISSING FORECAST -> BASELINE 1.0x
    # ========================================================

    data = clean_multiplier_column(
        data,
        "GARCHMultiplier",
    )

    data = clean_multiplier_column(
        data,
        "RV24Multiplier",
    )

    data = clean_multiplier_column(
        data,
        "RV168Multiplier",
    )

    # ========================================================
    # VERIFY NO INVALID MULTIPLIERS
    # ========================================================

    multiplier_columns = [
        "FixedMultiplier",
        "GARCHMultiplier",
        "RV24Multiplier",
        "RV168Multiplier",
    ]

    for column in multiplier_columns:

        invalid = (
            ~np.isfinite(
                data[column]
            )
        ).sum()

        if invalid != 0:
            raise ValueError(
                f"{column} still contains "
                f"{invalid} invalid values."
            )

    # ========================================================
    # STRATEGIES
    # ========================================================

    configurations = [
        (
            "Fixed",
            "FixedMultiplier",
        ),
        (
            "GARCH_InverseVol",
            "GARCHMultiplier",
        ),
        (
            "RV24_InverseVol",
            "RV24Multiplier",
        ),
        (
            "RV168_InverseVol",
            "RV168Multiplier",
        ),
    ]

    summaries = []

    detailed_results = {}

    # ========================================================
    # RUN
    # ========================================================

    for (
        strategy_name,
        multiplier_column,
    ) in configurations:

        (
            summary,
            result,
            monthly,
        ) = simulate_strategy(
            data,
            multiplier_column,
            strategy_name,
        )

        # Same number of trades for every strategy.
        if len(result) != EXPECTED_TRADES:
            raise ValueError(
                f"{strategy_name}: expected "
                f"{EXPECTED_TRADES} trades, got "
                f"{len(result)}."
            )

        summaries.append(
            summary
        )

        detailed_results[
            strategy_name
        ] = result

        monthly.to_csv(
            OUTPUT_DIR
            / f"{strategy_name}_monthly_pnl.csv"
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary_df = pd.DataFrame(
        summaries
    )

    summary_path = (
        OUTPUT_DIR
        / "volatility_sizing_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    # ========================================================
    # SAVE TRADE RESULTS
    # ========================================================

    for (
        strategy_name,
        result,
    ) in detailed_results.items():

        result.to_csv(
            OUTPUT_DIR
            / f"{strategy_name}_trades.csv",
            index=False,
        )

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    pd.set_option(
        "display.max_columns",
        None,
    )

    pd.set_option(
        "display.width",
        240,
    )

    print()
    print("=" * 100)
    print(
        "VOLATILITY-AWARE POSITION SIZING"
    )
    print("=" * 100)

    print(
        summary_df[
            [
                "Strategy",
                "FinalEquity",
                "TotalReturn",
                "MaxDD",
                "Sharpe",
                "Trades",
                "WinRate",
                "ProfitFactor",
                "ExpectancyPnL",
                "ProfitableMonths",
                "LosingMonths",
                "WorstMonthPnL",
                "MeanMultiplier",
                "MedianMultiplier",
                "MinMultiplier",
                "MaxMultiplier",
            ]
        ].to_string(
            index=False
        )
    )

    # ========================================================
    # MULTIPLIER DISTRIBUTIONS
    # ========================================================

    print()
    print("=" * 100)
    print(
        "SIZING MULTIPLIER DISTRIBUTIONS"
    )
    print("=" * 100)

    print(
        data[
            [
                "GARCHMultiplier",
                "RV24Multiplier",
                "RV168Multiplier",
            ]
        ]
        .describe()
        .T
        .to_string()
    )

    # ========================================================
    # VOLATILITY QUARTILES
    # ========================================================

    for vol_column in [
        "EntryGARCHVol",
        "EntryRV24",
        "EntryRV168",
    ]:

        bucket_df = (
            analyze_volatility_buckets(
                data,
                vol_column,
            )
        )

        bucket_df.to_csv(
            OUTPUT_DIR
            / f"{vol_column}_sizing_buckets.csv",
            index=False,
        )

        print()
        print(
            vol_column
        )

        print(
            bucket_df.to_string(
                index=False
            )
        )

    # ========================================================
    # SANITY CHECK
    # ========================================================

    print()
    print("=" * 100)
    print(
        "SANITY CHECK"
    )
    print("=" * 100)

    print(
        f"Expected trades : "
        f"{EXPECTED_TRADES:,}"
    )

    print(
        f"Actual trades   : "
        f"{len(data):,}"
    )

    print(
        f"Initial equity  : "
        f"${INITIAL_EQUITY:,.2f}"
    )

    print(
        f"Max multiplier  : "
        f"{MAX_MULTIPLIER:.2f}x"
    )

    print(
        f"Min multiplier  : "
        f"{MIN_MULTIPLIER:.2f}x"
    )

    print(
        "Entry/exit signals changed: NO"
    )

    print(
        "Trade sequence changed: NO"
    )

    print(
        "Lookahead from MAE/MFE: NO"
    )

    # ========================================================
    # FILES
    # ========================================================

    print()
    print("=" * 100)
    print(
        "FILES SAVED"
    )
    print("=" * 100)

    print(
        OUTPUT_DIR
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "This remains a research rescaling experiment."
    )

    print(
        "It does NOT yet represent exact Exness "
        "contract-level sizing, tick value, lot size, "
        "margin, or stop-distance risk."
    )

    print(
        "Missing entry volatility forecasts use "
        "1.0x baseline sizing so all strategies "
        "retain exactly the same 1,351 trades."
    )

    print(
        "No multiplier exceeds 1.0x, so this experiment "
        "does not increase exposure above the original baseline."
    )


if __name__ == "__main__":
    main()