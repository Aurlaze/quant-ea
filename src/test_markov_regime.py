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


from markov_regime import (
    build_regime_features,
    fit_hmm,
)


DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "mt5_h1"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "markov_regime"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# TEST UNIVERSE
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


# =========================================================
# DATA LOADER
# =========================================================

def load_data(symbol):

    path = (
        DATA_DIR
        / f"{symbol}_H1_2019_2026.csv"
    )

    df = pd.read_csv(path)

    # -----------------------------------------------------
    # Time
    # -----------------------------------------------------

    time_col = None

    for col in df.columns:

        if str(col).lower() in [
            "time",
            "timestamp",
            "datetime",
            "date",
        ]:

            time_col = col
            break

    if time_col is None:

        raise ValueError(
            f"{symbol}: time column missing"
        )

    # -----------------------------------------------------
    # OHLC
    # -----------------------------------------------------

    def find_column(name):

        for col in df.columns:

            if str(col).lower() == name:

                return col

        return None

    open_col = find_column("open")
    high_col = find_column("high")
    low_col = find_column("low")
    close_col = find_column("close")

    out = pd.DataFrame()

    out["Time"] = pd.to_datetime(
        df[time_col],
        utc=True,
    )

    out["Open"] = pd.to_numeric(
        df[open_col]
    )

    out["High"] = pd.to_numeric(
        df[high_col]
    )

    out["Low"] = pd.to_numeric(
        df[low_col]
    )

    out["Close"] = pd.to_numeric(
        df[close_col]
    )

    out = out.dropna()

    out = out.sort_values(
        "Time"
    )

    out = out.drop_duplicates(
        "Time"
    )

    out = out.reset_index(
        drop=True
    )

    return out


# =========================================================
# STATE ANALYSIS
# =========================================================

def analyze_states(
    df,
    model,
    states,
):

    features = build_regime_features(
        df
    )

    analysis = pd.DataFrame(
        {
            "Return":
                features["LogReturn"],

            "Volatility":
                features["Volatility24"],

            "State":
                np.nan,
        },
        index=df.index,
    )

    clean_index = (
        features.dropna()
        .index
    )

    analysis.loc[
        clean_index,
        "State",
    ] = states

    rows = []

    for state in sorted(
        analysis["State"]
        .dropna()
        .unique()
    ):

        state = int(state)

        group = analysis[
            analysis["State"] == state
        ].copy()

        returns = (
            group["Return"]
            .dropna()
        )

        # -------------------------------------------------
        # Autocorrelation
        # -------------------------------------------------

        if len(returns) > 2:

            autocorr_1 = (
                returns
                .autocorr(lag=1)
            )

        else:

            autocorr_1 = np.nan

        # -------------------------------------------------
        # Model parameters
        # -------------------------------------------------

        mean_return = (
            model.means_[state][0]
        )

        mean_volatility = (
            group["Volatility"]
            .mean()
        )

        rows.append(
            {
                "State": state,
                "Observations": len(group),

                "Fraction":
                    len(group)
                    / len(analysis),

                "MeanReturn":
                    returns.mean(),

                "ReturnStd":
                    returns.std(),

                "AbsReturnMean":
                    returns.abs().mean(),

                "ReturnAutocorrelation1":
                    autocorr_1,

                "MeanVolatility24":
                    mean_volatility,

                "HMMMeanFeature":
                    mean_return,
            }
        )

    return pd.DataFrame(rows)


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 100)
    print("MARKOV REGIME BASELINE TEST")
    print("=" * 100)

    all_results = []

    for symbol in INSTRUMENTS:

        print()
        print("#" * 100)
        print(f"TESTING {symbol}")
        print("#" * 100)

        try:

            df = load_data(symbol)

            print(
                f"Bars: {len(df):,}"
            )

            features = (
                build_regime_features(
                    df
                )
            )

            clean = features.dropna()

            print(
                f"Usable observations: "
                f"{len(clean):,}"
            )

            (
                model,
                states,
                mr_state,
                trend_state,
            ) = fit_hmm(
                features
            )

            state_df = analyze_states(
                df,
                model,
                states,
            )

            state_df.insert(
                0,
                "Symbol",
                symbol,
            )

            state_df[
                "MeanRevertingCandidate"
            ] = (
                state_df["State"]
                == mr_state
            )

            state_df[
                "TrendingCandidate"
            ] = (
                state_df["State"]
                == trend_state
            )

            state_df.to_csv(
                RESULTS_DIR
                / f"{symbol}_states.csv",
                index=False,
            )

            # -------------------------------------------------
            # Console
            # -------------------------------------------------

            print()
            print(
                state_df.to_string(
                    index=False
                )
            )

            print()
            print(
                f"Candidate MR state: "
                f"{mr_state}"
            )

            print(
                f"Candidate trend state: "
                f"{trend_state}"
            )

            all_results.append(
                state_df
            )

        except Exception as e:

            print(
                f"ERROR {symbol}: "
                f"{repr(e)}"
            )

    # =====================================================
    # COMBINED
    # =====================================================

    if all_results:

        combined = pd.concat(
            all_results,
            ignore_index=True,
        )

        combined.to_csv(
            RESULTS_DIR
            / "all_states.csv",
            index=False,
        )

    print()
    print("=" * 100)
    print("MARKOV BASELINE TEST COMPLETE")
    print("=" * 100)

    print()
    print(
        f"Results saved to:\n"
        f"{RESULTS_DIR}"
    )


if __name__ == "__main__":
    main()