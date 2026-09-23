from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = (
    BASE_DIR
    / "data"
    / "mt5_h1"
)

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "instrument_regime_diagnostic"
)

START_DATE = "2019-01-01"
END_DATE = "2026-09-19"

INITIAL_CAPITAL = 10_000.0


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

MAX_HOLD_HOURS = 24

RISK_PER_TRADE = 0.005
MAX_NOTIONAL_FRACTION = 1.0
ASSUMED_STOP_RETURN = 0.01


# ============================================================
# LOCKED MARKOV PARAMETERS
# ============================================================

HMM_WINDOW = 1000
MIN_OBSERVATIONS = 300
REFIT_EVERY = 24

MR_THRESHOLD = 0.60


# ============================================================
# EXACT MT5 SYMBOLS
# ============================================================

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


# ============================================================
# DATA COLUMN NORMALIZATION
# ============================================================

def find_column(
    columns,
    candidates,
):
    """
    Find a column using case-insensitive matching.
    """

    lookup = {
        str(column).strip().lower(): column
        for column in columns
    }

    for candidate in candidates:

        key = candidate.lower()

        if key in lookup:
            return lookup[key]

    return None


def normalize_mt5_columns(df, symbol):
    """
    Normalize common MT5 CSV schemas into:

        Time
        Open
        High
        Low
        Close
        SpreadPoints

    Handles common variations such as:

        Time / time / Date / Datetime / Timestamp
        Open / open
        High / high
        Low / low
        Close / close
        Spread / spread / SpreadPoints
    """

    original_columns = list(
        df.columns
    )

    print(
        f"{symbol} CSV columns:"
    )

    print(
        original_columns
    )

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    time_column = find_column(
        df.columns,
        [
            "Time",
            "time",
            "Datetime",
            "datetime",
            "DateTime",
            "Date",
            "date",
            "Timestamp",
            "timestamp",
        ],
    )

    if time_column is None:

        raise ValueError(
            f"{symbol}: unable to find timestamp column.\n"
            f"Available columns: {original_columns}"
        )

    # --------------------------------------------------------
    # OHLC
    # --------------------------------------------------------

    open_column = find_column(
        df.columns,
        [
            "Open",
            "open",
        ],
    )

    high_column = find_column(
        df.columns,
        [
            "High",
            "high",
        ],
    )

    low_column = find_column(
        df.columns,
        [
            "Low",
            "low",
        ],
    )

    close_column = find_column(
        df.columns,
        [
            "Close",
            "close",
        ],
    )

    missing_ohlc = []

    if open_column is None:
        missing_ohlc.append("Open")

    if high_column is None:
        missing_ohlc.append("High")

    if low_column is None:
        missing_ohlc.append("Low")

    if close_column is None:
        missing_ohlc.append("Close")

    if missing_ohlc:

        raise ValueError(
            f"{symbol}: missing OHLC columns "
            f"{missing_ohlc}.\n"
            f"Available columns: {original_columns}"
        )

    # --------------------------------------------------------
    # Spread
    # --------------------------------------------------------

    spread_column = find_column(
        df.columns,
        [
            "SpreadPoints",
            "spreadpoints",
            "Spread",
            "spread",
        ],
    )

    # If no spread exists, use zero rather than crashing.
    #
    # This should be visible in the output because the research
    # engine should ideally use actual MT5 spread data.
    if spread_column is None:

        print(
            f"WARNING: {symbol}: no spread column found. "
            f"Using 0 spread points."
        )

        spread_values = 0.0

    else:

        spread_values = df[
            spread_column
        ]

    # --------------------------------------------------------
    # Build normalized dataframe
    # --------------------------------------------------------

    result = pd.DataFrame(
        {
            "Time":
                df[time_column],

            "Open":
                df[open_column],

            "High":
                df[high_column],

            "Low":
                df[low_column],

            "Close":
                df[close_column],

            "SpreadPoints":
                spread_values,
        }
    )

    # --------------------------------------------------------
    # Parse timestamp
    # --------------------------------------------------------

    result["Time"] = pd.to_datetime(
        result["Time"],
        utc=True,
        errors="coerce",
    )

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    numeric_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "SpreadPoints",
    ]

    for column in numeric_columns:

        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Remove invalid rows
    # --------------------------------------------------------

    before = len(result)

    result = result.dropna(
        subset=[
            "Time",
            "Open",
            "High",
            "Low",
            "Close",
            "SpreadPoints",
        ]
    )

    removed = (
        before
        - len(result)
    )

    if removed > 0:

        print(
            f"{symbol}: removed "
            f"{removed:,} invalid rows."
        )

    # --------------------------------------------------------
    # Remove duplicate timestamps
    # --------------------------------------------------------

    duplicate_count = (
        result["Time"]
        .duplicated()
        .sum()
    )

    if duplicate_count > 0:

        print(
            f"{symbol}: removing "
            f"{duplicate_count:,} duplicate timestamps."
        )

        result = (
            result
            .drop_duplicates(
                subset="Time",
                keep="first",
            )
        )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    result = (
        result
        .sort_values("Time")
        .reset_index(drop=True)
    )

    return result


# ============================================================
# LOAD H1
# ============================================================

def load_h1(symbol):

    path = (
        DATA_DIR
        / f"{symbol}_H1_2019_2026.csv"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"\nH1 file not found:\n"
            f"{path}"
        )

    print(
        f"Loading: {path.name}"
    )

    raw = pd.read_csv(
        path
    )

    df = normalize_mt5_columns(
        raw,
        symbol,
    )

    # --------------------------------------------------------
    # Date range
    # --------------------------------------------------------

    df = df[
        (
            df["Time"]
            >= pd.Timestamp(
                START_DATE,
                tz="UTC",
            )
        )
        &
        (
            df["Time"]
            <= pd.Timestamp(
                END_DATE,
                tz="UTC",
            )
        )
    ].copy()

    # --------------------------------------------------------
    # Basic price validation
    # --------------------------------------------------------

    invalid_price = (
        (df["Open"] <= 0)
        |
        (df["High"] <= 0)
        |
        (df["Low"] <= 0)
        |
        (df["Close"] <= 0)
    )

    invalid_ohlc = (
        (df["High"] < df["Low"])
        |
        (df["High"] < df["Open"])
        |
        (df["High"] < df["Close"])
        |
        (df["Low"] > df["Open"])
        |
        (df["Low"] > df["Close"])
    )

    invalid = (
        invalid_price
        |
        invalid_ohlc
    )

    if invalid.any():

        print(
            f"WARNING: {symbol}: removing "
            f"{invalid.sum():,} invalid OHLC rows."
        )

        df = df.loc[
            ~invalid
        ].copy()

    df = df.reset_index(
        drop=True
    )

    if df.empty:

        raise ValueError(
            f"{symbol}: no H1 data remains "
            f"after filtering."
        )

    print(
        f"H1 bars: {len(df):,}"
    )

    print(
        f"Period: "
        f"{df['Time'].iloc[0]} "
        f"-> "
        f"{df['Time'].iloc[-1]}"
    )

    return df


# ============================================================
# KALMAN FILTER
# ============================================================

def walk_forward_kalman(
    prices,
    phi=PHI,
    q=Q,
    r=R,
    mean_window=MEAN_WINDOW,
):
    """
    Exact research Kalman implementation.
    """

    prices = pd.Series(
        prices,
        dtype=float,
    )

    n = len(prices)

    states = np.full(
        n,
        np.nan,
    )

    variances = np.full(
        n,
        np.nan,
    )

    residuals = np.full(
        n,
        np.nan,
    )

    means = np.full(
        n,
        np.nan,
    )

    x = prices.iloc[0]
    p = 1.0

    states[0] = x
    variances[0] = p

    for i in range(1, n):

        x_pred = (
            phi * x
        )

        p_pred = (
            phi
            * phi
            * p
            + q
        )

        y = prices.iloc[i]

        innovation = (
            y
            - x_pred
        )

        s = (
            p_pred
            + r
        )

        k = (
            p_pred
            / s
        )

        x = (
            x_pred
            + k * innovation
        )

        p = (
            1.0 - k
        ) * p_pred

        states[i] = x
        variances[i] = p

    residuals = (
        prices.values
        - states
    )

    residual_series = pd.Series(
        residuals,
        index=prices.index,
    )

    means = (
        residual_series
        .rolling(
            mean_window
        )
        .mean()
        .values
    )

    return (
        states,
        variances,
        residuals,
        means,
    )


def calculate_zscore(
    residuals,
    window=Z_WINDOW,
):
    """
    Exact causal Z-score.
    """

    if not isinstance(
        residuals,
        pd.Series,
    ):

        residuals = pd.Series(
            residuals
        )

    rolling_mean = (
        residuals
        .rolling(window)
        .mean()
        .shift(1)
    )

    rolling_std = (
        residuals
        .rolling(window)
        .std()
        .shift(1)
    )

    z = (
        residuals
        - rolling_mean
    ) / rolling_std

    return z


# ============================================================
# MARKOV FEATURES
# ============================================================

def calculate_regime_features(
    bars,
):

    result = bars.copy()

    result["LogReturn"] = np.log(
        result["Close"]
        / result["Close"].shift(1)
    )

    result["Volatility24"] = (
        result["LogReturn"]
        .rolling(24)
        .std()
    )

    result["Autocorr24"] = (
        result["LogReturn"]
        .rolling(24)
        .corr(
            result["LogReturn"].shift(1)
        )
    )

    return result


# ============================================================
# HMM STATE CLASSIFICATION
# ============================================================

def classify_hmm_states(
    model,
    X_raw,
    scaler,
):

    X_scaled = (
        scaler.transform(
            X_raw
        )
    )

    probabilities = (
        model.predict_proba(
            X_scaled
        )
    )

    states = (
        model.predict(
            X_scaled
        )
    )

    unique_states = sorted(
        np.unique(states)
    )

    if len(unique_states) < 2:

        return (
            states,
            np.full(
                len(states),
                np.nan,
            ),
            np.full(
                len(states),
                np.nan,
            ),
            None,
            None,
        )

    state_stats = {}

    for state in unique_states:

        mask = (
            states == state
        )

        state_stats[state] = {
            "autocorr":
                np.nanmean(
                    X_raw[
                        mask,
                        2,
                    ]
                ),

            "volatility":
                np.nanmean(
                    X_raw[
                        mask,
                        1,
                    ]
                ),
        }

    s0 = unique_states[0]
    s1 = unique_states[1]

    ac0 = state_stats[s0][
        "autocorr"
    ]

    ac1 = state_stats[s1][
        "autocorr"
    ]

    vol0 = state_stats[s0][
        "volatility"
    ]

    vol1 = state_stats[s1][
        "volatility"
    ]

    # --------------------------------------------------------
    # Existing V2 classification:
    #
    # lower autocorrelation -> MR
    #
    # if difference is too small:
    # lower volatility -> MR
    # --------------------------------------------------------

    if (
        np.isfinite(ac0)
        and np.isfinite(ac1)
    ):

        if (
            abs(ac0 - ac1)
            >= 0.005
        ):

            mr_state = (
                s0
                if ac0 < ac1
                else s1
            )

        else:

            mr_state = (
                s0
                if vol0 < vol1
                else s1
            )

    else:

        mr_state = (
            s0
            if vol0 < vol1
            else s1
        )

    trend_state = (
        s1
        if mr_state == s0
        else s0
    )

    classes = list(
        model.classes_
    )

    mr_index = classes.index(
        mr_state
    )

    trend_index = classes.index(
        trend_state
    )

    p_mr = probabilities[
        :,
        mr_index,
    ]

    p_trend = probabilities[
        :,
        trend_index,
    ]

    return (
        states,
        p_mr,
        p_trend,
        mr_state,
        trend_state,
    )


# ============================================================
# CAUSAL MARKOV / HMM
# ============================================================

def causal_markov_regime(
    bars,
):

    try:

        from hmmlearn.hmm import (
            GaussianHMM
        )

        from sklearn.preprocessing import (
            StandardScaler
        )

    except ImportError as exc:

        raise ImportError(
            "Missing dependencies.\n"
            "Install with:\n"
            "pip install hmmlearn scikit-learn"
        ) from exc

    data = (
        calculate_regime_features(
            bars
        )
    )

    feature_columns = [
        "LogReturn",
        "Volatility24",
        "Autocorr24",
    ]

    X = (
        data[
            feature_columns
        ]
        .values
    )

    n = len(data)

    hmm_state = np.full(
        n,
        np.nan,
    )

    p_mr = np.full(
        n,
        np.nan,
    )

    p_trend = np.full(
        n,
        np.nan,
    )

    mr_autocorr = np.full(
        n,
        np.nan,
    )

    trend_autocorr = np.full(
        n,
        np.nan,
    )

    mr_volatility = np.full(
        n,
        np.nan,
    )

    trend_volatility = np.full(
        n,
        np.nan,
    )

    current_model = None
    current_scaler = None

    current_mr_state = None
    current_trend_state = None

    last_fit = -REFIT_EVERY

    # ========================================================
    # WALK FORWARD
    # ========================================================

    for i in range(n):

        # ----------------------------------------------------
        # Refit
        # ----------------------------------------------------

        if (
            current_model is None
            or i - last_fit
            >= REFIT_EVERY
        ):

            train_start = max(
                0,
                i - HMM_WINDOW,
            )

            train_end = i

            X_train = X[
                train_start:train_end
            ]

            valid_mask = (
                np.isfinite(
                    X_train
                ).all(
                    axis=1
                )
            )

            X_train = X_train[
                valid_mask
            ]

            if (
                len(X_train)
                >= MIN_OBSERVATIONS
            ):

                scaler = (
                    StandardScaler()
                )

                X_train_scaled = (
                    scaler.fit_transform(
                        X_train
                    )
                )

                model = GaussianHMM(
                    n_components=2,
                    covariance_type="diag",
                    n_iter=150,
                    tol=1e-4,
                    random_state=42,
                    min_covar=1e-6,
                )

                try:

                    model.fit(
                        X_train_scaled
                    )

                    (
                        _,
                        _,
                        _,
                        mr_state,
                        trend_state,
                    ) = classify_hmm_states(
                        model,
                        X_train,
                        scaler,
                    )

                    if (
                        mr_state is not None
                        and trend_state is not None
                    ):

                        current_model = (
                            model
                        )

                        current_scaler = (
                            scaler
                        )

                        current_mr_state = (
                            mr_state
                        )

                        current_trend_state = (
                            trend_state
                        )

                        last_fit = i

                except Exception:

                    pass

        # ----------------------------------------------------
        # No valid model yet
        # ----------------------------------------------------

        if (
            current_model is None
            or current_scaler is None
        ):
            continue

        current_x = X[i]

        if not np.isfinite(
            current_x
        ).all():

            continue

        # ----------------------------------------------------
        # Current observation
        # ----------------------------------------------------

        try:

            current_scaled = (
                current_scaler.transform(
                    current_x.reshape(
                        1,
                        -1,
                    )
                )
            )

            probabilities = (
                current_model
                .predict_proba(
                    current_scaled
                )[0]
            )

            classes = list(
                current_model.classes_
            )

            mr_index = classes.index(
                current_mr_state
            )

            trend_index = classes.index(
                current_trend_state
            )

            state = (
                current_model
                .predict(
                    current_scaled
                )[0]
            )

            hmm_state[i] = state

            p_mr[i] = (
                probabilities[
                    mr_index
                ]
            )

            p_trend[i] = (
                probabilities[
                    trend_index
                ]
            )

            # ------------------------------------------------
            # Reconstruct state statistics from the
            # historical training window.
            # ------------------------------------------------

            train_start = max(
                0,
                i - HMM_WINDOW,
            )

            X_train = X[
                train_start:i
            ]

            valid_mask = (
                np.isfinite(
                    X_train
                ).all(
                    axis=1
                )
            )

            X_train = X_train[
                valid_mask
            ]

            if (
                len(X_train)
                >= MIN_OBSERVATIONS
            ):

                X_train_scaled = (
                    current_scaler.transform(
                        X_train
                    )
                )

                train_states = (
                    current_model.predict(
                        X_train_scaled
                    )
                )

                mr_mask = (
                    train_states
                    == current_mr_state
                )

                trend_mask = (
                    train_states
                    == current_trend_state
                )

                if mr_mask.any():

                    mr_autocorr[i] = (
                        np.nanmean(
                            X_train[
                                mr_mask,
                                2,
                            ]
                        )
                    )

                    mr_volatility[i] = (
                        np.nanmean(
                            X_train[
                                mr_mask,
                                1,
                            ]
                        )
                    )

                if trend_mask.any():

                    trend_autocorr[i] = (
                        np.nanmean(
                            X_train[
                                trend_mask,
                                2,
                            ]
                        )
                    )

                    trend_volatility[i] = (
                        np.nanmean(
                            X_train[
                                trend_mask,
                                1,
                            ]
                        )
                    )

        except Exception:

            continue

    result = data.copy()

    result["HMMState"] = (
        hmm_state
    )

    result["P_MeanReverting"] = (
        p_mr
    )

    result["P_Trending"] = (
        p_trend
    )

    result["MRState"] = (
        p_mr
        >= MR_THRESHOLD
    )

    result["MR_Autocorrelation"] = (
        mr_autocorr
    )

    result["Trend_Autocorrelation"] = (
        trend_autocorr
    )

    result["MR_Volatility"] = (
        mr_volatility
    )

    result["Trend_Volatility"] = (
        trend_volatility
    )

    # --------------------------------------------------------
    # Entry regime
    # --------------------------------------------------------

    result["Regime"] = np.where(
        result[
            "P_MeanReverting"
        ]
        >= MR_THRESHOLD,
        "MR",
        "TREND",
    )

    # If P(MR) is unavailable, mark UNKNOWN rather than
    # accidentally treating it as TREND.

    result.loc[
        result[
            "P_MeanReverting"
        ].isna(),
        "Regime",
    ] = "UNKNOWN"

    return result


# ============================================================
# POINT SIZE
# ============================================================

def point_size_for_symbol(
    symbol,
):

    if symbol in [
        "EURUSDm",
        "GBPUSDm",
    ]:

        return 0.00001

    if symbol == "USDJPYm":

        return 0.001

    if symbol == "XAUUSDm":

        return 0.001

    if symbol == "XAGUSDm":

        return 0.001

    if symbol in [
        "USTECm",
        "JP225m",
    ]:

        return 0.1

    if symbol in [
        "BTCUSDm",
        "ETHUSDm",
    ]:

        return 0.01

    if symbol == "USOILm":

        return 0.01

    raise ValueError(
        f"No point size configured "
        f"for {symbol}"
    )


# ============================================================
# BID / ASK
# ============================================================

def add_bid_ask(
    df,
    symbol,
):

    point_size = (
        point_size_for_symbol(
            symbol
        )
    )

    spread_price = (
        df["SpreadPoints"]
        * point_size
    )

    result = df.copy()

    result["Bid"] = (
        result["Close"]
        - spread_price / 2.0
    )

    result["Ask"] = (
        result["Close"]
        + spread_price / 2.0
    )

    return result


# ============================================================
# TRADE GENERATION
# ============================================================

def generate_trades(
    bars,
    symbol,
):

    bars = bars.copy()

    # ========================================================
    # KALMAN
    # ========================================================

    prices = (
        bars["Close"]
        .astype(float)
    )

    (
        states,
        variances,
        residuals,
        means,
    ) = walk_forward_kalman(
        prices
    )

    bars["KalmanState"] = (
        states
    )

    bars["KalmanVariance"] = (
        variances
    )

    bars["Residual"] = (
        residuals
    )

    residual_series = pd.Series(
        residuals,
        index=bars.index,
    )

    bars["ZScore"] = (
        calculate_zscore(
            residual_series,
            Z_WINDOW,
        )
    )

    # ========================================================
    # MARKOV
    # ========================================================

    bars = causal_markov_regime(
        bars
    )

    # ========================================================
    # BID / ASK
    # ========================================================

    bars = add_bid_ask(
        bars,
        symbol,
    )

    # ========================================================
    # TRADING
    # ========================================================

    equity = INITIAL_CAPITAL

    position = None

    entry_index = None
    entry_price = None
    entry_z = None
    entry_regime = None
    entry_time = None
    entry_p_mr = None

    trades = []

    # ========================================================
    # EVENT LOOP
    # ========================================================

    for i in range(
        len(bars) - 1
    ):

        row = bars.iloc[i]

        next_row = bars.iloc[
            i + 1
        ]

        current_z = row[
            "ZScore"
        ]

        if not np.isfinite(
            current_z
        ):

            continue

        current_regime = row[
            "Regime"
        ]

        p_mr = row[
            "P_MeanReverting"
        ]

        # ====================================================
        # EXISTING POSITION
        # ====================================================

        if position is not None:

            trading_hours = (
                i
                - entry_index
            )

            calendar_hours = (
                (
                    row["Time"]
                    - entry_time
                )
                .total_seconds()
                / 3600.0
            )

            should_exit = False

            exit_reason = None

            # ------------------------------------------------
            # LONG
            # ------------------------------------------------

            if position == "LONG":

                if (
                    current_z
                    >= -EXIT_Z
                ):

                    should_exit = True

                    exit_reason = (
                        "Z_EXIT"
                    )

                elif (
                    current_z
                    <= -STOP_Z
                ):

                    should_exit = True

                    exit_reason = (
                        "Z_STOP"
                    )

            # ------------------------------------------------
            # SHORT
            # ------------------------------------------------

            elif position == "SHORT":

                if (
                    current_z
                    <= EXIT_Z
                ):

                    should_exit = True

                    exit_reason = (
                        "Z_EXIT"
                    )

                elif (
                    current_z
                    >= STOP_Z
                ):

                    should_exit = True

                    exit_reason = (
                        "Z_STOP"
                    )

            # ------------------------------------------------
            # MAX HOLD
            # ------------------------------------------------

            if (
                not should_exit
                and trading_hours
                >= MAX_HOLD_HOURS
            ):

                should_exit = True

                exit_reason = (
                    "MAX_HOLD"
                )

            # ------------------------------------------------
            # EXIT NEXT BAR
            # ------------------------------------------------

            if should_exit:

                if position == "LONG":

                    exit_price = (
                        next_row["Bid"]
                    )

                    price_return = (
                        exit_price
                        / entry_price
                        - 1.0
                    )

                else:

                    exit_price = (
                        next_row["Ask"]
                    )

                    price_return = (
                        entry_price
                        / exit_price
                        - 1.0
                    )

                # Research notional.
                #
                # This script is diagnostic only and intentionally
                # keeps the same approximate execution framework.
                notional = (
                    equity
                    * MAX_NOTIONAL_FRACTION
                )

                pnl = (
                    notional
                    * price_return
                )

                equity += pnl

                trades.append(
                    {
                        "Symbol":
                            symbol,

                        "EntryTime":
                            entry_time,

                        "ExitTime":
                            next_row[
                                "Time"
                            ],

                        "Side":
                            position,

                        "EntryPrice":
                            entry_price,

                        "ExitPrice":
                            exit_price,

                        "EntryZ":
                            entry_z,

                        "ExitZ":
                            current_z,

                        "EntryRegime":
                            entry_regime,

                        "ExitRegime":
                            next_row[
                                "Regime"
                            ],

                        "EntryP_MR":
                            entry_p_mr,

                        "TradingHoursHeld":
                            trading_hours,

                        "CalendarHoursHeld":
                            calendar_hours,

                        "HoursHeld":
                            trading_hours,

                        "Return":
                            price_return,

                        "PnL":
                            pnl,

                        "ExitReason":
                            exit_reason,
                    }
                )

                position = None

                entry_index = None
                entry_price = None
                entry_z = None
                entry_regime = None
                entry_time = None
                entry_p_mr = None

                continue

        # ====================================================
        # NEW ENTRY
        # ====================================================

        if position is None:

            # ------------------------------------------------
            # LONG
            #
            # Existing Markov-short-filter strategy:
            # longs are unrestricted.
            # ------------------------------------------------

            if (
                current_z
                <= -ENTRY_Z
            ):

                position = "LONG"

                entry_index = (
                    i + 1
                )

                entry_price = (
                    next_row["Ask"]
                )

                entry_z = (
                    current_z
                )

                entry_regime = (
                    current_regime
                )

                entry_time = (
                    next_row["Time"]
                )

                entry_p_mr = p_mr

            # ------------------------------------------------
            # SHORT
            #
            # Existing Markov short filter:
            #
            # Z >= +2.25
            # AND
            # P(MR) >= 0.60
            # ------------------------------------------------

            elif (
                current_z
                >= ENTRY_Z
                and np.isfinite(
                    p_mr
                )
                and p_mr
                >= MR_THRESHOLD
            ):

                position = "SHORT"

                entry_index = (
                    i + 1
                )

                entry_price = (
                    next_row["Bid"]
                )

                entry_z = (
                    current_z
                )

                entry_regime = (
                    current_regime
                )

                entry_time = (
                    next_row["Time"]
                )

                entry_p_mr = p_mr

    return pd.DataFrame(
        trades
    )


# ============================================================
# METRICS
# ============================================================

def calculate_trade_metrics(
    trades,
):

    if trades.empty:

        return {
            "Trades": 0,
            "WinRate": np.nan,
            "ProfitFactor": np.nan,
            "ExpectancyPnL": np.nan,
            "TotalPnL": 0.0,
            "MeanReturn": np.nan,
            "StopRate": np.nan,
        }

    pnl = (
        trades["PnL"]
        .astype(float)
    )

    wins = (
        pnl > 0
    )

    gross_profit = (
        pnl[pnl > 0].sum()
    )

    gross_loss = (
        -pnl[pnl < 0].sum()
    )

    if gross_loss > 0:

        pf = (
            gross_profit
            / gross_loss
        )

    else:

        pf = np.inf

    return {
        "Trades":
            len(trades),

        "WinRate":
            wins.mean(),

        "ProfitFactor":
            pf,

        "ExpectancyPnL":
            pnl.mean(),

        "TotalPnL":
            pnl.sum(),

        "MeanReturn":
            trades[
                "Return"
            ].mean(),

        "StopRate":
            (
                trades[
                    "ExitReason"
                ]
                == "Z_STOP"
            ).mean(),
    }


# ============================================================
# DIRECTION × REGIME
# ============================================================

def analyze_direction_regime(
    trades,
):

    rows = []

    if trades.empty:

        return pd.DataFrame()

    for direction in [
        "LONG",
        "SHORT",
    ]:

        for regime in [
            "MR",
            "TREND",
            "UNKNOWN",
        ]:

            subset = trades[
                (
                    trades["Side"]
                    == direction
                )
                &
                (
                    trades[
                        "EntryRegime"
                    ]
                    == regime
                )
            ].copy()

            if subset.empty:

                continue

            metrics = (
                calculate_trade_metrics(
                    subset
                )
            )

            metrics.update(
                {
                    "Direction":
                        direction,

                    "Regime":
                        regime,
                }
            )

            rows.append(
                metrics
            )

    if not rows:

        return pd.DataFrame()

    result = pd.DataFrame(
        rows
    )

    return result[
        [
            "Direction",
            "Regime",
            "Trades",
            "WinRate",
            "ProfitFactor",
            "ExpectancyPnL",
            "TotalPnL",
            "MeanReturn",
            "StopRate",
        ]
    ]


# ============================================================
# YEAR × DIRECTION × REGIME
# ============================================================

def analyze_year_direction_regime(
    trades,
):

    if trades.empty:

        return pd.DataFrame()

    data = trades.copy()

    data["Year"] = (
        pd.to_datetime(
            data["EntryTime"],
            utc=True,
        ).dt.year
    )

    rows = []

    for (
        year,
        direction,
        regime,
    ), group in data.groupby(
        [
            "Year",
            "Side",
            "EntryRegime",
        ]
    ):

        metrics = (
            calculate_trade_metrics(
                group
            )
        )

        metrics.update(
            {
                "Year":
                    year,

                "Direction":
                    direction,

                "Regime":
                    regime,
            }
        )

        rows.append(
            metrics
        )

    if not rows:

        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        [
            [
                "Year",
                "Direction",
                "Regime",
                "Trades",
                "WinRate",
                "ProfitFactor",
                "ExpectancyPnL",
                "TotalPnL",
                "MeanReturn",
                "StopRate",
            ]
        ]
        .sort_values(
            [
                "Year",
                "Direction",
                "Regime",
            ]
        )
    )


# ============================================================
# YEARLY TOTAL
# ============================================================

def analyze_yearly(
    trades,
):

    if trades.empty:

        return pd.DataFrame()

    data = trades.copy()

    data["Year"] = (
        pd.to_datetime(
            data["EntryTime"],
            utc=True,
        ).dt.year
    )

    rows = []

    for (
        year,
        group,
    ) in data.groupby(
        "Year"
    ):

        metrics = (
            calculate_trade_metrics(
                group
            )
        )

        metrics["Year"] = year

        rows.append(
            metrics
        )

    if not rows:

        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        [
            [
                "Year",
                "Trades",
                "WinRate",
                "ProfitFactor",
                "ExpectancyPnL",
                "TotalPnL",
                "MeanReturn",
                "StopRate",
            ]
        ]
        .sort_values(
            "Year"
        )
    )


# ============================================================
# ANALYZE ONE INSTRUMENT
# ============================================================

def analyze_instrument(
    symbol,
):

    print()
    print("=" * 100)
    print(
        f"ANALYZING {symbol}"
    )
    print("=" * 100)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    bars = load_h1(
        symbol
    )

    # --------------------------------------------------------
    # Generate trades
    # --------------------------------------------------------

    trades = generate_trades(
        bars,
        symbol,
    )

    if trades.empty:

        print(
            "No trades generated."
        )

        return None

    # --------------------------------------------------------
    # Overall
    # --------------------------------------------------------

    metrics = (
        calculate_trade_metrics(
            trades
        )
    )

    metrics[
        "Symbol"
    ] = symbol

    # --------------------------------------------------------
    # Save trades
    # --------------------------------------------------------

    trades.to_csv(
        OUTPUT_DIR
        / f"{symbol}_trades.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Direction × regime
    # --------------------------------------------------------

    direction_regime = (
        analyze_direction_regime(
            trades
        )
    )

    if not direction_regime.empty:

        direction_regime[
            "Symbol"
        ] = symbol

        direction_regime.to_csv(
            OUTPUT_DIR
            / f"{symbol}_direction_regime.csv",
            index=False,
        )

    # --------------------------------------------------------
    # Year × direction × regime
    # --------------------------------------------------------

    yearly_direction_regime = (
        analyze_year_direction_regime(
            trades
        )
    )

    if not yearly_direction_regime.empty:

        yearly_direction_regime[
            "Symbol"
        ] = symbol

        yearly_direction_regime.to_csv(
            OUTPUT_DIR
            / f"{symbol}_year_direction_regime.csv",
            index=False,
        )

    # --------------------------------------------------------
    # Yearly
    # --------------------------------------------------------

    yearly = analyze_yearly(
        trades
    )

    if not yearly.empty:

        yearly[
            "Symbol"
        ] = symbol

        yearly.to_csv(
            OUTPUT_DIR
            / f"{symbol}_yearly.csv",
            index=False,
        )

    # --------------------------------------------------------
    # Print direction × regime
    # --------------------------------------------------------

    print()
    print(
        "DIRECTION × REGIME"
    )

    if direction_regime.empty:

        print(
            "No direction/regime groups."
        )

    else:

        print(
            direction_regime[
                [
                    "Direction",
                    "Regime",
                    "Trades",
                    "WinRate",
                    "ProfitFactor",
                    "ExpectancyPnL",
                    "TotalPnL",
                    "StopRate",
                ]
            ].to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Print yearly
    # --------------------------------------------------------

    print()
    print(
        "YEARLY"
    )

    if yearly.empty:

        print(
            "No yearly data."
        )

    else:

        print(
            yearly[
                [
                    "Year",
                    "Trades",
                    "WinRate",
                    "ProfitFactor",
                    "TotalPnL",
                    "StopRate",
                ]
            ].to_string(
                index=False
            )
        )

    return {
        "Symbol":
            symbol,

        **metrics,

        "DirectionRegime":
            direction_regime,

        "Yearly":
            yearly,

        "YearDirectionRegime":
            yearly_direction_regime,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_summary = []

    all_direction_regime = []

    all_yearly = []

    all_year_direction_regime = []

    # ========================================================
    # RUN ALL 10
    # ========================================================

    for symbol in INSTRUMENTS:

        try:

            result = (
                analyze_instrument(
                    symbol
                )
            )

            if result is None:

                continue

            # ------------------------------------------------
            # Summary
            # ------------------------------------------------

            summary = {
                key: value
                for key, value
                in result.items()
                if key not in [
                    "DirectionRegime",
                    "Yearly",
                    "YearDirectionRegime",
                ]
            }

            all_summary.append(
                summary
            )

            # ------------------------------------------------
            # Direction × regime
            # ------------------------------------------------

            if not (
                result[
                    "DirectionRegime"
                ].empty
            ):

                all_direction_regime.append(
                    result[
                        "DirectionRegime"
                    ].copy()
                )

            # ------------------------------------------------
            # Yearly
            # ------------------------------------------------

            if not (
                result[
                    "Yearly"
                ].empty
            ):

                all_yearly.append(
                    result[
                        "Yearly"
                    ].copy()
                )

            # ------------------------------------------------
            # Year × direction × regime
            # ------------------------------------------------

            if not (
                result[
                    "YearDirectionRegime"
                ].empty
            ):

                all_year_direction_regime.append(
                    result[
                        "YearDirectionRegime"
                    ].copy()
                )

        except Exception as exc:

            print()
            print(
                f"ERROR: {symbol}"
            )

            print(
                repr(exc)
            )

    # ========================================================
    # SAVE COMBINED RESULTS
    # ========================================================

    if all_summary:

        summary_df = pd.DataFrame(
            all_summary
        )

        summary_df.to_csv(
            OUTPUT_DIR
            / "all_instruments_summary.csv",
            index=False,
        )

    if all_direction_regime:

        direction_regime_df = (
            pd.concat(
                all_direction_regime,
                ignore_index=True,
            )
        )

        direction_regime_df.to_csv(
            OUTPUT_DIR
            / "all_instruments_direction_regime.csv",
            index=False,
        )

    if all_yearly:

        yearly_df = pd.concat(
            all_yearly,
            ignore_index=True,
        )

        yearly_df.to_csv(
            OUTPUT_DIR
            / "all_instruments_yearly.csv",
            index=False,
        )

    if all_year_direction_regime:

        year_direction_regime_df = (
            pd.concat(
                all_year_direction_regime,
                ignore_index=True,
            )
        )

        year_direction_regime_df.to_csv(
            OUTPUT_DIR
            / "all_instruments_year_direction_regime.csv",
            index=False,
        )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()
    print(
        "#" * 110
    )

    print(
        "10-INSTRUMENT "
        "DIRECTION × REGIME DIAGNOSTIC"
    )

    print(
        "#" * 110
    )

    # --------------------------------------------------------
    # Overall
    # --------------------------------------------------------

    if all_summary:

        summary_df = pd.DataFrame(
            all_summary
        )

        print()
        print(
            "OVERALL"
        )

        print(
            summary_df[
                [
                    "Symbol",
                    "Trades",
                    "WinRate",
                    "ProfitFactor",
                    "ExpectancyPnL",
                    "TotalPnL",
                    "MeanReturn",
                    "StopRate",
                ]
            ].to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Direction × regime
    # --------------------------------------------------------

    if all_direction_regime:

        direction_regime_df = (
            pd.concat(
                all_direction_regime,
                ignore_index=True,
            )
        )

        print()
        print(
            "DIRECTION × REGIME"
        )

        print(
            direction_regime_df[
                [
                    "Symbol",
                    "Direction",
                    "Regime",
                    "Trades",
                    "WinRate",
                    "ProfitFactor",
                    "ExpectancyPnL",
                    "TotalPnL",
                    "StopRate",
                ]
            ].to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Files
    # --------------------------------------------------------

    print()
    print(
        "RESULTS SAVED TO:"
    )

    print(
        OUTPUT_DIR
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "This script is diagnostic only."
    )

    print(
        "No parameters were optimized."
    )

    print(
        "Entry Z remains 2.25."
    )

    print(
        "Exit Z remains 0.25."
    )

    print(
        "Stop Z remains 3.0."
    )

    print(
        "Markov threshold remains 0.60."
    )

    print(
        "No GARCH or volatility filter was added."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()