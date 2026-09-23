from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


# =========================================================
# CONFIGURATION
# =========================================================

N_STATES = 2

HMM_WINDOW = 1000

MIN_OBSERVATIONS = 300

REFIT_EVERY = 24

RANDOM_STATE = 42

N_ITER = 150

TOL = 1e-4


# =========================================================
# FEATURE CONSTRUCTION
# =========================================================

def build_regime_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build causal regime features.

    Features:

    1. LogReturn
       Hourly log return.

    2. Volatility24
       Rolling 24-hour volatility.

    3. Autocorr24
       Rolling 24-hour lag-1 return autocorrelation.

    All features only use current/past observations.
    """

    close = (
        df["Close"]
        .astype(float)
    )

    log_return = np.log(
        close / close.shift(1)
    )

    volatility_24 = (
        log_return
        .rolling(24)
        .std()
    )

    autocorr_24 = (
        log_return
        .rolling(24)
        .corr(
            log_return.shift(1)
        )
    )

    features = pd.DataFrame(
        {
            "LogReturn": log_return,
            "Volatility24": volatility_24,
            "Autocorr24": autocorr_24,
        },
        index=df.index,
    )

    return features


# =========================================================
# FIT STANDARDIZER
# =========================================================

def fit_standardizer(
    features: pd.DataFrame,
):
    """
    Fit scaling parameters using ONLY the training window.
    """

    values = (
        features
        .values
        .astype(float)
    )

    mean = np.nanmean(
        values,
        axis=0,
    )

    std = np.nanstd(
        values,
        axis=0,
    )

    std = np.where(
        std > 1e-12,
        std,
        1.0,
    )

    return mean, std


# =========================================================
# APPLY STANDARDIZER
# =========================================================

def transform_features(
    features: pd.DataFrame,
    mean: np.ndarray,
    std: np.ndarray,
):
    """
    Transform using the scaler from the HMM's
    training window.

    IMPORTANT:
    Do not refit mean/std between HMM refits.
    """

    values = (
        features
        .values
        .astype(float)
    )

    return (
        values - mean
    ) / std


# =========================================================
# STATE DIAGNOSTICS
# =========================================================

def calculate_state_statistics(
    features: pd.DataFrame,
    states: np.ndarray,
):
    """
    Calculate observable statistics for each HMM state.

    The raw feature values are used here rather than the
    standardized values.
    """

    stats = []

    for state in range(
        N_STATES
    ):

        mask = (
            states == state
        )

        state_features = (
            features.loc[
                mask
            ]
        )

        if len(state_features) == 0:

            stats.append(
                {
                    "State":
                        state,

                    "MeanReturn":
                        np.nan,

                    "Volatility":
                        np.nan,

                    "Autocorrelation":
                        np.nan,

                    "Observations":
                        0,
                }
            )

            continue

        stats.append(
            {
                "State":
                    state,

                "MeanReturn":
                    state_features[
                        "LogReturn"
                    ].mean(),

                "Volatility":
                    state_features[
                        "Volatility24"
                    ].mean(),

                "Autocorrelation":
                    state_features[
                        "Autocorr24"
                    ].mean(),

                "Observations":
                    len(state_features),
            }
        )

    return pd.DataFrame(
        stats
    )


# =========================================================
# IDENTIFY REGIMES
# =========================================================

def identify_regimes(
    state_statistics: pd.DataFrame,
):
    """
    Identify the two states.

    Primary criterion:
        lower return autocorrelation
        = more mean-reverting.

    Higher return autocorrelation:
        = more persistent/trending.

    If autocorrelation values are extremely close,
    volatility is used as a secondary criterion.
    """

    stats = (
        state_statistics
        .copy()
        .set_index("State")
    )

    ac0 = stats.loc[
        0,
        "Autocorrelation",
    ]

    ac1 = stats.loc[
        1,
        "Autocorrelation",
    ]

    # -----------------------------------------------------
    # Normal case
    # -----------------------------------------------------

    if (
        np.isfinite(ac0)
        and np.isfinite(ac1)
        and abs(ac0 - ac1) >= 0.005
    ):

        if ac0 < ac1:

            mr_state = 0
            trend_state = 1

        else:

            mr_state = 1
            trend_state = 0

    # -----------------------------------------------------
    # Tie-breaker:
    # lower volatility is considered more
    # mean-reversion-like.
    # -----------------------------------------------------

    else:

        vol0 = stats.loc[
            0,
            "Volatility",
        ]

        vol1 = stats.loc[
            1,
            "Volatility",
        ]

        if vol0 <= vol1:

            mr_state = 0
            trend_state = 1

        else:

            mr_state = 1
            trend_state = 0

    return (
        int(mr_state),
        int(trend_state),
    )


# =========================================================
# FIT HMM
# =========================================================

def fit_hmm(
    features: pd.DataFrame,
):
    """
    Fit a two-state Gaussian HMM.

    Returns:

        model
        scaler_mean
        scaler_std
        mr_state
        trend_state
        state_statistics
    """

    clean = (
        features
        .dropna()
        .copy()
    )

    if len(clean) < MIN_OBSERVATIONS:

        raise ValueError(
            "Not enough observations "
            f"for HMM: {len(clean)}"
        )

    # -----------------------------------------------------
    # Fit scaler ONLY on this training window
    # -----------------------------------------------------

    scaler_mean, scaler_std = (
        fit_standardizer(
            clean
        )
    )

    X = transform_features(
        clean,
        scaler_mean,
        scaler_std,
    )

    # -----------------------------------------------------
    # HMM
    # -----------------------------------------------------

    model = GaussianHMM(
        n_components=N_STATES,
        covariance_type="diag",
        n_iter=N_ITER,
        tol=TOL,
        random_state=RANDOM_STATE,
        min_covar=1e-6,
    )

    with warnings.catch_warnings():

        warnings.filterwarnings(
            "ignore",
            message="Model is not converging.*",
        )

        model.fit(X)

    # -----------------------------------------------------
    # State assignment
    # -----------------------------------------------------

    states = model.predict(
        X
    )

    state_statistics = (
        calculate_state_statistics(
            clean,
            states,
        )
    )

    (
        mr_state,
        trend_state,
    ) = identify_regimes(
        state_statistics
    )

    return {
        "model":
            model,

        "scaler_mean":
            scaler_mean,

        "scaler_std":
            scaler_std,

        "mr_state":
            mr_state,

        "trend_state":
            trend_state,

        "state_statistics":
            state_statistics,
    }


# =========================================================
# CALCULATE CAUSAL REGIME PROBABILITY
# =========================================================

def calculate_regime_probability(
    df: pd.DataFrame,
    hmm_window: int = HMM_WINDOW,
    min_observations: int = MIN_OBSERVATIONS,
    refit_every: int = REFIT_EVERY,
):
    """
    Chronological / causal HMM regime calculation.

    At each refit:

        1. Use only observations <= current bar.
        2. Fit HMM.
        3. Fit scaler.
        4. Freeze scaler + HMM.
        5. Use them until next refit.

    The probability at bar t therefore never uses
    observations after t.
    """

    features = (
        build_regime_features(
            df
        )
    )

    result = pd.DataFrame(
        index=df.index
    )

    result[
        "HMMState"
    ] = np.nan

    result[
        "P_MeanReverting"
    ] = np.nan

    result[
        "P_Trending"
    ] = np.nan

    result[
        "MRState"
    ] = np.nan

    result[
        "TrendState"
    ] = np.nan

    result[
        "MR_Autocorrelation"
    ] = np.nan

    result[
        "Trend_Autocorrelation"
    ] = np.nan

    result[
        "MR_Volatility"
    ] = np.nan

    result[
        "Trend_Volatility"
    ] = np.nan

    hmm_info = None

    last_fit_position = (
        -refit_every
    )

    for i in range(
        len(df)
    ):

        # -------------------------------------------------
        # Need enough history
        # -------------------------------------------------

        if i < min_observations:

            continue

        # -------------------------------------------------
        # REFIT
        # -------------------------------------------------

        if (
            hmm_info is None
            or (
                i
                - last_fit_position
                >= refit_every
            )
        ):

            window_start = max(
                0,
                i - hmm_window + 1,
            )

            training_features = (
                features.iloc[
                    window_start:i + 1
                ]
                .dropna()
            )

            if (
                len(training_features)
                < min_observations
            ):

                continue

            try:

                hmm_info = fit_hmm(
                    training_features
                )

            except Exception:

                hmm_info = None

                continue

            last_fit_position = i

        # -------------------------------------------------
        # No valid model
        # -------------------------------------------------

        if hmm_info is None:

            continue

        # -------------------------------------------------
        # Current sequence
        #
        # IMPORTANT:
        # The SAME scaler from the HMM training window
        # is used.
        # -------------------------------------------------

        sequence_start = max(
            0,
            i - hmm_window + 1,
        )

        current_features = (
            features.iloc[
                sequence_start:i + 1
            ]
            .dropna()
        )

        if len(current_features) == 0:

            continue

        X_current = (
            transform_features(
                current_features,
                hmm_info[
                    "scaler_mean"
                ],
                hmm_info[
                    "scaler_std"
                ],
            )
        )

        try:

            probabilities = (
                hmm_info["model"]
                .predict_proba(
                    X_current
                )
            )

            states = (
                hmm_info["model"]
                .predict(
                    X_current
                )
            )

        except Exception:

            continue

        # -------------------------------------------------
        # Last observation = current bar
        # -------------------------------------------------

        current_index = (
            current_features.index[-1]
        )

        state = int(
            states[-1]
        )

        mr_state = int(
            hmm_info["mr_state"]
        )

        trend_state = int(
            hmm_info["trend_state"]
        )

        p_mr = float(
            probabilities[
                -1,
                mr_state,
            ]
        )

        p_trend = float(
            probabilities[
                -1,
                trend_state,
            ]
        )

        stats = (
            hmm_info[
                "state_statistics"
            ]
            .set_index("State")
        )

        result.loc[
            current_index,
            "HMMState",
        ] = state

        result.loc[
            current_index,
            "P_MeanReverting",
        ] = p_mr

        result.loc[
            current_index,
            "P_Trending",
        ] = p_trend

        result.loc[
            current_index,
            "MRState",
        ] = mr_state

        result.loc[
            current_index,
            "TrendState",
        ] = trend_state

        result.loc[
            current_index,
            "MR_Autocorrelation",
        ] = stats.loc[
            mr_state,
            "Autocorrelation",
        ]

        result.loc[
            current_index,
            "Trend_Autocorrelation",
        ] = stats.loc[
            trend_state,
            "Autocorrelation",
        ]

        result.loc[
            current_index,
            "MR_Volatility",
        ] = stats.loc[
            mr_state,
            "Volatility",
        ]

        result.loc[
            current_index,
            "Trend_Volatility",
        ] = stats.loc[
            trend_state,
            "Volatility",
        ]

    return result


# =========================================================
# ADD MARKOV REGIME
# =========================================================

def add_markov_regime(
    df: pd.DataFrame,
    hmm_window: int = HMM_WINDOW,
    min_observations: int = MIN_OBSERVATIONS,
    refit_every: int = REFIT_EVERY,
):
    """
    Add causal Markov regime probabilities to dataframe.
    """

    regime = calculate_regime_probability(
        df,
        hmm_window=hmm_window,
        min_observations=min_observations,
        refit_every=refit_every,
    )

    out = df.copy()

    for column in regime.columns:

        out[column] = regime[
            column
        ]

    return out