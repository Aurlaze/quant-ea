import numpy as np


class MeanRevertingKalman:
    """
    Scalar Kalman filter with a mean-reverting latent state.

    State equation:

        x_t = phi * x_(t-1) + (1 - phi) * mu + eta_t

    Observation equation:

        y_t = x_t + epsilon_t

    where:

        x_t  = latent/fair price
        y_t  = observed price
        phi  = persistence parameter
        mu   = long-run state mean
        Q    = process variance
        R    = observation variance
    """

    def __init__(
        self,
        phi: float = 0.995,
        q: float = 0.05,
        r: float = 0.25,
    ):
        if not 0 < phi < 1:
            raise ValueError("phi must be between 0 and 1.")

        if q <= 0:
            raise ValueError("q must be > 0.")

        if r <= 0:
            raise ValueError("r must be > 0.")

        self.phi = phi
        self.q = q
        self.r = r

        self.mu = None
        self.state = None
        self.variance = None

    def fit_transform(self, prices):
        """
        Run the Kalman filter over a price series.

        Returns:
            state_estimate
            state_variance
            residual
        """

        prices = np.asarray(prices, dtype=float)

        if len(prices) < 2:
            raise ValueError(
                "At least two price observations are required."
            )

        if np.any(~np.isfinite(prices)):
            raise ValueError(
                "Price series contains NaN or infinite values."
            )

        # Long-run mean used by the prototype.
        self.mu = np.mean(prices)

        n = len(prices)

        states = np.zeros(n)
        variances = np.zeros(n)
        residuals = np.zeros(n)

        # Initial state
        states[0] = prices[0]
        variances[0] = self.r
        residuals[0] = 0.0

        for t in range(1, n):

            # ==========================================
            # 1. PREDICT
            # ==========================================

            predicted_state = (
                self.phi * states[t - 1]
                + (1 - self.phi) * self.mu
            )

            predicted_variance = (
                self.phi ** 2
                * variances[t - 1]
                + self.q
            )

            # ==========================================
            # 2. INNOVATION
            # ==========================================

            innovation = (
                prices[t] - predicted_state
            )

            innovation_variance = (
                predicted_variance + self.r
            )

            # ==========================================
            # 3. KALMAN GAIN
            # ==========================================

            kalman_gain = (
                predicted_variance
                / innovation_variance
            )

            # ==========================================
            # 4. UPDATE
            # ==========================================

            states[t] = (
                predicted_state
                + kalman_gain * innovation
            )

            variances[t] = (
                (1 - kalman_gain)
                * predicted_variance
            )

            # ==========================================
            # 5. RESIDUAL
            # ==========================================

            residuals[t] = (
                prices[t] - states[t]
            )

        self.state = states[-1]
        self.variance = variances[-1]

        return states, variances, residuals