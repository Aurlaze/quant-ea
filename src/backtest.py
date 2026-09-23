"""
Event-driven research backtester for the Kalman mean-reversion strategy.

Strategy:
    Z <= -entry_z  -> LONG
    Z >= +entry_z  -> SHORT
    |Z| <= exit_z  -> EXIT
    |Z| >= stop_z  -> EXIT
    Trading hours >= max_hold_hours -> TIME_STOP

Execution:
    LONG entry  -> next bar Ask_Open
    LONG exit   -> next executable bar Bid_Open
    SHORT entry -> next bar Bid_Open
    SHORT exit  -> next executable bar Ask_Open

Holding-time convention:
    CalendarHoursHeld:
        Actual elapsed wall-clock time between entry and exit.

    TradingHoursHeld:
        Number of available hourly market bars held.

    HoursHeld:
        Alias for TradingHoursHeld for compatibility with
        the existing multi_month_backtest.py.

The maximum holding period is enforced using TRADING HOURS,
not calendar hours. Therefore weekends and market closures
do not artificially extend or consume the holding period.

IMPORTANT:
This is a research simulator.

It does NOT yet model:
    - exact MT5 contract specification
    - tick value
    - lot size
    - commission
    - swap
    - slippage
    - broker-specific execution behavior

Those will be incorporated during the MT5 stage.
"""

import numpy as np
import pandas as pd


class KalmanBacktester:

    def __init__(
        self,
        entry_z=2.0,
        exit_z=0.5,
        stop_z=3.5,
        max_hold_hours=24,
        initial_capital=10_000.0,
        risk_per_trade=0.005,
        max_notional_fraction=1.0,
        assumed_stop_return=0.01,
    ):

        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_z = stop_z

        self.max_hold_hours = max_hold_hours

        self.initial_capital = initial_capital

        self.risk_per_trade = risk_per_trade
        self.max_notional_fraction = max_notional_fraction
        self.assumed_stop_return = assumed_stop_return

    # ========================================================
    # POSITION SIZE
    # ========================================================

    def calculate_position_size(
        self,
        equity,
        entry_price,
    ):
        """
        Research position sizing.

        Risk budget:

            equity * risk_per_trade

        Temporary assumed stop:

            entry_price * assumed_stop_return

        No-leverage constraint:

            units * entry_price
                <= equity * max_notional_fraction

        This is NOT the final MT5 lot calculation.
        """

        if equity <= 0:
            return 0.0

        if entry_price <= 0:
            return 0.0

        # ----------------------------------------------------
        # Risk budget
        # ----------------------------------------------------

        risk_budget = (
            equity
            * self.risk_per_trade
        )

        # ----------------------------------------------------
        # Assumed risk per unit
        # ----------------------------------------------------

        risk_per_unit = (
            entry_price
            * self.assumed_stop_return
        )

        if risk_per_unit <= 0:
            return 0.0

        # ----------------------------------------------------
        # Risk-based units
        # ----------------------------------------------------

        risk_based_units = (
            risk_budget
            / risk_per_unit
        )

        # ----------------------------------------------------
        # No-leverage constraint
        # ----------------------------------------------------

        max_notional = (
            equity
            * self.max_notional_fraction
        )

        max_units = (
            max_notional
            / entry_price
        )

        units = min(
            risk_based_units,
            max_units,
        )

        return max(
            float(units),
            0.0,
        )

    # ========================================================
    # RUN BACKTEST
    # ========================================================

    def run(self, bars):
        """
        Run the event-driven backtest.

        Signal timing:

            Signal detected on bar i.

            Entry/exit executed on bar i+1.

        Time-stop:

            The position is allowed to remain open for at most
            max_hold_hours AVAILABLE HOURLY BARS.

            Missing timestamps caused by weekends or market
            closures do not count toward the holding limit.

        Example:

            Entry:
                Friday 15:00

            Market closes:
                Friday evening

            Weekend:
                No hourly bars

            Reopens:
                Monday

            The weekend does NOT count toward the 24-hour
            trading-hour limit.
        """

        data = bars.copy()

        # ====================================================
        # REQUIRED COLUMNS
        # ====================================================

        required_columns = [
            "Close",
            "Bid_Close",
            "Ask_Close",
            "Bid_Open",
            "Ask_Open",
            "ZScore",
        ]

        missing = (
            set(required_columns)
            - set(data.columns)
        )

        if missing:

            raise ValueError(
                f"Missing required columns: {missing}"
            )

        # ====================================================
        # CLEAN DATA
        # ====================================================

        data = data.dropna(
            subset=required_columns
        ).copy()

        data = data.sort_index()

        if len(data) < 2:

            return (
                pd.DataFrame(),
                pd.DataFrame(),
            )

        # ====================================================
        # DATETIME INDEX
        # ====================================================

        if not isinstance(
            data.index,
            pd.DatetimeIndex,
        ):

            data.index = pd.to_datetime(
                data.index,
                utc=True,
            )

        # ====================================================
        # ACCOUNT STATE
        # ====================================================

        cash = float(
            self.initial_capital
        )

        # ----------------------------------------------------
        # Position
        #
        # 0  = flat
        # +1 = long
        # -1 = short
        # ----------------------------------------------------

        position = 0

        units = 0.0

        # ----------------------------------------------------
        # Entry information
        # ----------------------------------------------------

        entry_price = np.nan

        entry_time = None

        entry_z = np.nan

        # ----------------------------------------------------
        # Trading-hour counter
        #
        # This is the important time-stop variable.
        #
        # It counts AVAILABLE hourly bars, not calendar hours.
        # ----------------------------------------------------

        hold_bars = 0

        # ====================================================
        # OUTPUTS
        # ====================================================

        trades = []

        equity_records = []

        # ====================================================
        # MARK TO MARKET
        # ====================================================

        def mark_to_market(
            current_bid,
            current_ask,
        ):

            # ------------------------------------------------
            # Flat
            # ------------------------------------------------

            if position == 0:

                return cash

            # ------------------------------------------------
            # Long
            # ------------------------------------------------

            if position == 1:

                unrealized_pnl = (
                    current_bid
                    - entry_price
                ) * units

            # ------------------------------------------------
            # Short
            # ------------------------------------------------

            else:

                unrealized_pnl = (
                    entry_price
                    - current_ask
                ) * units

            return (
                cash
                + unrealized_pnl
            )

        # ====================================================
        # CLOSE POSITION
        # ====================================================

        def close_position(
            exit_time,
            exit_price,
            exit_z,
            exit_reason,
        ):

            nonlocal cash
            nonlocal position
            nonlocal units
            nonlocal entry_price
            nonlocal entry_time
            nonlocal entry_z
            nonlocal hold_bars

            if position == 0:

                return

            # ------------------------------------------------
            # Side + PnL
            # ------------------------------------------------

            if position == 1:

                side = "LONG"

                pnl = (
                    exit_price
                    - entry_price
                ) * units

            else:

                side = "SHORT"

                pnl = (
                    entry_price
                    - exit_price
                ) * units

            # ------------------------------------------------
            # Calendar holding time
            # ------------------------------------------------

            calendar_hours_held = (
                exit_time
                - entry_time
            ).total_seconds() / 3600.0

            # ------------------------------------------------
            # Trading holding time
            # ------------------------------------------------

            trading_hours_held = float(
                hold_bars
            )

            # ------------------------------------------------
            # Trade return
            # ------------------------------------------------

            notional = (
                entry_price
                * units
            )

            if notional != 0:

                trade_return = (
                    pnl
                    / notional
                )

            else:

                trade_return = 0.0

            # ------------------------------------------------
            # Record trade
            # ------------------------------------------------

            trades.append(
                {
                    "EntryTime": entry_time,
                    "ExitTime": exit_time,
                    "Side": side,
                    "Units": units,
                    "EntryPrice": entry_price,
                    "ExitPrice": exit_price,
                    "EntryZ": entry_z,
                    "ExitZ": exit_z,

                    # Actual wall-clock duration.
                    "CalendarHoursHeld": (
                        calendar_hours_held
                    ),

                    # Actual tradable hourly bars.
                    "TradingHoursHeld": (
                        trading_hours_held
                    ),

                    # Compatibility with existing metrics.
                    "HoursHeld": (
                        trading_hours_held
                    ),

                    "Return": trade_return,
                    "PnL": pnl,
                    "ExitReason": exit_reason,
                }
            )

            # ------------------------------------------------
            # Realize PnL
            # ------------------------------------------------

            cash += pnl

            # ------------------------------------------------
            # Reset position
            # ------------------------------------------------

            position = 0

            units = 0.0

            entry_price = np.nan

            entry_time = None

            entry_z = np.nan

            hold_bars = 0

        # ====================================================
        # MAIN EVENT LOOP
        # ====================================================

        for i in range(
            len(data) - 1
        ):

            current = data.iloc[i]

            next_bar = data.iloc[i + 1]

            current_time = data.index[i]

            next_time = data.index[i + 1]

            z = current["ZScore"]

            # =================================================
            # CURRENT MARK-TO-MARKET
            # =================================================

            current_equity = mark_to_market(
                current["Bid_Close"],
                current["Ask_Close"],
            )

            # =================================================
            # INVALID Z-SCORE
            # =================================================

            if not np.isfinite(z):

                equity_records.append(
                    {
                        "Time": current_time,
                        "Cash": cash,
                        "Equity": current_equity,
                        "Position": position,
                        "Units": units,
                    }
                )

                continue

            # =================================================
            # EXISTING POSITION
            # =================================================

            if position != 0:

                # ------------------------------------------------
                # IMPORTANT:
                #
                # The current available bar represents one
                # additional tradable hourly observation.
                #
                # However, the entry bar itself starts at
                # hold_bars = 0.
                #
                # We evaluate the next executable bar.
                # ------------------------------------------------

                next_hold_bars = (
                    hold_bars + 1
                )

                exit_reason = None

                # =================================================
                # 1. HARD TRADING-HOUR TIME STOP
                # =================================================

                if (
                    next_hold_bars
                    >= self.max_hold_hours
                ):

                    exit_reason = "TIME_STOP"

                # =================================================
                # 2. Z EXIT
                # =================================================

                elif (
                    abs(z)
                    <= self.exit_z
                ):

                    exit_reason = "Z_EXIT"

                # =================================================
                # 3. Z STOP
                # =================================================

                elif (
                    abs(z)
                    >= self.stop_z
                ):

                    exit_reason = "Z_STOP"

                # =================================================
                # EXECUTE EXIT
                # =================================================

                if exit_reason is not None:

                    if position == 1:

                        exit_price = (
                            next_bar["Bid_Open"]
                        )

                    else:

                        exit_price = (
                            next_bar["Ask_Open"]
                        )

                    # Update the number of tradable
                    # hourly bars before closing.

                    hold_bars = (
                        next_hold_bars
                    )

                    close_position(
                        exit_time=next_time,
                        exit_price=exit_price,
                        exit_z=z,
                        exit_reason=exit_reason,
                    )

                    # ------------------------------------------------
                    # Record realized equity
                    # ------------------------------------------------

                    equity_records.append(
                        {
                            "Time": current_time,
                            "Cash": cash,
                            "Equity": cash,
                            "Position": 0,
                            "Units": 0.0,
                        }
                    )

                    continue

                # ------------------------------------------------
                # No exit.
                #
                # Advance trading-hour counter.
                # ------------------------------------------------

                hold_bars = (
                    next_hold_bars
                )

            # =================================================
            # NEW ENTRY
            # =================================================

            if position == 0:

                # =================================================
                # LONG
                # =================================================

                if (
                    z
                    <= -self.entry_z
                ):

                    new_entry_price = (
                        next_bar["Ask_Open"]
                    )

                    new_units = (
                        self.calculate_position_size(
                            cash,
                            new_entry_price,
                        )
                    )

                    if new_units > 0:

                        position = 1

                        units = new_units

                        entry_price = (
                            new_entry_price
                        )

                        entry_time = (
                            next_time
                        )

                        entry_z = z

                        # Entry starts the counter at zero.
                        hold_bars = 0

                # =================================================
                # SHORT
                # =================================================

                elif (
                    z
                    >= self.entry_z
                ):

                    new_entry_price = (
                        next_bar["Bid_Open"]
                    )

                    new_units = (
                        self.calculate_position_size(
                            cash,
                            new_entry_price,
                        )
                    )

                    if new_units > 0:

                        position = -1

                        units = new_units

                        entry_price = (
                            new_entry_price
                        )

                        entry_time = (
                            next_time
                        )

                        entry_z = z

                        # Entry starts the counter at zero.
                        hold_bars = 0

            # =================================================
            # MARK TO MARKET
            # =================================================

            equity = mark_to_market(
                current["Bid_Close"],
                current["Ask_Close"],
            )

            equity_records.append(
                {
                    "Time": current_time,
                    "Cash": cash,
                    "Equity": equity,
                    "Position": position,
                    "Units": units,
                }
            )

        # ====================================================
        # END OF DATA
        # ====================================================

        if (
            position != 0
            and len(data) > 0
        ):

            last_time = data.index[-1]

            last_bar = data.iloc[-1]

            if position == 1:

                exit_price = (
                    last_bar["Bid_Close"]
                )

            else:

                exit_price = (
                    last_bar["Ask_Close"]
                )

            last_z = last_bar["ZScore"]

            close_position(
                exit_time=last_time,
                exit_price=exit_price,
                exit_z=last_z,
                exit_reason="END_OF_DATA",
            )

            equity_records.append(
                {
                    "Time": last_time,
                    "Cash": cash,
                    "Equity": cash,
                    "Position": 0,
                    "Units": 0.0,
                }
            )

        # ====================================================
        # EQUITY DATAFRAME
        # ====================================================

        equity = pd.DataFrame(
            equity_records
        )

        if not equity.empty:

            equity = equity.set_index(
                "Time"
            )

            equity = equity[
                ~equity.index.duplicated(
                    keep="last"
                )
            ]

            equity = equity.sort_index()

        # ====================================================
        # TRADES DATAFRAME
        # ====================================================

        trades = pd.DataFrame(
            trades
        )

        if not trades.empty:

            trades["EntryTime"] = (
                pd.to_datetime(
                    trades["EntryTime"],
                    utc=True,
                )
            )

            trades["ExitTime"] = (
                pd.to_datetime(
                    trades["ExitTime"],
                    utc=True,
                )
            )

            trades = (
                trades
                .sort_values(
                    "EntryTime"
                )
                .reset_index(
                    drop=True
                )
            )

        return equity, trades


# ============================================================
# PERFORMANCE REPORT
# ============================================================

def performance_report(
    equity,
    trades,
):
    """
    Print standard performance statistics.
    """

    print("\n")
    print("=" * 70)
    print(
        "PROPER MARK-TO-MARKET BACKTEST"
    )
    print("=" * 70)

    # ========================================================
    # NORMALIZE EQUITY
    # ========================================================

    if isinstance(
        equity,
        pd.DataFrame,
    ):

        if "Equity" in equity.columns:

            equity_series = (
                equity["Equity"]
            )

        elif len(equity.columns) == 1:

            equity_series = (
                equity.iloc[:, 0]
            )

        else:

            raise ValueError(
                "Cannot identify equity column."
            )

    else:

        equity_series = equity

    equity_series = (
        pd.Series(
            equity_series
        )
        .dropna()
        .astype(float)
    )

    if len(equity_series) == 0:

        print(
            "No equity data."
        )

        return

    # ========================================================
    # RETURN
    # ========================================================

    initial = (
        equity_series.iloc[0]
    )

    final = (
        equity_series.iloc[-1]
    )

    total_return = (
        final / initial
        - 1.0
    )

    # ========================================================
    # DRAWDOWN
    # ========================================================

    running_max = (
        equity_series.cummax()
    )

    drawdown = (
        equity_series
        / running_max
        - 1.0
    )

    max_dd = drawdown.min()

    # ========================================================
    # SHARPE
    # ========================================================

    returns = (
        equity_series
        .pct_change()
        .dropna()
    )

    if (
        len(returns) > 1
        and returns.std() > 0
    ):

        sharpe = (
            returns.mean()
            / returns.std()
            * np.sqrt(24 * 252)
        )

    else:

        sharpe = np.nan

    # ========================================================
    # TRADE STATISTICS
    # ========================================================

    if (
        trades is not None
        and not trades.empty
    ):

        pnl = pd.to_numeric(
            trades["PnL"],
            errors="coerce",
        ).dropna()

        trade_count = len(pnl)

        wins = pnl[
            pnl > 0
        ]

        losses = pnl[
            pnl < 0
        ]

        if trade_count > 0:

            win_rate = (
                len(wins)
                / trade_count
            )

            expectancy = (
                pnl.mean()
            )

        else:

            win_rate = np.nan
            expectancy = np.nan

        gross_profit = (
            wins.sum()
        )

        gross_loss = abs(
            losses.sum()
        )

        if gross_loss > 0:

            profit_factor = (
                gross_profit
                / gross_loss
            )

        elif gross_profit > 0:

            profit_factor = np.inf

        else:

            profit_factor = np.nan

        # ----------------------------------------------------
        # Trading-hours holding time
        # ----------------------------------------------------

        if "TradingHoursHeld" in trades.columns:

            avg_hold = pd.to_numeric(
                trades["TradingHoursHeld"],
                errors="coerce",
            ).mean()

        elif "HoursHeld" in trades.columns:

            avg_hold = pd.to_numeric(
                trades["HoursHeld"],
                errors="coerce",
            ).mean()

        else:

            avg_hold = np.nan

    else:

        trade_count = 0
        win_rate = np.nan
        profit_factor = np.nan
        expectancy = np.nan
        avg_hold = np.nan

    # ========================================================
    # PRINT
    # ========================================================

    print(
        f"Initial equity : "
        f"${initial:,.2f}"
    )

    print(
        f"Final equity   : "
        f"${final:,.2f}"
    )

    print(
        f"Total return   : "
        f"{total_return * 100:.2f}%"
    )

    print(
        f"Maximum DD     : "
        f"{max_dd * 100:.2f}%"
    )

    if np.isfinite(sharpe):

        print(
            f"Sharpe         : "
            f"{sharpe:.3f}"
        )

    else:

        print(
            "Sharpe         : NaN"
        )

    print(
        f"Trades         : "
        f"{trade_count}"
    )

    if np.isfinite(win_rate):

        print(
            f"Win rate       : "
            f"{win_rate * 100:.2f}%"
        )

    else:

        print(
            "Win rate       : NaN"
        )

    if np.isfinite(
        profit_factor
    ):

        print(
            f"Profit factor  : "
            f"{profit_factor:.3f}"
        )

    elif profit_factor == np.inf:

        print(
            "Profit factor  : inf"
        )

    else:

        print(
            "Profit factor  : NaN"
        )

    if np.isfinite(
        expectancy
    ):

        print(
            f"Expectancy     : "
            f"${expectancy:.2f}"
        )

    else:

        print(
            "Expectancy     : NaN"
        )

    if np.isfinite(
        avg_hold
    ):

        print(
            f"Avg hold       : "
            f"{avg_hold:.2f} trading hours"
        )

    else:

        print(
            "Avg hold       : NaN"
        )

    # ========================================================
    # EXIT REASONS
    # ========================================================

    if (
        trades is not None
        and not trades.empty
        and "ExitReason" in trades.columns
    ):

        print("\nExit reasons:")

        print(
            trades[
                "ExitReason"
            ]
            .value_counts()
            .to_string()
        )

    # ========================================================
    # MONTHLY RETURNS
    # ========================================================

    if len(equity_series) > 0:

        monthly_equity = (
            equity_series
            .resample("ME")
            .last()
        )

        monthly_returns = (
            monthly_equity
            .pct_change()
        )

        if len(monthly_returns) > 0:

            monthly_returns.iloc[0] = (
                monthly_equity.iloc[0]
                / initial
                - 1.0
            )

        print("\n")
        print("=" * 70)
        print(
            "MONTHLY RETURNS"
        )
        print("=" * 70)

        print(
            (
                monthly_returns
                * 100
            )
            .to_frame(
                "Return (%)"
            )
            .to_string()
        )