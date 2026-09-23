from data import load_exness_ticks, ticks_to_bars


CSV_PATH = "../data/Exness_XAUUSDm_2026_09.csv"


def main():

    # -----------------------------
    # Load tick data
    # -----------------------------

    ticks = load_exness_ticks(CSV_PATH)

    print("\n========== TICK DATA ==========")

    print("Number of ticks:", len(ticks))

    print(
        "Start:",
        ticks["Timestamp"].min()
    )

    print(
        "End:",
        ticks["Timestamp"].max()
    )

    print("\nColumns:")
    print(ticks.columns.tolist())

    # -----------------------------
    # Spread analysis
    # -----------------------------

    print("\n========== SPREAD ==========")

    print(
        ticks["Spread"].describe()
    )

    # -----------------------------
    # Convert to hourly bars
    # -----------------------------

    bars = ticks_to_bars(
        ticks,
        timeframe="1h"
    )

    print("\n========== HOURLY DATA ==========")

    print("Number of bars:", len(bars))

    print(
        "Start:",
        bars.index.min()
    )

    print(
        "End:",
        bars.index.max()
    )

    print("\nFirst 5 bars:")
    print(bars.head())

    print("\nReturn statistics:")
    print(
        bars["LogReturn"].describe()
    )


if __name__ == "__main__":
    main()