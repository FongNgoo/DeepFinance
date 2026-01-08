import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class threshold:
    def __init__(self):
        pass

    # =========================
    # 1. RETURN COMPUTATION
    # =========================
    def grow_rate(self, data, window_size=1):
        """
        r_t = p_t / p_{t-1} - 1
        (used for distribution & threshold estimation)
        """
        df = data.copy()
        df["return"] = df["close"] / df["close"].shift(1) - 1
        return df["return"]

    # =========================
    # 2. LABEL STATISTICS
    # =========================
    def label_statistics(self, growth_rate, flat_th):
        """
        Compute label count & ratio
        0 = DOWN, 1 = FLAT, 2 = UP
        """
        labels = np.select(
            [
                growth_rate < -flat_th,
                (growth_rate >= -flat_th) & (growth_rate <= flat_th),
                growth_rate > flat_th
            ],
            [0, 1, 2]
        )

        label_series = pd.Series(labels, name="label")

        stats = pd.DataFrame({
            "count": label_series.value_counts().sort_index(),
            "ratio": label_series.value_counts(normalize=True).sort_index()
        })

        stats.index = ["DOWN", "FLAT", "UP"]
        return stats

    # =========================
    # 3. DISTRIBUTION + THRESHOLD
    # =========================
    def distribution(self, data, future_size=1, bins=100, flat_ratio=33):
        """
        Percentile-based flat threshold
        """
        growth_rate = self.grow_rate(
            data, window_size=future_size
        ).dropna()

        # Adaptive flat threshold (percentile-based)
        flat_th = np.percentile(
            np.abs(growth_rate), flat_ratio
        )

        # ===== Plot =====
        plt.figure(figsize=(8, 5))
        plt.hist(growth_rate, bins=bins, density=True, alpha=0.7)

        plt.axvline(flat_th, linestyle="--",
                    label=f"+flat ({flat_th:.4f})")
        plt.axvline(-flat_th, linestyle="--",
                    label=f"-flat ({-flat_th:.4f})")

        plt.axvline(0, linestyle=":")
        plt.xlabel("Growth Rate")
        plt.ylabel("Density")
        plt.title(f"Growth Rate Distribution (window={future_size})")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # ===== Label statistics =====
        label_stats = self.label_statistics(
            growth_rate, flat_th
        )

        return flat_th, label_stats


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    # =========================
    # LOAD DATA
    # =========================
    data_path = r"D:\Project\NCKH\data\env_data\synchronized_data.pkl"
    Data = pd.read_pickle(data_path)

    stock_name = ["TSLA", "NFLX", "MSFT", "AMZN"]

    rows = []
    for d, content in Data.items():
        for stock in stock_name:
            row = {
                "date": d,
                "stock": stock,
                **content["price"][stock],
                **content["macro"],
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    df.sort_values(["stock", "date"], inplace=True)

    thr = threshold()

    # =========================
    # 1. GLOBAL THRESHOLD
    # =========================
    print("===== GLOBAL THRESHOLD =====")

    global_price_df = df[["open", "high", "close"]]

    global_th, global_stats = thr.distribution(
        global_price_df,
        future_size=1,
        flat_ratio=30
    )

    print(f"GLOBAL flat threshold: {global_th:.4f}")
    print("GLOBAL label statistics:")
    print(global_stats)

    # =========================
    # 2. PER-STOCK THRESHOLD
    # =========================
    print("\n===== PER-STOCK THRESHOLD =====")

    stock_thresholds = {}

    for stock in stock_name:
        print(f"\n--- {stock} ---")

        price_df = df[df["stock"] == stock][
            ["open", "high", "close"]
        ]

        th, stats = thr.distribution(
            price_df,
            future_size=1,
            flat_ratio=30
        )

        stock_thresholds[stock] = th

        print(f"{stock} flat threshold: {th:.4f}")
        print("Label statistics:")
        print(stats)

    # =========================
    # SUMMARY
    # =========================
    print("\n===== SUMMARY OF THRESHOLDS =====")
    for stock, th in stock_thresholds.items():
        print(f"{stock}: {th:.4f}")
