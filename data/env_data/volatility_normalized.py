import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class VolatilityThreshold:
    """
    Volatility-normalized threshold estimation
    """

    def __init__(self, vol_window=20, k=0.5, min_vol=1e-4):
        """
        vol_window : rolling window for volatility
        k          : volatility scaling factor
        min_vol    : numerical stability
        """
        self.vol_window = vol_window
        self.k = k
        self.min_vol = min_vol

    # =========================
    # 1. RETURN COMPUTATION
    # =========================
    def past_return(self, df):
        """
        Past return (for volatility estimation only)
        r_t = p_t / p_{t-1} - 1
        """
        return df["close"] / df["close"].shift(1) - 1

    def future_return(self, df, horizon=1):
        """
        Future return (for labeling / analysis)
        r_t = p_{t+h} / p_t - 1
        """
        return df["close"].shift(-horizon) / df["close"] - 1

    # =========================
    # 2. VOLATILITY
    # =========================
    def rolling_volatility(self, past_ret):
        """
        Rolling volatility using past returns ONLY
        """
        vol = (
            past_ret
            .rolling(self.vol_window)
            .std()
            .shift(1)
        )

        return vol.clip(lower=self.min_vol)

    # =========================
    # 3. VOL-NORMALIZED THRESHOLD
    # =========================
    def compute_threshold(self, price_df, horizon=1):
        """
        Compute volatility-normalized threshold
        """
        df = price_df.copy()

        df["past_return"] = self.past_return(df)
        df["volatility"] = self.rolling_volatility(df["past_return"])
        df["future_return"] = self.future_return(df, horizon)

        df.dropna(inplace=True)

        df["threshold"] = self.k * df["volatility"]

        return df

    # =========================
    # 4. DISTRIBUTION PLOT
    # =========================
    def plot_distribution(self, df, title="", bins=100):
        """
        Plot future return distribution + dynamic thresholds
        """
        plt.figure(figsize=(8, 5))

        plt.hist(df["future_return"], bins=bins, density=True, alpha=0.6)

        mean_th = df["threshold"].mean()

        plt.axvline(mean_th, linestyle="--", label=f"+k·σ ({mean_th:.4f})")
        plt.axvline(-mean_th, linestyle="--", label=f"-k·σ ({-mean_th:.4f})")

        plt.axvline(0, linestyle=":")
        plt.xlabel("Future Return")
        plt.ylabel("Density")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # =========================
    # 5. LABEL STATISTICS
    # =========================
    def label_stats(self, df):
        """
        Compute label distribution
        """
        labels = np.select(
            [
                df["future_return"] < -df["threshold"],
                (df["future_return"] >= -df["threshold"]) &
                (df["future_return"] <= df["threshold"]),
                df["future_return"] > df["threshold"]
            ],
            [0, 1, 2]  # DOWN, FLAT, UP
        )

        return pd.Series(labels).value_counts(normalize=True)


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
            rows.append({
                "date": d,
                "stock": stock,
                **content["price"][stock],
                **content["macro"]
            })

    df = pd.DataFrame(rows)
    df.sort_values(["stock", "date"], inplace=True)

    vt = VolatilityThreshold(
        vol_window=20,
        k=0.5
    )

    # =========================
    # 1. GLOBAL THRESHOLD
    # =========================
    print("===== GLOBAL VOLATILITY-NORMALIZED THRESHOLD =====")

    global_price_df = df[["date", "close"]].sort_values("date")

    global_df = vt.compute_threshold(
        global_price_df,
        horizon=1
    )

    vt.plot_distribution(
        global_df,
        title="Global Future Return Distribution (Vol-Normalized)"
    )

    print("GLOBAL label distribution:")
    print(vt.label_stats(global_df))

    print(f"GLOBAL mean threshold: {global_df['threshold'].mean():.4f}")

    # =========================
    # 2. PER-STOCK THRESHOLD
    # =========================
    print("\n===== PER-STOCK VOLATILITY-NORMALIZED THRESHOLD =====")

    stock_thresholds = {}

    for stock in stock_name:
        print(f"\n--- {stock} ---")

        stock_df = df[df["stock"] == stock][["date", "close"]]
        stock_df.sort_values("date", inplace=True)

        stock_ret_df = vt.compute_threshold(
            stock_df,
            horizon=1
        )

        vt.plot_distribution(
            stock_ret_df,
            title=f"{stock} Future Return Distribution (Vol-Normalized)"
        )

        stats = vt.label_stats(stock_ret_df)

        stock_thresholds[stock] = stock_ret_df["threshold"].mean()

        print("Label distribution:")
        print(stats)
        print(f"{stock} mean threshold: {stock_thresholds[stock]:.4f}")
