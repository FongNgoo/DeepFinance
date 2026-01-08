import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler


class data_prepare:
    def __init__(self, data_path) -> None:
        self.data_path = data_path

    # ======================================================
    # LABEL: GIỮ NGUYÊN CÔNG THỨC RETURN
    # r_t = close_t / close_{t-1} - 1
    # ======================================================
    def create_return(self, price_df):
        df = price_df.copy()
        df["return"] = df["close"] / df["close"].shift(1) - 1
        df.dropna(inplace=True)
        return df[["return"]]

    def make_window(self, data, window_size):
        """
        data: numpy array (T, D)
        return: (N, window_size, D)
        """
        X = []
        for i in range(len(data) - window_size + 1):
            X.append(data[i:i + window_size])
        return np.array(X)

    def prepare_data(
        self,
        stock_name,
        window_size=20,
        future_days=1,
        train_ratio=0.8,
        flat_ratio=30
    ):
        # ==========================
        # LOAD DATA
        # ==========================
        Data = pd.read_pickle(self.data_path)

        df = pd.DataFrame({
            d: {
                **content["price"][stock_name],
                **content["macro"],
            }
            for d, content in Data.items()
        }).T

        price_df = df[["open", "high", "close"]]
        macro_df = df[
            ["vix", "yield_spread_10y_2y",
             "sp500", "sp500_return", "dxy", "wti"]
        ]

        # ==========================
        # RETURN (PAST RETURN – GIỮ NGUYÊN)
        # ==========================
        return_df = self.create_return(price_df)

        # align
        price_df = price_df.loc[return_df.index]
        macro_df = macro_df.loc[return_df.index]

        # ==========================
        # NORMALIZATION
        # ==========================
        # price: log-return (INPUT)
        price_df = np.log(price_df / price_df.shift(1))
        price_df.dropna(inplace=True)

        macro_df = macro_df.loc[price_df.index]
        return_df = return_df.loc[price_df.index]

        # macro clean
        macro_df = macro_df.replace([np.inf, -np.inf], np.nan)
        macro_df = macro_df.fillna(method="ffill").fillna(method="bfill")

        # ==========================
        # NUMPY
        # ==========================
        price_np = price_df.values
        macro_np = macro_df.values
        return_np = return_df.values  # (T, 1)

        # ==========================
        # WINDOWING
        # ==========================
        price_win = self.make_window(price_np, window_size)
        macro_win = self.make_window(macro_np, window_size)

        # ❗ LABEL = RETURN SAU WINDOW (KHÔNG LEAK)
        label_raw = return_np[window_size - 1 + future_days:]

        price_win = price_win[:-future_days]
        macro_win = macro_win[:-future_days]

        # ==========================
        # SPLIT (TIME-SERIES SAFE)
        # ==========================
        split_idx = int(len(price_win) * train_ratio)

        # ==========================
        # THRESHOLD (FIT ON TRAIN ONLY)
        # ==========================
        train_returns = label_raw[:split_idx].flatten()
        threshold = np.percentile(np.abs(train_returns), flat_ratio)

        def map_label(r):
            if r < -threshold:
                return 0  # DOWN
            elif r > threshold:
                return 2  # UP
            else:
                return 1  # FLAT

        label_all = np.array([map_label(r[0]) for r in label_raw])

        # ==========================
        # MACRO NORMALIZATION (TRAIN ONLY)
        # ==========================
        macro_mean = macro_win[:split_idx].mean(axis=(0, 1), keepdims=True)
        macro_std = macro_win[:split_idx].std(axis=(0, 1), keepdims=True) + 1e-6
        macro_win = (macro_win - macro_mean) / macro_std

        # ==========================
        # TORCH TENSORS
        # ==========================
        train_data = {
            "s_o": torch.tensor(price_win[:split_idx, :, 0:1], dtype=torch.float32),
            "s_h": torch.tensor(price_win[:split_idx, :, 1:2], dtype=torch.float32),
            "s_c": torch.tensor(price_win[:split_idx, :, 2:3], dtype=torch.float32),
            "s_m": torch.tensor(macro_win[:split_idx], dtype=torch.float32),
            "label": torch.tensor(label_all[:split_idx], dtype=torch.long),
        }

        test_data = {
            "s_o": torch.tensor(price_win[split_idx:, :, 0:1], dtype=torch.float32),
            "s_h": torch.tensor(price_win[split_idx:, :, 1:2], dtype=torch.float32),
            "s_c": torch.tensor(price_win[split_idx:, :, 2:3], dtype=torch.float32),
            "s_m": torch.tensor(macro_win[split_idx:], dtype=torch.float32),
            "label": torch.tensor(label_all[split_idx:], dtype=torch.long),
        }

        return train_data, test_data
