"""Đọc dữ liệu thật và tạo dữ liệu giả (data_fake)."""
import os

import numpy as np
import pandas as pd

from config import SEED


def load_data(data_dir):
    """Đọc train.csv, test.csv, sample_submission.csv trong data_dir."""
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))
    sample_sub = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))
    print(f"Train: {train.shape} | Test: {test.shape} | Sample submission: {sample_sub.shape}")
    return train, test, sample_sub


def make_fake_data(real_dir, fake_dir, n_train=300, n_test=100):
    """
    Tạo bộ dữ liệu giả nhỏ có CÙNG TÊN CỘT với dữ liệu thật.
    Mỗi cột: bốc ngẫu nhiên giá trị từ cột thật (giữ kiểu dữ liệu và cả giá trị thiếu).
    SalePrice: tự sinh bằng công thức đơn giản (chất lượng, diện tích, năm xây) + nhiễu.
    """
    os.makedirs(fake_dir, exist_ok=True)
    rng = np.random.RandomState(SEED)
    real_train = pd.read_csv(os.path.join(real_dir, "train.csv"))
    real_test = pd.read_csv(os.path.join(real_dir, "test.csv"))

    def sample_features(n, start_id):
        df = pd.DataFrame({"Id": np.arange(start_id, start_id + n)})
        for col in real_test.columns:
            if col != "Id":
                df[col] = rng.choice(real_train[col].values, size=n, replace=True)
        df["GrLivArea"] = df["1stFlrSF"] + df["2ndFlrSF"]  # cho hợp lý hơn
        return df

    fake_train = sample_features(n_train, 1)
    fake_test = sample_features(n_test, n_train + 1)
    fake_train["SalePrice"] = (
        30000
        + 18000 * fake_train["OverallQual"]
        + 55 * fake_train["GrLivArea"]
        + 25 * fake_train["TotalBsmtSF"].fillna(0)
        + 300 * (fake_train["YearBuilt"] - 1900)
        + rng.normal(0, 15000, n_train)
    ).clip(lower=40000).round(0).astype(int)
    sample_sub = pd.DataFrame({"Id": fake_test["Id"], "SalePrice": 180000})

    fake_train.to_csv(os.path.join(fake_dir, "train.csv"), index=False)
    fake_test.to_csv(os.path.join(fake_dir, "test.csv"), index=False)
    sample_sub.to_csv(os.path.join(fake_dir, "sample_submission.csv"), index=False)
    print(f"[data_fake] Đã tạo: train={fake_train.shape}, test={fake_test.shape} -> {fake_dir}")
