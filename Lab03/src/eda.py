"""EDA: lưu biểu đồ (.png) và các bảng thống kê (.csv) vào thư mục outputs."""
import os

import matplotlib

matplotlib.use("Agg")  # chỉ lưu ảnh, không mở cửa sổ
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")


def run_eda(train, out_dir):
    print("\n===== EDA =====")
    num_cols = train.select_dtypes(include=[np.number]).columns
    cat_cols = train.select_dtypes(exclude=[np.number]).columns
    print(f"Số cột số: {len(num_cols)} | Số cột phân loại: {len(cat_cols)}")

    # --- Thống kê mô tả (CSV) ---
    train[num_cols].describe().T.to_csv(os.path.join(out_dir, "eda_describe_numeric.csv"))

    # --- Phân phối SalePrice trước/sau log ---
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(train["SalePrice"], kde=True, ax=ax[0])
    ax[0].set_title(f"SalePrice gốc (skew = {train['SalePrice'].skew():.2f})")
    sns.histplot(np.log1p(train["SalePrice"]), kde=True, ax=ax[1], color="orange")
    ax[1].set_title(f"log1p(SalePrice) (skew = {np.log1p(train['SalePrice']).skew():.2f})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "eda_saleprice_dist.png"), dpi=120)
    plt.close()
    print("-> SalePrice lệch phải, log1p làm phân phối gần chuẩn hơn => dự đoán log(SalePrice).")

    # --- Giá trị thiếu ---
    missing = train.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    miss_df = pd.DataFrame({"missing_count": missing,
                            "missing_pct": (missing / len(train) * 100).round(2)})
    miss_df.index.name = "column"
    miss_df.to_csv(os.path.join(out_dir, "eda_missing_values.csv"))
    if len(missing) > 0:
        plt.figure(figsize=(9, 5))
        miss_df["missing_pct"].plot(kind="bar", color="tomato")
        plt.ylabel("% thiếu")
        plt.title("Tỉ lệ giá trị thiếu theo cột (train)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "eda_missing_values.png"), dpi=120)
        plt.close()
        print(f"-> Có {len(missing)} cột bị thiếu. Top 5:\n{missing.head(5)}")

    # --- Tương quan với SalePrice ---
    corr = train[num_cols].corr()["SalePrice"].drop("SalePrice").sort_values(ascending=False)
    corr.rename("corr_with_SalePrice").to_csv(os.path.join(out_dir, "eda_correlation_with_saleprice.csv"))
    print("\nTop 10 biến số tương quan mạnh nhất với SalePrice:")
    print(corr.head(10).round(3))
    top = corr.abs().sort_values(ascending=False).head(10).index.tolist() + ["SalePrice"]
    plt.figure(figsize=(9, 7))
    sns.heatmap(train[top].corr(), annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Ma trận tương quan - Top 10 biến + SalePrice")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "eda_correlation_top10.png"), dpi=120)
    plt.close()

    # --- Ngoại lệ ---
    out_mask = train["GrLivArea"] > 4000
    train.loc[out_mask, ["Id", "GrLivArea", "SalePrice", "SaleCondition"]].to_csv(
        os.path.join(out_dir, "eda_outliers.csv"), index=False)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].scatter(train["GrLivArea"], train["SalePrice"], s=10, alpha=0.6)
    ax[0].scatter(train.loc[out_mask, "GrLivArea"], train.loc[out_mask, "SalePrice"],
                  color="red", s=40, label="GrLivArea > 4000")
    ax[0].set_xlabel("GrLivArea")
    ax[0].set_ylabel("SalePrice")
    ax[0].legend()
    ax[0].set_title("Ngoại lệ: GrLivArea vs SalePrice")
    sns.boxplot(x="OverallQual", y="SalePrice", data=train, ax=ax[1])
    ax[1].set_title("SalePrice theo OverallQual")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "eda_outliers.png"), dpi=120)
    plt.close()
    print(f"\n-> Có {out_mask.sum()} nhà GrLivArea > 4000 (bài báo De Cock khuyên loại bỏ).")
