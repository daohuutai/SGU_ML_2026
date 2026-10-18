"""Tiền xử lý (missing values, outlier, encoding) + Feature Engineering."""
import numpy as np
import pandas as pd
from scipy.stats import skew

# Các cột mà NaN thật ra nghĩa là "không có" (No basement, No garage, ...) theo data_description.txt
NA_MEANS_NONE = [
    "Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "PoolQC", "Fence", "MiscFeature", "MasVnrType",
]

# Các cột chất lượng có thứ tự -> mã hoá thành số (Ordinal/Label Encoding)
QUAL_MAP = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
ORDINAL_QUAL_COLS = [
    "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC", "KitchenQual",
    "FireplaceQu", "GarageQual", "GarageCond", "PoolQC",
]
ORDINAL_OTHER = {
    "BsmtExposure": {"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4},
    "BsmtFinType1": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "BsmtFinType2": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "GarageFinish": {"None": 0, "Unf": 1, "RFn": 2, "Fin": 3},
    "LandSlope": {"Sev": 1, "Mod": 2, "Gtl": 3},
    "LotShape": {"IR3": 1, "IR2": 2, "IR1": 3, "Reg": 4},
    "PavedDrive": {"N": 0, "P": 1, "Y": 2},
    "CentralAir": {"N": 0, "Y": 1},
}


def fill_missing(df):
    """Xử lý giá trị thiếu."""
    # (a) NaN nghĩa là "không có" -> "None"
    for col in NA_MEANS_NONE:
        if col in df.columns:
            df[col] = df[col].fillna("None")

    # (b) Cột số liên quan basement/garage/veneer: không có thì = 0
    zero_cols = ["MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
                 "BsmtFullBath", "BsmtHalfBath", "GarageCars", "GarageArea"]
    for col in zero_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # (c) GarageYrBlt: không có garage -> dùng năm xây nhà
    df["GarageYrBlt"] = df["GarageYrBlt"].fillna(df["YearBuilt"])

    # (d) LotFrontage: median theo Neighborhood (nhà cùng khu thường có mặt tiền tương tự)
    df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(lambda s: s.fillna(s.median()))
    df["LotFrontage"] = df["LotFrontage"].fillna(df["LotFrontage"].median())

    # (e) Cột phân loại còn thiếu ít (MSZoning, Electrical, ...) -> mode
    for col in df.select_dtypes(include=["object"]).columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode()[0])
    # (f) Cột số còn thiếu -> median
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    return df


def add_features(df):
    """Feature Engineering: tạo đặc trưng mới."""
    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]       # tổng diện tích
    df["HouseAge"] = (df["YrSold"] - df["YearBuilt"]).clip(lower=0)          # tuổi nhà lúc bán
    df["RemodAge"] = (df["YrSold"] - df["YearRemodAdd"]).clip(lower=0)       # số năm từ lần sửa gần nhất
    df["IsRemodeled"] = (df["YearRemodAdd"] != df["YearBuilt"]).astype(int)
    df["TotalBath"] = (df["FullBath"] + 0.5 * df["HalfBath"]
                       + df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"])
    df["TotalPorchSF"] = (df["OpenPorchSF"] + df["EnclosedPorch"]
                          + df["3SsnPorch"] + df["ScreenPorch"] + df["WoodDeckSF"])
    df["HasPool"] = (df["PoolArea"] > 0).astype(int)
    df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
    df["HasBsmt"] = (df["TotalBsmtSF"] > 0).astype(int)
    df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)
    df["OverallScore"] = df["OverallQual"] * df["OverallCond"]
    print("Đã tạo thêm: TotalSF, HouseAge, RemodAge, IsRemodeled, TotalBath, TotalPorchSF, "
          "HasPool/Garage/Bsmt/Fireplace, OverallScore")
    return df


def encode(df):
    """Encoding: ordinal -> số, log1p cho biến lệch, One-Hot cho biến phân loại."""
    # (a) Ordinal encoding cho các cột chất lượng có thứ tự
    for col in ORDINAL_QUAL_COLS:
        df[col] = df[col].map(QUAL_MAP).fillna(0).astype(int)
    for col, mapping in ORDINAL_OTHER.items():
        df[col] = df[col].map(mapping).fillna(0).astype(int)

    # (b) Các cột số thật chất là phân loại -> chuỗi để One-Hot
    for col in ["MSSubClass", "MoSold", "YrSold"]:
        df[col] = df[col].astype(str)

    # (c) Giảm độ lệch bằng log1p (bỏ qua cột nhị phân 0/1)
    num_cols = df.select_dtypes(include=[np.number]).columns
    skewness = df[num_cols].apply(lambda s: skew(s.dropna()))
    skewed = [c for c in skewness[abs(skewness) > 0.75].index if df[c].nunique() > 2]
    df[skewed] = np.log1p(df[skewed])
    print(f"Log1p cho {len(skewed)} biến số bị lệch (|skew| > 0.75)")

    # (d) One-Hot Encoding
    df = pd.get_dummies(df, drop_first=True, dtype=int)
    print(f"Sau One-Hot: {df.shape[1]} đặc trưng")
    return df


def preprocess(train, test, remove_outliers=True):
    """
    Trả về: X_train, y_train (đã log1p), X_test, test_ids.
    Gộp train + test để xử lý đồng nhất (các bước này không dùng SalePrice nên không bị rò rỉ nhãn).
    """
    print("\n===== TIỀN XỬ LÝ =====")
    train = train.copy()
    test = test.copy()

    # Loại ngoại lệ (theo bài báo De Cock: nhà > 4000 sqft)
    if remove_outliers:
        n_before = len(train)
        train = train[train["GrLivArea"] < 4000].reset_index(drop=True)
        print(f"Loại {n_before - len(train)} ngoại lệ (GrLivArea >= 4000). Còn {len(train)} dòng.")

    # Log-transform biến mục tiêu
    y = np.log1p(train["SalePrice"])
    train = train.drop(columns=["SalePrice"])

    test_ids = test["Id"].copy()
    n_train = len(train)
    df = pd.concat([train, test], axis=0, ignore_index=True).drop(columns=["Id"])

    df = fill_missing(df)
    df = add_features(df)
    df = encode(df)

    X_train = df.iloc[:n_train].reset_index(drop=True)
    X_test = df.iloc[n_train:].reset_index(drop=True)
    return X_train, y.reset_index(drop=True), X_test, test_ids
