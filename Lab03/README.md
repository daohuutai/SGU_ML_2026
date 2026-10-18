# Lab 03 – Dự đoán giá nhà Ames (House Prices)

**Lớp:** DCT124C3

## Thành viên nhóm

| STT | MSSV | Họ và Tên | Vai trò |
|:---:|:----------:|----------------------|--------------|
| 1 | 3124411038 | Trần Lê Gia Bảo | **Nhóm trưởng** |
| 2 | 3124411337 | Nguyễn Thanh Tuấn | Thành viên |
| 3 | 3124411261 | Đào Hữu Tài | Thành viên |
| 4 | 3124411245 | Huỳnh Nhật Quang | Thành viên |
| 5 | 3124411198 | Tô Thành Nhân | Thành viên |

---

## 1. Giới thiệu

Bài lab giải bài toán **hồi quy (regression)**: dự đoán giá bán nhà (`SalePrice`) ở thành phố Ames, bang Iowa (Mỹ) dựa trên **79 đặc trưng** mô tả ngôi nhà (diện tích, chất lượng, năm xây, garage, tầng hầm, khu vực, ...). Dữ liệu lấy từ cuộc thi Kaggle *House Prices – Advanced Regression Techniques*.

Pipeline gồm các bước:

1. **EDA** – khám phá dữ liệu, vẽ biểu đồ, thống kê.
2. **Tiền xử lý** – xử lý giá trị thiếu, loại ngoại lệ, mã hoá biến phân loại, giảm độ lệch.
3. **Feature Engineering** – tạo đặc trưng mới.
4. **Huấn luyện & đánh giá** nhiều mô hình (Cross-Validation + Train/Validation).
5. **Blend** các mô hình tốt nhất và **xuất `submission.csv`**.

---

## 2. Cấu trúc thư mục

```
Lab03/
├── data/                  # Dữ liệu thật (Kaggle)
│   ├── train.csv              (1460 dòng, có SalePrice)
│   ├── test.csv               (1459 dòng, không có SalePrice)
│   ├── sample_submission.csv  (mẫu định dạng nộp bài)
│   └── data_description.txt   (mô tả ý nghĩa từng cột)
├── data_fake/             # Dữ liệu giả (tự sinh, nhỏ) để chạy thử nhanh
├── outputs/               # Kết quả khi chạy trên dữ liệu thật
├── outputs_fake/          # Kết quả khi chạy trên dữ liệu giả
├── src/                   # Mã nguồn
│   ├── config.py              # Đường dẫn + hằng số (SEED = 42)
│   ├── data_utils.py          # Đọc dữ liệu + tạo dữ liệu giả
│   ├── eda.py                 # Phân tích khám phá dữ liệu (EDA)
│   ├── preprocess.py          # Tiền xử lý + Feature Engineering
│   ├── models.py              # Huấn luyện, đánh giá, vẽ biểu đồ mô hình
│   └── main.py                # File chạy chính
├── tutorial/
│   └── house-prices-eda-to-ml-beginner.ipynb   # Notebook hướng dẫn tham khảo
└── README.md
```

---

## 3. Cài đặt & chạy

### Yêu cầu
- Python 3.8+
- Thư viện: `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn
```

### Chạy chương trình
Chạy **từ thư mục gốc `Lab03/`**:

```bash
python src/main.py              # chạy cả dữ liệu giả + dữ liệu thật (mặc định)
python src/main.py --mode fake  # chỉ chạy dữ liệu giả (test nhanh)
python src/main.py --mode real  # chỉ chạy dữ liệu thật
```

- Chế độ `fake` tự tạo lại `data_fake/` (300 dòng train, 100 dòng test) và chạy với số cây ít hơn (60) để nhanh.
- Chế độ `real` chạy đầy đủ (300 cây cho RandomForest / GradientBoosting).

---

## 4. Mô tả chi tiết các bước

### 4.1 EDA (`src/eda.py`)
- Thống kê mô tả các cột số → `eda_describe_numeric.csv`
- Phân phối `SalePrice` trước/sau `log1p` → `eda_saleprice_dist.png`. `SalePrice` lệch phải nên **dự đoán `log1p(SalePrice)`**.
- Giá trị thiếu theo cột → `eda_missing_values.csv/.png` (nhiều nhất: `PoolQC` 99.5%, `MiscFeature` 96.3%, `Alley` 93.8%, `Fence` 80.8%).
- Tương quan với `SalePrice` → `eda_correlation_with_saleprice.csv`, `eda_correlation_top10.png`. Mạnh nhất: `OverallQual` (0.79), `GrLivArea` (0.71), `GarageCars` (0.64), `GarageArea` (0.62), `TotalBsmtSF` (0.61).
- Ngoại lệ → `eda_outliers.csv/.png`: có **4 nhà** `GrLivArea > 4000` (khuyến nghị loại bỏ theo bài báo của De Cock).

### 4.2 Tiền xử lý (`src/preprocess.py`)
| Bước | Nội dung |
|------|----------|
| Loại ngoại lệ | Bỏ các nhà có `GrLivArea >= 4000` |
| Biến mục tiêu | `y = log1p(SalePrice)` |
| Gộp train + test | Xử lý đồng nhất (các bước không dùng `SalePrice` nên không rò rỉ nhãn) |
| Giá trị thiếu | `NaN` nghĩa là "không có" (garage, basement, hồ bơi, ...) → `"None"`; cột số liên quan → `0`; `GarageYrBlt` → `YearBuilt`; `LotFrontage` → median theo `Neighborhood`; còn lại → mode (phân loại) / median (số) |
| Ordinal encoding | Các cột chất lượng có thứ tự (`Po < Fa < TA < Gd < Ex`, `BsmtExposure`, `GarageFinish`, `LotShape`, ...) → số nguyên |
| Biến số thực chất là phân loại | `MSSubClass`, `MoSold`, `YrSold` → chuỗi để One-Hot |
| Giảm độ lệch | `log1p` cho các biến số có `|skew| > 0.75` (bỏ qua cột nhị phân) |
| One-Hot Encoding | `pd.get_dummies(drop_first=True)` cho biến phân loại còn lại |

### 4.3 Feature Engineering
| Đặc trưng mới | Ý nghĩa |
|---------------|---------|
| `TotalSF` | Tổng diện tích = tầng hầm + tầng 1 + tầng 2 |
| `HouseAge` | Tuổi nhà lúc bán (`YrSold − YearBuilt`) |
| `RemodAge` | Số năm từ lần sửa chữa gần nhất |
| `IsRemodeled` | Nhà có được sửa chữa hay không |
| `TotalBath` | Tổng số phòng tắm (phòng nửa tính 0.5) |
| `TotalPorchSF` | Tổng diện tích hiên / sàn gỗ |
| `HasPool`, `HasGarage`, `HasBsmt`, `HasFireplace` | Có hồ bơi / garage / tầng hầm / lò sưởi |
| `OverallScore` | `OverallQual × OverallCond` |

### 4.4 Mô hình (`src/models.py`)
- **Mô hình:** `LinearRegression`, `Ridge (alpha=15)`, `Lasso (alpha=0.0005)` (3 mô hình tuyến tính dùng `RobustScaler`), `RandomForest`, `GradientBoosting`.
- **Đánh giá:** 5-fold Cross-Validation (RMSE trên log) + tách Train/Validation 80/20 (RMSE, R²).
- **Blend:** trung bình dự đoán out-of-fold của **Ridge + Lasso + GradientBoosting**.
- **Dự đoán cuối:** chọn mô hình có CV RMSE thấp nhất, huấn luyện lại trên toàn bộ train, dự đoán test rồi đổi về giá thật bằng `expm1`.

---

##  5. Kết quả (dữ liệu thật)

Sắp xếp theo CV RMSE (trên log, **càng thấp càng tốt**):

| Mô hình | CV RMSE (mean) | CV RMSE (std) | Val RMSE | Val R² |
|---------|:---:|:---:|:---:|:---:|
| **Blend (Ridge + Lasso + GB)** | **0.1113** | – | – | 0.9209* |
| Lasso | 0.1133 | 0.0082 | 0.1212 | 0.9070 |
| Ridge | 0.1134 | 0.0086 | 0.1210 | 0.9074 |
| GradientBoosting | 0.1205 | 0.0083 | 0.1240 | 0.9026 |
| LinearRegression | 0.1249 | 0.0048 | 0.1235 | 0.9034 |
| RandomForest | 0.1330 | 0.0103 | 0.1424 | 0.8717 |

\* R² của Blend tính trên dự đoán out-of-fold toàn bộ tập train.

 **Blend** cho kết quả tốt nhất và được dùng để tạo `submission.csv`.

**Top đặc trưng quan trọng (Gradient Boosting):** `TotalSF` (0.362), `OverallQual` (0.300), `OverallScore` (0.055), `TotalBath` (0.043), `GarageCars` (0.019). Hai đặc trưng đầu chiếm hơn 60% độ quan trọng – trong đó `TotalSF` là đặc trưng do nhóm tự tạo.

---

## 6. Các file kết quả trong `outputs/`

| File | Nội dung |
|------|----------|
| `eda_describe_numeric.csv` | Thống kê mô tả các biến số |
| `eda_missing_values.csv / .png` | Tỉ lệ giá trị thiếu |
| `eda_correlation_with_saleprice.csv`, `eda_correlation_top10.png` | Tương quan với giá nhà |
| `eda_saleprice_dist.png` | Phân phối `SalePrice` gốc và sau log |
| `eda_outliers.csv / .png` | Các điểm ngoại lệ và boxplot theo `OverallQual` |
| `model_results.csv`, `model_comparison.png` | So sánh các mô hình |
| `cv_predictions.csv` | Dự đoán out-of-fold (thực tế vs dự đoán) |
| `pred_vs_actual.png` | Biểu đồ dự đoán vs thực tế + residual plot |
| `feature_importance.csv / .png` | Độ quan trọng đặc trưng |
| `submission.csv` | **File nộp bài** (cột `Id`, `SalePrice`) |

> Thư mục `outputs_fake/` có cấu trúc tương tự nhưng chạy trên dữ liệu giả, **chỉ dùng để kiểm tra pipeline**, kết quả không có ý nghĩa thực tế.

---

## 7. Ghi chú

- `SEED = 42` được dùng xuyên suốt để kết quả có thể tái lập.
- Dữ liệu giả (`data_fake/`) có **cùng tên cột** với dữ liệu thật, giá trị được bốc ngẫu nhiên từ cột thật; `SalePrice` giả sinh bằng công thức đơn giản từ chất lượng, diện tích, năm xây cộng nhiễu.
- Notebook trong `tutorial/` là tài liệu tham khảo (EDA → ML cho người mới bắt đầu).
