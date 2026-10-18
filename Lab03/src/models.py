"""Huấn luyện, đánh giá (CV + train/val) và vẽ biểu đồ kết quả mô hình."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

from config import SEED


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def get_models(fast=False):
    """Các mô hình từ cơ bản -> nâng cao. Mô hình tuyến tính cần scale nên dùng pipeline + RobustScaler."""
    n_trees = 60 if fast else 300
    return {
        "LinearRegression": make_pipeline(RobustScaler(), LinearRegression()),
        "Ridge": make_pipeline(RobustScaler(), Ridge(alpha=15)),
        "Lasso": make_pipeline(RobustScaler(), Lasso(alpha=0.0005, max_iter=50000)),
        "RandomForest": RandomForestRegressor(n_estimators=n_trees, min_samples_leaf=2,
                                              n_jobs=-1, random_state=SEED),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=n_trees, learning_rate=0.05,
                                                      max_depth=3, subsample=0.8, random_state=SEED),
    }


def evaluate_models(X, y, out_dir, fast=False):
    print("\n===== HUẤN LUYỆN & ĐÁNH GIÁ =====")
    models = get_models(fast)
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)

    rows = []
    for name, model in models.items():
        cv_scores = -cross_val_score(model, X, y, cv=kf, scoring="neg_root_mean_squared_error")
        model.fit(X_tr, y_tr)
        pred_val = model.predict(X_val)
        rows.append({"Model": name, "CV_RMSE_mean": cv_scores.mean(), "CV_RMSE_std": cv_scores.std(),
                     "Val_RMSE": rmse(y_val, pred_val), "Val_R2": r2_score(y_val, pred_val)})
        print(f"{name:18s} | CV RMSE(log) = {cv_scores.mean():.4f} ± {cv_scores.std():.4f} "
              f"| Val RMSE = {rows[-1]['Val_RMSE']:.4f} | Val R² = {rows[-1]['Val_R2']:.4f}")

    # Blend: trung bình dự đoán out-of-fold của Ridge + Lasso + GradientBoosting
    blend_names = ["Ridge", "Lasso", "GradientBoosting"]
    oof = np.column_stack([cross_val_predict(models[n], X, y, cv=kf) for n in blend_names])
    blend_oof = oof.mean(axis=1)
    rows.append({"Model": "Blend(Ridge+Lasso+GB)", "CV_RMSE_mean": rmse(y, blend_oof), "CV_RMSE_std": np.nan,
                 "Val_RMSE": np.nan, "Val_R2": r2_score(y, blend_oof)})
    print(f"{'Blend':18s} | CV RMSE(log) = {rows[-1]['CV_RMSE_mean']:.4f} "
          f"| R² (out-of-fold) = {rows[-1]['Val_R2']:.4f}")

    results = pd.DataFrame(rows).sort_values("CV_RMSE_mean").reset_index(drop=True)
    results.to_csv(os.path.join(out_dir, "model_results.csv"), index=False)
    print("\nBảng kết quả (sắp theo CV RMSE):")
    print(results.round(4).to_string(index=False))

    plt.figure(figsize=(9, 4))
    sns.barplot(data=results, x="CV_RMSE_mean", y="Model", color="steelblue")
    plt.xlabel("CV RMSE (log SalePrice) - càng thấp càng tốt")
    plt.title("So sánh các mô hình")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "model_comparison.png"), dpi=120)
    plt.close()

    return models, results, blend_names, blend_oof


def save_final_outputs(models, X, y, blend_oof, out_dir):
    """Lưu dự đoán out-of-fold (CSV), biểu đồ pred vs actual, và độ quan trọng đặc trưng (CSV + PNG)."""
    # Dự đoán out-of-fold (CSV)
    pd.DataFrame({"actual_log": y, "pred_log": blend_oof,
                  "actual_price": np.expm1(y), "pred_price": np.expm1(blend_oof)}
                 ).to_csv(os.path.join(out_dir, "cv_predictions.csv"), index=False)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].scatter(y, blend_oof, s=10, alpha=0.5)
    lims = [min(y.min(), blend_oof.min()), max(y.max(), blend_oof.max())]
    ax[0].plot(lims, lims, "r--")
    ax[0].set_xlabel("log1p(SalePrice) thực tế")
    ax[0].set_ylabel("Dự đoán (out-of-fold)")
    ax[0].set_title("Dự đoán vs Thực tế (Blend)")
    ax[1].scatter(blend_oof, y - blend_oof, s=10, alpha=0.5)
    ax[1].axhline(0, color="r", linestyle="--")
    ax[1].set_xlabel("Dự đoán")
    ax[1].set_ylabel("Residual")
    ax[1].set_title("Residual plot")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pred_vs_actual.png"), dpi=120)
    plt.close()

    # Đặc trưng quan trọng (Gradient Boosting)
    gb = models["GradientBoosting"]
    gb.fit(X, y)
    imp_all = pd.Series(gb.feature_importances_, index=X.columns).sort_values(ascending=False)
    imp_all.rename("importance").to_csv(os.path.join(out_dir, "feature_importance.csv"), index_label="feature")
    imp = imp_all.head(15)
    plt.figure(figsize=(8, 6))
    imp[::-1].plot(kind="barh", color="seagreen")
    plt.title("Top 15 đặc trưng quan trọng (Gradient Boosting)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "feature_importance.png"), dpi=120)
    plt.close()
    print("\nTop 10 đặc trưng quan trọng:")
    print(imp.head(10).round(4))


def fit_and_predict(models, results, blend_names, X, y, X_test):
    """Huấn luyện lại trên toàn bộ train rồi dự đoán test. Trả về giá nhà (đô la)."""
    best_name = results.loc[0, "Model"]
    print(f"\nMô hình tốt nhất theo CV: {best_name}")
    if best_name.startswith("Blend"):
        preds = []
        for n in blend_names:
            models[n].fit(X, y)
            preds.append(models[n].predict(X_test))
        pred_log = np.mean(preds, axis=0)
    else:
        models[best_name].fit(X, y)
        pred_log = models[best_name].predict(X_test)
    return np.expm1(pred_log)  # đổi từ log về giá thật
