"""
Lab 03 - Dự đoán giá nhà Ames. File chạy chính.

Chạy từ thư mục gốc Lab03/:
    python src/main.py              # chạy cả dữ liệu giả + dữ liệu thật
    python src/main.py --mode fake  # chỉ chạy dữ liệu giả (test nhanh)
    python src/main.py --mode real  # chỉ chạy dữ liệu thật
"""
import argparse
import os

import pandas as pd

from config import DATA_DIR, DATA_FAKE_DIR, OUTPUT_DIR, OUTPUT_FAKE_DIR
from data_utils import load_data, make_fake_data
from eda import run_eda
from models import evaluate_models, fit_and_predict, save_final_outputs
from preprocess import preprocess


def run_pipeline(data_dir, out_dir, fast=False, tag="REAL"):
    print("\n" + "=" * 70)
    print(f"CHẠY PIPELINE [{tag}]  data: {data_dir}  ->  outputs: {out_dir}")
    print("=" * 70)
    os.makedirs(out_dir, exist_ok=True)

    train, test, sample_sub = load_data(data_dir)
    run_eda(train, out_dir)
    X, y, X_test, test_ids = preprocess(train, test)

    models, results, blend_names, blend_oof = evaluate_models(X, y, out_dir, fast=fast)
    save_final_outputs(models, X, y, blend_oof, out_dir)
    pred_price = fit_and_predict(models, results, blend_names, X, y, X_test)

    # Xuất submission.csv đúng thứ tự Id + format sample_submission
    submission = pd.DataFrame({"Id": test_ids, "SalePrice": pred_price})
    submission = sample_sub[["Id"]].merge(submission, on="Id", how="left")
    assert submission["SalePrice"].notna().all(), "Có Id bị thiếu dự đoán!"
    assert list(submission.columns) == list(sample_sub.columns), "Sai format cột so với sample_submission"
    submission.to_csv(os.path.join(out_dir, "submission.csv"), index=False)
    print(f"\nĐã lưu submission.csv ({len(submission)} dòng) vào {out_dir}")
    print(submission.head())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fake", "real", "both"], default="both")
    args = parser.parse_args()

    if args.mode in ("fake", "both"):
        make_fake_data(DATA_DIR, DATA_FAKE_DIR)           # tự tạo data_fake/
        run_pipeline(DATA_FAKE_DIR, OUTPUT_FAKE_DIR, fast=True, tag="FAKE")
    if args.mode in ("real", "both"):
        run_pipeline(DATA_DIR, OUTPUT_DIR, fast=False, tag="REAL")


if __name__ == "__main__":
    main()
