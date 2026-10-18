"""Cấu hình chung: đường dẫn thư mục + hằng số."""
import os

SEED = 42

# Thư mục gốc Lab03/ (cha của src/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")                # dữ liệu thật
DATA_FAKE_DIR = os.path.join(BASE_DIR, "data_fake")      # dữ liệu giả
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")           # kết quả chạy dữ liệu thật
OUTPUT_FAKE_DIR = os.path.join(BASE_DIR, "outputs_fake") # kết quả chạy dữ liệu giả
