# ============================================================
# ✅ STEP 0. 上傳 platesv2.zip
# ============================================================
from google.colab import files
uploaded = files.upload()  # 選擇 platesv2.zip 上傳

# 安裝 YOLOv8
!pip install -q ultralytics
# ============================================================
# ✅ STEP 1. 匯入套件
# ============================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pandas as pd
import numpy as np
import glob, os
from PIL import Image
import shutil

# ============================================================
# ✅ STEP 2. 解壓 platesv2.zip
# ============================================================
import zipfile, os

# 外層 ZIP
outer_zip_path = "/content/platesv2.zip"
extract_outer_path = "/content/platesv2"

# ✅ 解壓外層 zip
with zipfile.ZipFile(outer_zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_outer_path)

print("✅ 外層 ZIP 解壓完成")

# 若裡面還有 plates.zip，再執行這段
inner_zip_path = "/content/platesv2/plates.zip"
if os.path.exists(inner_zip_path):
    extract_inner_path = "/content/platesv2/plates"
    with zipfile.ZipFile(inner_zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_inner_path)
    print("✅ 內層 ZIP 解壓完成")
else:
    print("⚠️ 未找到內層 plates.zip（可能資料已直接展開）")

# 檢查最終內容
for root, dirs, files in os.walk("/content/platesv2"):
    print("📁", root)
    print("   資料夾:", dirs)
    print("   檔案:", files)

# ============================================================
# ✅ STEP 3. 建立資料集結構 (修正版)
# ============================================================
dataset_path = "/content/datasets/platesv2"
os.makedirs(dataset_path, exist_ok=True)

# 複製 train/test，排除 __MACOSX 和 .DS_Store
train_src = "/content/platesv2/plates/plates/train"
test_src  = "/content/platesv2/plates/plates/test"

shutil.copytree(train_src, f"{dataset_path}/train", dirs_exist_ok=True)
shutil.copytree(test_src,  f"{dataset_path}/test", dirs_exist_ok=True)

# 建立 data.yaml
data_yaml = f"""
path: {dataset_path}
train: train
val: train

names:
  0: cleaned
  1: dirty
"""

with open(f"{dataset_path}/data.yaml", "w") as f:
    f.write(data_yaml)

print("✅ YOLO 資料結構建立完成")

# ============================================================
# ✅ STEP 4. 分類 訓練
# ============================================================
from ultralytics import YOLO

train_dir = "/content/datasets/platesv2/train"  # train 資料夾

model = YOLO("yolov8m-cls.pt")  # 分類模型

model.train(
    data=train_dir,       # 訓練資料夾
    epochs=50,            # 訓練輪數
    batch=16,             # batch size
    imgsz=320,            # 圖片尺寸
    name="plates_cls_yolov8_opt",
    project="/content",
    augment=True,         # 資料增強
    val=True,             # 自動切驗證集
    save=True             # 自動儲存最佳模型
)

print("✅ 訓練完成")

# ============================================================
# ✅ STEP 5. 在測試集上推論
# ============================================================
test_dir = f"/content/datasets/platesv2/test"
results = model.predict(source=test_dir)

submission = []
for idx, r in enumerate(results):
    label_id = int(r.probs.top1)                  # 最高機率類別索引
    label_name = "cleaned" if label_id == 0 else "dirty"
    submission.append([idx, label_name])

submission_df = pd.DataFrame(submission, columns=["id", "label"])
submission_df.to_csv("/content/submission.csv", index=False)
print("✅ submission.csv 已生成！")

# ============================================================
# ✅ STEP 6. 生成 submission.csv 
# ============================================================
submission = []

for idx, r in enumerate(results):
    label_id = int(r.probs.top1)                  # 取最高機率類別索引
    label_name = "cleaned" if label_id == 0 else "dirty"
    submission.append([idx, label_name])

submission_df = pd.DataFrame(submission, columns=["id", "label"])
submission_df.to_csv("/content/submission.csv", index=False)

print("✅ submission.csv 已生成！")
print(submission_df.head())

# ============================================================
# ✅ STEP 7. 下載
# ============================================================
from google.colab import files

# 下載到本機
files.download("/content/submission.csv")