# 盤子清潔度分類：Cleaned vs Dirty V2

Kaggle 競賽 — 判斷盤子是否已清洗乾淨的二元影像分類任務。

## 競賽簡介

**平台：** [Kaggle](https://kaggle.com/competitions/platesv2)  
**評估指標：** 準確率（Accuracy）

### 問題描述

這是一個少樣本學習（Few-shot Learning）競賽。訓練集僅提供 20 張乾淨盤子與 20 張髒盤子的圖片，測試集則有數百張圖片需分類。目標是訓練一個分類器，判斷每張圖片中的盤子是否已清洗。

- **類別：** `cleaned`（已清洗）、`dirty`（未清洗）
- **訓練樣本：** 40 張（各類別 20 張）
- **挑戰：** 少樣本、需善用資料增強與遷移學習

## 解題方法

### 模型架構

使用 **YOLOv8m-cls**（YOLOv8 中型分類模型）進行遷移學習，充分利用預訓練權重以應對少樣本問題。

### 訓練設定

| 參數 | 設定值 |
|---|---|
| 模型 | `yolov8m-cls.pt` |
| Epochs | 50 |
| Batch Size | 16 |
| 圖片尺寸 | 320 × 320 |
| 資料增強 | 啟用（`augment=True`） |
| 驗證集 | 自動從訓練集切分 |

### 推論流程

1. 對測試集所有圖片執行預測
2. 取最高機率對應的類別（`top1`）
3. 輸出 `submission.csv`

## 檔案說明

| 檔案 | 說明 |
|---|---|
| `code.py` | 完整流程：解壓資料、建立 YOLO 資料結構、訓練與推論 |
| `platesv2.zip` | 原始資料集（訓練集 + 測試集） |
| `submission.csv` | 提交至 Kaggle 的最終預測結果 |

## 相依套件

```
ultralytics
torch
torchvision
pandas
numpy
Pillow
tqdm
```

## 執行方式

本程式碼設計於 **Google Colab** 環境中執行：

1. 上傳 `platesv2.zip` 至 Colab
2. 依序執行各步驟（STEP 0 ～ STEP 7）
3. 訓練完成後自動下載 `submission.csv`

## 引用

Igor.Slinko. *Cleaned vs Dirty V2*. https://kaggle.com/competitions/platesv2, 2019. Kaggle.
