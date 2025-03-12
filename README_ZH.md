# 🎙️ SVM 語音端點偵測 (EPD)

## 📁 資料夾結構
```
hw03
├── data                        # 📚 資料集
│   ├── dataset_cache.pt        # 🚀 資料集前處理後快取
│   ├── wavefiles-2008
│   ├── wavefiles-ex
│   └── wavefiles-all
├── models                      # 💻 模型輸出資料夾
│   └── 20250309_151313         # 📅 訓練時間戳
│       ├── svm_model.pth       # 📦 模型檔案
│       ├── test.log
│       └── train.log
└── src                         # 🧩 程式碼
    ├── config.py
    ├── dataset.py
    ├── epd.py
    ├── model.py
    ├── test.py
    └── train.py
```

## 🛠️ 環境

> python `3.11.5`, cuda `12.4`
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## 🧷 使用方式

```bash
# 訓練
python src/train.py

# 測試 Frame 分數
python src/test.py --model <模型路徑>

# 測試 EPD 分數
python src/epd.py --model <模型路徑> --score true --dataset <音檔資料夾>

# 視覺化
python src/epd.py --model <模型路徑>
```

## ⚙️ 前處理

檔案命名規則：`<字母>_<起始 sample>_<結束 sample>.wav`，並透過 `wavefiles-all` 以 8:2 比例切分測試與訓練集。

### ✂️ 音框 (Frame) 切割
由於 SVM 需要固定大小的輸入特徵，需先將音檔切割成固定大小的 Frame。

```
# 2008 dataset
frame_size = 400 samples
hop_size = 80 samples

# ex dataset
frame_size = 256 samples
hop_size = 128 samples
```

### 📐 維度調整
透過 `torchaudio.load()` 讀取後，維度為 `(1, num_samples)`。
使用 `unsqueeze(0)` 擴增維度至 `(1, 1, frame_size)` 以符合 `torchaudio.transforms.MFCC` 的需求。

### 🎚️ MFCC 特徵轉換
將 `n_mfcc` 設定為 `13`，與模型輸入維度同步，輸出維度為 `(1, n_mfcc, num_frames)`。

### 🔍 最終維度轉換
移除 `batch` 維度後，最終維度為 `(n_mfcc, num_frames)`。

## 🧠 SVM 模型

SVM 只接受 `2D` 特徵，因此將 `(n_mfcc, num_frames)` 展開為 `mean(n_mfcc) `，並且與 `label` 組合成 `(mfcc, label)`。

透過 `nn.Linear` 模擬：

$$
f(x) = Wx + b
$$

## 📉 損失函數 (Hinge Loss)

$$
L = max(0, 1−y⋅f(x))
$$

分類正確時 (同號)，Loss 為 0；分類錯誤時 (不同號)，Loss 增大，引導模型學習邊界。

## 🚀 訓練方式

### 🎲 Stochastic Gradient Decent (隨機梯度下降)
使用 PyTorch 內建的 `SGD` 訓練，更新參數 `W` 和 `b`。

### 📉 Reduce Learning Rate (學習率下降)
透過 `torch.optim.lr_scheduler.ReduceLROnPlateau` 降低學習率，進一步降低 loss，如果 loss 在連續 3 個 epoch 內沒有改善，學習率將被降低為原來的 0.5 倍。

### 🚧 Early Stopping (提前停止)
當連續 5 個 epoch 都沒有改善，則提前停止訓練。

## 📊 輸出
```
model_timestamp
├── best_svm_model.pth  # 📦 模型檔案
├── config.yaml         # ⚙️ 設定檔
├── epd_score.csv       # 📊 偵測音檔的評估分數
├── test.log            # 📝 依照測試集的評估分數
└── train.log           # 📝 訓練紀錄
```

## 🎯 驗證

### 🎵 測試 Frame Label
以測試 frame 的 mfcc 與 label 進行預測，並計算準確率。

✨ [最佳模型](./models/best_model_ex/test.log)準確率達 `95.12%`。

### 🎼 測試 End-point Sample
以測試音檔的起始 sample 與結束 sample 進行預測，並計算準確率。

✨ [最佳模型](./models/best_model_ex/epd_score.csv)準確率達 `90.29%`。

#### wavfiles-ex 測試結果
```csv
Speaker,Correct,Total,EPD Score
123456_Zhehui,70.5,72,97.91666666666666
654321_Baiway,131.0,144,90.97222222222221
921510_Roger,163.0,180,90.55555555555556
921588_Leon,206.0,252,81.74603174603175
Average,0,0,90.29761904761904
```

## 📈 視覺化
透過 Gradio + Plotly 實現互動視覺化，可即時上傳音訊檔案進行分析、標記音段並播放。

![視覺化展示](./asset/image1.png)
