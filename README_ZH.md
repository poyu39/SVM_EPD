# 🎙️ SVM 語音端點偵測 (EPD)

## 📁 1. 資料夾結構
```
hw03
├── data                        # 📚 資料集
│   ├── dataset_cache.pt        # 🚀 資料集前處理後快取
│   ├── waveFiles_2008
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

## ⚙️ 2. 前處理

檔案命名規則：`<字母>_<起始 sample>_<結束 sample>.wav`，並透過 `wavefiles-all` 以 8:2 比例切分測試與訓練集。

### ✂️ 2.1. 音框 (Frame) 切割
由於 SVM 需要固定大小的輸入特徵，需先將音檔切割成固定大小的 Frame。

```
frame_size = 400 samples
hop_size = 80 samples
```

### 📐 2.2. 維度調整
透過 `torchaudio.load()` 讀取後，維度為 `(1, num_samples)`。
使用 `unsqueeze(0)` 擴增維度至 `(1, 1, frame_size)` 以符合 `torchaudio.transforms.MFCC` 的需求。

### 🎚️ 2.3. MFCC 特徵轉換
將 `n_mfcc` 設定為 `13`，與模型輸入維度同步，輸出維度為 `(1, n_mfcc, num_frames)`。

### 🔍 2.4. 最終維度轉換
移除 `batch` 維度後，最終維度為 `(n_mfcc, num_frames)`。

## 🧠 3. SVM 模型

因 SVM 僅接受 `1D` 特徵，因此將 `(13, 3)` 攤平為 `(39,)` 作為輸入。

透過 `nn.Linear` 模擬：

$$
f(x) = Wx + b
$$

## 📉 4. 損失函數 (Hinge Loss)

$$
L = max(0, 1−y⋅f(x))
$$

分類正確時 (同號)，Loss 為 0；分類錯誤時 (不同號)，Loss 增大，引導模型學習邊界。

## 🚀 5. 訓練方式 - SGD (隨機梯度下降)
使用 PyTorch 內建的 `SGD` 訓練，更新參數 `W` 和 `b`。

## 📊 5. 訓練日誌
詳細記錄每 epoch 和 Loss 資訊，最終模型儲存為 `svm_model.pth`。

## 🎯 6. 訓練成果
最終測試集準確率達 `91.87%`。

## 📈 7. 視覺化
透過 Gradio + Plotly 實現互動視覺化，可即時上傳音訊檔案進行分析、標記音段並播放。

![視覺化展示](./asset/image1.png)

