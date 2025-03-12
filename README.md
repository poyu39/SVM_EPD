# 🎙️ SVM Speech Endpoint Detection (EPD)

> [README_ZH.md](./README_ZH.md)

## 📁 Folder Structure
```
hw03
├── data                        # 📚 Dataset
│   ├── dataset_cache.pt        # 🚀 Cached preprocessed dataset
│   ├── waveFiles_2008
│   └── wavefiles-all
├── models                      # 💻 Model output folder
│   └── 20250309_151313         # 📅 Training timestamp
│       ├── svm_model.pth       # 📦 Model file
│       ├── test.log
│       └── train.log
└── src                         # 🧩 Source code
    ├── config.py
    ├── dataset.py
    ├── epd.py
    ├── model.py
    ├── test.py
    └── train.py
```

## 🛠️ Environment

> python `3.11.5`, cuda `12.4`
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## 🧷 Usage

```bash
# Training
python src/train.py

# Test Frame Scores
python src/test.py --model <model_path>

# Test EPD Scores
python src/epd.py --model <model_path> --score true --dataset <audio_folder>

# Visualization
python src/epd.py --model <model_path>
```

## ⚙️ Preprocessing

File naming convention: `<letter>_<start_sample>_<end_sample>.wav`. The dataset is split into training and testing sets in an 8:2 ratio using `wavefiles-all`.

### ✂️ Frame Segmentation
Since SVM requires fixed-size input features, audio files are segmented into frames of a fixed size.

```
frame_size = 400 samples
hop_size = 80 samples
```

### 📐 Dimension Adjustment
After reading the file using `torchaudio.load()`, the dimension is `(1, num_samples)`.  
Using `unsqueeze(0)`, the dimension is expanded to `(1, 1, frame_size)`, which aligns with `torchaudio.transforms.MFCC` requirements.

### 🎚️ MFCC Feature Extraction
`n_mfcc` is set to `13`, matching the model's input dimension, resulting in an output dimension of `(1, n_mfcc, num_frames)`.

### 🔍 Final Dimension Transformation
After removing the `batch` dimension, the final dimension is `(n_mfcc, num_frames)`.

## 🧠 SVM Model

SVM only accepts `2D` features, so `(n_mfcc, num_frames)` is flattened into `mean(n_mfcc)`, combined with labels to form `(mfcc, label)`.

Simulated using `nn.Linear`:

$$
f(x) = Wx + b
$$

## 📉 Loss Function (Hinge Loss)

$$
L = max(0, 1−y⋅f(x))
$$

When classification is correct (same sign), the loss is 0. When incorrect (different sign), the loss increases, guiding the model to learn boundaries.

## 🚀 Training Methodology

### 🎲 Stochastic Gradient Descent (SGD)
Training is performed using PyTorch’s built-in `SGD`, updating parameters `W` and `b`.

### 📉 Reduce Learning Rate
Using `torch.optim.lr_scheduler.ReduceLROnPlateau`, the learning rate is reduced when the loss does not improve for 3 consecutive epochs, reducing it to 0.5 times the original value.

### 🚧 Early Stopping
If there is no improvement for 5 consecutive epochs, training is stopped early.

## 📊 Output
```
model_timestamp
├── best_svm_model.pth  # 📦 Model file
├── config.yaml         # ⚙️ Configuration file
├── epd_score.csv       # 📊 Evaluation scores for detected audio files
├── test.log            # 📝 Evaluation scores based on the test set
└── train.log           # 📝 Training records
```

## 🎯 Validation

### 🎵 Frame Label Testing
Predictions are made based on MFCC features and labels of test frames, and accuracy is calculated.

✨ [Best Model](./models/best_model/test.log) achieved an accuracy of `92.58%`.

### 🎼 End-point Sample Testing
Predictions are made based on the start and end samples of test audio files, and accuracy is calculated.

✨ [Best Model](./models/best_model/epd_score.csv) achieved an accuracy of `82.21%`.

#### wavfiles-2008 Test Results
```csv
Speaker,Correct,Total,EPD Score
931915_yylee,57.5,72,79.86111111111111
932008_zylee,121.5,144,84.375
932017_zjye,189.5,216,87.73148148148148
942027_zwxiao,236.5,288,82.11805555555556
9565506_hyzhuang,304.0,360,84.44444444444444
960003_Jens,361.5,432,83.68055555555556
9662613_yhjang,369.5,504,73.31349206349206
Average,0,0,82.2177343159486
```

## 📈 Visualization
Interactive visualization is implemented using Gradio + Plotly, allowing real-time audio file uploads for analysis, segment labeling, and playback.

![Visualization Demo](./asset/image1.png)
