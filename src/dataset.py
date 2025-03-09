import torchaudio
from torch.utils.data import Dataset
import torch
import os
from pathlib import Path
import logging
from config import DATASET_PATH, DATASET_CACHE_PATH, TRAIN_RATIO, FRAME_SIZE, HOP_SIZE, N_MFCC

class EPD_FrameDataset(Dataset):
    def __init__(self,
                 data_dir       = DATASET_PATH,
                 mode           = "train",
                 train_ratio    = TRAIN_RATIO,
                 frame_size     = FRAME_SIZE,
                 hop_size       = HOP_SIZE,
                 n_mfcc         = N_MFCC):
        """
        Args:
            data_dir (str):         資料夾路徑
            mode (str):             "train" 或 "test"
            train_ratio (float):    訓練集比例 (剩餘 1 - train_ratio 當作測試集)
            frame_size (int):       每個 frame 的大小
            hop_size (int):         frame 的移動步長
            n_mfcc (int):           MFCC 的維度
        """
        self.mode = mode
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.train_ratio = train_ratio
        
        # 如果有快取檔，直接載入
        if os.path.exists(DATASET_CACHE_PATH):
            logging.info(f"已找到資料集快取 {DATASET_CACHE_PATH}，正在載入...")
            cached_data = torch.load(DATASET_CACHE_PATH)
            self.train_samples = cached_data["train"]
            self.test_samples = cached_data["test"]
            logging.info(f"載入完成：train={len(self.train_samples)} 筆, test={len(self.test_samples)} 筆")
        else:
            logging.info("沒有找到快取檔，開始讀取並處理每個音檔...")
            
            self.mfcc_transform = torchaudio.transforms.MFCC(
                sample_rate=16000,
                n_mfcc=n_mfcc,
                melkwargs={
                    "n_fft": frame_size,
                    "hop_length": hop_size,
                    "n_mels": 40,
                }
            )
            
            # 分別儲存每檔案整理完的 frame
            self.train_samples = []
            self.test_samples = []
            
            directory_path = Path(data_dir)
            
            now_speaker = ""
            
            for file_path in directory_path.glob("**/*.wav"):
                # 從檔名中解析
                _, start_str, end_str = file_path.stem.split('_')
                start_sample = int(start_str)
                end_sample   = int(end_str)
                
                if now_speaker != file_path.parent.name:
                    logging.info(f"正在處理 {file_path.parent.name} 的音檔")
                    now_speaker = file_path.parent.name
                
                
                abs_path = os.path.abspath(file_path)
                
                waveform, sample_rate = torchaudio.load(abs_path)
                
                # resample 到 16000 Hz
                if sample_rate != 16000:
                    resample = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                    waveform = resample(waveform)
                
                signal = waveform[0]
                length = signal.shape[0]
                
                file_frames = []
                for start in range(0, length, hop_size):
                    end = start + frame_size
                    if end > length:
                        break
                    
                    # 1D -> 2D，因為 MFCC 需要 2D 輸入
                    # (num_samples,) -> (1, num_samples)
                    frame = signal[start:end].unsqueeze(0)
                    
                    # 2D -> 3D -> 2D
                    # (1, num_samples) -> (1, n_mfcc, num_frames) -> (n_mfcc, num_frames)
                    mfcc = self.mfcc_transform(frame).squeeze(0)
                    
                    # 算 num_frames 的平均
                    # (n_mfcc, num_frames) -> (n_mfcc,)
                    mfcc_mean = mfcc.mean(dim=-1)
                    
                    # 直接標記成 -1 / 1
                    label = 1 if (start_sample <= start <= end_sample) else -1
                    
                    if mfcc_mean.shape[0] == n_mfcc:
                        file_frames.append((mfcc_mean, label))
                
                # frame 隨機排列，再分割
                if len(file_frames) > 0:
                    indices = torch.randperm(len(file_frames))
                    train_count = int(self.train_ratio * len(file_frames))
                    
                    train_indices = indices[:train_count]
                    test_indices  = indices[train_count:]
                    
                    for i in train_indices:
                        self.train_samples.append(file_frames[i])
                    for i in test_indices:
                        self.test_samples.append(file_frames[i])
            
            torch.save({"train": self.train_samples, "test": self.test_samples}, DATASET_CACHE_PATH)
            logging.info(f"已完成音檔 {round(self.train_ratio, 1)}, {round(1 - self.train_ratio, 1)} 分割並快取，train={len(self.train_samples)}, test={len(self.test_samples)}")
        
        if mode == "train":
            self.samples = self.train_samples
            logging.info(f"載入訓練集，共 {len(self.samples)} 筆")
        elif mode == "test":
            self.samples = self.test_samples
            logging.info(f"載入測試集，共 {len(self.samples)} 筆")
        else:
            raise ValueError("mode 必須是 'train' 或 'test'")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]
