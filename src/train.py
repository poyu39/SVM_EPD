import torch
import os
from torch.utils.data import DataLoader
from dataset import EPD_FrameDataset
from model import SVM_Model, hinge_loss
import logging
from datetime import datetime
from config import *

from torch.optim.lr_scheduler import ReduceLROnPlateau

if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"models/{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    log_path = os.path.join(save_dir, "train.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    
    # 紀錄訓練參數
    with open(os.path.join(save_dir, "config.yaml"), "w") as f:
        f.write(f"DATASET_PATH: {DATASET_PATH}\n")
        f.write(f"TRAIN_RATIO: {TRAIN_RATIO}\n")
        f.write(f"EPOCHS: {EPOCHS}\n")
        f.write(f"BATCH_SIZE: {BATCH_SIZE}\n")
        f.write(f"FRAME_SIZE: {FRAME_SIZE}\n")
        f.write(f"HOP_SIZE: {HOP_SIZE}\n")
        f.write(f"N_MFCC: {N_MFCC}\n")
    
    logging.info("開始訓練...")
    
    # 加載數據集
    dataset = EPD_FrameDataset(mode="train")
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SVM_Model().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Early Stopping 參數
    best_loss = float('inf')
    patience = 5  # 允許 loss 停滯的 epoch 數
    counter = 0
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for mfccs, labels in train_loader:
            mfccs, labels = mfccs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(mfccs).squeeze()
            loss = hinge_loss(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        logging.info(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, lr: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Early Stopping 檢查
        if avg_loss < best_loss:
            best_loss = avg_loss
            counter = 0
            best_model_path = os.path.join(save_dir, "best_svm_model.pth")
            
            # 儲存最佳模型
            torch.save(model.state_dict(), best_model_path)
        else:
            counter += 1
            if counter >= patience:
                logging.info(f"Loss 停滯 {patience} epochs，提前停止訓練")
                break
        
        # 調整學習率
        scheduler.step(avg_loss)
    
    logging.info("訓練結束")
