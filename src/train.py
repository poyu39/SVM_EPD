import torch
import os
from torch.utils.data import DataLoader, random_split
from dataset import EPD_FrameDataset
from model import SVM_Model, hinge_loss
import logging
from datetime import datetime

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
    
    logging.info("開始訓練...")
    
    dataset = EPD_FrameDataset(mode="train")
    
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SVM_Model().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
    
    for epoch in range(10):
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
        logging.info(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")
    
    model_path = os.path.join(save_dir, "svm_model.pth")
    torch.save(model.state_dict(), model_path)
    logging.info(f"模型已儲存至 {model_path}")
