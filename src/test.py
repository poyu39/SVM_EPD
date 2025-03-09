import torch
import logging
import argparse
import os
from torch.utils.data import DataLoader
from model import SVM_Model
from dataset import EPD_FrameDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="要測試的模型路徑"
    )
    args = parser.parse_args()
    
    save_dir = os.path.dirname(args.model)
    log_path = os.path.join(save_dir, "test.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logging.info("開始測試...")
    dataset = EPD_FrameDataset(mode="test")
    
    test_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    model = SVM_Model().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    logging.info(f"模型已載入: {args.model}")
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for mfccs, labels in test_loader:
            mfccs: torch.Tensor
            labels: torch.Tensor
            
            mfccs, labels = mfccs.to(device), labels.to(device)
            
            # 2D -> 1D
            outputs = model(mfccs).squeeze()
            
            # 預測值取 sign (-1 or 1)
            preds = torch.sign(outputs)
            
            # 把布林值轉換成 int 0 1
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total * 100
    logging.info(f"測試集 accuracy: {accuracy:.2f}%")
