import torch
import torch.nn as nn
from config import N_MFCC

def hinge_loss(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    SVM Loss Function
    
    Hinge Loss 公式
        L = mean( max(0, 1 - y * f(x)) )
    
    - y = label (-1 or 1）
    - f(x) = 模型的輸出（logits）
    - y, f(x) 符號相同，則 loss = 0（表示分類正確），符號相反，則 loss 會較大
    
    Args:
        outputs (torch.Tensor):     模型的輸出
        labels (torch.Tensor):      真實標籤 -1/1
    
    Returns:
        torch.Tensor: Hinge Loss 值
    """
    
    labels = labels.float()
    
    loss = torch.clamp(1 - outputs.view(-1) * labels, min=0)
    
    return torch.mean(loss)

class SVM_Model(nn.Module):
    def __init__(self, input_dim=N_MFCC):
        """
        SVM 線性分類模型，用 Linear 模擬 SVM
        
        Args:
            input_dim (int): 輸入特徵維度，依照 n_mfcc
        """
        super(SVM_Model, self).__init__()
        
        # 用 nn.Linear 模擬 y = Wx + b
        self.fc = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        """
        前向傳播計算輸出
        
        Args:
            x (torch.Tensor): 輸入特徵 (batch_size, input_dim)
        
        Returns:
            torch.Tensor: 模型輸出 (batch_size,)
        """
        
        # (batch_size, 1) -> (batch_size,)
        return self.fc(x).squeeze(1)
