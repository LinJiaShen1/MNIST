# MNIST 訓練
1. 載入 MNIST 資料集 (手寫圖片)
2. 資料前處理 (Normalization)
3. **建立模型架構 (CNN)**
4. **訓練模型 (fit)**
5. **評估模型 (evaluate)**
6. **儲存模型 (save)**

## 3.建立模型架構 (CNN)
### 前言
#### 安裝 PyTorch 的前置條件
首先，確保你的 macOS 上已經安裝了 Python。Python 可以通過官方網站、Homebrew 或 Anaconda 來安裝。如果你使用的是 Anaconda，那麼管理 Python 環境和庫會更加方便。  
#### 使用 Conda 安裝 PyTorch
如果你已經安裝了 Anaconda，可以使用 Conda 來安裝 PyTorch，這通常是最簡單的方式，因為 Conda 會自動處理所有依賴性。打開終端並輸入以下命令：  
```
conda install pytorch torchvision torchaudio -c pytorch
```
這個命令會從 PyTorch 的官方 channel 安裝最新版本的 PyTorch，以及 torchvision 和 torchaudio，這兩個庫常用於圖像和音頻處理。  
#### 使用 Pip 安裝 PyTorch
如果你傾向於使用 pip，可以按照以下步驟進行。首先，打開你的終端，然後輸入以下命令：  
```
pip install torch torchvision torchaudio
```
#### 選擇適合你硬件的安裝選項
如果你的 Mac 支持 GPU，並且你希望 PyTorch 能夠利用這一點來加速計算，則需要確保安裝了適用於 Mac 的 GPU 版本的 PyTorch。目前，PyTorch 對 Apple Silicon (M1/M2 芯片) 的支持正在進行中，並且已經有了一些進展，例如可以使用為 Apple Silicon 優化的 PyTorch 版本。詳情可以查看 PyTorch 官方網站或相關社區的更新。
#### 驗證安裝
```
import torch
print(torch.__version__)
print(torch.cuda.is_available())
```
這將顯示你安裝的 PyTorch 版本，並檢查 CUDA 是否可用（對於使用 NVIDIA GPU 的用戶）。對於 Mac 用戶，通常 torch.cuda.is_available() 會返回 False，除非你使用的是外接的 GPU 或特殊配置。  

### CNN 模型
這是一個簡易的CNN模型，可以參考Kaggle上其他人的模型建置。
```
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 14 * 14)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

## 4. 訓練模型 (fit)
訓練模型需要準備訓練循環和優化器。這裡是一個基本的訓練循環示例：  
```
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# 假設 train_loader 是你的數據加載器
# 初始化模型、損失函數和優化器
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 調用訓練函數
train_model(model, train_loader, criterion, optimizer)
```
## 5.評估模型 (evaluate)
模型評估可以通過以下方式進行：
```
def evaluate_model(model, test_loader):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    print(f'Accuracy: {100 * correct / total}%')

# 假設 test_loader 是你的測試數據加載器
evaluate_model(model, test_loader)
```

## 6.儲存模型 (save)
最後，你可能想要儲存你訓練好的模型，以便之後使用或進行部署：
```
def save_model(model, path='model.pth'):
    torch.save(model.state_dict(), path)

save_model(model)
```
