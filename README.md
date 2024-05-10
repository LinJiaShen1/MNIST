# MNIST 資料集

訓練流程:

1. **載入 MNIST 資料集 (手寫圖片)**
2. **資料前處理 (Normalization)**
3. 建立模型架構 (Sequential)
4. 訓練模型 (fit)
5. 評估模型 (evaluate)
6. 儲存模型 (save)

## 1. 載入 MNIST 資料集

### 1.0 前言(準備資料集):

#### 定義資料集: 

X = 輸入的資料(圖片)  
y = 輸出的資料(標籤)  

舉例而言，我今天要辨識貓/狗，X 就是圖片，y 就是 0/1 (貓/狗)。  
所以在資料準備的時候會對每一張照片都做標籤。

假設欲判斷的字有20個，標籤就會有20個，但我們並不會在數學內標0~20，而是用矩陣 one-hot encoding的方式來表示。  
[one hot encoding](https://medium.com/@PatHuang/%E5%88%9D%E5%AD%B8python%E6%89%8B%E8%A8%98-3-%E8%B3%87%E6%96%99%E5%89%8D%E8%99%95%E7%90%86-label-encoding-one-hot-encoding-85c983d63f87)

#### 建構資料集:

在準備資料集時，可以使用folder來進行不同資料的分類。  
舉例而言，我有三個物件(貓、狗、牛)，在資料夾內我們可以這麼做。  
Data  
|----  cat  
|----  ---- cat_photo0.jpg  
|----  ---- ...  
|----  dog  
|----  ---- dog_photo0.jpg  
|----  ---- ...  
|----  cow  
|----  ---- cow_photo0.jpg  
|----  ---- ...  
因此在讀取的時候會一邊讀取文件內資料，一邊加上對應的圖片的標籤。  
```
# travel過data資料夾內所有的文件
import os
import cv2
data_directory = 'data'
items = os.lsitdir(data_directory)
# 篩選出目錄，即訓練的LABEL
labels = [item for item in items if os.path.isdir(os.path.join(data_directory, item))]
print("訓練標籤（資料夾）有：", labels)
X_data = []
Y_data = []
# 遍歷每個標籤目錄，並列出其中的文件
for label in labels:
    folder_path = os.path.join(data_directory, label)
    files = os.listdir(folder_path)
    # 進行每個文件的讀取
    for file in files:
        file_path = os.path.join(folder_path, file)
        # 確保文件是圖片格式
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            X_data.append(cv2.imread(file_path))
            Y_data.append(label)
```

### 1.1 讀取圖片
#### 安裝Library
首先，你需要確保已經安裝了 cv2，你可以通過以下命令安裝：
```
pip install opencv-python
pip install numpy
pip install -U scikit-learn=1.3.0
```
#### 讀取圖片
接著，你可以用以下程式碼來讀取圖片：
```
import cv2

# 讀取圖片的路徑
image_path = 'path/to/your/image.jpg'

# 使用 cv2.imread() 函數讀取圖片，cv2.IMREAD_COLOR 表示讀取彩色圖片
image = cv2.imread(image_path, cv2.IMREAD_COLOR)

# 檢查圖片是否正確讀取
if image is None:
    print("無法讀取圖片")
else:
    print("圖片已成功讀取")
```
#### 對圖片做簡單的處理
##### 轉換為灰階圖片
將彩色圖片轉換成灰階圖片可以幫助減少處理的複雜性，尤其是在進行影像處理或機器學習任務時。  
你可以使用 OpenCV 的 cv2.cvtColor() 函數來實現這一點：  
```
# 轉換圖片到灰階
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 顯示灰階圖片
cv2.imshow('Gray Image', gray_image)
cv2.waitKey(0)  # 等待按鍵事件
cv2.destroyAllWindows()  # 關閉所有 OpenCV 開啟的窗口
```
##### 進行二值化處理
二值化是將圖片轉換成僅包含黑白兩種顏色的過程，這在許多影像處理場景中非常有用，比如在文本識別或邊緣檢測中。在 OpenCV 中，你可以使用 cv2.threshold() 函數來實現二值化：  
```
# 設定閾值
thresh = 127
# 最大值
max_value = 255

# 二值化處理
ret, binary_image = cv2.threshold(gray_image, thresh, max_value, cv2.THRESH_BINARY)

# 顯示二值化圖片
cv2.imshow('Binary Image', binary_image)
cv2.waitKey(0)  # 等待按鍵事件
cv2.destroyAllWindows()  # 關閉所有 OpenCV 開啟的窗口
```
### 1.2 分成訓練資料、驗證資料與測試資料
#### 介紹訓練資料
在機器學習項目中，通常會將數據集分成三部分：訓練資料、驗證資料和測試資料。這樣做的目的是為了在不同階段對模型進行訓練、調參和評估，以確保模型的泛化能力。以下是這一過程的一個基本指南：  
1. 訓練資料（Training Data）: 用於訓練模型，使模型學習如何從輸入數據預測或分類輸出。
2. 驗證資料（Validation Data）: 用於模型訓練過程中的參數調整，驗證資料幫助確認不同超參數的設置對模型效果的影響。
3. 測試資料（Test Data）: 在模型開發過程中保持獨立，用於最終評估模型的效能，以此來模擬模型對新數據的反應如何。
#### 對Y資料進行 One-hot Encoding
One-hot 編碼是一種處理類別標籤的方法，它將每個類別標籤轉換為一個只一位是 1，其餘位都是 0 的二進制表示。這種方法對於類別輸出的神經網絡非常有用，特別是在進行多類分類時。
```
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False)
y_one_hot = encoder.fit_transform(y.reshape(-1, 1))
```
#### 實現數據集分割
可以使用 Python 的 sklearn 庫來輕鬆地分割數據。首先，確保安裝了 sklearn：
```
pip install scikit-learn
```
然後，可以使用 train_test_split 函數來分割數據。以下是一個如何分割數據的範例：
```
from sklearn.model_selection import train_test_split

# 假設 X 是輸入特徵數據，y 是標籤
# 先分割出測試集，一般大小為整個數據集的 20%
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# 再從剩下的數據中分割出訓練集和驗證集，這裡將驗證集設為 25%（即原始數據的 20%）
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

# 現在，X_train, y_train 是訓練集；X_val, y_val 是驗證集；X_test, y_test 是測試集
```
其中，
1. 數據分層：在使用 train_test_split 時，你可以透過 stratify 參數來保證訓練集、驗證集和測試集中的類別分佈與原始數據集保持一致。
2. 隨機種子（Random Seed）：設置 random_state 參數可以確保每次執行代碼時，數據分割的方式都保持一致。

## 2.資料前處理 (Normalization)

### 2.0 前言
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

### 2.1 轉換數據為Tensor
將數據轉換成 PyTorch tensors 是實現模型訓練的一個必要步驟，因為PyTorch 使用 tensor 來進行所有計算。  
```
import torch

# 假設 X_train, X_val, X_test 已經是預處理好的特徵數據
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# 將 y_one_hot 轉換為 tensor
y_train_tensor = torch.tensor(y_train_one_hot, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val_one_hot, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_one_hot, dtype=torch.float32)
```

### 2.2 使用DataLoader
```
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        '初始化數據集，這裡的 features 和 labels 應該是已經轉換好的 tensors'
        self.features = features
        self.labels = labels

    def __len__(self):
        '返回數據集中的樣本數量'
        return len(self.features)

    def __getitem__(self, index):
        '按照給定的索引 index 返回一個樣本和其標籤'
        return self.features[index], self.labels[index]
```
定義完後可以宣告
```
from torch.utils.data import DataLoader

# 假設 X_train_tensor 和 y_train_tensor 已經是處理好的訓練數據和標籤的 tensors
train_dataset = CustomDataset(X_train_tensor, y_train_tensor)
# 創建 DataLoader，設置批次大小和是否洗牌
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = CustomDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

test_dataset = CustomDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
```

