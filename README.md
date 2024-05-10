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

假設欲判斷的字有20個，標籤就會有20個，但我們並不會在數學內標0~20，而是用矩陣-one hot encoding的方式來表示。  
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
