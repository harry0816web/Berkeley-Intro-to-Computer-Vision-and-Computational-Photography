import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
from scipy.signal import convolve2d

# 1. 讀取圖片 (記得換成你的圖片路徑)
image_path = 'test.png' 
img = iio.imread(image_path)
if img.ndim == 3: img = np.dot(img[..., :3], [0.299, 0.587, 0.114])
img = img.astype(float)

# 2. 定義差分算子 (根據題目要求)
# Dx: [1, 0, -1] -> 偵測垂直邊緣 (左右數值變化大)
Dx = np.array([[1, 0, -1]])
# Dy: [1, 0, -1]^T -> 偵測水平邊緣 (上下數值變化大)
Dy = np.array([[1], [0], [-1]])

# 3. 計算偏微分 (Partial Derivatives)
# 使用 scipy.signal.convolve2d，mode='same' 維持大小
# 這裡不需要手動 flip kernel，因為我們是用現成的函數 (它內部會處理或視為 correlation，但在這裡效果一致)
Ix = convolve2d(img, Dx, mode='same', boundary='symm')
Iy = convolve2d(img, Dy, mode='same', boundary='symm')

# 4. 計算梯度幅值 (Gradient Magnitude)
# 公式: sqrt(Ix^2 + Iy^2)
grad_mag = np.sqrt(Ix**2 + Iy**2)

# 5. 二值化 (Binarization / Thresholding)
# 這個 threshold 需要你自己調整！試試看 20, 50, 80...
# 目標：看到清楚的邊緣線條，但不要太多雜訊白點
threshold = 30
edges = grad_mag > threshold

# --- 顯示結果 ---
plt.figure(figsize=(15, 8))

# 第一排：原始微分結果
plt.subplot(2, 3, 1)
plt.imshow(Ix, cmap='gray')
plt.title('Partial Derivative X (Ix)')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(Iy, cmap='gray')
plt.title('Partial Derivative Y (Iy)')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(grad_mag, cmap='gray')
plt.title('Gradient Magnitude')
plt.axis('off')

# 第二排：二值化結果 (Edge Map)與原圖比較
plt.subplot(2, 3, 4)
plt.imshow(edges, cmap='gray')
plt.title(f'Edge Map (Threshold={threshold})')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.hist(grad_mag.flatten(), bins=50, log=True)
plt.title('Gradient Magnitude Histogram')
plt.xlabel('Magnitude Value')

plt.tight_layout()
plt.show()