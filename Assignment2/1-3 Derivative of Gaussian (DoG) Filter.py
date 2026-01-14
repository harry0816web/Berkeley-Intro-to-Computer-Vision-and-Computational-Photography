import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import imageio.v3 as iio
from scipy.signal import convolve2d
import cv2


# bonus:
def visualize_gradient_orientation(Ix, Iy, magnitude):
    # 1. 計算梯度方向 (角度)，範圍為 [-pi, pi]
    # np.arctan2(y, x)
    orientations = np.arctan2(Iy, Ix)
    
    # 2. 將角度映射到 [0, 1] 區間以符合 HSV 的 Hue 標準
    # (orientations + pi) / (2 * pi)
    hue = (orientations + np.pi) / (2 * np.pi)
    
    # 3. 準備 HSV 圖片
    # H (Hue): 方向
    # S (Saturation): 全滿
    # V (Value): 梯度強度 (需正規化到 0~1)
    
    # 正規化 Magnitude 到 0~1，這會決定邊緣的亮度
    v_min, v_max = magnitude.min(), magnitude.max()
    value = (magnitude - v_min) / (v_max - v_min + 1e-8)
    
    # 建立 HSV 矩陣 (H, W, 3)
    hsv_img = np.zeros((Ix.shape[0], Ix.shape[1], 3))
    hsv_img[..., 0] = hue           # 色調
    hsv_img[..., 1] = 1.0           # 飽和度
    hsv_img[..., 2] = value         # 亮度
    
    # 4. 轉成 RGB 並畫出來
    rgb_img = hsv_to_rgb(hsv_img)
    return rgb_img

# ==========================================
# 1. 準備工作：讀取圖片與定義基礎算子
# ==========================================
def load_and_gray(path):
    try:
        img = iio.imread(path)
        if img.ndim == 3:
            img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
        return img.astype(float)
    except:
        from skimage import data
        return data.camera().astype(float)

img = load_and_gray('test.png') # 請確保路徑正確

Dx = np.array([[1, 0, -1]])
Dy = np.array([[1], [0], [-1]])

# ==========================================
# 2. 建立 2D 高斯核 (Gaussian Kernel)
# ==========================================
# ksize 是窗口大小，sigma 是標準差（數值越大越模糊）
ksize = 15
sigma = 2
g_1d = cv2.getGaussianKernel(ksize, sigma)
G = np.outer(g_1d, g_1d.T) 

# ==========================================
# 3. 方法一：先模糊再微分 (Blur then Diff)
# ==========================================
# Step 1: 模糊原圖
img_blurred = convolve2d(img, G, mode='same', boundary='symm')

# Step 2: 對模糊圖做微分
Ix_method1 = convolve2d(img_blurred, Dx, mode='same', boundary='symm')
Iy_method1 = convolve2d(img_blurred, Dy, mode='same', boundary='symm')

# Step 3: 計算梯度幅值
mag_method1 = np.sqrt(Ix_method1**2 + Iy_method1**2)

# ==========================================
# 4. 方法二：使用高斯導數濾波器 (DoG Filter)
# ==========================================
# Step 1: 先讓Gaussian Kernel 跟 Finite Difference Operators 捲積，生成 DoG 算子
DoG_x = convolve2d(G, Dx, mode='same', boundary='symm')
DoG_y = convolve2d(G, Dy, mode='same', boundary='symm')

# Step 2: 直接拿 DoG 算子去捲積原圖 (一次捲積完成)
Ix_method2 = convolve2d(img, DoG_x, mode='same', boundary='symm')
Iy_method2 = convolve2d(img, DoG_y, mode='same', boundary='symm')


# Step 3: 計算梯度幅值
mag_method2 = np.sqrt(Ix_method2**2 + Iy_method2**2)

# ==========================================
# 5. 驗證與視覺化
# ==========================================
# 驗證兩者是否一致
difference = np.sum(np.abs(mag_method1 - mag_method2))
print(f"兩組方法之間的數值差異: {difference:.2e}")

# 挑選一個適當的 Threshold 來轉成 Edge Image (需手動調整)
threshold = 20
edge_final = mag_method2 > threshold

plt.figure(figsize=(16, 10))

# 顯示 DoG Filters
plt.subplot(2, 4, 1); plt.imshow(DoG_x, cmap='gray'); plt.title('DoG X Filter')
plt.subplot(2, 4, 2); plt.imshow(DoG_y, cmap='gray'); plt.title('DoG Y Filter')

# 顯示中間過程
plt.subplot(2, 4, 3); plt.imshow(img_blurred, cmap='gray'); plt.title('Step 1: Blurred Image')
plt.subplot(2, 4, 4); plt.imshow(mag_method1, cmap='gray'); plt.title('Step 2-1: Blur then Diff Magnitude')
plt.subplot(2, 4, 5); plt.imshow(mag_method2, cmap='gray'); plt.title('Step 2-2: DoG Filter Magnitude')


# 顯示最終結果與比較
plt.subplot(2, 4, 6); plt.imshow(img, cmap='gray'); plt.title('Original Image')
plt.subplot(2, 4, 7); plt.imshow(edge_final, cmap='gray'); plt.title(f'Final Edges (Thres={threshold})')

# 回顧 Part 1.2 的結果 (為了觀察差異，這裡模擬沒做 Gaussian 的樣子)
Ix_no_blur = convolve2d(img, Dx, mode='same', boundary='symm')
Iy_no_blur = convolve2d(img, Dy, mode='same', boundary='symm')
mag_no_blur = np.sqrt(Ix_no_blur**2 + Iy_no_blur**2)
plt.subplot(2, 4, 8); plt.imshow(mag_no_blur > threshold, cmap='gray'); plt.title('Comparison: Part 1.2 (No Blur)')

plt.tight_layout()
plt.show()


# bonus:
grad_orientation_rgb = visualize_gradient_orientation(Ix_method2, Iy_method2, mag_method2)
plt.figure(figsize=(10, 5))
plt.imshow(grad_orientation_rgb)
plt.title('Gradient Orientation')
plt.axis('off')
plt.show()