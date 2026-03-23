import numpy as np
import matplotlib.pyplot as plt
import cv2
import imageio.v3 as iio

# ==========================================
# 1. 基礎工具：讀圖
# ==========================================
def load_image_color(path):
    try:
        img = iio.imread(path)
        # 處理 RGBA -> RGB
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[..., :3]
        return img.astype(float) / 255.0
    except Exception as e:
        print(f"讀取失敗 {path}: {e}")
        return None

# ==========================================
# 2. 核心：建立 Stacks
# ==========================================
def get_gaussian_stack(image, levels=5, sigma_start=2):
    """
    產生 Gaussian Stack
    levels: 總層數
    sigma_start: 第一層的模糊程度 (之後每層通常會加倍)
    """
    stack = [image] # 第 0 層是原圖
    
    for i in range(levels):
        # 每一層都比上一層更模糊
        sigma = sigma_start * (2 ** i)
        # 每一層把高頻訊號濾掉，最後只剩模糊輪廓
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
        
        stack.append(blurred)
        
    return stack

def get_laplacian_stack(gaussian_stack):
    """
    產生 Laplacian Stack
    L_i = G_i - G_{i+1}
    最後一層直接 L_i = G_N (G_N 一路把高頻訊號差(L_i)補回來) => 最後還原回原圖
    """
    laplacian_stack = []
    
    for i in range(len(gaussian_stack) - 1):
        # 計算細節層 (High Frequency at this level)
        low_freq_detail = gaussian_stack[i] - gaussian_stack[i+1]
        laplacian_stack.append(low_freq_detail)
    
    # 最後一層 L_N = G_N
    laplacian_stack.append(gaussian_stack[-1])
    
    return laplacian_stack

# ==========================================
# 3. 視覺化與驗證
# ==========================================
# 請準備兩張圖：'apple.jpeg' 和 'orange.jpeg'
# 如果沒有，可以用任何兩張圖片代替
img_path = 'apple.png' 
img = load_image_color(img_path)

if img is not None:
    # 建立 Stacks
    levels = 5
    sigma = 2
    
    g_stack = get_gaussian_stack(img, levels, sigma)
    l_stack = get_laplacian_stack(g_stack)
    
    # --- 顯示 Gaussian Stack ---
    plt.figure(figsize=(12, 6))
    for i in range(len(g_stack)):
        plt.subplot(1, len(g_stack), i+1)
        plt.imshow(np.clip(g_stack[i], 0, 1))
        plt.title(f"G_{i}")
        plt.axis('off')
    plt.suptitle("Gaussian Stack (Progressively Blurry)")
    plt.show()
    
    # --- 顯示 Laplacian Stack ---
    # 注意：Laplacian 會有負值，顯示時需要正規化或加 0.5 變成灰色基底
    plt.figure(figsize=(12, 6))
    for i in range(len(l_stack)):
        plt.subplot(1, len(l_stack), i+1)
        
        # 為了視覺化方便，將數值 +0.5 讓 0 對應到灰色
        # 這樣才能看到負值的細節 (變暗) 和正值的細節 (變亮)
        vis_layer = np.clip(l_stack[i] + 0.5, 0, 1)
        
        plt.imshow(vis_layer)
        plt.title(f"L_{i}")
        plt.axis('off')
    plt.suptitle("Laplacian Stack (Band-pass Details)")
    plt.show()

    # --- 驗證重建 (Sanity Check) ---
    # 把 Laplacian Stack 全部加起來，應該要等於原圖
    reconstructed = np.zeros_like(img)
    for layer in l_stack:
        reconstructed += layer
        
    # 計算誤差
    diff = np.sum(np.abs(img - reconstructed))
    print(f"重建誤差 (應該接近 0): {diff:.5f}")
    
    # 顯示重建結果
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1); plt.imshow(img); plt.title("Original")
    plt.subplot(1, 2, 2); plt.imshow(np.clip(reconstructed, 0, 1)); plt.title("Reconstructed from Sum(L_i)")
    plt.show()