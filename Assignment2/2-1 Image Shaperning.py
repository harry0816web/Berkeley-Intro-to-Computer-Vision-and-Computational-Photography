import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
import cv2
from scipy.signal import convolve2d

def load_and_gray(path):
    try:
        img = iio.imread(path)
        if img.ndim == 3:
            img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
        return img.astype(float)
    except:
        print(f"找不到圖片 {path}，使用範例圖。")
        from skimage import data
        return data.camera().astype(float)

def get_gaussian_kernel(ksize, sigma):
    g_1d = cv2.getGaussianKernel(ksize, sigma)
    return np.outer(g_1d, g_1d.T)

# ==========================================
# 核心函數：建立銳化濾波器
# ==========================================
def create_sharpen_filter(ksize, sigma, alpha):
    # 1. 取得 Gaussian Kernel
    G = get_gaussian_kernel(ksize, sigma)
    
    # 2. 建立 Unit Impulse (Identity) Kernel
    # 大小必須跟 Gaussian 一樣
    unit_impulse = np.zeros((ksize, ksize))
    center = ksize // 2
    unit_impulse[center, center] = 1.0
    
    # 3. 根據公式組合: (1 + alpha) * e - alpha * G
    sharpen_filter = (1 + alpha) * unit_impulse - alpha * G
    
    return sharpen_filter

# ==========================================
# 任務 1: 銳化模糊圖片
# ==========================================
# img_path = 'bro.webp' 
img_path = 'test.png' 
img = load_and_gray(img_path)

# ==========================================
# 評估函數：使用拉普拉斯方差評估銳化質量
# ==========================================
def evaluate_sharpness(img_sharpened):
    """
    使用拉普拉斯方差 (Laplacian Variance) 評估圖像銳化程度
    數值越高表示圖像越銳利（邊緣越清晰）
    """
    # 拉普拉斯算子 (Laplacian Kernel)
    laplacian_kernel = np.array([[0, -1, 0],
                                 [-1, 4, -1],
                                 [0, -1, 0]])
    
    # 計算拉普拉斯響應
    laplacian_response = convolve2d(img_sharpened, laplacian_kernel, mode='same', boundary='symm')
    
    # 計算方差（方差越大，表示邊緣越明顯，圖像越銳利）
    variance = np.var(laplacian_response)
    return variance

def find_best_parameters(img, ksize=15, sigma_range=(0, 5, 0.2), alpha_range=(0, 5, 0.2)):
    """
    在給定的參數範圍內搜索最佳的 sigma 和 alpha 組合
    
    參數:
        img: 輸入圖像
        ksize: Gaussian 視窗大小
        sigma_range: (start, end, step) sigma 的搜索範圍
        alpha_range: (start, end, step) alpha 的搜索範圍
    
    返回:
        best_sigma: 最佳 sigma 值
        best_alpha: 最佳 alpha 值
        best_score: 最佳評估分數
        results: 所有參數組合的結果字典
    """
    sigma_start, sigma_end, sigma_step = sigma_range
    alpha_start, alpha_end, alpha_step = alpha_range
    
    # 生成參數網格
    sigma_values = np.arange(sigma_start, sigma_end + sigma_step/2, sigma_step)
    alpha_values = np.arange(alpha_start, alpha_end + alpha_step/2, alpha_step)
    
    best_sigma = None
    best_alpha = None
    best_score = -1
    results = []
    
    print(f"開始搜索最佳參數...")
    print(f"Sigma 範圍: {sigma_start} 到 {sigma_end} (間隔 {sigma_step})")
    print(f"Alpha 範圍: {alpha_start} 到 {alpha_end} (間隔 {alpha_step})")
    print(f"總共需要測試 {len(sigma_values) * len(alpha_values)} 個組合\n")
    
    total_combinations = len(sigma_values) * len(alpha_values)
    current = 0
    
    for sigma in sigma_values:
        for alpha in alpha_values:
            current += 1
            # 建立銳化濾波器
            sharpen_kernel = create_sharpen_filter(ksize, sigma, alpha)
            
            # 執行銳化
            img_sharpened = convolve2d(img, sharpen_kernel, mode='same', boundary='symm')
            img_sharpened = np.clip(img_sharpened, 0, 255)
            
            # 評估銳化質量
            score = evaluate_sharpness(img_sharpened)
            
            results.append({
                'sigma': sigma,
                'alpha': alpha,
                'score': score
            })
            
            # 更新最佳參數
            if score > best_score:
                best_score = score
                best_sigma = sigma
                best_alpha = alpha
            
            # 顯示進度
            if current % 50 == 0 or current == total_combinations:
                print(f"進度: {current}/{total_combinations} ({100*current/total_combinations:.1f}%) - "
                      f"當前最佳: sigma={best_sigma:.1f}, alpha={best_alpha:.1f}, score={best_score:.2f}")
    
    print(f"\n搜索完成！")
    print(f"最佳參數: sigma={best_sigma:.1f}, alpha={best_alpha:.1f}")
    print(f"最佳評估分數: {best_score:.2f}\n")
    
    return best_sigma, best_alpha, best_score, results

# 設定參數 (可以調整 alpha 看看效果)
ksize = 15    # Gaussian 視窗大小
sigma = 2    # 模糊程度
alpha = 5.0   # 銳化強度 (越大越銳利，但也越多雜訊)

# 【可選】執行參數搜索：固定 sigma=3，搜索 alpha (0 到 5，間隔 0.2)
# best_sigma, best_alpha, best_score, all_results = find_best_parameters(
#     img, 
#     ksize=ksize,
#     sigma_range=(3, 3, 0.2),  # 固定 sigma=3
#     alpha_range=(0, 5, 0.2)   # 搜索 alpha 從 0 到 5
# )
# sigma, alpha = best_sigma, best_alpha  # 使用找到的最佳參數

# 建立濾波器並執行卷積
sharpen_kernel = create_sharpen_filter(ksize, sigma, alpha)
img_sharpened = convolve2d(img, sharpen_kernel, mode='same', boundary='symm')

# 2. 【關鍵步驟】截斷數值 (Clipping)!!!!!
# 這樣可以保證數值回到合法的圖片範圍，不會讓顯示器混亂
# 如果你的圖片是 0~255 的範圍：
img_sharpened = np.clip(img_sharpened, 0, 255)

# 顯示結果
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1); plt.imshow(img, cmap='gray'); plt.title('Original (Blurry)')
plt.subplot(1, 2, 2); plt.imshow(img_sharpened, cmap='gray'); plt.title(f'Sharpened (alpha={alpha})')
plt.show()

# ==========================================
# 任務 2: Evaluation (清楚 -> 模糊 -> 銳化)
# ==========================================
# 讀取一張清楚的照片
sharp_img_path = 'clear.jpg' # 換成你自己的清楚照片
img_sharp_original = load_and_gray(sharp_img_path)

# 1. 先把它弄模糊
gaussian = get_gaussian_kernel(15, 2) # 用強一點的模糊
img_simulated_blur = convolve2d(img_sharp_original, gaussian, mode='same', boundary='symm')

# 2. 再嘗試把它救回來 (銳化)
# 這裡 alpha 可以設大一點來對抗強模糊
sharpen_kernel_eval = create_sharpen_filter(15, 5, alpha=5) 
img_restored = convolve2d(img_simulated_blur, sharpen_kernel_eval, mode='same', boundary='symm')

# same
img_restored = np.clip(img_restored, 0, 255)

# 比較
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1); plt.imshow(img_sharp_original, cmap='gray'); plt.title('Original Sharp Image')
plt.subplot(1, 3, 2); plt.imshow(img_simulated_blur, cmap='gray'); plt.title('Blurred Version')
plt.subplot(1, 3, 3); plt.imshow(img_restored, cmap='gray'); plt.title('Restored (Re-sharpened)')
plt.show()