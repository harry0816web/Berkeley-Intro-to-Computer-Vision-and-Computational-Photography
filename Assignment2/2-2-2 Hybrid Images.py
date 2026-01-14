import numpy as np
import matplotlib.pyplot as plt
import cv2
import imageio.v3 as iio

# ==========================================
# 1. 基礎工具：讀圖 (強制轉灰階) 與 FFT 顯示
# ==========================================
def load_image_gray(path):
    """
    讀取圖片並強制轉為灰階 (2D array)
    回傳值範圍歸一化到 0~1
    """
    try:
        # 讀取圖片
        img = iio.imread(path)
        
        # 判斷是否為彩色 (3維陣列: H, W, Channels)
        if img.ndim == 3:
            # 如果是 RGBA (4通道)，先去掉 Alpha 通道
            if img.shape[2] == 4:
                img = img[..., :3]
            
            # 使用亮度公式轉灰階: 0.299*R + 0.587*G + 0.114*B
            img = np.dot(img[..., :3], [0.299, 0.587, 0.114])
        
        # 如果原本就是灰階 (2維陣列)，則不需處理直接正規化
        
        return img.astype(float) / 255.0
        
    except Exception as e:
        print(f"讀取圖片失敗 {path}: {e}")
        return None

def vis_fft(image, title="FFT"):
    """
    作業要求的頻率分析視覺化
    """
    # 因為輸入已經保證是灰階，這裡直接算 FFT 即可
    fft_mag = np.log(np.abs(np.fft.fftshift(np.fft.fft2(image))) + 1e-8)
    
    plt.imshow(fft_mag, cmap='gray')
    plt.title(title)
    plt.axis('off')

# ==========================================
# 2. 核心：Hybrid Image 生成
# ==========================================
def create_hybrid_image(img1, img2, sigma1, sigma2):
    """
    img1: 低頻成分 (Blur)
    img2: 高頻成分 (Original - Blur)
    """
    
    # --- 1. Low-pass Filter ---
    # 對灰階圖做高斯模糊
    low_pass = cv2.GaussianBlur(img1, (0, 0), sigmaX=sigma1, sigmaY=sigma1)
    
    # --- 2. High-pass Filter ---
    img2_blurred = cv2.GaussianBlur(img2, (0, 0), sigmaX=sigma2, sigmaY=sigma2)
    high_pass = img2 - img2_blurred
    
    # --- 3. Combine ---
    hybrid = 0.5 * low_pass + 0.5 * high_pass
    
    # 數值截斷 (Clipping) 確保在 0~1 之間
    hybrid = np.clip(hybrid, 0, 1)
    
    return low_pass, high_pass, hybrid

# ==========================================
# 3. 主執行區塊
# ==========================================
# 請確認這兩個檔案存在，或者換成你對齊好的圖片
img_low_path = 'aligned_person.png'   
img_high_path = 'cat.png' 

# 載入圖片 (使用新寫的灰階函數)
im1 = load_image_gray(img_low_path)
im2 = load_image_gray(img_high_path)

if im1 is not None and im2 is not None:
    # 確保兩張圖尺寸一樣 (簡單裁切至最小值)
    h = min(im1.shape[0], im2.shape[0])
    w = min(im1.shape[1], im2.shape[1])
    im1 = im1[:h, :w]
    im2 = im2[:h, :w]

    # --- 參數調整 ---
    # 灰階圖片的頻率響應可能跟彩色略有不同，建議重新微調 sigma
    sigma1 = 5  # 低頻模糊程度
    sigma2 = 5   # 高頻過濾程度

    low_pass, high_pass, hybrid = create_hybrid_image(im1, im2, sigma1, sigma2)

    # --- 顯示結果 ---
    plt.figure(figsize=(15, 10))
    
    # Row 1: 圖片效果 (注意這裡都要加上 cmap='gray')
    plt.subplot(2, 3, 1)
    plt.imshow(im1, cmap='gray')
    plt.title("Original Low (Grayscale)")
    
    plt.subplot(2, 3, 2)
    plt.imshow(im2, cmap='gray')
    plt.title("Original High (Grayscale)")
    
    plt.subplot(2, 3, 3)
    plt.imshow(hybrid, cmap='gray')
    plt.title("Hybrid Image")
    
    # Row 2: 頻率分析
    plt.subplot(2, 3, 4); vis_fft(im1, "FFT: Input 1")
    plt.subplot(2, 3, 5); vis_fft(high_pass, "FFT: High-pass Component")
    plt.subplot(2, 3, 6); vis_fft(hybrid, "FFT: Hybrid Result")
    
    plt.tight_layout()
    plt.show()

    # 存檔 (必須指定 cmap='gray' 或是轉成 uint8 後儲存)
    # 這裡將 0~1 的 float 轉為 0~255 的 uint8 存檔
    iio.imwrite("hybrid_result_gray.png", (hybrid * 255).astype(np.uint8))
    print("Hybrid image saved as hybrid_result_gray.png")