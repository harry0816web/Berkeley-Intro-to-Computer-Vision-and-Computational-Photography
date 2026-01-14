import numpy as np
import matplotlib.pyplot as plt
import cv2
import imageio.v3 as iio

# ==========================================
# 1. 讀取彩色圖片 (不轉灰階)
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
# 2. 核心：支援色彩選擇的 Hybrid Image
# ==========================================
def create_hybrid_color(img1, img2, sigma1, sigma2, color_mode='low'):
    """
    color_mode: 
      'low'  -> img1 (低頻) 彩色, img2 (高頻) 灰階 (推薦!)
      'high' -> img1 (低頻) 灰階, img2 (高頻) 彩色
      'both' -> 兩者皆彩色
    """
    
    # 內部小函數：轉灰階並疊回 3 通道 (H, W, 3) 以便相加
    def to_gray_3ch(img):
        if img.ndim == 3:
            # RGB 轉灰階公式
            gray = np.dot(img[..., :3], [0.299, 0.587, 0.114])
            # 疊加 3 層變成 (H, W, 3)
            return np.stack([gray, gray, gray], axis=-1)
        return img

    # --- 根據模式準備圖片 ---
    if color_mode == 'low':
        input1 = img1  # 彩色
        input2 = to_gray_3ch(img2) # 灰階
    elif color_mode == 'high':
        input1 = to_gray_3ch(img1) # 灰階
        input2 = img2 # 彩色
    elif color_mode == 'both':
        input1 = img1 # 彩色
        input2 = img2 # 彩色
    else: # 預設全灰階
        input1 = to_gray_3ch(img1)
        input2 = to_gray_3ch(img2)

    # --- 1. Low-pass Filter (對 input1) ---
    # cv2.GaussianBlur 支援彩色 (會分別對 RGB 做模糊)
    low_pass = cv2.GaussianBlur(input1, (0, 0), sigmaX=sigma1, sigmaY=sigma1)
    
    # --- 2. High-pass Filter (對 input2) ---
    img2_blurred = cv2.GaussianBlur(input2, (0, 0), sigmaX=sigma2, sigmaY=sigma2)
    high_pass = input2 - img2_blurred
    
    # --- 3. Combine ---
    hybrid = low_pass + high_pass
    hybrid = np.clip(hybrid, 0, 1)
    
    return hybrid

# ==========================================
# 3. 執行與比較
# ==========================================
# 載入你對齊好的彩色圖片
# 注意：這裡要用彩色讀取函數，不要用上一段的灰階函數
img_low_path = 'aligned_person.png'   
img_high_path = 'cat.png' # 建議使用你之前存的 aligned_img2.jpg

im1 = load_image_color(img_low_path)
im2 = load_image_color(img_high_path)

if im1 is not None and im2 is not None:
    # 簡單對齊尺寸
    h = min(im1.shape[0], im2.shape[0])
    w = min(im1.shape[1], im2.shape[1])
    im1 = im1[:h, :w]
    im2 = im2[:h, :w]

    # 設定參數 (沿用你覺得效果最好的)
    sigma1 = 8
    sigma2 = 8

    # --- 生成三種版本 ---
    res_color_low = create_hybrid_color(im1, im2, sigma1, sigma2, color_mode='low')
    res_color_high = create_hybrid_color(im1, im2, sigma1, sigma2, color_mode='high')
    res_color_both = create_hybrid_color(im1, im2, sigma1, sigma2, color_mode='both')

    # --- 顯示與存檔 ---
    plt.figure(figsize=(15, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(res_color_low)
    plt.title("Color Low + Gray High (Best?)")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(res_color_high)
    plt.title("Gray Low + Color High")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(res_color_both)
    plt.title("Color Both")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # 存檔 (你可以選一張最喜歡的放進報告)
    iio.imwrite("hybrid_color_low.png", (res_color_low * 255).astype(np.uint8))
    print("Done! Saved hybrid_color_low.png")