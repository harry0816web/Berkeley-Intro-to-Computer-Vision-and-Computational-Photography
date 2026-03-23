import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
from scipy.signal import convolve2d
from skimage import data  # 用於備用圖片

# ==========================================
# 1. 基礎工具函數 (Padding & Loading)
# ==========================================

def zero_pad(image, pad_height, pad_width):
    """ 幫圖片周圍補零 """
    H, W = image.shape
    padded = np.zeros((H + 2 * pad_height, W + 2 * pad_width))
    padded[pad_height:pad_height+H, pad_width:pad_width+W] = image
    return padded

def load_img(path):
    """
    讀取圖片並轉為灰階 (2D float array)
    """
    try:
        # 讀取圖片 (通常是 RGB)
        img = iio.imread(path)
        print(f"[Info] 成功讀取圖片: {path}, 原始形狀: {img.shape}")
    except Exception as e:
        print(f"[Warning] 讀取失敗 ({e})，改用內建 Cameraman 圖片。")
        img = data.camera()

    # 如果是彩色 (H, W, 3)，轉為灰階 (H, W)
    if img.ndim == 3:
        # 使用標準權重: 0.299 R + 0.587 G + 0.114 B
        img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
    
    # 轉為 float 並正規化 (0~255 區間)
    return img.astype(float)

# ==========================================
# 2. 你的實作 (Part 1.1)
# ==========================================

def conv_nested(image, kernel):
    """ 4層迴圈版本 (慢，但在 C++ 中是基礎) """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    Ho, Wo = Hi, Wi
    
    # 準備 Padding
    pad_h, pad_w = Hk // 2, Wk // 2
    image_pad = zero_pad(image, pad_h, pad_w)
    output = np.zeros((Ho, Wo))
    
    # *** 關鍵：Flip Kernel 以符合捲積定義 ***
    # 如果沒有flip kernel，就是correlation，但還是不太懂
    kernel_flipped = np.flip(kernel)
    
    # 4 Loops
    for i in range(Ho):
        for j in range(Wo):
            val = 0
            for m in range(Hk):
                for n in range(Wk):
                    # i, j represent image patch
                    val += image_pad[i + m, j + n] * kernel_flipped[m, n]
            output[i, j] = val
            
    return output

def conv_fast(image, kernel):
    """ 2層迴圈版本 (使用 Numpy Slicing / Patch) """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    Ho, Wo = Hi, Wi
    
    pad_h, pad_w = Hk // 2, Wk // 2
    image_pad = zero_pad(image, pad_h, pad_w)
    output = np.zeros((Ho, Wo))
    
    kernel_flipped = np.flip(kernel)
    
    # 2 Loops (只跑圖片位置)
    for i in range(Ho):
        for j in range(Wo):
            # 取出與 Kernel 同大小的 Patch
            # i從 cur_i -> cur_i + kernel height, j從 cur_j -> cur_j + kernel width
            patch = image_pad[i : i + Hk, j : j + Wk]
            # Element-wise 乘法後加總
            output[i, j] = np.sum(patch * kernel_flipped)
            
    return output

# ==========================================
# 3. 驗證函數
# ==========================================

def verify_with_scipy(image, kernel, my_result, filter_name):
    """ 
    呼叫 scipy.signal.convolve2d 並計算誤差 
    """
    # mode='same' 維持大小, boundary='fill' 補零
    scipy_result = convolve2d(image, kernel, mode='same', boundary='fill')
    
    # 計算絕對誤差總和
    diff = np.sum(np.abs(my_result - scipy_result))
    
    print(f"--- 驗證 {filter_name} ---")
    if diff < 1e-5:
        print(f"✅ 通過! 誤差極小: {diff:.2e}")
    else:
        print(f"❌ 失敗! 誤差過大: {diff:.2e}")
        
    return scipy_result

# ==========================================
# 4. Main 執行區
# ==========================================

if __name__ == "__main__":
    # --- A. 設定參數 ---
    my_image_path = "test.png"  # <--- 請確保這裡有你的圖片，或是程式會自動用備用圖
    
    # --- B. 載入圖片 ---
    img = load_img(my_image_path)

    # --- C. 定義 Filters (依據截圖要求) ---
    # 1. Box Filter (9x9)
    box_filter = np.ones((9, 9)) / 81.0
    
    # 2. Dx (Central Difference) -> shape (1, 3)
    Dx = np.array([[1, 0, -1]])
    
    # 3. Dy (Transpose of Dx) -> shape (3, 1)
    Dy = np.array([[1], [0], [-1]])

    # --- D. 執行捲積與驗證 ---
    print("\n開始執行捲積運算 (使用 conv_fast)...")
    
    # 1. Box Filter
    img_box = conv_fast(img, box_filter)
    verify_with_scipy(img, box_filter, img_box, "Box Filter")

    # 2. Dx
    img_dx = conv_fast(img, Dx)
    verify_with_scipy(img, Dx, img_dx, "Dx Filter")

    # 3. Dy
    img_dy = conv_fast(img, Dy)
    verify_with_scipy(img, Dy, img_dy, "Dy Filter")

    # --- E. 視覺化結果 ---
    plt.figure(figsize=(15, 5))

    titles = ['Original', 'Box Blurred', 'Dx (Vertical Edge)', 'Dy (Horizontal Edge)']
    images = [img, img_box, img_dx, img_dy]

    for i in range(4):
        plt.subplot(1, 4, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()
    plt.show()