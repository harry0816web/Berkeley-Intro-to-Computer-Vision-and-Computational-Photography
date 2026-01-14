import numpy as np
import matplotlib.pyplot as plt
import cv2
import imageio.v3 as iio

# ==========================================
# 1. 基礎函數 (讀圖與 Stack)
# ==========================================
def load_image(path):
    try:
        img = iio.imread(path)
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[..., :3]
        return img.astype(float) / 255.0
    except:
        return None

def get_gaussian_stack(image, levels=5, sigma_start=2):
    stack = [image]
    for i in range(levels):
        # 每一層 sigma 加倍
        sigma = sigma_start * (2 ** i)
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
        stack.append(blurred)
    return stack

def get_laplacian_stack(gaussian_stack):
    laplacian_stack = []
    for i in range(len(gaussian_stack) - 1):
        detail = gaussian_stack[i] - gaussian_stack[i+1]
        laplacian_stack.append(detail)
    # 最後一層保留低頻殘差
    laplacian_stack.append(gaussian_stack[-1])
    return laplacian_stack

# ==========================================
# 2. 核心：多重解析度混合
# ==========================================
def multiresolution_blending(img1, img2, mask, levels=5, sigma_start=2):
    """
    img1: 左邊/前景圖 (對應 Mask 為 1 的部分)
    img2: 右邊/背景圖 (對應 Mask 為 0 的部分)
    mask: 0~1 的遮罩圖片
    """
    
    # 1. 建立兩張圖的 Laplacian Stacks
    g1 = get_gaussian_stack(img1, levels, sigma_start)
    l1 = get_laplacian_stack(g1)
    
    g2 = get_gaussian_stack(img2, levels, sigma_start)
    l2 = get_laplacian_stack(g2)
    
    # 2. 建立 Mask 的 Gaussian Stack (關鍵！)
    # Mask 必須跟著一起變模糊，這樣低頻才會融合得平滑
    mask_g_stack = get_gaussian_stack(mask, levels, sigma_start)
    
    # 3. 每一層分別混合
    blended_stack = []
    for i in range(len(l1)):
        # Formula: L_blend = GM * L1 + (1 - GM) * L2
        # 注意: Mask 可能是單通道，若圖片是 RGB 需擴展維度
        m = mask_g_stack[i]
        if img1.ndim == 3 and m.ndim == 2:
            m = np.stack([m, m, m], axis=-1)
            
        layer_blend = m * l1[i] + (1 - m) * l2[i]
        blended_stack.append(layer_blend)
        
    # 4. 疊加重建 (Collapse the stack)
    final_image = np.zeros_like(img1)
    for layer in blended_stack:
        final_image += layer
        
    return np.clip(final_image, 0, 1), blended_stack, mask_g_stack, l1, l2

# ==========================================
# 3. 主執行區塊
# ==========================================
# 載入圖片 (請確保尺寸一致，或是程式內裁切)
path1 = 'apple.png'  # 蘋果
path2 = 'orange.png' # 橘子

im1 = load_image(path1)
im2 = load_image(path2)

if im1 is not None and im2 is not None:
    # 簡單裁切確保尺寸一致
    h = min(im1.shape[0], im2.shape[0])
    w = min(im1.shape[1], im2.shape[1])
    im1 = im1[:h, :w]
    im2 = im2[:h, :w]
    
    # --- A. 建立垂直遮罩 (Vertical Seam Mask) ---
    # 左半邊是 1 (Apple)，右半邊是 0 (Orange)
    mask = np.zeros((h, w), dtype=float)
    mask[:, :w//2] = 1.0
    
    # --- B. 執行混合 ---
    levels = 5
    sigma = 2
    result, blend_stack, mask_stack, l1, l2 = multiresolution_blending(im1, im2, mask, levels, sigma)
    
    # --- C. 顯示結果 (The Oraple) ---
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1); plt.imshow(im1); plt.title("Apple")
    plt.subplot(1, 3, 2); plt.imshow(im2); plt.title("Orange")
    plt.subplot(1, 3, 3); plt.imshow(result); plt.title("The Oraple (Multiresolution Blend)")
    plt.show()
    
    # --- D. 視覺化過程 (類似論文 Figure 10) ---
    # 畫出前 3 層的 Laplacian 和混合結果
    show_levels = 3
    plt.figure(figsize=(12, 8))
    
    for i in range(show_levels):
        # Mask 模糊程度
        curr_mask = mask_stack[i]
        
        # 顯示 Layer i 的混合狀況
        # 左: Apple Detail masked, 中: Orange Detail masked, 右: Blended Detail
        
        # 視覺化調整: 加 0.5 讓細節清楚顯示
        layer_apple = (l1[i] * np.stack([curr_mask]*3, axis=-1)) + 0.5
        layer_orange = (l2[i] * (1 - np.stack([curr_mask]*3, axis=-1))) + 0.5
        layer_blend = blend_stack[i] + 0.5
        
        plt.subplot(show_levels, 3, i*3 + 1)
        plt.imshow(np.clip(layer_apple, 0, 1))
        plt.ylabel(f"Level {i}")
        if i == 0: plt.title("Apple Laplacian * Mask")
        plt.axis('off')

        plt.subplot(show_levels, 3, i*3 + 2)
        plt.imshow(np.clip(layer_orange, 0, 1))
        if i == 0: plt.title("Orange Laplacian * (1-Mask)")
        plt.axis('off')

        plt.subplot(show_levels, 3, i*3 + 3)
        plt.imshow(np.clip(layer_blend, 0, 1))
        if i == 0: plt.title("Blended Laplacian")
        plt.axis('off')
        
    plt.suptitle("Blending Process Visualization (Levels 0-2)")
    plt.tight_layout()
    plt.show()

    # 存檔
    iio.imwrite("oraple_result.jpg", (result * 255).astype(np.uint8))
    print("Oraple saved!")