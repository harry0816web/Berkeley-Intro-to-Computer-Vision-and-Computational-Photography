import matplotlib.pyplot as plt
import numpy as np
import cv2
import imageio.v3 as iio

def get_points(image, num_points=2):
    """ 顯示圖片並讓使用者點擊 num_points 個點 """
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.title(f"Click {num_points} points (e.g., Left Eye, then Right Eye)")
    plt.axis('off')
    
    # ginput 是一個互動函式，會等待使用者點擊
    # timeout=-1 表示無限等待直到點擊完畢
    points = plt.ginput(num_points, timeout=-1)
    plt.close()
    
    return np.array(points)

def align_images(img1, img2):
    """ 將 img2 對齊到 img1 """
    print("請在第一張圖點兩點 (左眼, 右眼)...")
    pts1 = get_points(img1)
    
    print("請在第二張圖點兩點 (左眼, 右眼)...")
    pts2 = get_points(img2)
    
    # --- 計算變換矩陣 ---
    # 我們使用 OpenCV 的 estimateAffinePartial2D
    # 它會計算最佳的 "旋轉 + 縮放 + 平移" 矩陣，讓 pts2 對齊到 pts1
    # 這裡只需要傳入對應的點座標
    M, _ = cv2.estimateAffinePartial2D(pts2, pts1)
    
    # --- 執行變換 ---
    # dsize 設為 img1 的大小，確保疊圖時尺寸一致
    h, w = img1.shape[:2]
    img2_aligned = cv2.warpAffine(img2, M, (w, h))
    
    return img2_aligned

# === 使用範例 ===
# 1. 讀圖
im1 = iio.imread('cat.png') # 你的圖1
im2 = iio.imread('person.png') # 你的圖2

# 2. 執行對齊
# 這裡會跳出視窗讓你點點，點完視窗會自動關閉
im2_aligned = align_images(im1, im2)

# 3. 檢查結果 (簡單疊加看看)
plt.imshow(im1 * 0.5 + im2_aligned * 0.5)
plt.show()

# 4. 存檔後就可以拿去跑 Hybrid Image 了
iio.imwrite('aligned_img2.png', im2_aligned.astype(np.uint8))