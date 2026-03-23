import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio

# ==========================================
# 第一部分：A.2 基礎工具 (選點與計算 H)
# ==========================================

def get_single_image_points(im, n_points=4):
    """
    互動式工具：在單張圖片上點選角點
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(im)
    plt.title(f"請依序點選 {n_points} 個角落 (建議順序: 左上->右上->右下->左下)\n點完後請按 Enter 或中鍵")
    plt.axis('off')
    # timeout=0 表示無限等待直到使用者操作完成
    pts = plt.ginput(n=n_points, timeout=0)
    plt.close()
    
    if len(pts) != n_points:
        raise ValueError(f"您需要點選正好 {n_points} 個點，但您點了 {len(pts)} 個。")
        
    return np.array(pts)

def computeH(im1_pts, im2_pts):
    """
    計算單應性矩陣 H，使得 p' = H p
    使用最小平方法解 Ah = b
    """
    num_pts = im1_pts.shape[0]
    if num_pts < 4:
        raise ValueError("至少需要 4 組對應點才能計算 Homography")

    A = []
    b = []

    for i in range(num_pts):
        x, y = im1_pts[i]
        xp, yp = im2_pts[i] # p' (target)

        # Row 1: ax + by + c - gxx' - hyx' = x'
        A.append([x, y, 1, 0, 0, 0, -x*xp, -y*xp])
        b.append(xp)
        # Row 2: dx + ey + f - gxy' - hyy' = y'
        A.append([0, 0, 0, x, y, 1, -x*yp, -y*yp])
        b.append(yp)

    A = np.array(A)
    b = np.array(b)

    # 解線性系統 Ah = b，rcond=None 用於處理奇異矩陣
    h, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    # 重組 H 矩陣 (補上最後一個元素 1)
    H = np.append(h, 1).reshape(3, 3)
    return H

# ==========================================
# 第二部分：A.3 核心變形工具 (插值與 Inverse Warping)
# ==========================================

def interpolate_nearest_neighbor(img, x, y):
    """
    最近鄰插值
    """
    x_nearest = np.round(x).astype(int)
    y_nearest = np.round(y).astype(int)
    
    h, w = img.shape[:2]
    # 建立遮罩，找出仍在原始圖片範圍內的點
    valid_mask = (x_nearest >= 0) & (x_nearest < w) & (y_nearest >= 0) & (y_nearest < h)
    
    out = np.zeros_like(x, dtype=img.dtype)
    if img.ndim == 3:
        out = np.zeros((x.shape[0], x.shape[1], img.shape[2]), dtype=img.dtype)
    
    # 填值
    out[valid_mask] = img[y_nearest[valid_mask], x_nearest[valid_mask]]
    return out

def interpolate_bilinear(img, x, y):
    """
    雙線性插值 (自行實作版)
    """
    h, w = img.shape[:2]
    
    # 1. 找出四周整數座標
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1
    
    # 2. 邊界檢查：只要碰到邊界就算出界 (簡化處理)
    valid_mask = (x0 >= 0) & (x1 < w) & (y0 >= 0) & (y1 < h)
    
    # 3. 計算權重小數
    a = x - x0
    b = y - y0
    
    # 只選取有效區域進行運算
    x0_v, x1_v = x0[valid_mask], x1[valid_mask]
    y0_v, y1_v = y0[valid_mask], y1[valid_mask]
    a_v, b_v = a[valid_mask], b[valid_mask]
    
    # 支援彩色廣播運算
    if img.ndim == 3:
        a_v = a_v[..., np.newaxis]
        b_v = b_v[..., np.newaxis]
    
    # 4. 取值
    Ia = img[y0_v, x0_v] # Top-left
    Ib = img[y1_v, x0_v] # Bottom-left
    Ic = img[y0_v, x1_v] # Top-right
    Id = img[y1_v, x1_v] # Bottom-right
    
    # 5. 插值公式
    top = (1 - a_v) * Ia + a_v * Ic
    bottom = (1 - a_v) * Ib + a_v * Id
    pixel_values = (1 - b_v) * top + b_v * bottom
    
    # 6. 填回輸出
    out = np.zeros((x.shape[0], x.shape[1], img.shape[2] if img.ndim==3 else 1), dtype=img.dtype)
    out[valid_mask] = pixel_values
    
    # 確保型別正確 (若是 uint8 需要截斷)
    if img.dtype == np.uint8:
        out = np.clip(out, 0, 255).astype(np.uint8)
        
    return out

def warpImage(im, H, output_shape=None, interpolation='bilinear'):
    """
    主變形函式：執行 Inverse Warping
    """
    h, w = im.shape[:2]
    
    # 1. 決定輸出畫布大小與範圍
    if output_shape is None:
        # 如果沒指定，就自動計算變形後的 Bounding Box
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]]).T
        corners_homo = np.vstack([corners, np.ones((1, 4))])
        warped_corners = H @ corners_homo
        warped_corners /= (warped_corners[2, :] + 1e-8) # 避免除零
        
        min_x = np.floor(np.min(warped_corners[0, :])).astype(int)
        max_x = np.ceil(np.max(warped_corners[0, :])).astype(int)
        min_y = np.floor(np.min(warped_corners[1, :])).astype(int)
        max_y = np.ceil(np.max(warped_corners[1, :])).astype(int)
        
        out_w = max_x - min_x
        out_h = max_y - min_y
        offset_x, offset_y = min_x, min_y
    else:
        # 如果有指定 (例如 Rectification)，通常希望左上角在 (0,0)
        out_h, out_w = output_shape
        offset_x, offset_y = 0, 0
        max_x, max_y = out_w, out_h # 用於生成 meshgrid

    print(f"Output image shape: {out_h}x{out_w}, Offset: ({offset_x}, {offset_y})")

    # 2. 產生目標圖片的網格座標 (Meshgrid)
    # 這裡產生的座標是相對於整個世界的座標
    xv, yv = np.meshgrid(np.arange(offset_x, offset_x + out_w), 
                         np.arange(offset_y, offset_y + out_h))
    
    # 3. Inverse Warping: 映射回源圖片座標
    ones = np.ones_like(xv)
    coords = np.stack([xv, yv, ones]) # 3 x H_out x W_out
    coords_flat = coords.reshape(3, -1)
    
    # 關鍵：計算反矩陣
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        print("Error: H matrix is singular and cannot be inverted.")
        return np.zeros_like(im)

    # 映射回去
    src_coords_homo = H_inv @ coords_flat
    
    # 歸一化 (除以 w)
    src_x = src_coords_homo[0, :] / (src_coords_homo[2, :] + 1e-8)
    src_y = src_coords_homo[1, :] / (src_coords_homo[2, :] + 1e-8)
    
    # Reshape 回網格形狀
    src_x = src_x.reshape(out_h, out_w)
    src_y = src_y.reshape(out_h, out_w)
    
    # 4. 插值取色
    if interpolation == 'nearest':
        print("Performing Nearest Neighbor Interpolation...")
        out_img = interpolate_nearest_neighbor(im, src_x, src_y)
    else:
        print("Performing Bilinear Interpolation...")
        out_img = interpolate_bilinear(im, src_x, src_y)
        
    return out_img

# ==========================================
# 第三部分：主執行流程 (Rectification Demo)
# ==========================================

if __name__ == "__main__":
    # 1. 設定圖片路徑 (請替換成你的圖片)
    image_path = 'data/warp-ipad.jpeg'

    try:
        # 讀取圖片
        im = iio.imread(image_path)
        print(f"成功讀取圖片: {image_path}, 大小: {im.shape}")

        # 2. 步驟 A：獲取來源點 (使用者點選)
        print("--- 開始點選角點 ---")
        pts_src = get_single_image_points(im)
        print("來源點座標 (Source Points):\n", pts_src)

        # 3. 步驟 B：定義目標點 (完美的矩形)
        # 這裡我們定義一個寬 300，高 400 的矩形
        # 順序必須與點選順序一致 (TL -> TR -> BR -> BL)
        target_w, target_h = 300, 400
        pts_dst = np.array([
            [0, 0],            # 左上
            [target_w, 0],     # 右上
            [target_w, target_h], # 右下
            [0, target_h]      # 左下
        ], dtype=np.float32)
        print("目標點座標 (Destination Points):\n", pts_dst)

        # 4. 步驟 C：計算 H 矩陣
        H = computeH(pts_src, pts_dst)
        print("計算出的 Homography 矩陣 H:\n", H)

        # 5. 步驟 D：執行變形 (Warping)
        # 我們分別用兩種插值法跑一次來比較
        rectified_nn = warpImage(im, H, output_shape=(target_h, target_w), interpolation='nearest')
        rectified_bil = warpImage(im, H, output_shape=(target_h, target_w), interpolation='bilinear')

        # 6. 顯示結果比較
        plt.figure(figsize=(15, 10))
        
        # 原圖與點位
        plt.subplot(1, 3, 1)
        plt.imshow(im)
        plt.scatter(pts_src[:, 0], pts_src[:, 1], c='r', s=40, marker='x')
        for i, (x, y) in enumerate(pts_src):
            plt.text(x+5, y+5, str(i+1), color='yellow', fontsize=12)
        plt.title("Original Image with Input Points")
        plt.axis('off')

        # 最近鄰插值結果
        plt.subplot(1, 3, 2)
        plt.imshow(rectified_nn)
        plt.title(f"Rectified (Nearest Neighbor)\n{target_w}x{target_h}")
        plt.axis('off')

        # 雙線性插值結果
        plt.subplot(1, 3, 3)
        plt.imshow(rectified_bil)
        plt.title(f"Rectified (Bilinear)\n{target_w}x{target_h}")
        plt.axis('off')

        plt.tight_layout()
        print("顯示結果中...")
        plt.show()

    except FileNotFoundError:
        print(f"錯誤：找不到圖片 '{image_path}'。請確保圖片檔案在正確的目錄下。")
    except Exception as e:
        print(f"發生錯誤: {e}")