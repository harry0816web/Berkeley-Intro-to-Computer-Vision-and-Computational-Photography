import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
from scipy.ndimage import distance_transform_edt

# 引入你之前寫好的函式 (假設它們在同一個檔案或已定義)
# from your_script import computeH, warpImage, get_points

def get_points(im1, im2, n_points=4):
    """
    互動式工具：讓使用者在兩張圖上點選對應點
    """
    print(f"請在第一張圖點選 {n_points} 個點，然後按 Enter 或中鍵結束...")
    plt.imshow(im1)
    plt.axis('off')
    pts1 = plt.ginput(n=n_points, timeout=0)
    plt.close()

    print(f"請在第二張圖依序點選對應的 {n_points} 個點...")
    plt.imshow(im2)
    plt.axis('off')
    pts2 = plt.ginput(n=n_points, timeout=0)
    plt.close()

    return np.array(pts1), np.array(pts2)

def computeH(im1_pts, im2_pts):
    """
    計算單應性矩陣 H，使得 p' = H p
    im1_pts: N x 2 矩陣，來源圖片座標 (x, y)
    im2_pts: N x 2 矩陣，目標圖片座標 (x', y')
    """
    num_pts = im1_pts.shape[0]
    if num_pts < 4:
        raise ValueError("至少需要 4 組對應點才能計算 Homography")

    # 1. 建構 A 矩陣 (2n * 8) 和 b 向量 (2n)
    A = []
    b = []

    for i in range(num_pts):
        x, y = im1_pts[i]
        xp, yp = im2_pts[i] # p' (target)

        # 根據公式填入兩列
        # Row 1: ax + by + c - gxx' - hyx' = x'
        A.append([x, y, 1, 0, 0, 0, -x*xp, -y*xp])
        b.append(xp)

        # Row 2: dx + ey + f - gxy' - hyy' = y'
        A.append([0, 0, 0, x, y, 1, -x*yp, -y*yp])
        b.append(yp)

    A = np.array(A)
    b = np.array(b)

    # 2. 解線性系統 Ah = b
    # 使用最小平方法 (Least Squares)
    h, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

    # 3. 重組 H 矩陣
    # h 是一個長度為 8 的向量 [a, b, c, d, e, f, g, h]
    # 我們需要把它變成 3x3，並補上最後一個元素 1
    H = np.append(h, 1).reshape(3, 3)

    return H

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

def get_canvas_dimensions(image1, image2, H):
    """
    計算拼接後的畫布大小和平移矩陣
    """
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    
    # Im1 的四個角 (還沒動)
    corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    
    # Im2 的四個角 (經過 H 變換)
    corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    
    # 手動投影 corners2 (不依賴 cv2.perspectiveTransform)
    # H (3x3) * point (3x1)
    corners2_trans = []
    for pt in corners2:
        x, y = pt[0]
        v = np.dot(H, np.array([x, y, 1]))
        v = v / v[2] # 歸一化
        corners2_trans.append([[v[0], v[1]]])
    corners2_trans = np.array(corners2_trans)

    # 合併所有角落，找出最大範圍
    all_corners = np.concatenate((corners1, corners2_trans), axis=0)
    
    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    
    # 計算平移量 (如果 xmin 是負的，就要往右移)
    translation_dist = [-xmin, -ymin]
    
    # 建立平移矩陣 T
    H_translation = np.array([
        [1, 0, translation_dist[0]],
        [0, 1, translation_dist[1]],
        [0, 0, 1]
    ])
    
    # 畫布大小
    output_shape = (ymax - ymin, xmax - xmin)
    
    return output_shape, H_translation

def make_weight_mask(img):
    """
    產生權重遮罩：中心為 1，邊緣為 0
    使用 Distance Transform 產生平滑過渡
    """
    if img.ndim == 3:
        # 轉成灰階來算遮罩 (只要有像素的地方就是 True)
        gray = np.any(img > 0, axis=2)
    else:
        gray = img > 0
        
    # distance_transform_edt 計算每個像素到最近的「零像素」的距離
    # 這樣圖片中心的距離最大，邊緣最小
    mask = distance_transform_edt(gray)
    
    # 歸一化到 0~1
    if mask.max() > 0:
        mask = mask / mask.max()
        
    return mask

def stitch_images(img1, img2, pts1, pts2):
    """
    拼接兩張圖片
    img1: 基準圖 (不會變形，只會平移)
    img2: 變形圖 (會變形對齊 img1)
    pts1, pts2: 對應點
    """
    print("1. 計算 Homography...")
    # 計算 Im2 -> Im1 的變換 (注意方向：我們要將 Im2 變形成 Im1 的視角)
    # 所以 pts2 (source) -> pts1 (dest)
    H = computeH(pts2, pts1)
    
    print("2. 計算畫布大小與平移矩陣...")
    output_shape, T = get_canvas_dimensions(img1, img2, H)
    out_h, out_w = output_shape
    print(f"   畫布大小: {out_w}x{out_h}")
    
    print("3. 執行 Warping...")
    # 變形 Im1 (只需要平移 T)
    # 注意：這裡使用我們自己寫的 warpImage
    # T 是平移矩陣，我們將 img1 放到畫布的正確位置
    warped_img1 = warpImage(img1, T, output_shape=(out_h, out_w))
    
    # 變形 Im2 (先 H 再 T -> T * H)
    H_final = np.dot(T, H)
    warped_img2 = warpImage(img2, H_final, output_shape=(out_h, out_w))
    
    # ---- 新增：把選的點也變到全景圖座標 ----
    # pts1 在 img1，經過平移 T
    pts1_h = np.hstack([pts1, np.ones((pts1.shape[0], 1))])          # N x 3
    pts1_on_canvas_h = (T @ pts1_h.T).T                              # N x 3
    pts1_on_canvas = pts1_on_canvas_h[:, :2] / pts1_on_canvas_h[:, 2:3]

    # pts2 在 img2，經過 H_final = T * H
    pts2_h = np.hstack([pts2, np.ones((pts2.shape[0], 1))])
    pts2_on_canvas_h = (H_final @ pts2_h.T).T
    pts2_on_canvas = pts2_on_canvas_h[:, :2] / pts2_on_canvas_h[:, 2:3]
    # ----------------------------------------

    print("4. 執行 Blending (Weighted Averaging)...")
    # 產生遮罩
    mask1 = make_weight_mask(warped_img1)
    mask2 = make_weight_mask(warped_img2)
    
    # 疊加 (Alpha Blending Logic)
    # output = (img1 * w1 + img2 * w2) / (w1 + w2)
    # 為了避免除以 0，分母加一個小數 epsilon
    numerator = (warped_img1.astype(float) * mask1[..., None] + 
                 warped_img2.astype(float) * mask2[..., None])
    denominator = (mask1[..., None] + mask2[..., None]) + 1e-8
    
    mosaic = numerator / denominator
    
    # 轉回 uint8
    mosaic = np.clip(mosaic, 0, 255).astype(np.uint8)
    
    # 回傳：結果影像、兩張 warped 圖、對應點在全景圖中的座標，以及對應的 alpha mask
    return mosaic, warped_img1, warped_img2, pts1_on_canvas, pts2_on_canvas, mask1, mask2

# ==========================================
# 主程式：執行拼接
# ==========================================
# 1-4 Blend the Images into a Mosaic.py
if __name__ == "__main__":
    # 1. 讀取兩張有重疊的照片
    # 建議先用手機拍兩張：左邊一張 (img1)，右邊一張 (img2)
    # 確保 img1 是你想保留視角的基準圖
    try:
        im1 = iio.imread('../data/panorama1-left.jpeg')  # 請準備圖片
        im2 = iio.imread('../data/panorama1-right.jpeg')
        
        # 2. 選點 (至少 4 點，建議 6 點)
        # 注意：請先在 img1 (基準) 點，再在 img2 (變形) 點
        print("請選點：先點左圖 (基準)，再點右圖 (要拼接的)")
        pts1, pts2 = get_points(im1, im2, n_points=6)
        
        # 3. 拼接
        # 原本: result, w1, w2 = stitch_images(im1, im2, pts1, pts2)
        result, w1, w2, pts1_on_canvas, pts2_on_canvas, mask1, mask2 = stitch_images(im1, im2, pts1, pts2)
        
        # 4. 顯示結果與對應點
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1); plt.imshow(w1); plt.title("Warped Image 1")
        plt.axis('off')
        plt.subplot(1, 3, 2); plt.imshow(w2); plt.title("Warped Image 2")
        plt.axis('off')
        plt.subplot(1, 3, 3)
        plt.imshow(result)
        # 在最後的全景圖上疊上 warped 點
        plt.scatter(pts1_on_canvas[:, 0], pts1_on_canvas[:, 1],
                    c='r', s=40, marker='x', label='pts1 (from left)')
        plt.scatter(pts2_on_canvas[:, 0], pts2_on_canvas[:, 1],
                    c='cyan', s=40, marker='+', label='pts2 (from right)')
        plt.title("Final Mosaic with Correspondences")
        plt.axis('off')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.show()

        # 5. 顯示 alpha mask（權重遮罩）
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(mask1, cmap='viridis')
        plt.title("Alpha Mask 1 (from left image)")
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(mask2, cmap='viridis')
        plt.title("Alpha Mask 2 (from right image)")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        # 若想把包含 alpha mask 的圖一起存起來，也可以在這裡額外做 plt.savefig(...)
        
        # 保留原本純 mosaic 的存檔（不含點）
        iio.imwrite('mosaic_result.jpg', result)
        print("全景圖已儲存！")
        
    except FileNotFoundError:
        print("錯誤：找不到 'left.jpg' 或 'right.jpg'，請先準備照片。")
    except Exception as e:
        print(f"發生錯誤: {e}")