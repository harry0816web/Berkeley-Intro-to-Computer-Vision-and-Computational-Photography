import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
import random
from scipy.ndimage import gaussian_filter, distance_transform_edt
from scipy.spatial.distance import cdist


def load_gray(img_path):
    img = iio.imread(img_path)
    if img.ndim == 3:
        gray = np.dot(img[...,:3], [0.299, 0.587, 0.114])
    else:
        gray = img
    return gray

# ===== Part 1: Homography Estimation and Warp Image =====

def computeH(im1_pts, im2_pts):
    """
    計算單應性矩陣 H，使得 p' = H p
    im1_pts: N x 2 矩陣，來源圖片座標 (x, y)
    im2_pts: N x 2 矩陣，目標圖片座標 (x', y')
    """
    num_pts = im1_pts.shape[0]
    if num_pts < 4:
        raise ValueError("至少需要 4 組對應點才能計算 Homography")

    # Ah = b
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

    # 2. least square to solve 線性系統 Ah = b
    h, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

    # 3. reshape H to 3x3 with right down corner = 1
    H = np.append(h, 1).reshape(3, 3)

    return H

def interpolate_bilinear(img, x, y):
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
    if img.ndim == 3:
        out = np.zeros((x.shape[0], x.shape[1], img.shape[2]), dtype=img.dtype)
    else:
        out = np.zeros((x.shape[0], x.shape[1]), dtype=img.dtype)
    out[valid_mask] = pixel_values
    
    # 確保型別正確 (若是 uint8 需要截斷)
    if img.dtype == np.uint8:
        out = np.clip(out, 0, 255).astype(np.uint8)
        
    return out

def warpImage(im, H, output_shape=None, interpolation='bilinear'):
    """
    Inverse Warping
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

# ===== Part 2: ANMS of Harris Corners  =====

def get_harris_corners(img, sigma=1, min_distance=10, threshold=0.01):
    """
    計算 Harris Corner Response
    img: 灰階圖片 (H, W)
    """
    # 1. 計算影像梯度 (Derivatives)
    # 使用 Sobel 或簡單的差分
    dy, dx = np.gradient(img)
    
    # 2. 計算梯度乘積 (Ixx, Ixy, Iyy)
    Ixx = dx**2
    Ixy = dx*dy
    Iyy = dy**2
    
    # 3. 對乘積做高斯模糊 (Sum of products window)
    # 這相當於在視窗內加總
    A = gaussian_filter(Ixx, sigma)
    B = gaussian_filter(Ixy, sigma)
    C = gaussian_filter(Iyy, sigma)
    
    # 4. 計算 Harris Response R
    # R = det(M) - k * trace(M)^2
    # det(M) = A*C - B^2
    # trace(M) = A + C
    # k 通常取 0.04 - 0.06
    k = 0.04
    detM = A * C - B**2
    traceM = A + C
    harris_response = detM - k * (traceM**2)
    
    return harris_response

def get_local_maxima(harris_map, min_distance=10, threshold=0.01):
    """
    簡單的 peak finding，找出局部最大值作為候選點
    這是為了減少後續 ANMS 的計算量
    """
    from skimage.feature import peak_local_max
    
    # 閾值處理：只保留強度大於最大值一定比例的點
    abs_threshold = harris_map.max() * threshold
    
    # 使用 skimage 快速找出局部峰值
    coords = peak_local_max(harris_map, min_distance=min_distance, threshold_abs=abs_threshold)
    
    # coords 是 (row, col) 格式
    return coords

def adaptive_non_maximal_suppression(corners, harris_map, N_best=500, c_robust=0.9):
    """
    實作 Brown et al. 的 ANMS 演算法
    corners: 候選角點座標 (N, 2) -> (y, x)
    harris_map: 對應的角點強度圖
    N_best: 最後要留下的點數量
    c_robust: 穩健係數 (通常 0.9)
    """
    num_corners = corners.shape[0]
    
    # 取得每個候選點的強度值
    # corners 是 (y, x)，注意 numpy 索引順序
    corner_strengths = harris_map[corners[:, 0], corners[:, 1]]
    
    # 初始化半徑 r_i 為無限大
    radii = np.full(num_corners, np.inf)
    
    # --- 演算法核心 ---
    # 對於每個點 i，我們要找「最近的」且「比它強很多」的點 j
    # 條件：f(x_j) > c_robust * f(x_i)
    # 這裡為了效率，我們可以用矩陣運算，但雙層迴圈比較好理解邏輯
    
    # 排序：為了加速，我們可以先對強度做排序 (由大到小)
    # 但標準算法是對每個點 i 找 j
    
    for i in range(num_corners):
        # 當前點的強度
        response_i = corner_strengths[i]
        pos_i = corners[i]
        
        # 找出所有強度顯著大於點 i 的點 j 的索引
        # f(x_j) > f(x_i) * 0.9
        stronger_candidates_idx = np.where(corner_strengths > (response_i * c_robust))[0]
        
        if len(stronger_candidates_idx) > 0:
            # 取出這些更強點的座標
            stronger_coords = corners[stronger_candidates_idx]
            
            # 計算點 i 到所有更強點的距離
            # (N_strong, 2) - (2,) -> (N_strong, 2)
            dists = np.sqrt(np.sum((stronger_coords - pos_i)**2, axis=1))
            
            # 最小距離就是該點的抑制半徑 r_i
            radii[i] = np.min(dists)
        
        # 如果沒有人比它強 (它是全局最大值)，r_i 保持無限大
            
    # --- 選出前 N 個半徑最大的點 ---
    # argsort 是由小到大，所以我們反轉取最後 N 個
    best_indices = np.argsort(radii)[::-1][:N_best]
    
    return corners[best_indices]

# ===== Part 3: Feature Descriptor Extraction =====

def extract_features(img, corners, patch_size=40, descriptor_size=8):
    """
    提取特徵描述子
    img: 灰階圖片
    corners: 角點座標 (N, 2) -> (y, x)
    patch_size: 採樣視窗大小 (預設 40)
    descriptor_size: 最終描述子大小 (預設 8)
    
    Returns:
    descriptors: (N, 64) 的正規化描述子列表
    valid_corners: 沒被邊界濾掉的角點 (M, 2)
    """
    descriptors = []
    valid_corners = []
    
    # 半徑 (40 // 2 = 20)
    r = patch_size // 2
    
    # 縮放比例 (40 -> 8, step = 5)
    step = patch_size // descriptor_size
    
    h, w = img.shape
    
    for y, x in corners:
        y, x = int(y), int(x)
        
        # 1. 邊界檢查：如果視窗超出圖片範圍，就丟掉這個點
        if y - r < 0 or y + r > h or x - r < 0 or x + r > w:
            continue
            
        # 2. 切出 40x40 的大視窗
        # 注意 numpy slicing 範圍是 [start:end]，不包含 end
        patch = img[y-r : y+r, x-r : x+r]
        
        # 3. 高斯模糊 (Blurring)
        # 這一步是為了避免 Aliasing，通常 sigma 取 1~2 即可
        blurred_patch = gaussian_filter(patch, sigma=1)
        
        # 4. 降採樣 (Subsampling) 到 8x8
        # 使用間隔切片 [::step]
        # 確保結果剛好是 8x8
        small_patch = blurred_patch[::step, ::step]
        
        # 萬一因為整除問題形狀不對，強制 resize (很少發生)
        if small_patch.shape != (descriptor_size, descriptor_size):
            # 簡單的 fallback，或者直接 skip
            continue
            
        # 5. 正規化 (Bias/Gain Normalization)
        # (x - mean) / std
        mean = np.mean(small_patch)
        std = np.std(small_patch)
        
        if std == 0:
            # 避免除以 0 (如果該區域完全純色)
            normalized_patch = small_patch - mean
        else:
            normalized_patch = (small_patch - mean) / std
            
        # 展平成一維向量 (64,) 或是保持 (8,8) 看你後續習慣
        # 為了 matching 方便，通常展平
        descriptors.append(normalized_patch.flatten())
        valid_corners.append([y, x])
        
    return np.array(descriptors), np.array(valid_corners)

# ===== Part 4: Feature Matching =====
def plot_matches(img1, img2, corners1, corners2, matches):
    """
    將兩張圖並排，並畫出配對的連線
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # 建立一個大畫布把兩張圖拼在一起
    vis_h = max(h1, h2)
    vis_w = w1 + w2
    vis_img = np.zeros((vis_h, vis_w), dtype=img1.dtype)
    
    # 貼上圖片
    vis_img[:h1, :w1] = img1
    vis_img[:h2, w1:w1+w2] = img2
    
    plt.figure(figsize=(15, 8))
    plt.imshow(vis_img, cmap='gray')
    
    # 畫線
    # matches 是 [[idx1, idx2], [idx1, idx2], ...]
    for idx1, idx2 in matches:
        pt1 = corners1[idx1] # (y, x)
        pt2 = corners2[idx2] # (y, x)
        
        # 注意 pyplot 的 plot 是 (x, y)
        x1, y1 = pt1[1], pt1[0]
        x2, y2 = pt2[1] + w1, pt2[0] # 右邊的圖 x 要加上 w1 的偏移量
        
        plt.plot([x1, x2], [y1, y2], 'c-', alpha=0.6, linewidth=0.8)
        plt.plot(x1, y1, 'r.', markersize=4)
        plt.plot(x2, y2, 'r.', markersize=4)
        
    plt.title(f"Feature Matches (Total: {len(matches)})")
    plt.axis('off')
    plt.show()

def match_features(desc1, desc2, threshold=0.75):
    """
    特徵配對：使用 Lowe's Ratio Test
    desc1: 圖1 的描述子 (N1, 64)
    desc2: 圖2 的描述子 (N2, 64)
    threshold: Lowe's ratio 閾值 (通常 0.7~0.8)
    
    Returns:
    matches: 配對成功的索引列表 (M, 2)
             每一列是 [index_in_desc1, index_in_desc2]
    """
    N1 = desc1.shape[0]
    matches = []
    
    # 1. 計算所有點對所有點的距離 (Distance Matrix)
    # cdist 會回傳 (N1, N2) 的矩陣，dist_matrix[i, j] 是 desc1[i] 和 desc2[j] 的距離
    # metric='euclidean' 或 'sqeuclidean' 都可以，這裡用歐式距離
    dists = cdist(desc1, desc2, metric='euclidean')
    
    # 2. 對每一個圖1的點，找圖2中最好的兩個對象
    for i in range(N1):
        # 取得第 i 個點對圖2所有點的距離向量
        dists_i = dists[i]
        
        # 排序找到最小的兩個索引
        # argsort 會回傳索引，由距離小到大排
        sorted_indices = np.argsort(dists_i)
        
        best_idx = sorted_indices[0]
        second_best_idx = sorted_indices[1]
        
        best_dist = dists_i[best_idx]
        second_best_dist = dists_i[second_best_idx]
        
        # 3. Lowe's Ratio Test
        # 避免除以 0 的保護
        if second_best_dist == 0:
            ratio = 1.0 # 這種情況很怪，當作不可靠
        else:
            ratio = best_dist / second_best_dist
            
        if ratio < threshold:
            # 是一個好配對！紀錄下來
            matches.append([i, best_idx])
            
    return np.array(matches)

# ===== Part 5: RANSAC Homography =====
def ransac_homography(pts1, pts2, num_iters=2000, threshold=5.0):
    """
    使用 RANSAC 找出最佳的單應性矩陣 H
    pts1: 來源點座標 (N, 2)
    pts2: 目標點座標 (N, 2)
    num_iters: 迭代次數
    threshold: 判定為 Inlier 的距離閾值 (pixels)
    """
    N = pts1.shape[0]
    if N < 4:
        print("配對點數量不足 4 個，無法執行 RANSAC")
        return None, []

    max_inliers = []
    
    # point to homogeneous coordinates
    pts1_homo = np.hstack([pts1, np.ones((N, 1))])
    
    for i in range(num_iters):
        # 1. random sample 4 points
        idx = np.random.choice(N, 4, replace=False)
        p1_sample = pts1[idx]
        p2_sample = pts2[idx]
        
        # 2. compute H with 4 sample points
        try:
            H_curr = computeH(p1_sample, p2_sample)
        except np.linalg.LinAlgError:
            continue
            
        # 3. cal error of current H
        # p1' = H * p1
        # 這裡用矩陣乘法一次算完所有點: (3, 3) @ (3, N) -> (3, N)
        projected_homo = (H_curr @ pts1_homo.T).T
        
        # 歸一化: x' = x/w, y' = y/w
        # 避免除以 0
        w = projected_homo[:, 2:3] + 1e-10
        projected_pts = projected_homo[:, :2] / w
        
        # 計算歐式距離誤差 || p1' - p2 ||
        errors = np.linalg.norm(projected_pts - pts2, axis=1)
        
        # 4. find inliners
        current_inlier_indices = np.where(errors < threshold)[0]
        
        # 5. update best record if getting more inliners
        if len(current_inlier_indices) > len(max_inliers):
            max_inliers = current_inlier_indices
            
    
    print(f"RANSAC 完成。最大 Inliers 數量: {len(max_inliers)} / {N}")
    
    if len(max_inliers) < 4:
        print("警告：找不到足夠的 Inliers。")
        return None, []
        
    # 6. Refine: 用所有的 Inliers 重新計算最精準的 H
    final_pts1 = pts1[max_inliers]
    final_pts2 = pts2[max_inliers]
    H_final = computeH(final_pts1, final_pts2)
    
    return H_final, max_inliers

def auto_stitch(im1, im2):
    print("=== Step 1: Harris Corner Detection & ANMS ===")
    corners1 = get_harris_corners(im1)
    corners2 = get_harris_corners(im2)
    local_maxima1 = get_local_maxima(corners1, min_distance=1, threshold=0.001)
    local_maxima2 = get_local_maxima(corners2, min_distance=1, threshold=0.001)
    anms_corners1 = adaptive_non_maximal_suppression(local_maxima1, corners1, N_best=500)
    anms_corners2 = adaptive_non_maximal_suppression(local_maxima2, corners2, N_best=500)

    print("=== Step 2: Feature Descriptor Extraction ===")
    desc1, valid_corners1 = extract_features(im1, anms_corners1)
    desc2, valid_corners2 = extract_features(im2, anms_corners2)
    
    print("=== Step 3: Feature Matching (Lowe's Ratio) ===")
    # get index of matched descriptor in desc1 and desc2
    matches = match_features(desc1, desc2, threshold=0.7)

    # swap x and y to (x, y)
    print(valid_corners1[0].shape)
    matched_pts1 = valid_corners1[matches[:, 0]]
    matched_pts2 = valid_corners2[matches[:, 1]]

    print(matched_pts1[0].shape)
    matched_pts1 = matched_pts1[:, [1, 0]]
    matched_pts2 = matched_pts2[:, [1, 0]]

    print("=== Step 4: RANSAC Homography ===")
    # converting img1(dest) to img2(src)
    H_robust, inliers_idx = ransac_homography(matched_pts2, matched_pts1) # computeH(src, dest)
    
    print("=== Step 5: Warping and Blending ===")
    # 1. Use H_robust from RANSEC to warp images
    output_shape, T = get_canvas_dimensions(im1, im2, H_robust)
    out_h, out_w = output_shape
    print(f"Output image shape: {out_h}x{out_w}")
    
    warped_im1 = warpImage(im1, T, output_shape=(out_h, out_w))
    H_final = T @ H_robust
    warped_im2 = warpImage(im2, H_final, output_shape=(out_h, out_w))
    
    # 2.Alpha Blending with Weight Masks
    mask1 = make_weight_mask(warped_im1)
    mask2 = make_weight_mask(warped_im2)
    if warped_im1.ndim == 3:
        # expand dims for broadcasting
        mask1_expanded = mask1[..., None]
        mask2_expanded = mask2[..., None]
    else:
        mask1_expanded = mask1
        mask2_expanded = mask2

    # mosaic image = weighted average sum of two warped images
    weighted_sum = (warped_im1.astype(float) * mask1_expanded + 
                 warped_im2.astype(float) * mask2_expanded)
    mosaic = weighted_sum / (mask1_expanded + mask2_expanded + 1e-8) # prevent division by zero
    mosaic = np.clip(mosaic, 0, 255).astype(np.uint8)
    
    return mosaic

if __name__ == "__main__":
    img1 = load_gray('../data/panorama2-left.jpeg')
    img2 = load_gray('../data/panorama2-right.jpeg')
    final_mosaic = auto_stitch(img1, img2)
    plt.imshow(final_mosaic, cmap='gray')
    plt.show()