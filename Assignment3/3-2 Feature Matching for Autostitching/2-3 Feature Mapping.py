import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist

def load_gray(img_path):
    img = iio.imread(img_path)
    if img.ndim == 3:
        gray = np.dot(img[...,:3], [0.299, 0.587, 0.114])
    else:
        gray = img
    return gray

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

# ==========================================
# Part 2: ANMS (作業核心)
# ==========================================
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

# ==========================================
# 視覺化工具：畫出配對連線
# ==========================================
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

# ==========================================
# 主程式 (假設已有 descriptors1, descriptors2, corners1, corners2)
# ==========================================
# 請自行準備兩張圖並跑完 B.1, B.2 流程
# img1, img2 = ...
# descriptors1, corners1 = extract_features(img1, anms_corners1)
# descriptors2, corners2 = extract_features(img2, anms_corners2)
if __name__ == "__main__":
    img1 = load_gray('../data/panorama1-left.jpeg')
    img2 = load_gray('../data/panorama1-right.jpeg')
    corners1 = get_harris_corners(img1)
    corners2 = get_harris_corners(img2)
    local_maxima1 = get_local_maxima(corners1, min_distance=1, threshold=0.001)
    local_maxima2 = get_local_maxima(corners2, min_distance=1, threshold=0.001)
    anms_corners1 = adaptive_non_maximal_suppression(local_maxima1, corners1, N_best=500)
    anms_corners2 = adaptive_non_maximal_suppression(local_maxima2, corners2, N_best=500)
    descriptors1, valid_corners1 = extract_features(img1, anms_corners1)
    descriptors2, valid_corners2 = extract_features(img2, anms_corners2)

    if 'descriptors1' in locals() and 'descriptors2' in locals():
        print("執行特徵配對...")
        # 0.75 是一個經驗值，你可以根據 Figure 6b 調整
        matches = match_features(descriptors1, descriptors2, threshold=0.75)
        
        print(f"找到 {len(matches)} 組配對。")
        
        # 顯示結果
        plot_matches(img1, img2, valid_corners1, valid_corners2, matches)
    else:
        print("請先執行 B.2 取得 descriptors 和 corners。")

