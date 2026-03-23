import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
from scipy.ndimage import gaussian_filter

# ==========================================
# Part 1: Harris Corner Detector
# ==========================================
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

# ==========================================
# 主程式
# ==========================================
# 讀取圖片 (請換成你的圖片路徑)
# 建議讀成灰階
img = iio.imread('../data/panorama1-left.jpeg')
if img.ndim == 3:
    gray = np.dot(img[...,:3], [0.299, 0.587, 0.114])
else:
    gray = img

print("1. 計算 Harris Response...")
harris_map = get_harris_corners(gray)

print("2. 取得初步候選點 (Local Maxima)...")
# 先取多一點點，比如 2000 個，再給 ANMS 篩選
candidate_corners = get_local_maxima(harris_map, min_distance=1, threshold=0.001)
print(f"   初步找到 {len(candidate_corners)} 個候選點")

print("3. 執行 ANMS...")
N_final = 500
anms_corners = adaptive_non_maximal_suppression(candidate_corners, harris_map, N_best=N_final)
print(f"   ANMS 篩選出 {len(anms_corners)} 個均勻分佈的點")

# ==========================================
# 視覺化比較 (Deliverables)
# ==========================================
plt.figure(figsize=(12, 6))

# 圖 1: 原始 Harris Corners (取前 N 個最強的作為對照)
# 如果單純只取最強的，點會擠在一起
plt.subplot(1, 2, 1)
plt.imshow(gray, cmap='gray')
# 簡單取強度前 N 名的點來畫
top_indices = np.argsort(harris_map[candidate_corners[:, 0], candidate_corners[:, 1]])[::-1][:N_final]
top_corners = candidate_corners[top_indices]
plt.plot(top_corners[:, 1], top_corners[:, 0], 'r.', markersize=3)
plt.title(f'Top {N_final} Strongest Corners (Clustered)')
plt.axis('off')

# 圖 2: ANMS 結果
plt.subplot(1, 2, 2)
plt.imshow(gray, cmap='gray')
plt.plot(anms_corners[:, 1], anms_corners[:, 0], 'r.', markersize=3)
plt.title(f'Top {N_final} ANMS Corners (Distributed)')
plt.axis('off')

plt.tight_layout()
plt.show()