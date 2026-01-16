import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
from scipy.ndimage import gaussian_filter

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

# ==========================================
# 主程式
# ==========================================
if __name__ == "__main__":
    gray = load_gray('../data/panorama2-left.jpeg')

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

    print("提取特徵描述子...")
    descriptors, final_corners = extract_features(gray, anms_corners)
    print(f"提取完成。原始點數: {len(anms_corners)}, 有效點數: {len(final_corners)}")
    print(f"描述子形狀: {descriptors.shape}") # 應該是 (N, 64)

    # 視覺化：從原始影像的角點指向對應的描述子
    num_samples = min(5, len(descriptors))  # 選擇前 5 個角點來顯示
    
    # 創建一個大畫布：左邊是原始影像，右邊是描述子
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(111)
    
    h, w = gray.shape
    descriptor_scale = 20  # 描述子放大倍數（8x8 -> 160x160 像素顯示）
    descriptor_size = 8 * descriptor_scale
    
    # 計算右邊描述子區域的起始位置
    img_width = w
    img_height = h
    descriptor_start_x = img_width + 50  # 原始影像和描述子之間的間距
    
    # 1. 顯示原始影像（左邊）
    ax.imshow(gray, cmap='gray', extent=[0, img_width, img_height, 0])
    
    # 2. 標示所有角點（灰色小點）
    ax.plot(final_corners[:, 1], final_corners[:, 0], 'o', 
            color='gray', markersize=2, alpha=0.3, label='All corners')
    
    # 3. 顯示選中的角點和對應的描述子
    colors = plt.cm.tab10(np.linspace(0, 1, num_samples))
    
    for i in range(num_samples):
        corner_y, corner_x = final_corners[i]
        descriptor = descriptors[i].reshape(8, 8)
        
        # 計算描述子在右邊的位置（垂直排列）
        descriptor_y = i * (descriptor_size + 20)  # 每個描述子之間間距 20
        descriptor_x = descriptor_start_x
        
        # 顯示描述子（放大顯示）
        descriptor_display = np.kron(descriptor, np.ones((descriptor_scale, descriptor_scale)))
        ax.imshow(descriptor_display, cmap='gray', 
                 extent=[descriptor_x, descriptor_x + descriptor_size,
                        descriptor_y + descriptor_size, descriptor_y],
                 alpha=0.9)
        
        # 畫線從角點指向描述子
        ax.plot([corner_x, descriptor_x], 
                [corner_y, descriptor_y + descriptor_size // 2],
                color=colors[i], linewidth=1.5, alpha=0.6)
        
        # 標示選中的角點（彩色大點）
        ax.plot(corner_x, corner_y, 'o', 
               color=colors[i], markersize=6, markeredgecolor='white', 
               markeredgewidth=1)
    
    # 設置座標軸範圍
    total_width = descriptor_start_x + descriptor_size + 50
    total_height = max(img_height, num_samples * (descriptor_size + 20))
    ax.set_xlim(-50, total_width)
    ax.set_ylim(total_height, -50)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'Feature Descriptors: Corners → Descriptors (showing {num_samples} samples)', 
                fontsize=12, pad=20)
    
    plt.tight_layout()
    plt.show()
