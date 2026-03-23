import matplotlib.pyplot as plt
import numpy as np
import imageio.v3 as iio

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

def test_homography(H, pt1, pt2):
    # 齊次座標:補上 1
    p1 = np.array([pt1[0], pt1[1], 1])
    
    # 矩陣乘法 p' = Hp
    p_prime = np.dot(H, p1)
    
    # 轉回歐氏座標 (除以 w)
    p_prime = p_prime / p_prime[2]

    # 計算誤差
    diff = np.linalg.norm(np.array(pt2) - p_prime[:2])
    print(f"誤差: {diff}")
    print(f"原點: {pt1}")
    print(f"目標點 (真值): {pt2}")
    print(f"轉換點 (預測): {p_prime[:2]}")

if __name__ == "__main__":
    img1 = iio.imread('data/panorama1-left.jpeg')
    img2 = iio.imread('data/panorama1-right.jpeg')
    pts1, pts2 = get_points(img1, img2, n_points=6)
    print('pts1:', pts1)
    print('pts2:', pts2)
    H = computeH(pts1, pts2)
    print('H:', H)
    test_homography(H, pts1[0], pts2[0])