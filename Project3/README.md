# Project 3 — Face Morphing

本專案是 Berkeley CS180/280A 的 Computational Photography 作業，主題是
**人臉 morphing（臉部變形）**。目標是透過臉部特徵點、三角剖分、仿射變形與顏色融合，
讓一張臉平滑地變成另一張臉；接著利用一組人臉資料計算平均臉，並從平均臉外插出自己的 caricature。

## 主要工作

1. **標記對應特徵點**：在自己的照片與目標人臉上，以相同順序標記眼睛、鼻子、嘴巴、下巴、耳朵等位置。
2. **建立三角網格**：使用特徵點建立 Delaunay triangulation，作為後續變形的固定網格。
3. **製作 midway face**：計算兩張臉的平均形狀，對每個三角形做 affine warp，再將兩張變形後的圖片平均融合。
4. **產生 morph sequence**：逐步改變形狀比例與顏色融合比例，輸出從自己的臉變成目標臉的動畫或 GIF。
5. **計算 population mean face**：使用附有標註的人臉資料集，將每張臉變形到平均幾何形狀，計算並展示族群平均臉。
6. **製作 caricature**：將自己的臉相對於族群平均臉的形狀差異外插，產生誇張化的人臉。

## 實作重點

- 使用 inverse warp，逐個三角形處理像素區域；不要對每個像素再寫額外的全域迴圈。
- affine transformation 必須自行計算，不使用內建的幾何轉換函式。
- 兩張圖片的特徵點順序必須一致，且 morph 過程中要使用同一份 triangulation。
- 自己的照片與目標照片最好有相同尺寸、比例、背景與臉部位置。

## 預期成果

- 特徵點與三角剖分圖
- 原始圖片 A、原始圖片 B，以及 midway face
- 從圖片 A 到圖片 B 的 morph GIF 或影片連結
- 資料集中的平均臉、變形範例，以及自己的臉與平均臉互相變形的結果
- 一張自己的 caricature
- 至少一個 bells and whistles 延伸項目，例如改變年齡／性別／表情、PCA face space，或互動式臉部變形工具

完整作業說明：[CS180 Project 3 — Computational Photography](https://cal-cs180.github.io/fa24/hw/proj3/index.html)

## 目前進度

- `eleven.jpeg`：原始的 Eleven 照片（200 × 250）
- `eleven_aligned.jpg`：放大到與 George 相同的 602 × 750 工作版本
- `george_small.jpg`：目標照片
- `select_points.py`：互動式特徵點標記與 Delaunay triangulation 工具

從 `Project3` 目錄執行：

```bash
../.venv/bin/python select_points.py
```

程式會依序要求在 Eleven 和 George 上點選同一個特徵。請仔細遵守畫面上的標籤順序；完成後會產生 `face_points.npz` 與 `results/landmarks_and_triangulation.png`。

建立 midway face：

```bash
../.venv/bin/python morph.py
```

這一步會將兩張臉分別 warp 到平均形狀，再以 50% 比例融合。結果會儲存在 `results/midway_face.jpg`，並同時輸出兩張單獨 warp 後的圖片與三張圖片的比較圖。

建立 46 幀的完整 morph 動畫：

```bash
../.venv/bin/python make_morph_sequence.py
```

程式會輸出 `results/morph_frames/frame_00.png` 到 `frame_45.png`，並建立 `results/eleven_to_george.gif`。

## Part 4：IMM population mean face

`IMM-Face` 根目錄包含 240 張圖片與各自的 58 個 ASF 特徵點標註。計算平均臉時使用根目錄的 240 組 JPG/ASF 配對，不包含 `IMM-Face/data/` 內的重複子集。程式會先平均所有人的 landmark geometry，再將每張臉 warp 到平均形狀，最後逐像素平均。

```bash
../.venv/bin/python compute_imm_mean.py
```

輸出會放在 `results/imm_mean/`：

- `imm_mean_face.jpg`：IMM dataset 的 population mean face
- `imm_mean_shape.png`：平均臉與 triangulation
- `imm_warped_examples.png`：六張 warp 到平均形狀的範例
- `imm_mean_geometry.npz`：平均 landmark、三角網格與資料集資訊

資料集本身已加入根目錄 `.gitignore`，不會被提交到 Git；計算結果仍會保留在專案中。

## Mean face 與 morphing 的比較

原本的 `Eleven → George` morph 不使用 IMM dataset，因此計算 population mean 不會改變原本的 midway face。若要比較兩者，執行：

```bash
../.venv/bin/python compare_morphs.py
```

程式會比較 `Eleven → George` 與 `IMM 01-1m → IMM population mean`，輸出到 `results/morph_comparison/`。

## Part 5：Eleven caricature

IMM 使用 58 個 landmark，與前面 Eleven→George 使用的點集不同。因此先執行：

```bash
../.venv/bin/python select_imm_points.py
```

程式左側會要求你在 Eleven 上點 58 個點，右側會同步顯示 IMM mean face 的對應點。完成後會產生 `eleven_imm_points.npz`，再執行：

```bash
../.venv/bin/python repair_imm_points.py
../.venv/bin/python make_caricature.py
```

修正工具會檢查並修復會造成三角形翻轉的 outlier click。`make_caricature.py` 使用 `P_caricature = P_mean + alpha * (P_Eleven - P_mean)` 外插形狀；`alpha=1` 是原始 Eleven，`alpha>1` 會放大 Eleven 相對於 population mean 的臉型差異，結果會放在 `results/caricature/`。
