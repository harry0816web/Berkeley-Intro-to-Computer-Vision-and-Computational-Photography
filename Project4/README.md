# CS180 Project 4: Image Mosaics and Feature Matching

This project builds panoramic image mosaics in two stages: first with manually selected correspondences, and then with an automatic feature-matching pipeline.

![Image mosaic result](result/Figure_1-4-3.jpg)

## Part 1: Image mosaics

### Recovering homographies

A homography maps points from one image plane to another. Given corresponding points in two images, the project solves for the (3 \times 3) projective transformation using a linear system.

### Warping images

The estimated homography is used to warp an image into the coordinate system of a reference image. Inverse mapping and bilinear interpolation are used to sample the source image while avoiding holes in the output.

### Blending mosaics

After both images are placed on a common canvas, the overlapping region is blended with distance-based weight masks. This reduces hard seams and produces a smoother panorama.

## Part 2: Feature matching for autostitching

The automatic stitching pipeline contains the following steps:

1. **Harris corner detection** finds pixels with strong local corner responses.
2. **Adaptive Non-Maximal Suppression (ANMS)** keeps strong points that are also spatially distributed across the image.
3. **Feature descriptor extraction** creates normalized local patches around the selected points.
4. **Feature matching** compares descriptors and filters unreliable matches.
5. **RANSAC** repeatedly estimates homographies from random subsets and selects the model with the most inliers.
6. **Warping and blending** use the robust homography to create the final mosaic.

## Results

Manual mosaicing results:

- Homography and correspondence visualization: `result/Figure_1-3.png`
- Warping and blending examples: `result/Figure_1-4-1.png`–`result/Figure_1-4-3.jpg`

Automatic autostitching results:

- Harris corner detection: `result/Figure_2-1.png`
- Feature descriptor extraction: `result/Figure_2-2.png`
- Feature matching: `result/Figure_2-3.png`
- RANSAC and final mosaics: `result/Figure_2-4-1.png`, `result/Figure_2-4-2.jpg`

## Installation

The scripts use NumPy, Matplotlib, ImageIO, and SciPy.

```bash
pip install numpy matplotlib imageio scipy
```

## Usage

Run scripts from the `Project4` directory. The scripts are organized by assignment part:

```bash
python "4-1 Image Mosaic/1-2 Recover Homographies.py"
python "4-1 Image Mosaic/1-3 Warp the Images.py"
python "4-1 Image Mosaic/1-4 Blend the Images into a Mosaic.py"

python "4-2 Feature Matching for Autostitching/2-1 Harris Corner Detection.py"
python "4-2 Feature Matching for Autostitching/2-2 Feature Descriptor Extraction.py"
python "4-2 Feature Matching for Autostitching/2-3 Feature Mapping.py"
python "4-2 Feature Matching for Autostitching/2-4 RANSAC for Robust Homography.py"
```

The input photographs are in `data/`, and generated figures are stored in `result/`. Some scripts contain experiment-specific point selections and parameters that can be adjusted for new image pairs.

## File structure

```text
Project4/
├── 4-1 Image Mosaic/
├── 4-2 Feature Matching for Autostitching/
├── data/
└── result/
```

