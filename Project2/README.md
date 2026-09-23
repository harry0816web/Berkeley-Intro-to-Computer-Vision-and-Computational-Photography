# CS180 Project 2: Fun with Filters and Frequencies

This project explores how convolution and frequency decomposition can be used for image filtering, edge detection, sharpening, hybrid images, and multiresolution blending.

## What is included

### 1. Convolution from scratch

The project implements 2D convolution with NumPy and compares the result with `scipy.signal.convolve2d`. Both a straightforward four-loop implementation and a faster two-loop implementation using array operations are included.

A normalized box filter is used to blur an image, while finite-difference filters estimate horizontal and vertical derivatives.

### 2. Finite differences and derivative of Gaussian

The (D_x) and (D_y) finite-difference operators are applied to compute image gradients. Their responses are combined into a gradient-magnitude image and thresholded to obtain an edge map.

The derivative-of-Gaussian experiment first smooths the image and then detects edges. This reduces small-scale noise and produces cleaner contours than applying finite differences directly.

### 3. Image sharpening

A blurred image is sharpened by adding a scaled high-frequency component back to the image:

```text
sharpened = image + alpha * (image - GaussianBlur(image))
```

The scripts also compare the original sharp image, a simulated blurred image, and the restored result.

### 4. Hybrid images

Hybrid images combine the low frequencies of one image with the high frequencies of another. The input images are aligned first, then Gaussian blur is used to separate their frequency components. The result changes appearance depending on viewing distance.

The project also visualizes the Fourier magnitude spectrum of the input and hybrid images.

### 5. Gaussian and Laplacian stacks

Gaussian stacks show progressively blurred versions of an image. Laplacian stacks store the band-pass details between neighboring Gaussian levels, making it possible to reconstruct the original image from the sum of its frequency bands.

### 6. Multiresolution blending

Two images are blended with a mask at multiple resolutions. Apple and orange are combined into the “oraple” result while preserving a smooth transition between the two halves.

![Multiresolution blending result](result/Figure_2-4-1.png)

## Results

- Filtering and finite-difference visualizations: `result/Figure_1-1.png`, `result/Figure_1-2.png`
- Derivative-of-Gaussian edge detection: `result/Figure_1-3.png`
- Sharpening: `result/Figure_2-1-1.png`
- Hybrid images and frequency analysis: `result/Figure_2-2-1.png`, `result/Figure_2-2-2.png`
- Gaussian/Laplacian stack reconstruction: `result/Figure_2-3-1.png`–`result/Figure_2-3-3.png`
- Multiresolution blending: `result/Figure_2-4-1.png`, `result/Figure_2-4-2.png`

## Installation

The scripts use NumPy, Matplotlib, ImageIO, SciPy, scikit-image, and OpenCV.

```bash
pip install numpy matplotlib imageio scipy scikit-image opencv-python
```

## Usage

Run scripts from the `Project2` directory. File names containing spaces should be quoted.

```bash
python "1-1 Conv from Scratch.py"
python "1-2 Finite Difference.py"
python "1-3 Derivative of Gaussian (DoG) Filter.py"
python "2-1 Image Shaperning.py"
python "2-2-1 Alignment.py"
python "2-2-2 Hybrid Images.py"
python "2-2-3 Color Hign or Low.py"
python "2-3 Gaussian and Laplacian Stacks.py"
python "2-4 Multiresolution Blending.py"
```

Most scripts use the sample images stored directly in this directory and display their results with Matplotlib. Adjust the input paths and parameters near the bottom of each script when experimenting with other images.

## File structure

```text
Project2/
├── 1-1 Conv from Scratch.py
├── 1-2 Finite Difference.py
├── 1-3 Derivative of Gaussian (DoG) Filter.py
├── 2-1 Image Shaperning.py
├── 2-2-1 Alignment.py
├── 2-2-2 Hybrid Images.py
├── 2-2-3 Color Hign or Low.py
├── 2-3 Gaussian and Laplacian Stacks.py
├── 2-4 Multiresolution Blending.py
└── result/
```
