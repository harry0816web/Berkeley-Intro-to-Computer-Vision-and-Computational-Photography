# CS180 Computer Vision Project #1: Prokudin-Gorskii Colorization

Basic starter code for colorizing Prokudin-Gorskii glass plate images. **You need to implement the alignment functions yourself.**

## What's Provided

- **Image Loading**: Load and preprocess glass plate images
- **Channel Splitting**: Split into Blue, Green, Red channels (BGR order)
- **Utility Functions**: Image shifting, distance metrics (L2, NCC)
- **Result Creation**: Combine aligned channels into RGB image
- **Visualization**: Display original channels and final result

## What You Need to Implement

- **`align_channels_single_scale()`**: Single-scale alignment with exhaustive search
- **`align_channels_pyramid()`**: Multi-scale pyramid alignment for large images

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from colorize_prokudin_gorskii import ProkudinGorskiiColorizer

colorizer = ProkudinGorskiiColorizer()
colorizer.load_image("monastery.jpg")
colorizer.split_channels()

# TODO: Implement these functions!
g_offset, r_offset = colorizer.align_channels_single_scale(search_range=15, metric='ncc')
aligned_image = colorizer.create_aligned_image(g_offset, r_offset)
colorizer.display_results("result.jpg")
```

## Key Points

- Filter order: **BGR** (Blue, Green, Red) from top to bottom
- Use `shift_image()` to move channels by (dx, dy) pixels
- Use `l2_distance()` or `normalized_cross_correlation()` as metrics
- NCC generally works better than L2
- Start with small .jpg images, then use pyramid for large .tif files

## File Structure

```
CS180-CV/
├── colorize_prokudin_gorskii.py    # Main file - implement alignment functions
├── example_usage.py                # Simple usage example
├── requirements.txt                # Dependencies
└── README.md                       # This file
```

Good luck with your implementation! 🚀
