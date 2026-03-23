"""
CS180 Computer Vision Project #1: Colorizing Prokudin-Gorskii Photos
Basic starter code - you need to implement the alignment functions

Author: CS180 Student
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import skimage
from skimage.metrics import structural_similarity as ssim
import os


class ProkudinGorskiiColorizer:
    """Basic class for colorizing Prokudin-Gorskii glass plate images"""
    
    def __init__(self):
        self.image = None
        self.channels = None
        self.aligned_image = None
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load and preprocess image
        
        Args:
            image_path: Path to the glass plate image
            
        Returns:
            Loaded image as numpy array
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Load image as grayscale (Prokudin-Gorskii plates are stacked grayscale)
        self.image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if self.image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Note: Prokudin-Gorskii images are grayscale, no color conversion needed
        
        # Convert to float and normalize to [0, 1]
        self.image = self.image.astype(np.float64) / 255.0
        
        print(f"Loaded image with shape: {self.image.shape}")
        return self.image
    
    def split_channels(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split the glass plate image into three color channels
        Note: Filter order is BGR (Blue, Green, Red) from top to bottom
        
        Returns:
            Tuple of (blue, green, red) channel images
        """
        if self.image is None:
            raise ValueError("No image loaded. Call load_image() first.")
        
        height = self.image.shape[0]
        channel_height = height // 3
        
        # Split into three equal parts (each is 2D grayscale)
        blue_channel = self.image[:channel_height, :]
        green_channel = self.image[channel_height:2*channel_height, :]
        red_channel = self.image[2*channel_height:3*channel_height, :]
        
        self.channels = (blue_channel, green_channel, red_channel)
        
        print(f"Split channels - Blue: {blue_channel.shape}, Green: {green_channel.shape}, Red: {red_channel.shape}")
        return self.channels
    
    def shift_image(self, image: np.ndarray, dx: int, dy: int) -> np.ndarray:
        """
        Shift image by (dx, dy) pixels using numpy.roll
        
        Args:
            image: Input image
            dx: Horizontal shift (positive = right)
            dy: Vertical shift (positive = down)
            
        Returns:
            Shifted image
        """
        shifted = np.roll(image, shift=(dy, dx), axis=(0, 1))
        return shifted

    def _crop_border(self, image: np.ndarray, border_ratio: float = 0.1) -> np.ndarray:
        h, w = image.shape[:2]
        by = int(h * border_ratio)
        bx = int(w * border_ratio)
        if by * 2 >= h or bx * 2 >= w:
            return image
        return image[by:h-by, bx:w-bx]
    
    def _apply_sobel(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Sobel edge detection to enhance structural features
        
        Args:
            image: Input grayscale image
            
        Returns:
            Edge-enhanced image
        """
        # Convert to uint8 for Sobel operator
        img_uint8 = (image * 255).astype(np.uint8)
        
        # Apply Sobel operator in both x and y directions
        # Gx Gy
        sobel_x = cv2.Sobel(img_uint8, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img_uint8, cv2.CV_64F, 0, 1, ksize=3)
        
        # Compute magnitude of gradient
        # Gx^2 + Gy^2
        sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Normalize back to [0, 1]
        sobel_magnitude = sobel_magnitude / 255.0
        
        return sobel_magnitude
    
    def display_edge_images(self, save_path: Optional[str] = None):
        """
        Display the original channels and their Sobel edge versions
        
        Args:
            save_path: Optional path to save the edge images
        """
        if self.channels is None:
            raise ValueError("No channels loaded. Call split_channels() first.")
        
        blue, green, red = self.channels
        
        # Apply Sobel edge detection
        blue_edges = self._apply_sobel(blue)
        green_edges = self._apply_sobel(green)
        red_edges = self._apply_sobel(red)
        
        # Create figure with subplots (2 rows, 3 columns)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Display original channels (top row)
        axes[0, 0].imshow(blue, cmap='gray')
        axes[0, 0].set_title('Original Blue Channel')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(green, cmap='gray')
        axes[0, 1].set_title('Original Green Channel')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(red, cmap='gray')
        axes[0, 2].set_title('Original Red Channel')
        axes[0, 2].axis('off')
        
        # Display edge images (bottom row)
        axes[1, 0].imshow(blue_edges, cmap='gray')
        axes[1, 0].set_title('Blue Channel Edges (Sobel)')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(green_edges, cmap='gray')
        axes[1, 1].set_title('Green Channel Edges (Sobel)')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(red_edges, cmap='gray')
        axes[1, 2].set_title('Red Channel Edges (Sobel)')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Edge images saved to: {save_path}")
        
        plt.show()

    def l2_distance(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Compute L2 (Euclidean) distance between two images
        
        Args:
            img1, img2: Input images (same shape)
            
        Returns:
            L2 distance (lower = better match)
        """
        if img1.shape != img2.shape:
            raise ValueError("Images must have the same shape for L2 distance")
        
        # pixel wise difference
        diff = img1 - img2
        l2_dist = np.sqrt(np.sum(diff ** 2))
        return l2_dist
    
    def normalized_cross_correlation(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Compute Normalized Cross Correlation (NCC) between two images.
        Higher values (closer to 1) indicate stronger similarity.
        """
        if img1.shape != img2.shape:
            raise ValueError("Images must have the same shape for NCC")
        
        # Flatten
        std1 = np.std(img1)
        std2 = np.std(img2)
        mean1 = np.mean(img1)
        mean2 = np.mean(img2)
        ncc = np.mean(np.multiply((img1-mean1),(img2-mean2))) / (std1 * std2)
        return ncc
    
    def align_channels_single_scale(self, 
                                   search_range: int = 15,
                                   metric: str = 'ssim',
                                   use_edges: bool = False
                                   ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        TODO: Implement single-scale alignment
        Align green and red channels to blue channel using exhaustive search
        
        Args:
            search_range: Search range in pixels (e.g., 15 means search from -15 to +15)
            metric: Distance metric ('l2' or 'ncc' or 'ssim')
            use_edges: Whether to apply Sobel edge detection before alignment
            
        Returns:
            Tuple of ((gx, gy), (rx, ry)) displacement vectors
        """
        if self.channels is None:
            raise ValueError("No channels loaded. Call split_channels() first.")
        
        blue, green, red = self.channels
        
        # Apply Sobel edge detection if requested
        if use_edges:
            print("Using Sobel edge detection for alignment")
            blue = self._apply_sobel(blue)
            green = self._apply_sobel(green)
            red = self._apply_sobel(red)
        
        # TODO: Implement your alignment algorithm here
        # You need to:
        # 1. Choose the appropriate metric function (l2_distance or normalized_cross_correlation)
        # 2. Search over all possible displacements in [-search_range, search_range]
        # 3. For each displacement, shift the green/red channel and compute the metric with blue
        # 4. Find the displacement that gives the best score
        # 5. Return the best displacements for green and red channels
        
        print("TODO: Implement single-scale alignment")
        
        # Placeholder - replace with your implementation
        g_offset = (0, 0)
        r_offset = (0, 0)

        if metric == 'l2':
            best_score_g = float('inf')
            best_score_r = float('inf')
        elif metric == 'ncc':
            best_score_g = float('-inf')
            best_score_r = float('-inf')
        elif metric == 'ssim':
            best_score_g = float('-inf')
            best_score_r = float('-inf')

        # Crop reference once to avoid roll artifacts impacting the score
        blue_cropped = self._crop_border(blue, 0.1)

        for dx in range(-search_range, search_range + 1):
            for dy in range(-search_range, search_range + 1):
                shifted_green = self.shift_image(green, dx, dy)
                shifted_red = self.shift_image(red, dx, dy)
                sg = self._crop_border(shifted_green, 0.1)
                sr = self._crop_border(shifted_red, 0.1)
                if metric == 'l2':
                    # print(f"Using L2 metric")
                    score_g = self.l2_distance(sg, blue_cropped)
                    score_r = self.l2_distance(sr, blue_cropped)
                    if score_g < best_score_g:
                        print(f"New best score: {score_g}")
                        best_score_g = score_g
                        g_offset = (dx, dy)
                    if score_r < best_score_r:
                        print(f"New best score: {score_r}")
                        best_score_r = score_r
                        r_offset = (dx, dy)
                elif metric == 'ncc':
                    # print(f"Using NCC metric")
                    score_g = self.normalized_cross_correlation(sg, blue_cropped)
                    score_r = self.normalized_cross_correlation(sr, blue_cropped)
                    if score_g > best_score_g:
                        print(f"New best score: {score_g}")
                        best_score_g = score_g
                        g_offset = (dx, dy)
                    if score_r > best_score_r:
                        print(f"New best score: {score_r}")
                        best_score_r = score_r
                        r_offset = (dx, dy)
                elif metric == 'ssim':
                    score_g = ssim(sg, blue_cropped, data_range=1.0)
                    score_r = ssim(sr, blue_cropped, data_range=1.0)
                    if score_g > best_score_g:
                        best_score_g = score_g
                        g_offset = (dx, dy)
                    if score_r > best_score_r:
                        best_score_r = score_r
                        r_offset = (dx, dy)
        print(f"Green offset: {g_offset}")
        print(f"Red offset: {r_offset}")
        return g_offset, r_offset

    def downsample_image(self, image: np.ndarray, scale_factor: float) -> np.ndarray:
        """
        Downsample image by gaussian blur using provided scale_factor
        """
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        new_w = max(1, int(image.shape[1] * scale_factor))
        new_h = max(1, int(image.shape[0] * scale_factor))
        downsampled = cv2.resize(
            blurred,
            (new_w, new_h),
            interpolation=cv2.INTER_AREA if scale_factor < 1.0 else cv2.INTER_CUBIC,
        )
        return downsampled
    
    def align_each_level(self, channels: Tuple[np.ndarray, np.ndarray, np.ndarray], g_offset: Tuple[int, int], r_offset: Tuple[int, int], search_range: int, metric: str) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Align each level of the image
        """
        blue_downsampled = self.downsample_image(channels[0], 0.5)
        green_downsampled = self.downsample_image(channels[1], 0.5)
        red_downsampled = self.downsample_image(channels[2], 0.5)

        if metric == 'l2':
            best_score_g = float('inf')
            best_score_r = float('inf')
        elif metric == 'ncc':
            best_score_g = float('-inf')
            best_score_r = float('-inf')
        elif metric == 'ssim':
            best_score_g = float('-inf')
            best_score_r = float('-inf')

        # g_offset
        best_g = g_offset
        for dx in range(-search_range + g_offset[0], search_range + 1 + g_offset[0]):
            for dy in range(-search_range + g_offset[1], search_range + 1 + g_offset[1]):
                shifted_green = self.shift_image(green_downsampled, dx, dy)
                if metric == 'l2':
                    score_g = self.l2_distance(shifted_green, blue_downsampled)
                    if score_g < best_score_g:
                        best_score_g = score_g
                        best_g = (dx, dy)
                elif metric == 'ncc':
                    score_g = self.normalized_cross_correlation(shifted_green, blue_downsampled)
                    if score_g > best_score_g:
                        best_score_g = score_g
                        best_g = (dx, dy)
                elif metric == 'ssim':
                    score_g = ssim(shifted_green, blue_downsampled, data_range=1.0)
                    if score_g > best_score_g:
                        best_score_g = score_g
                        best_g = (dx, dy)
        # r_offset
        for dx in range(-search_range + r_offset[0], search_range + 1 + r_offset[0]):
            for dy in range(-search_range + r_offset[1], search_range + 1 + r_offset[1]):
                shifted_red = self.shift_image(red_downsampled, dx, dy)
                if metric == 'l2':
                    score_r = self.l2_distance(shifted_red, blue_downsampled)
                    if score_r < best_score_r:
                        best_score_r = score_r
                        best_r = (dx, dy)
                elif metric == 'ncc':
                    score_r = self.normalized_cross_correlation(shifted_red, blue_downsampled)
                    if score_r > best_score_r:
                        best_score_r = score_r
                        best_r = (dx, dy)
                elif metric == 'ssim':
                    score_r = ssim(shifted_red, blue_downsampled, data_range=1.0)
                    if score_r > best_score_r:
                        best_score_r = score_r
                        best_r = (dx, dy)
        return (best_g, best_r)

    def recursive_align(self, channels: Tuple[np.ndarray, np.ndarray, np.ndarray], offset_last_level: Tuple[Tuple[int, int], Tuple[int, int]], search_range: int, metric: str) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Deprecated recursive align (kept for reference). Use align_channels_pyramid instead.
        """
        return offset_last_level
    
    def align_channels_pyramid(self, 
                              num_levels: int = 4,
                              search_range: int = 15,
                              metric: str = 'ncc',
                              use_edges: bool = False) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        TODO: Implement multi-scale pyramid alignment
        Align channels using image pyramid for efficiency with large images
        
        Args:
            num_levels: Number of pyramid levels
            search_range: Search range at finest level
            metric: Distance metric ('l2' or 'ncc' or 'ssim')
            use_edges: Whether to apply Sobel edge detection before alignment
            
        Returns:
            Tuple of ((gx, gy), (rx, ry)) displacement vectors
        """
        if self.channels is None:
            raise ValueError("No channels loaded. Call split_channels() first.")
        
        # TODO: Implement pyramid alignment here
        # You need to:
        # 1. Create image pyramid by downsampling
        # 2. Start with coarse alignment at the top level
        # 3. Refine alignment at each level going down the pyramid
        # 4. Scale up offsets between levels
        
        print("Running pyramid alignment...")
        blue, green, red = self.channels

        # Apply Sobel edge detection if requested
        if use_edges:
            print("Using Sobel edge detection for pyramid alignment")
            blue = self._apply_sobel(blue)
            green = self._apply_sobel(green)
            red = self._apply_sobel(red)

        # Build pyramids (index 0 = largest/original, last = smallest)
        pyr_b = [blue]
        pyr_g = [green]
        pyr_r = [red]
        for _ in range(1, num_levels):
            pyr_b.append(self.downsample_image(pyr_b[-1], 0.5))
            pyr_g.append(self.downsample_image(pyr_g[-1], 0.5))
            pyr_r.append(self.downsample_image(pyr_r[-1], 0.5))

        g_offset = (0, 0)
        r_offset = (0, 0)

        # Iterate from smallest to largest (coarse-to-fine)
        for level in reversed(range(num_levels)):
            bL = pyr_b[level]
            gL = pyr_g[level]
            rL = pyr_r[level]

            # Apply current offsets to this level
            gL = self.shift_image(gL, g_offset[0], g_offset[1])
            rL = self.shift_image(rL, r_offset[0], r_offset[1])

            # Use a small local search window at each level
            local_range = max(1, search_range // (2 ** (num_levels - 1 - level)))

            bL_c = self._crop_border(bL, 0.1)

            best_g = g_offset
            best_r = r_offset
            if metric == 'l2':
                best_g_score = float('inf')
                best_r_score = float('inf')
            else:
                best_g_score = float('-inf')
                best_r_score = float('-inf')

            for dx in range(-local_range, local_range + 1):
                for dy in range(-local_range, local_range + 1):
                    sg = self.shift_image(gL, dx, dy)
                    sr = self.shift_image(rL, dx, dy)
                    sg_c = self._crop_border(sg, 0.1)
                    sr_c = self._crop_border(sr, 0.1)

                    if metric == 'l2':
                        score_g = self.l2_distance(sg_c, bL_c)
                        score_r = self.l2_distance(sr_c, bL_c)
                        if score_g < best_g_score:
                            best_g_score = score_g
                            best_g = (g_offset[0] + dx, g_offset[1] + dy)
                        if score_r < best_r_score:
                            best_r_score = score_r
                            best_r = (r_offset[0] + dx, r_offset[1] + dy)
                    elif metric == 'ncc':
                        score_g = self.normalized_cross_correlation(sg_c, bL_c)
                        score_r = self.normalized_cross_correlation(sr_c, bL_c)
                        if score_g > best_g_score:
                            best_g_score = score_g
                            best_g = (g_offset[0] + dx, g_offset[1] + dy)
                        if score_r > best_r_score:
                            best_r_score = score_r
                            best_r = (r_offset[0] + dx, r_offset[1] + dy)
                    else:  # ssim
                        score_g = ssim(sg_c, bL_c, data_range=1.0)
                        score_r = ssim(sr_c, bL_c, data_range=1.0)
                        if score_g > best_g_score:
                            best_g_score = score_g
                            best_g = (g_offset[0] + dx, g_offset[1] + dy)
                        if score_r > best_r_score:
                            best_r_score = score_r
                            best_r = (r_offset[0] + dx, r_offset[1] + dy)

            g_offset = best_g
            r_offset = best_r

            # Scale offsets up for the next larger level
            if level > 0:
                g_offset = (g_offset[0] * 2, g_offset[1] * 2)
                r_offset = (r_offset[0] * 2, r_offset[1] * 2)

        return g_offset, r_offset
    
    def resize_image(self, image: np.ndarray, scale_factor: float) -> np.ndarray:
        """
        Resize image by scale factor using OpenCV
        
        Args:
            image: Input image
            scale_factor: Scale factor (e.g., 0.5 for half size)
            
        Returns:
            Resized image
        """
        if scale_factor >= 1.0:
            interpolation = cv2.INTER_CUBIC
        else:
            interpolation = cv2.INTER_AREA
        
        new_width = int(image.shape[1] * scale_factor)
        new_height = int(image.shape[0] * scale_factor)
        
        resized = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
        return resized
    
    def create_aligned_image(self, g_offset: Tuple[int, int], r_offset: Tuple[int, int]) -> np.ndarray:
        """
        Create the final aligned color image
        
        Args:
            g_offset: (dx, dy) offset for green channel
            r_offset: (dx, dy) offset for red channel
            
        Returns:
            Aligned RGB image
        """
        if self.channels is None:
            raise ValueError("No channels loaded. Call split_channels() first.")
        
        blue, green, red = self.channels
        
        # Apply offsets
        aligned_green = self.shift_image(green, g_offset[0], g_offset[1])
        aligned_red = self.shift_image(red, r_offset[0], r_offset[1])
        
        # Stack channels to create RGB image
        self.aligned_image = np.stack([aligned_red, aligned_green, blue], axis=2)
        
        print(f"Created aligned image with shape: {self.aligned_image.shape}")
        return self.aligned_image
    
    def display_results(self, save_path: Optional[str] = None):
        """
        Display the original channels and final aligned image
        
        Args:
            save_path: Optional path to save the result image
        """
        if self.channels is None or self.aligned_image is None:
            raise ValueError("No aligned image available. Run alignment first.")
        
        blue, green, red = self.channels
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Display individual channels
        axes[0, 0].imshow(blue, cmap='gray')
        axes[0, 0].set_title('Blue Channel')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(green, cmap='gray')
        axes[0, 1].set_title('Green Channel')
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(red, cmap='gray')
        axes[1, 0].set_title('Red Channel')
        axes[1, 0].axis('off')
        
        # Display aligned result
        axes[1, 1].imshow(self.aligned_image)
        axes[1, 1].set_title('Aligned Color Image')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Results saved to: {save_path}")
        
        plt.show()


def main():
    """Example usage - implement your alignment functions"""
    
    # Initialize colorizer
    colorizer = ProkudinGorskiiColorizer()
    
    # Example usage - replace with your actual image path
    data_path = "data/"
    image_path = "harvesters.tif"  # or "cathedral.jpg" or any .tif file
    
    try:
        # Load image
        print("Loading image...")
        colorizer.load_image(data_path + image_path)
        
        # Split into channels
        print("Splitting channels...")
        colorizer.split_channels()
        
        # TODO: Implement and test your alignment functions
        print("\n=== Choose alignment function ===")
        scale = input("Enter 1 for pyramid, 2 for single-scale: ")
        use_edges = input("Use Sobel edge detection? (y/n): ").lower() == 'y'
        
        # Show edge images if using edge detection
        if use_edges:
            print("\n=== Displaying Edge Images ===")
            colorizer.display_edge_images(f"{data_path}{image_path}_edges.jpg")
        
        if scale == '1':
            g_offset, r_offset = colorizer.align_channels_pyramid(
                num_levels=4,
                search_range=15,
                metric='ssim',
                use_edges=use_edges
            )
        elif scale == '2':
            g_offset, r_offset = colorizer.align_channels_single_scale(
                search_range=15,
                metric='ssim',
                use_edges=use_edges
            )
        
        # Create aligned image
        aligned_image = colorizer.create_aligned_image(g_offset, r_offset)
        
        # Display results
        colorizer.display_results(f"{data_path}{image_path}_aligned.jpg")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure the image file exists in the current directory.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
