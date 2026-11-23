#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
INT6011 – Assignment 1
The Digital Camera Pipeline 

This file is structured to mirror the sections in the assignment PDF:

  2. Raw Image Extraction           → load_raw_tiff()
  3. Linearization                  → linearize()
  4. The Bayer Pattern              → _build_bayer_masks(), bayer_pattern
  5. White Balancing                → white_balance(...)
  6. Demosaicing                    → demosaic_bilinear()
  7. Color Space Correction         → color_space_correction(...)
  8. Brightness & Gamma Encoding    → brightness_and_gamma()
  9. Compression                    → save_png_jpeg(...)

Your job:
  • Fill in the TODOs using the instructions from the PDF.
  • Experiment with different Bayer patterns and WB methods, and report your observations.
"""

import os
import numpy as np
from skimage import io, color, img_as_ubyte
from scipy.ndimage import convolve
import matplotlib.pyplot as plt

from skimage import io

SUPPORTED_PATTERNS = ("RGGB", "BGGR", "GRBG", "GBRG")


class RawCameraPipeline:
    """
    RAW → sRGB pipeline skeleton.

    You are expected to implement each stage according to the assignment text.
    """

    def __init__(self,
                 tiff_path: str,
                 black_level: int,
                 white_level: int,
                 wb_multipliers,
                 bayer_pattern: str = "RGGB"):
        """
        Parameters
        ----------
        tiff_path      : path to the TIFF produced 
        black_level    : "darkness" value from dcraw
        white_level    : "saturation" value from dcraw
        wb_multipliers : [r_scale, g_scale, b_scale, g_scale] from dcraw
        bayer_pattern  : one of 'RGGB', 'BGGR', 'GRBG', 'GBRG'
        """
        self.tiff_path = tiff_path
        self.black = float(black_level)
        self.white = float(white_level)
        self.wb_mults = np.array(wb_multipliers, dtype=np.float64)

        self.pattern = bayer_pattern.upper()
        if self.pattern not in SUPPORTED_PATTERNS:
            raise ValueError(f"Unsupported Bayer pattern: {self.pattern}")

        # Buffers for intermediate results
        self.raw_image = None      
        self.linear_image = None   
        self.rgb_linear = None     
        self.rgb_srgb = None       

    # ------------------------------------------------------------------
    # Section 2: Raw Image Extraction
    # ------------------------------------------------------------------
    def load_raw_tiff(self):
        """
        TODO (Section 2.2 – Generating the Linear TIFF):

        1. Load the TIFF using skimage.io.imread.
        2. Convert the array to double-precision floats (np.float64).
        3. Print:
           - width and height
           - min and max raw values
           - approximate bits per pixel (hint: log2(max_value + 1))
        4. Store the loaded image in self.raw_image.
        """
        # TODO: Implement this function.
        self.raw_image = io.imread(self.tiff_path).astype(np.float64)
        h, w = self.raw_image.shape
        min_val = np.min(self.raw_image)
        max_val = np.max(self.raw_image)
        bits_per_pixel = np.log2(max_val + 1)

        print(f"Width: {w}, Height: {h}")
        print(f"Min raw value: {min_val}, Max raw value: {max_val}")
        print(f"Approximate bits per pixel: {bits_per_pixel}")

    # ------------------------------------------------------------------
    # Section 3: Linearization
    # ------------------------------------------------------------------
    def linearize(self):
        """
        TODO (Section 3 – Linearization):

        Implement the formula to map raw pixel P_raw to P_linear:

            P_linear = (P_raw - black) / (white - black)

        with:
            - values <= black mapped to 0
            - values >= white mapped to 1
            - everything in between linearly scaled to [0,1]

        Steps:
        1. Use self.raw_image as input.
        2. Apply the formula.
        3. Clip the result to [0, 1].
        4. Store the result in self.linear_image.
        """
        # TODO: Implement this function.
        if self.raw_image is None:
            raise ValueError("Raw image not loaded. Call load_raw_tiff() first.")

        self.linear_image = (self.raw_image - self.black) / (self.white - self.black)
        self.linear_image = np.clip(self.linear_image, 0, 1)

    # ------------------------------------------------------------------
    # Section 4: The Bayer Pattern
    # ------------------------------------------------------------------
    @staticmethod
    def _build_bayer_masks(h, w, pattern):
        """
        TODO (Section 4 – Pattern Phase Identification):

        Build boolean masks for where the R, G, and B pixels live
        in the Bayer mosaic.

        The Bayer tile is 2x2:
            pattern[0] pattern[1]
            pattern[2] pattern[3]

        Example for 'RGGB':

            R G
            G B

        Steps:
        1. Create three boolean arrays: mask_r, mask_g, mask_b of shape (h, w).
        2. Construct boolean masks, setting True where each color channel is located.        
        3. Return mask_r, mask_g, mask_b.
        """
        # TODO: Implement this function.
        mask_r = np.zeros((h, w), dtype=bool)
        mask_g = np.zeros((h, w), dtype=bool)
        mask_b = np.zeros((h, w), dtype=bool)

        # Fill the masks based on the Bayer pattern
        for i in range(h):
            for j in range(w):
                tile_row = i % 2
                tile_col = j % 2
                color = pattern[tile_row * 2 + tile_col]
                if color == 'R':
                    mask_r[i, j] = True
                elif color == 'G':
                    mask_g[i, j] = True
                elif color == 'B':
                    mask_b[i, j] = True

        return mask_r, mask_g, mask_b

    # ------------------------------------------------------------------
    # Section 5: White Balancing
    # ------------------------------------------------------------------
    def white_balance(self, method='camera', manual_region=None):
        """
        Task 4: White Balance.
        
        Implements 4 strategies to compute scaling factors (s_r, s_g, s_b).
        The Green channel is usually kept as the baseline (s_g = 1).
        
        Args:
            method (str): One of 'camera', 'white_world', 'gray_world', 'manual'.
            manual_region (tuple): (y_start, y_end, x_start, x_end) for 'manual' method.
            
        Formulas:
            1. White World (Max-RGB):
               s_r = max(G) / max(R)
               s_b = max(G) / max(B)
               
            2. Gray World:
               s_r = mean(G) / mean(R)
               s_b = mean(G) / mean(B)
               
            3. Camera Presets:
               Use self.wb_mults [m1, m2, m3, m4].
               Normalize so Green is 1.0: s_r = m1/m2, s_b = m3/m2.
               
            4. Manual:
               Calculate means of R, G, B within the manual_region.
               s_r = mean(G_patch) / mean(R_patch)
               s_b = mean(G_patch) / mean(B_patch)
        """
        print(f"Step 4: White Balancing ({method})...")
        if self.linear_image is None:
            raise ValueError("Image must be linearized first.")
        
        img = self.linear_image.copy()
        h, w = img.shape
        
        # TODO: 1. Build R/G/B masks using self._build_bayer_masks()
        mask_r, mask_g, mask_b = self._build_bayer_masks(h, w, self.pattern)

        # TODO: 2. Compute scalars based on the selected method
        if method == 'white_world':
            max_r = np.percentile(img[mask_r], 99)
            max_g = np.percentile(img[mask_g], 99)
            max_b = np.percentile(img[mask_b], 99)
            s_r = max_g / max_r if max_r != 0 else 1.0
            s_b = max_g / max_b if max_b != 0 else 1.0
            s_g = 1.0
        elif method == 'gray_world':
            mean_r = np.mean(img[mask_r])
            mean_g = np.mean(img[mask_g])
            mean_b = np.mean(img[mask_b])
            s_r = mean_g / mean_r if mean_r != 0 else 1.0
            s_b = mean_g / mean_b if mean_b != 0 else 1.0
            s_g = 1.0
        elif method == 'camera':   
            m1, m2, m3, m4 = self.wb_mults
            s_r = m1 / m2 if m2 != 0 else 1.0
            s_b = m3 / m2 if m2 != 0 else 1.0
            s_g = 1.0
        elif method == 'manual':
            if manual_region is None:
                raise ValueError("manual_region must be provided for 'manual' method.")
            y_start, y_end, x_start, x_end = manual_region
            patch = img[y_start:y_end, x_start:x_end]
            mask_r_patch, mask_g_patch, mask_b_patch = self._build_bayer_masks(
                y_end - y_start, x_end - x_start, self.pattern)
            mean_r = np.mean(patch[mask_r_patch])
            mean_g = np.mean(patch[mask_g_patch])
            mean_b = np.mean(patch[mask_b_patch])
            s_r = mean_g / mean_r if mean_r != 0 else 1.0
            s_b = mean_g / mean_b if mean_b != 0 else 1.0
            s_g = 1.0
        # TODO: 3. Apply scalars to the pixels in 'img'
        img[mask_r] *= s_r
        img[mask_g] *= s_g  
        img[mask_b] *= s_b
        
        # TODO: 4. Clip values to [0, 1] and update self.linear_image
        img = np.clip(img, 0, 1)
        self.linear_image = img

    # ------------------------------------------------------------------
    # Section 6: Demosaicing (Bilinear Interpolation)
    # ------------------------------------------------------------------
    def demosaic_bilinear_forij(self):
        """
        TODO (Section 6 – Demosaicing):

        Implement bilinear demosaicing using convolution as in the figure
        and text (Fig. "Demosaicing Logic").

        Use self.linear_image (white-balanced mosaic) and produce a full
        H x W x 3 RGB image in self.rgb_linear.

        Hints:
        - Use _build_bayer_masks(...) to get mask_r, mask_g, mask_b.
        - For Green:
            Known at mask_g locations.
            At R/B locations, interpolate from 4 cross neighbors (up/down/left/right).
              You can use a cross-shaped kernel, e.g.:

                  [0, 1, 0]
                  [1, 0, 1] / 4
                  [0, 1, 0]

        - For Red at Blue:
            Interpolate from 4 diagonal neighbors (kernel with diagonals).

        - For Red/Blue at Green:
            Interpolate from horizontal/vertical neighbors (only 2 neighbors),
              so be careful with normalization.

        - Mirror padding (mode="mirror") is acceptable.
        """
        # TODO: Implement this function.
        h, w = self.linear_image.shape
        mask_r, mask_g, mask_b = self._build_bayer_masks(h, w, self.pattern)
        
        rgb_image = np.zeros((h, w, 3), dtype=np.float64)

        for i in range(h):
            for j in range(w):
                if mask_r[i, j]:
                    rgb_image[i, j, 0] = self.linear_image[i, j]  # Red
                elif mask_g[i, j]:
                    rgb_image[i, j, 1] = self.linear_image[i, j]  # Green
                elif mask_b[i, j]:
                    rgb_image[i, j, 2] = self.linear_image[i, j]  # Blue

        for i in range(h):
            for j in range(w):
                if mask_g[i, j]:
                    # if green pixel, make up red and blue :v
                    neighbors = []
                    if i > 0:
                        neighbors.append(rgb_image[i - 1, j, 0])
                    if i < h - 1:
                        neighbors.append(rgb_image[i + 1, j, 0])
                    # Interpolate red channel at green pixel
                    rgb_image[i, j, 0] = np.mean(neighbors) if neighbors else rgb_image[i, j, 0]

                    neighbors = []
                    if j > 0:
                        neighbors.append(rgb_image[i, j - 1, 2])
                    if j < w - 1:
                        neighbors.append(rgb_image[i, j + 1, 2])
                    # Interpolate blue channel at green pixel
                    rgb_image[i, j, 2] = np.mean(neighbors) if neighbors else rgb_image[i, j, 2]

                elif mask_r[i, j]:
                    # If red pixel, make up green and blue :))
                    neighbors = []
                    if i > 0:
                        neighbors.append(rgb_image[i - 1, j, 1])
                    if i < h - 1:
                        neighbors.append(rgb_image[i + 1, j, 1])
                    if j > 0:
                        neighbors.append(rgb_image[i, j - 1, 1])
                    if j < w - 1:
                        neighbors.append(rgb_image[i, j + 1, 1])
                    # Interpolate green channel at red pixel
                    rgb_image[i, j, 1] = np.mean(neighbors) if neighbors else rgb_image[i, j, 1]

                    neighbors = []
                    if i > 0 and j > 0:
                        neighbors.append(rgb_image[i - 1, j - 1, 2])
                    if i > 0 and j < w - 1:
                        neighbors.append(rgb_image[i - 1, j + 1, 2])   
                    if i < h - 1 and j > 0:
                        neighbors.append(rgb_image[i + 1, j - 1, 2])
                    if i < h - 1 and j < w - 1:
                        neighbors.append(rgb_image[i + 1, j + 1, 2])
                    # Interpolate blue channel at red pixel
                    rgb_image[i, j, 2] = np.mean(neighbors) if neighbors else rgb_image[i, j, 2]

                elif mask_b[i, j]:
                    # If blue pixel, make up green and red :>
                    neighbors = []
                    if i > 0:
                        neighbors.append(rgb_image[i - 1, j, 1])
                    if i < h - 1:
                        neighbors.append(rgb_image[i + 1, j, 1])
                    if j > 0:
                        neighbors.append(rgb_image[i, j - 1, 1])
                    if j < w - 1:
                        neighbors.append(rgb_image[i, j + 1, 1])
                    # Interpolate green channel at blue pixel
                    rgb_image[i, j, 1] = np.mean(neighbors) if neighbors else rgb_image[i, j, 1]

                    neighbors = []
                    if i > 0 and j > 0:
                        neighbors.append(rgb_image[i - 1, j - 1, 0])
                    if i > 0 and j < w - 1:
                        neighbors.append(rgb_image[i - 1, j + 1, 0])
                    if i < h - 1 and j > 0:
                        neighbors.append(rgb_image[i + 1, j - 1, 0])
                    if i < h - 1 and j < w - 1:
                        neighbors.append(rgb_image[i + 1, j + 1, 0])
                    # Interpolate red channel at blue pixel
                    rgb_image[i, j, 0] = np.mean(neighbors) if neighbors else rgb_image[i, j, 0]

        self.rgb_linear = rgb_image

    def demosaic_bilinear(self):
        """
        Section 6 – Demosaicing (Sử dụng Convolution để tối ưu hiệu năng)
        """
        if self.linear_image is None:
            raise ValueError("Image must be white-balanced first.")

        h, w = self.linear_image.shape
        mask_r, mask_g, mask_b = self._build_bayer_masks(h, w, self.pattern)

        # 1. Trích xuất các kênh màu tại các vị trí đã biết (Mảng thưa)
        r_known = np.where(mask_r, self.linear_image, 0.0)
        g_known = np.where(mask_g, self.linear_image, 0.0)
        b_known = np.where(mask_b, self.linear_image, 0.0)

        # 2. Định nghĩa các Kernels nội suy tuyến tính (Bilinear Filters)
        # Lọc kênh Green tại các vị trí Red/Blue (4 điểm xóm giềng hình chữ thập)
        k_green = np.array([
            [0.0, 0.25, 0.0],
            [0.25, 0.0, 0.25],
            [0.0, 0.25, 0.0]
        ], dtype=np.float64)

        # Lọc Red tại điểm Blue (hoặc Blue tại điểm Red) (4 điểm xóm giềng chéo)
        k_diag = np.array([
            [0.25, 0.0, 0.25],
            [0.0,  0.0, 0.0],
            [0.25, 0.0, 0.25]
        ], dtype=np.float64)

        # Lọc Red/Blue tại các điểm Green (2 điểm xóm giềng ngang hoặc dọc)
        k_cross = np.array([
            [0.0, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.0]
        ], dtype=np.float64)

        # 3. Nội suy kênh Green
        g_interpolated = convolve(g_known, k_green, mode='mirror')
        g_channel = np.where(mask_g, g_known, g_interpolated)

        # 4. Nội suy kênh Red
        r_at_b = convolve(r_known, k_diag, mode='mirror')
        r_at_g = convolve(r_known, k_cross, mode='mirror')
        r_channel = np.where(mask_r, r_known, np.where(mask_b, r_at_b, r_at_g))

        # 5. Nội suy kênh Blue
        b_at_r = convolve(b_known, k_diag, mode='mirror')
        b_at_g = convolve(b_known, k_cross, mode='mirror')
        b_channel = np.where(mask_b, b_known, np.where(mask_r, b_at_r, b_at_g))

        # 6. Ghép 3 kênh lại thành ảnh RGB

        self.rgb_linear = np.stack([r_channel, g_channel, b_channel], axis=-1)


    # ------------------------------------------------------------------
    # Section 7: Color Space Correction
    # ------------------------------------------------------------------
    def color_space_correction(self, M_xyz_to_cam: np.ndarray):
        """
        TODO (Section 7 – Color Space Correction):

        Convert from camera RGB to sRGB using CIE XYZ as an intermediate space.

        Given:
          - M_xyz_to_cam : M_{XYZ→Cam}, camera-specific (from dcraw/adobe_coeff).
          - M_srgb_to_xyz: fixed standard sRGB→XYZ matrix.

        You must:
        1. Compute M_srgb_to_cam = M_xyz_to_cam · M_srgb_to_xyz.
        2. Invert it to get M_cam_to_srgb.
        3. For each pixel in self.rgb_linear (camera RGB), multiply by M_cam_to_srgb.
        4. Store the result back into self.rgb_linear (now in sRGB linear space).
        5. Clip values to [0, 1].

        Note:
        - You can reshape the image to (N, 3) for a single matrix multiplication.
        """
        # TODO: Implement this function.
        if self.rgb_linear is None:
            raise ValueError("Image must be demosaiced first.")
        # Standard sRGB to XYZ matrix
        M_srgb_to_xyz = np.array([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ], dtype=np.float64)

        M_srgb_to_cam = M_xyz_to_cam @ M_srgb_to_xyz
        M_cam_to_srgb = np.linalg.inv(M_srgb_to_cam)

        M_cam_to_srgb = M_cam_to_srgb / M_cam_to_srgb.sum(axis=1, keepdims=True)

        # Reshape the image to (N, 3) for matrix multiplication
        h, w, c = self.rgb_linear.shape
        rgb_flat = self.rgb_linear.reshape(-1, 3)
        rgb_corrected_flat = np.dot(rgb_flat, M_cam_to_srgb.T)
        self.rgb_linear = rgb_corrected_flat.reshape(h, w, 3)

        self.rgb_linear = np.clip(self.rgb_linear, 0, 1)    


    # ------------------------------------------------------------------
    # Section 8: Brightness and Gamma Encoding
    # ------------------------------------------------------------------
    
    def brightness_and_gamma(self, target_mean: float = 0.25):
        """
        TODO (Section 8 – Brightness and Gamma):

        Step 1: Brightness Scaling
        - Convert self.rgb_linear to grayscale using color.rgb2gray.
        - Compute the current mean gray level.
        - Compute a scale factor so that the mean gray becomes target_mean (~0.25).
        - Multiply the RGB image by this factor and clip to [0, 1].

        Step 2: sRGB Gamma Encoding
        - For each pixel/channel C_linear in [0,1], apply:

            if C_linear <= 0.0031308:
                C_nonlin = 12.92 * C_linear
            else:
                C_nonlin = 1.055 * C_linear^(1/2.4) - 0.055

        - Store the gamma-encoded result in self.rgb_srgb.
        """
        # TODO: Implement this function.
        if self.rgb_linear is None:
            raise ValueError("Image must be color-corrected first.")
        
        # Step 1: Brightness Scaling
        gray_image = color.rgb2gray(self.rgb_linear)
        current_mean = np.mean(gray_image)
        scale_factor = target_mean / current_mean if current_mean != 0 else 1.
        scaled_rgb = self.rgb_linear * scale_factor
        scaled_rgb = np.clip(scaled_rgb, 0, 1)

        # Step 2: sRGB Gamma Encoding
        self.rgb_srgb = np.where(scaled_rgb <= 0.0031308,
                                 12.92 * scaled_rgb,
                                 1.055 * np.power(scaled_rgb, 1/2.4) - 0.055)
        self.rgb_srgb = np.clip(self.rgb_srgb, 0, 1)


    def extract_srgb_channels(self, channel='all'):
        """
        Tách 3 kênh màu từ self.rgb_srgb.
        
        Args:
            mode (str): 'grayscale' (trả về 3 mảng 2D) 
                        hoặc 'color' (trả về 3 mảng 3D chỉ giữ 1 màu).
            channel (str): 'all', 'red', 'green', 'blue'
        """
        if self.rgb_srgb is None:
            raise ValueError("Call brightness_and_gamma() first.")
            
        r = self.rgb_srgb[:, :, 0]
        g = self.rgb_srgb[:, :, 1]
        b = self.rgb_srgb[:, :, 2]
        
        zeros = np.zeros_like(r)
        r_3d = np.stack([r, zeros, zeros], axis=-1)
        g_3d = np.stack([zeros, g, zeros], axis=-1)
        b_3d = np.stack([zeros, zeros, b], axis=-1)
        if channel == 'red':
            self.rgb_srgb = r_3d
        elif channel == 'green':
            self.rgb_srgb = g_3d
        elif channel == 'blue':
            self.rgb_srgb = b_3d
        else:
            self.rgb_srgb = np.stack([r, g, b], axis=-1)

    # ------------------------------------------------------------------
    # Section 9: Compression
    # ------------------------------------------------------------------
    def save_png_jpeg(self, prefix: str, quality: int = 95):
        """
        TODO (Section 9 – Compression):

        1. Convert self.rgb_srgb (float [0,1]) to 8-bit using img_as_ubyte.
        2. Save as:
            prefix + ".png"  (lossless)
            prefix + ".jpg"  (with given quality, e.g., 95)
        3. Use os.path.getsize(...) to get file sizes.
        4. Print:
            - PNG size in KB
            - JPEG size in KB
            - Compression ratio = PNG_size / JPEG_size

        For the report:
        - You can call this multiple times with different JPEG quality values
          and visually inspect which quality is "indistinguishable" from PNG.
        """
        # TODO: Implement this function.

        img_8bit = img_as_ubyte(self.rgb_srgb)
        png_path = f"{prefix}.png"
        jpeg_path = f"{prefix}.jpg"
        io.imsave(png_path, img_8bit)
        io.imsave(jpeg_path, img_8bit, quality=quality)

        png_size = os.path.getsize(png_path) / 1024.0  # in KB
        jpeg_size = os.path.getsize(jpeg_path) / 1024.0 # in KB
        compression_ratio = png_size / jpeg_size if jpeg_size != 0 else 0

        print(f"PNG size: {png_size:.2f} KB")
        print(f"JPEG size: {jpeg_size:.2f} KB")
        print(f"Compression ratio: {compression_ratio:.2f}")

    def show(self, title="Final Developed Image"):
        """
        Optional helper: show the final sRGB image with matplotlib.

        You can use this to visually compare different Bayer patterns and
        white balance methods.
        """
        if self.rgb_srgb is None:
            raise RuntimeError("Call brightness_and_gamma() first.")
        plt.figure(figsize=(6, 6))
        plt.imshow(self.rgb_srgb)
        plt.axis("off")
        plt.title(title)
        plt.show()


if __name__ == "__main__":
    # === Fill these from dcraw -4 -d -v -w -T campus.CR2 ===
    BLACK = 2048          # replace with "darkness"
    WHITE = 15488     # replace with "saturation"
    WB_MULTS = [1.858398, 1.0, 1.723633, 1.0]   # replace with [R G B G] from dcraw

    # Try different Bayer patterns here: 'RGGB', 'BGGR', 'GRBG', 'GBRG'
    BAYER_PATTERN = "RGGB"

    # Example XYZ→Camera matrix (from dcraw/adobe_coeff, for a Canon body)
    # Values: { 6722,-635,-963,-4287,12460,2028,-908,2162,5668 } / 10000
    M_XYZ_CAM = np.array([
        [6722,  -635,  -963],
        [-4287, 12460,  2028],
        [ -908,  2162,  5668]
    ], dtype=np.float64) / 10000.0

    pipeline = RawCameraPipeline(
        tiff_path="data/campus.tiff",
        black_level=BLACK,
        white_level=WHITE,
        wb_multipliers=WB_MULTS,
        bayer_pattern=BAYER_PATTERN
    )

    # Once you implement each step, uncomment these lines one by one to test.
    pipeline.load_raw_tiff()
    pipeline.linearize()
    # pipeline.white_balance(method="manual", manual_region=(690, 700, 2920, 2930))   # or "gray_world", "white_world", "manual"
    pipeline.white_balance(method="camera")  # or "gray_world", "white_world", "manual"
    pipeline.demosaic_bilinear()
    pipeline.color_space_correction(M_XYZ_CAM)
    pipeline.brightness_and_gamma()

    pipeline.extract_srgb_channels(channel='all')  # or 'red', 'green', 'blue'

    os.makedirs("results", exist_ok=True)
    pipeline.save_png_jpeg("results/campus_output", quality=95)
    pipeline.show()
