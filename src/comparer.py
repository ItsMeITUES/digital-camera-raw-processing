import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from scipy.ndimage import convolve


class DemosaicComparisonRunner:
    def __init__(self, tiff_path, black, white, wb_mults, pattern="RGGB"):
        self.tiff_path = tiff_path
        self.black = float(black)
        self.white = float(white)
        self.wb_mults = np.array(wb_mults, dtype=np.float64)
        self.pattern = pattern.upper()

        self.raw_image = None
        self.linear_image = None
        self.rgb_linear = None

    def load_and_linearize(self):
        # 1. Load RAW TIFF
        self.raw_image = io.imread(self.tiff_path).astype(np.float64)
        
        # 2. Linearize
        self.linear_image = (self.raw_image - self.black) / (self.white - self.black)
        self.linear_image = np.clip(self.linear_image, 0, 1)

    @staticmethod
    def build_bayer_masks(h, w, pattern):
        mask_r = np.zeros((h, w), dtype=bool)
        mask_g = np.zeros((h, w), dtype=bool)
        mask_b = np.zeros((h, w), dtype=bool)

        p = pattern.upper()
        masks = {'R': mask_r, 'G': mask_g, 'B': mask_b}

        masks[p[0]][0::2, 0::2] = True
        masks[p[1]][0::2, 1::2] = True
        masks[p[2]][1::2, 0::2] = True
        masks[p[3]][1::2, 1::2] = True

        return mask_r, mask_g, mask_b

    def white_balance_camera(self):
        h, w = self.linear_image.shape
        mask_r, mask_g, mask_b = self.build_bayer_masks(h, w, self.pattern)
        
        m1, m2, m3, _ = self.wb_mults
        s_r = m1 / m2 if m2 != 0 else 1.0
        s_b = m3 / m2 if m2 != 0 else 1.0
        s_g = 1.0

        img = self.linear_image.copy()
        img[mask_r] *= s_r
        img[mask_g] *= s_g
        img[mask_b] *= s_b

        self.linear_image = np.clip(img, 0, 1)

    def demosaic_bilinear(self):
        h, w = self.linear_image.shape
        mask_r, mask_g, mask_b = self.build_bayer_masks(h, w, self.pattern)

        r_known = np.where(mask_r, self.linear_image, 0.0)
        g_known = np.where(mask_g, self.linear_image, 0.0)
        b_known = np.where(mask_b, self.linear_image, 0.0)

        k_green = np.array([[0.0, 0.25, 0.0],
                            [0.25, 0.0, 0.25],
                            [0.0, 0.25, 0.0]], dtype=np.float64)

        k_diag = np.array([[0.25, 0.0, 0.25],
                           [0.0,  0.0, 0.0],
                           [0.25, 0.0, 0.25]], dtype=np.float64)

        k_cross = np.array([[0.0, 0.5, 0.0],
                            [0.5, 0.0, 0.5],
                            [0.0, 0.5, 0.0]], dtype=np.float64)

        # Interpolate Green
        g_interpolated = convolve(g_known, k_green, mode='mirror')
        g_channel = np.where(mask_g, g_known, g_interpolated)

        # Interpolate Red
        r_at_b = convolve(r_known, k_diag, mode='mirror')
        r_at_g = convolve(r_known, k_cross, mode='mirror')
        r_channel = np.where(mask_r, r_known, np.where(mask_b, r_at_b, r_at_g))

        # Interpolate Blue
        b_at_r = convolve(b_known, k_diag, mode='mirror')
        b_at_g = convolve(b_known, k_cross, mode='mirror')
        b_channel = np.where(mask_b, b_known, np.where(mask_r, b_at_r, b_at_g))

        # Output shape: (H, W, 3)
        self.rgb_linear = np.stack([r_channel, g_channel, b_channel], axis=-1)

    def plot_sparse_vs_dense_comparison(self, crop_region=(2500, 2530, 4450, 4480), channel='Red'):
        """
        Crop và vẽ so sánh dạng thưa (Sparse) vs dạng mịn (Dense) cho kênh màu được chọn.
        crop_region: (y_start, y_end, x_start, x_end)
        channel: 'Red', 'Green', hoặc 'Blue'
        """
        y1, y2, x1, x2 = crop_region
        h, w = self.linear_image.shape
        mask_r, mask_g, mask_b = self.build_bayer_masks(h, w, self.pattern)

        channel_map = {
            'Red': (0, mask_r),
            'Green': (1, mask_g),
            'Blue': (2, mask_b)
        }
        ch_idx, mask = channel_map[channel]

        # 1. Dạng THƯA (Sparse): Chỉ giữ lại pixel màu hiện tại, các pixel khác = 0
        sparse_channel = np.where(mask, self.linear_image, 0.0)
        crop_sparse = sparse_channel[y1:y2, x1:x2]

        # 2. Dạng MỊN (Dense): Lấy kênh tương ứng sau khi Bilinear Demosaicing
        dense_channel = self.rgb_linear[:, :, ch_idx]
        crop_dense = dense_channel[y1:y2, x1:x2]

        # 3. Vẽ đồ thị so sánh
        fig, axes = plt.subplots(1, 2, figsize=(13, 6))
        vmax_val = np.max(crop_dense) if np.max(crop_dense) > 0 else 1.0

        # Bên trái: Sparse
        im0 = axes[0].imshow(crop_sparse, cmap='gray', vmin=0, vmax=vmax_val)
        axes[0].set_title(f"Sparse {channel} Channel (Before Demosaic)\n[Grid rows: {y1}:{y2}, cols: {x1}:{x2}]", fontsize=11)
        axes[0].set_xlabel("Pixel X Offset")
        axes[0].set_ylabel("Pixel Y Offset")
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        # Bên phải: Dense
        im1 = axes[1].imshow(crop_dense, cmap='gray', vmin=0, vmax=vmax_val)
        axes[1].set_title(f"Dense {channel} Channel (After Demosaic)\n[Grid rows: {y1}:{y2}, cols: {x1}:{x2}]", fontsize=11)
        axes[1].set_xlabel("Pixel X Offset")
        axes[1].set_ylabel("Pixel Y Offset")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        plt.suptitle(f"Bayer Demosaicing Comparison ({channel} Channel)", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # --- Thông số đầu vào ---
    BLACK = 2048
    WHITE = 15488
    WB_MULTS = [1.858398, 1.0, 1.723633, 1.0]
    BAYER_PATTERN = "RGGB"
    TIFF_FILE = "data/campus.tiff"

    # Khởi tạo và chạy pipeline đến bước Demosaicing
    runner = DemosaicComparisonRunner(
        tiff_path=TIFF_FILE,
        black=BLACK,
        white=WHITE,
        wb_mults=WB_MULTS,
        pattern=BAYER_PATTERN
    )

    print("1. Loading & Linearizing...")
    runner.load_and_linearize()

    print("2. White Balancing...")
    runner.white_balance_camera()

    print("3. Demosaicing...")
    runner.demosaic_bilinear()

    # --- Vẽ đồ thị so sánh cho Báo cáo (Report) ---
    print("4. Plotting comparison...")
    # Bạn có thể điều chỉnh vùng crop (y_start, y_end, x_start, x_end) tùy theo ảnh
    CROP_BOX = (2500, 2530, 4450, 4480)
    
    # Hiển thị so sánh cho kênh Red
    runner.plot_sparse_vs_dense_comparison(crop_region=CROP_BOX, channel='Red')