# Digital Camera RAW-to-sRGB Processing Pipeline

A Python implementation of a digital camera RAW processing pipeline, developed for the INT3404 Image Processing course (Assignment 1). The project processes a single-channel linear TIFF extracted from a Canon CR2 RAW image and converts it into an sRGB image.

## Pipeline

1. RAW TIFF loading and linearization using black/white levels.
2. Bayer CFA phase evaluation (RGGB, BGGR, GRBG, GBRG).
3. White balance using camera presets, Gray World, White World (99th percentile), or a manually selected neutral patch.
4. Bilinear demosaicing using convolution kernels.
5. Camera RGB to linear sRGB color correction through CIE XYZ.
6. Global brightness scaling and sRGB gamma encoding.
7. PNG/JPEG export and file-size comparison.

## Experimental setup

The submitted experiment used a Canon CR2 image converted to a 16-bit linear TIFF. The report records an image size of 5796 x 3870 pixels (approximately 22.4 MP), a 14-bit sensor range, black level 2048, white level 15488, camera white-balance multipliers `[1.858398, 1.0, 1.723633, 1.0]`, and RGGB as the selected Bayer phase.

The repository does **not** include the original CR2/TIFF image or the course report. Place your own permitted input file at `data/campus.tiff` before running the scripts. The TIFF should be a single-channel, linear, 16-bit Bayer mosaic; do not use a demosaiced TIFF.

## Repository layout

```text
src/
  assign_code.py   # Main RAW-to-sRGB pipeline
  comparer.py      # Sparse/dense Bayer channel visualization
  helper.py        # Generates experiment figures and numerical results

data/
  README.md        # Input data preparation notes

results/
  figures/         # Place selected, shareable result figures here

requirements.txt
```

## Requirements

- Python 3.10 or later
- NumPy
- SciPy
- scikit-image
- Matplotlib

Install dependencies:

```bash
python -m venv .venv

# Windows PowerShell
.venv\\Scripts\\Activate.ps1

# macOS / Linux
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Preparing the input TIFF

The original experiment used `dcraw` to inspect the CR2 metadata and extract a linear Bayer TIFF. The commands documented in the assignment report were:

```bash
dcraw -4 -d -v -w -T campus.CR2
dcraw -4 -D -T campus.CR2
```

Use the metadata for your own RAW file and verify its black level, white level, and white-balance multipliers before changing the constants in the scripts. Copy the resulting linear TIFF to `data/campus.tiff`.

> The numeric camera parameters currently in the scripts belong to the original Canon sample used for this coursework. They are not universal settings for other cameras or RAW files.

## Running

Run commands from the repository root.

### Main pipeline

```bash
python -m src.assign_code
```

The script writes `results/campus_output.png` and `results/campus_output.jpg` and displays the developed image.

### Generate report experiment outputs

```bash
python -m src.helper
```

This generates comparison images and `results/report_outputs/results.txt`. The helper script evaluates the four Bayer phases and four white-balance methods, and records compression measurements.

### Compare sparse and demosaiced channels

```bash
python -m src.comparer
```

This opens a plot comparing a selected crop of the sparse Bayer channel with its demosaiced counterpart. Adjust `CROP_BOX` in `src/comparer.py` if the crop does not suit your input image.

## Notes and limitations

- The implementation uses bilinear interpolation; it is an educational pipeline, not a production-grade RAW developer.
- The global brightness scaling can clip highlights.
- The White World implementation uses the 99th percentile rather than the absolute maximum to reduce sensitivity to isolated saturated pixels.
- The camera-to-sRGB transform includes row normalization, as implemented in the coursework code. This is an experimental color correction choice, not a general-purpose color-management solution.
- Input dimensions and Bayer phase must match the metadata and CFA layout of the source camera.

## Development note

This is an academic coursework project. As disclosed in the accompanying assignment submission, AI tools assisted with understanding parts of the implementation and drafting the comparison/helper scripts and report. The author ran the pipeline and checked the reported outputs.

## Author

**Nguyen Manh Dung**  
VNU University of Engineering and Technology (VNU-UET)  
GitHub: [@ItsMeITUES](https://github.com/ItsMeITUES)

## Academic context

This repository contains code developed for INT3404 – Image Processing, Assignment 1. It is presented as an academic implementation and learning project.
