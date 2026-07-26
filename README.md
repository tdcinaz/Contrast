# Contrast Residence Analyzer

Desktop interface for comparing contrast residence in paired fluoroscopy videos of a benchtop aneurysm model.

## Run

```bash
uv sync
uv run python main.py
```

The app loads these videos by default when they are present in the project folder:

- `PPI150_PreDeployment_Contrast.mov`
- `PPI150_PostDeployment_Contrast.mov`

Use the **File** menu to choose different videos if needed.

## Workflow

1. Use the playback controls to navigate to a frame where the aneurysm sac is visible. Adjust the playback speed slider as needed.
2. Drag with the mouse on each video to draw an ROI box around the saccular aneurysm.
3. Enable enhancement stages in order. Each change rebuilds both full videos, updates the video panels, and passes the cumulative result to every enabled downstream stage.
4. Click **Analyze ROIs**.
5. Compare the normalized contrast curves, raw ROI brightness curves, and residence-time cards.
6. Adjust the clearance threshold if needed. The metrics update from the already measured curves.
7. Export a CSV for downstream analysis.

Quality controls are available:

- **Automatic pillarbox crop** runs when each video is loaded. If black side bars are detected, frames are cropped to the fluoroscope content before ROI drawing, enhancement, and analysis.

- **NGC FFDNet (Docker)** is the default display-enhancement backend. It runs FFDNet in the local `nvcr.io/nvidia/pytorch:26.06-py3` image and exchanges configurable batches with the desktop process through shared memory. The container stays alive while its model, precision, and batch settings remain selected and is removed when those settings change or the app closes.
- **NGC DnCNN 15/25/50** provides three fixed-noise alternatives using the official grayscale KAIR checkpoints. `15` preserves the most fine structure, `25` is the balanced preset, and `50` applies the strongest denoising. Unlike FFDNet, each DnCNN checkpoint has a fixed trained noise level, so the FFDNet noise-sigma control is disabled for these models.
- **Native FFDNet (GPU)** runs the same model and checkpoint through the ARM CUDA 13 PyTorch packages locked in this project. Keep it as a fallback when Docker or the NGC image is unavailable. NGC and native output differ by no more than one 8-bit intensity level in validation.
- **FFDNet noise sigma** represents the assumed noise standard deviation on the 0-255 intensity scale. `10` is a conservative default for the included videos; lower values preserve more texture, while higher values produce a smoother but increasingly plastic image. The available range is `0` to `50`.
- **Batch frames** controls how many video frames are sent to the GPU together. The NGC default is `4`, which gave the best DnCNN throughput on full-resolution frames from the included videos; other GPUs and frame sizes may benefit from a different value. Larger batches consume more GPU and shared memory. **Precision** defaults to `FP16` for throughput, while `FP32` is available for numerical comparisons.
- **Enhancement pipeline** starts with all stages disabled. Its checkboxes are always applied top to bottom: gain stabilization, scanline correction, spatial denoising, motion-aware temporal filtering, CLAHE local contrast, and final Gaussian smoothing. Enable stages one at a time to inspect their cumulative effect. Unchecking a stage removes only that operation while preserving enabled downstream stages.
- **Spatial denoising** uses the selected deep model, or an edge-preserving bilateral filter when **Classical** is selected. Model, sigma, batch, and precision controls affect only this stage.
- Every pipeline change rebuilds enhanced caches for both complete videos with an on-screen progress overlay and immediately refreshes both panels. **Show original videos** clears all stages. Enhanced frames are JPEG-compressed in memory and reused during playback, paused viewing, and scrubbing. Display enhancement does not change the measured ROI values.
- **Correct gain drift in analysis** measures a reference region around the ROI on every frame and normalizes the ROI intensity against that reference before calculating contrast residence.

The locked native deep-learning runtime targets the NVIDIA GB10 (`aarch64`, CUDA 13). NGC mode additionally requires Docker with NVIDIA Container Toolkit support and the `nvcr.io/nvidia/pytorch:26.06-py3` image already pulled. First use downloads the selected official grayscale KAIR checkpoint into `models/` and verifies its pinned SHA-256 checksum. The model weights are not committed to the repository.

## Measurement

For each ROI, the app computes mean grayscale brightness frame by frame. When gain correction is enabled, the ROI brightness is first normalized against a surrounding reference region to reduce frame-to-frame fluoroscopy gain fluctuation. The reference and corrected ROI traces are then despiked with a short median filter and smoothed with a symmetric Gaussian filter. This reduces analog noise without adding the timing lag of a causal filter, though it intentionally trades a small amount of temporal precision for cleaner curves.

Because iodinated contrast appears darker in fluoroscopy, the contrast signal is calculated as:

```text
baseline ROI brightness - current ROI brightness
```

The signal is normalized by its peak. Residence time is measured from first threshold crossing to clearance below the selected normalized threshold after the peak. The default threshold is `0.20`.
