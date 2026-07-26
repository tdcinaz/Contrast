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
3. Choose **NGC FFDNet (Docker)**, **Native FFDNet (GPU)**, or **Classical** enhancement and click **Enable video enhancement** when an enhanced display is useful.
4. Click **Analyze ROIs**.
5. Compare the normalized contrast curves, raw ROI brightness curves, and residence-time cards.
6. Adjust the clearance threshold if needed. The metrics update from the already measured curves.
7. Export a CSV for downstream analysis.

Quality controls are available:

- **Automatic pillarbox crop** runs when each video is loaded. If black side bars are detected, frames are cropped to the fluoroscope content before ROI drawing, enhancement, and analysis.

- **NGC FFDNet (Docker)** is the default display-enhancement backend. It runs FFDNet in the local `nvcr.io/nvidia/pytorch:26.06-py3` image and exchanges four-frame batches with the desktop process through shared memory. The container stays alive while the backend is selected and is removed when the backend changes or the app closes. The included pre-deployment clip processes at about `20.7 fps` end to end on the GB10, versus `10.7 fps` through the native backend.
- **Native FFDNet (GPU)** runs the same model and checkpoint through the ARM CUDA 13 PyTorch packages locked in this project. Keep it as a fallback when Docker or the NGC image is unavailable. NGC and native output differ by no more than one 8-bit intensity level in validation.
- Both FFDNet modes apply the pretrained grayscale denoiser after gain stabilization and scanline correction. The strength control represents the assumed noise standard deviation on the 0-255 intensity scale. `10` is calibrated as a conservative default for the included videos; lower values preserve more texture, while higher values produce a smoother but increasingly plastic image. The supported range is `5` to `25`.
- **Classical enhancement** remains available for comparison and does not require PyTorch. It uses an edge-preserving bilateral filter in place of FFDNet.
- **Enhance video display** starts disabled. Click **Enable video enhancement** to build the selected enhancement cache with an on-screen loading bar. Both modes combine adjacent frames with centered, motion-aware weights before local contrast enhancement. Enhanced frames are JPEG-compressed in memory and reused during playback, paused viewing, and scrubbing. Changing the model or strength invalidates this cache. Display enhancement does not change the measured ROI values.
- **Correct gain drift in analysis** measures a reference region around the ROI on every frame and normalizes the ROI intensity against that reference before calculating contrast residence.

The locked native deep-learning runtime targets the NVIDIA GB10 (`aarch64`, CUDA 13). NGC mode additionally requires Docker with NVIDIA Container Toolkit support and the `nvcr.io/nvidia/pytorch:26.06-py3` image already pulled. The first use of FFDNet downloads the official grayscale checkpoint from the KAIR release into `models/ffdnet_gray.pth` and verifies its SHA-256 checksum. The model weights are not committed to the repository.

## Measurement

For each ROI, the app computes mean grayscale brightness frame by frame. When gain correction is enabled, the ROI brightness is first normalized against a surrounding reference region to reduce frame-to-frame fluoroscopy gain fluctuation. The reference and corrected ROI traces are then despiked with a short median filter and smoothed with a symmetric Gaussian filter. This reduces analog noise without adding the timing lag of a causal filter, though it intentionally trades a small amount of temporal precision for cleaner curves.

Because iodinated contrast appears darker in fluoroscopy, the contrast signal is calculated as:

```text
baseline ROI brightness - current ROI brightness
```

The signal is normalized by its peak. Residence time is measured from first threshold crossing to clearance below the selected normalized threshold after the peak. The default threshold is `0.20`.
