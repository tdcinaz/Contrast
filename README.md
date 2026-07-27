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
3. Enable live processing stages in order. Each change rebuilds both full videos, updates the video panels, and passes the cumulative result to every enabled downstream stage.
4. Click **Analyze ROIs**.
5. Compare the normalized contrast curves, raw ROI brightness curves, and residence-time cards.
6. Adjust the clearance threshold if needed. The metrics update from the already measured curves.
7. Export a CSV for downstream analysis.

Quality controls are available:

- **Automatic fluoroscope crop** runs when each video is loaded. Circular collimator fields are segmented across sampled frames, then cropped to the largest centered square with at least 99.5% in-field coverage and a size divisible by 32. This removes dark margins, gives paired videos consistent accelerator-friendly shapes, and applies before ROI drawing, enhancement, and analysis. Non-circular videos retain the pillarbox-only fallback.

- **NGC FFDNet (Docker)** is the default display-enhancement backend. It runs FFDNet in the local `nvcr.io/nvidia/pytorch:26.06-py3` ARM64 image and exchanges configurable batches with the desktop process through shared memory. One persistent NGC worker is assigned to each concurrently enhanced video, and mapped batches are passed directly to PyTorch without an intermediate host copy.
- **NGC DnCNN 15/25/50** provides three fixed-noise alternatives using the official grayscale KAIR checkpoints. `15` preserves the most fine structure, `25` is the balanced preset, and `50` applies the strongest denoising. Unlike FFDNet, each DnCNN checkpoint has a fixed trained noise level, so the FFDNet noise-sigma control is disabled for these models.
- **Native FFDNet (GPU)** uses the ARM CUDA 13 PyTorch packages locked in this project. The current PyTorch 2.11 wheel expects cuDNN 9.13, while NVIDIA's cu13 Python index currently provides cuDNN 9.12 for aarch64, so NGC is the validated GB10 path until a matching native cuDNN package is available.
- **FFDNet noise sigma** represents the assumed noise standard deviation on the 0-255 intensity scale. `10` is a conservative default for the included videos; lower values preserve more texture, while higher values produce a smoother but increasingly plastic image. The available range is `0` to `50`.
- **Batch frames** controls how many video frames are sent to the GPU together. The NGC default is `4`; on GB10 it sustained about 91 fps through the FFDNet shared-memory round trip at the auto-cropped `800x800` shape, while larger batches reduced throughput despite the available 128GB unified pool. Larger batches can still help other models or frame sizes. **Precision** defaults to `FP16`; measured BF16 was slightly slower and changed output by as much as seven 8-bit levels, while FP32/TF32 was substantially slower.
- **Live processing pipeline** starts with all stages disabled. Its checkboxes are always applied top to bottom: gain stabilization, scanline correction, spatial denoising, motion-aware temporal filtering, CLAHE local contrast, final Gaussian smoothing, and brightness-coded contrast segmentation. Enable stages one at a time to inspect their cumulative effect. Unchecking a stage removes only that operation while preserving enabled downstream stages.
- **Brightness-coded contrast segmentation** uses adaptive local thresholding and connected-component filtering to identify dark contrast-filled regions. Each retained component stores its median grayscale brightness rather than a binary foreground value, allowing regions at different brightness levels to remain distinguishable. Its neighborhood, sensitivity, and minimum-area controls can be tuned per video set. The stage captures its component map from the cumulative frame at its current pipeline position, then passes that frame through unchanged so downstream enhancement and ROI analysis are not altered.
- **Mask overlay** in the playback bar blends cached component maps over only the enhanced video using a shared brightness color scale. Toggle it at any time without rebuilding the pipeline; maps remain available during playback and scrubbing.
- **Frame workers** are shared by both videos and every enhancement stage. The pool grows on demand, bounds both submitted work and inter-stage queues, and discovers the highest-frequency CPU tier from Linux sysfs. On Grace, ten workers are pinned one per Cortex-X925 performance core while decode, UI, and orchestration remain free to use the Cortex-A725 cores. Set `CONTRAST_FRAME_WORKERS` to a positive integer to override the detected worker count.
- **CUDA Graphs** cache fixed batch/frame shapes and replay the FFDNet or DnCNN convolution chain with lower launch overhead. Up to eight shapes are retained so differently cropped paired videos and final partial batches do not recapture continuously. Set `CONTRAST_CUDA_GRAPHS=0` only for diagnostics.
- **Spatial denoising** uses the selected deep model, or an edge-preserving bilateral filter when **Classical** is selected. Model, sigma, batch, and precision controls affect only this stage.
- Every pipeline change rebuilds enhanced caches for both complete videos with an on-screen progress overlay and immediately refreshes both panels. **Show original videos** clears all stages. Enhanced frames are JPEG-compressed in memory and reused during playback, paused viewing, and scrubbing. Display enhancement does not change the measured ROI values.
- **Correct gain drift in analysis** measures a reference region around the ROI on every frame and normalizes the ROI intensity against that reference before calculating contrast residence.

The validated GB10 runtime is NGC 26.06 (`aarch64`, CUDA 13.3, cuDNN 9.23). NGC mode requires Docker with NVIDIA Container Toolkit support and the `nvcr.io/nvidia/pytorch:26.06-py3` image already pulled. First use downloads the selected official grayscale KAIR checkpoint into `models/` and verifies its pinned SHA-256 checksum. The model weights are not committed to the repository.

## Measurement

For each ROI, the app computes mean grayscale brightness frame by frame. When gain correction is enabled, the ROI brightness is first normalized against a surrounding reference region to reduce frame-to-frame fluoroscopy gain fluctuation. The reference and corrected ROI traces are then despiked with a short median filter and smoothed with a symmetric Gaussian filter. This reduces analog noise without adding the timing lag of a causal filter, though it intentionally trades a small amount of temporal precision for cleaner curves.

Because iodinated contrast appears darker in fluoroscopy, the contrast signal is calculated as:

```text
baseline ROI brightness - current ROI brightness
```

The signal is normalized by its peak. Residence time is measured from first threshold crossing to clearance below the selected normalized threshold after the peak. The default threshold is `0.20`.
