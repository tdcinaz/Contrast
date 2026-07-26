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
3. Click **Analyze ROIs**.
4. Compare the normalized contrast curves, raw ROI brightness curves, and residence-time cards.
5. Adjust the clearance threshold if needed. The metrics update from the already measured curves.
6. Export a CSV for downstream analysis.

Two quality controls are available:

- **Automatic pillarbox crop** runs when each video is loaded. If black side bars are detected, frames are cropped to the fluoroscope content before ROI drawing, enhancement, and analysis.

- **Enhance video display** starts disabled. Click **Enable video enhancement** to run a stationary-scene enhancement pass with an on-screen loading bar. It stabilizes global gain, suppresses row-wise scanline excursions, applies edge-preserving spatial denoising, and combines adjacent frames with motion-aware weights before local contrast enhancement. The centered temporal filter avoids favoring either the preceding or following frame. Enhanced frames are JPEG-compressed in memory and reused during playback, paused viewing, and scrubbing. Display enhancement does not change the measured ROI values.
- **Correct gain drift in analysis** measures a reference region around the ROI on every frame and normalizes the ROI intensity against that reference before calculating contrast residence.

## Measurement

For each ROI, the app computes mean grayscale brightness frame by frame. When gain correction is enabled, the ROI brightness is first normalized against a surrounding reference region to reduce frame-to-frame fluoroscopy gain fluctuation. The reference and corrected ROI traces are then despiked with a short median filter and smoothed with a symmetric Gaussian filter. This reduces analog noise without adding the timing lag of a causal filter, though it intentionally trades a small amount of temporal precision for cleaner curves.

Because iodinated contrast appears darker in fluoroscopy, the contrast signal is calculated as:

```text
baseline ROI brightness - current ROI brightness
```

The signal is normalized by its peak. Residence time is measured from first threshold crossing to clearance below the selected normalized threshold after the peak. The default threshold is `0.20`.
