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

Choose **File > Save as default pipeline settings** to make the current source and live pipeline stages, their order, enabled states, and parameter values the startup defaults. These defaults are stored in [configs/default_pipeline.json](configs/default_pipeline.json) and deliberately do not include video paths.

For live fluoroscope enhancement, choose **File > Switch to live camera mode** and select a video to use as a looping camera simulation. The source crop is measured once from that stream and retained for every live frame. When **DSA mask subtraction** is enabled, each newly detected recording waits one second after fluoroscopy starts, captures a fresh mask frame, and subtracts later frames from it. Sequence-dependent stages, including temporal alignment and motion-aware temporal filtering, are unavailable in this mode; the remaining compatible stages are applied directly to each incoming frame.

## Headless Streaming

Run the live enhancement service with a stream configuration:

```bash
uv run python main.py --headless --config configs/headless_stream_config.json
```

`POST /ingest` accepts one `image/jpeg` frame per request. The service samples the configured number of initial frames to calculate a fixed auto-crop, then processes and publishes each following frame. `GET /egress.mjpg` returns the enhanced stream as MJPEG, and `GET /health` reports crop readiness and frame counters. The most recent enhanced frame is retained, so a slow egress consumer never delays ingest.

For example, a camera bridge that can produce JPEG snapshots can forward frames with:

```bash
curl --data-binary @frame.jpg -H 'Content-Type: image/jpeg' http://localhost:8080/ingest
```

View the enhanced output at `http://localhost:8080/egress.mjpg`. [configs/headless_stream_config.json](configs/headless_stream_config.json) is a ready-to-edit example. Headless mode rejects stages that require a full temporal sequence, including temporal alignment, brightness stabilization, ROI extraction, motion-aware filtering, and ROI residence analysis.

### Desktop Streaming

Run the desktop application normally:

```bash
uv run python main.py
```

Select **Live camera** to start the network stream service. It listens on `0.0.0.0:8080`, samples 24 frames for auto-crop, emits JPEG at quality 92, and accepts frames up to 16 MiB. The desktop pipeline drawer controls the processing applied to subsequent `/ingest` frames, and updates take effect without restarting the service. Stages that need a full temporal sequence remain available for file analysis in the GUI but are omitted from the single-frame network stream.

Live camera opens with source and enhanced views side by side. Drag on the source image to select the legacy rectangular ROI; temporal ROI extraction remains unavailable for live input. Add and enable **ROI residence analysis** and/or **Frame brightness analysis** in the pipeline drawer to update their plots from the latest 60 seconds of received frames.

The service stops when the desktop application closes. Use [configs/headless_stream_config.json](configs/headless_stream_config.json) only for the standalone `--headless` service when you need to customize its listener or encoding settings.

## Pipeline Architecture

Pipeline contracts are frontend-neutral and live in `contrast_pipeline/`:

- `models.py` defines immutable stage instances, parameters, and enhancement requests shared by the desktop and headless entry points.
- `stages.py` is the stage registry. Each definition owns its key, display name, execution shape, cache signature, performance estimates, live compatibility, and optional frame processor.
- `executor.py` runs live-compatible stages in order and injects runtime services such as denoising without coupling a stage to HTTP, Qt, or a concrete accelerator backend.

To add a stateless filter, implement a frame processor in `contrast_pipeline/stages.py`, register one `StageDefinition`, and add its Qt parameter controls/template. The desktop scheduler and headless service then share its processing, cache metadata, display name, and compatibility rules. Batch, temporal, full-sequence, source, observer, and analysis stages declare an `ExecutionShape`; specialized sequence scheduling remains in the desktop orchestration until a dedicated executor is provided for that shape.

## Workflow

1. Enable live processing stages in order. The default stage order now places **Aneurysm ROI extraction** second, immediately after gain / brightness stabilization.
2. Enable **Aneurysm ROI extraction** to build an ROI mask from the current upstream video state. Use **Refresh ROI extraction** to rerun it after adjusting upstream stages or extraction parameters.
3. For a two-video comparison, add **Align ROI / needle baseline levels** after ROI extraction. It expands whichever video has the narrower pre-injection needle-to-ROI range, then shifts its brightness to the wider-range video.
4. Review the detected mask during playback. Drag on a video only when a detected ROI needs correction.
5. Enable **ROI residence analysis** only after the upstream extraction stage is enabled and has produced masks for both videos.
6. Compare the normalized contrast curves, raw ROI brightness curves, and residence-time cards.
7. Adjust the clearance threshold if needed. The metrics update from the already measured curves.
8. Export a CSV for downstream analysis.

Quality controls are available:

- **Aneurysm ROI extraction** is now a pipeline stage that runs on the current upstream enhanced video rather than as a separate load-time step. It stabilizes frame-wide intensity, measures directional darkening from the pre-injection baseline through the trimmed video, and searches multiple response levels for compact circular components. Candidates are ranked by temporal contrast response, area, circularity, and fill. The extracted mask is cached as a downstream artifact for ROI residence analysis.
- **Mask softening / expansion** is optional inside the ROI extraction stage. When enabled, the detected contour is rounded and slightly expanded before analysis. Use **Softening radius** and **Soft mask threshold** to control how much the extracted blob grows beyond the tight raw contour.
- **Automatic fluoroscope crop** runs when each video is loaded. Circular collimator fields are segmented across sampled frames, then cropped to the largest centered square with at least 99.5% in-field coverage and a size divisible by 32. This removes dark margins, gives paired videos consistent accelerator-friendly shapes, and applies before ROI drawing, enhancement, and analysis. Non-circular videos retain the pillarbox-only fallback.

- **NGC FFDNet (Docker)** is the default display-enhancement backend. It runs FFDNet in the local `nvcr.io/nvidia/pytorch:26.06-py3` ARM64 image and exchanges configurable batches with the desktop process through shared memory. One persistent NGC worker is assigned to each concurrently enhanced video, and mapped batches are passed directly to PyTorch without an intermediate host copy.
- **Native FFDNet (GPU)** uses the ARM CUDA 13 PyTorch packages locked in this project. The current PyTorch 2.11 wheel expects cuDNN 9.13, while NVIDIA's cu13 Python index currently provides cuDNN 9.12 for aarch64, so NGC is the validated GB10 path until a matching native cuDNN package is available.
- **FFDNet noise sigma** represents the assumed noise standard deviation on the 0-255 intensity scale. `10` is a conservative default for the included videos; lower values preserve more texture, while higher values produce a smoother but increasingly plastic image. The available range is `0` to `50`.
- **Batch frames** controls how many video frames are sent to the GPU together. The NGC default is `4`; on GB10 it sustained about 91 fps through the FFDNet shared-memory round trip at the auto-cropped `800x800` shape, while larger batches reduced throughput despite the available 128GB unified pool. Larger batches can still help other models or frame sizes. **Precision** defaults to `FP16`; measured BF16 was slightly slower and changed output by as much as seven 8-bit levels, while FP32/TF32 was substantially slower.
- **Live processing pipeline** starts with all stages disabled. Its checkboxes are always applied top to bottom: gain / brightness stabilization, aneurysm ROI extraction, ROI / needle baseline alignment, median gain normalization, scanline correction, spatial denoising, motion-aware temporal filtering, quantum mottle reduction, CLAHE local contrast, image adjustments, final Gaussian smoothing, and terminal analysis stages. Enable stages one at a time to inspect their cumulative effect. Unchecking a stage removes only that operation while preserving enabled downstream stages.
- **Gain / brightness stabilization** is the first stage. It robustly aligns upper-histogram probes from each frame to the video's reference histogram, fitting both gain and offset while excluding the darker intensity population affected by contrast passage. This corrects multiplicative and additive exposure jitter in one operation without flattening the contrast trace or allocating a full-resolution temporal stack.
- **ROI / needle baseline alignment** is a file-only comparison stage. It measures the pre-injection mean level inside each extracted aneurysm ROI and the stable core of each segmented needle. The video with the narrower absolute needle-to-ROI range receives gain greater than or equal to one until both ranges match, then receives a brightness offset that aligns both baseline anchors with the wider-range video. Place it after an enabled ROI extraction stage; downstream stages consume the aligned pixels. The **ROI alignment** analysis tab plots each video's ROI brightness before and after the transform, with labeled ROI and needle baseline levels for both states.
- **Image adjustments** is a general-purpose enhancement stage for common finishing tweaks. It exposes brightness offset, contrast gain, sharpen amount, and gamma so you can quickly tune visual emphasis without switching denoising backends.
- **Brightness-coded contrast segmentation** now supports two segmentation bases:
	- **Dark contrast (per frame)** keeps the original adaptive local threshold workflow and uses neighborhood + sensitivity controls.
	- **Temporal brightness change (full video)** computes one per-pixel change map from the full trimmed video (`P90 - P10` brightness per pixel), then keeps connected components above **Change threshold**. This is useful when you want regions that changed meaningfully over time rather than regions that are simply dark in one frame.
	In both modes, **Brightness tolerance** groups nearby component levels and **Minimum component area** removes small regions. The stage captures its component map at its pipeline position and passes frames through unchanged so downstream enhancement and ROI analysis are not altered.
- **Mask overlay** in the playback bar blends cached component maps over only the enhanced video using a shared brightness color scale. Toggle it at any time without rebuilding the pipeline; maps remain available during playback and scrubbing.
- **ROI residence analysis** now has a hard dependency on the upstream **Aneurysm ROI extraction** stage. If that stage is disabled or cannot produce a mask for either video, the analysis stage reports a failure and skips curve generation.
- **Frame brightness analysis** compares the mean pixel value of every original frame with its enhanced counterpart in the Analysis drawer's **Frame brightness** tab. It is a terminal pipeline stage and does not require an ROI mask.
- **Needle segmentation and brightness** estimates one stationary needle mask from only the enhanced frames before contrast injection. It retains the single darkest connected region, overlays that fixed mask on the enhanced video, and plots its average enhanced pixel brightness across the complete video in the Analysis drawer's **Needle brightness** tab.
- **Contrast residence heatmap** removes frame-wide fluoroscope gain drift, estimates a pre-contrast baseline, and incrementally integrates only local darkening beyond baseline noise. Enable **Heatmap** beside **Show source** to display it beside the enhanced video; the two display modes are mutually exclusive. Time-integrated contrast burden determines color while peak vessel contrast controls brightness. The stage offers Hot, Inferno, Turbo, Viridis, and Cividis color maps, which can be switched without recomputing the analysis. Comparison heatmaps use shared, outlier-resistant scales so persistent aneurysm contrast appears bright without static acquisition texture washing out the vessels. Memory use for the calculation is independent of video duration. It is a terminal, file-only stage and does not require an ROI mask.
- **Frame workers** are shared by both videos and every enhancement stage. The pool grows on demand, bounds both submitted work and inter-stage queues, and discovers the highest-frequency CPU tier from Linux sysfs. On Grace, ten workers are pinned one per Cortex-X925 performance core while decode, UI, and orchestration remain free to use the Cortex-A725 cores. Set `CONTRAST_FRAME_WORKERS` to a positive integer to override the detected worker count.
- **CUDA Graphs** cache fixed batch/frame shapes and replay the FFDNet convolution chain with lower launch overhead. Up to eight shapes are retained so differently cropped paired videos and final partial batches do not recapture continuously. Set `CONTRAST_CUDA_GRAPHS=0` only for diagnostics.
- **Spatial denoising** offers FFDNet, **NGC Tensor NLM (GPU)**, and **Non-local means (CPU)**. Tensor NLM runs a chunked PyTorch implementation in the optimized NGC 26.06 container, keeps accumulation in FP32, and uses bounded GPU memory instead of materializing the full search tensor. Both NLM backends use the strength control as the luminance-filter `h` value with 7x7 template and 21x21 search windows; strength `5` is a useful starting point after temporal mottle reduction. On the GB10 at the included video's 672x672 crop, Tensor NLM matched OpenCV's grain reduction and edge retention but measured about 18 fps versus roughly 25 fps for the four-worker CPU backend, so the CPU option remains preferable for this particular frame size.
- **Quantum mottle reduction** uses a short centered temporal window and admits neighboring pixels according to their intensity similarity to the current frame. The default five-frame window and similarity sigma of `12` target uncorrelated fluoroscopy grain while rejecting changing anatomy and injected contrast; lower sigma values preserve faster changes, while longer windows or higher sigma values remove more grain.
- Every pipeline change rebuilds enhanced caches for both complete videos with an on-screen progress overlay and immediately refreshes both panels. **Show original videos** clears all stages. Enhanced frames are JPEG-compressed in memory and reused during playback, paused viewing, and scrubbing. Display enhancement does not change the measured ROI values.
- **Correct gain drift in analysis** measures a reference region around the ROI on every frame and normalizes the ROI intensity against that reference before calculating contrast residence.

The validated GB10 runtime is NGC 26.06 (`aarch64`, CUDA 13.3, cuDNN 9.23). NGC mode requires Docker with NVIDIA Container Toolkit support and the `nvcr.io/nvidia/pytorch:26.06-py3` image already pulled. First use downloads the selected official grayscale KAIR checkpoint into `models/` and verifies its pinned SHA-256 checksum. The model weights are not committed to the repository.

## Measurement

For each ROI, the app computes mean grayscale brightness frame by frame. The enhanced ROI trace is then despiked with a short median filter and smoothed with a symmetric Gaussian filter. This reduces analog noise without adding the timing lag of a causal filter, though it intentionally trades a small amount of temporal precision for cleaner curves.

Because iodinated contrast appears darker in fluoroscopy, the contrast signal is calculated as:

```text
baseline ROI brightness - current ROI brightness
```

For each video, the pre-injection baseline is the median brightness from the initial baseline window. Subtracting that baseline aligns the pre-injection signal to `0.0`:

```text
contrast signal = max(baseline ROI brightness - current ROI brightness, 0)
```

When comparing videos, the app finds the strongest contrast-darkening moment across the complete set of analyzed videos and maps that one shared peak to `1.0`. Every other curve uses the same scale, so a weaker peak remains proportionally below `1.0` instead of being independently stretched to the top of the graph. For a single-video analysis, that video's own peak is the shared peak.

Residence time is measured from the first crossing of the selected shared normalized threshold to clearance below it after the peak. The default threshold is `0.20`.
