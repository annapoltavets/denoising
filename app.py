"""Streamlit app — Interactive noise detection and denoising.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import sys
import os
import tempfile
from typing import Optional

import cv2
import numpy as np
import streamlit as st

# ---------------------------------------------------------------------------
# Ensure the repo src package is importable when the app is run from any CWD
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))

from src.noise import (  # noqa: E402
    WhiteGaussianNoise,
    ShotNoise,
    ScaledPoissonNoise,
    PoissonGaussianNoise,
    HeteroskedasticGaussianNoise,
    MultiplicativeNoise,
    ThermalNoise,
    CrossTalkNoise,
    FixedPatternNoise,
    RowColumnStripingNoise,
    QuantizationNoise,
    ClippingSaturationNoise,
    PRNUNoise,
    GainNonuniformityNoise,
    FlickerNoise,
    LensFlareNoise,
    ColoredCorrelatedNoise,
    RealCameraRawNoise,
    TemporalBlockOutlierNoise,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Denoising Explorer",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Denoising Explorer")
st.caption(
    "Upload an image or video, tune each noise detector's parameters in the "
    "sidebar, and compare the original frame with the denoised result."
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    """Convert a BGR uint8 image to RGB for display in Streamlit."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_video_frames(path: str) -> list[np.ndarray]:
    """Read all frames from a video file into a list of BGR arrays."""
    cap = cv2.VideoCapture(path)
    frames: list[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


# ---------------------------------------------------------------------------
# Sidebar — noise type toggles and parameter controls
# ---------------------------------------------------------------------------

st.sidebar.header("⚙️ Noise detectors")
st.sidebar.markdown("Enable detectors and tune their parameters.")

show_mask = st.sidebar.toggle("🔴 Overlay noise mask", value=False)

st.sidebar.divider()

# Each entry: (label, enabled_default, constructor)
# We collect user-configured noise instances into `active_noises`.
active_noises: list = []

# ------ Row/Column Striping ------
with st.sidebar.expander("Row / Column Striping", expanded=False):
    rcn_on = st.checkbox("Enable", value=True, key="rcn_on")
    rcn_thresh = st.slider(
        "Stripe threshold (σ of row/col means)",
        min_value=0.5, max_value=20.0, value=2.0, step=0.5, key="rcn_thresh",
    )
if rcn_on:
    active_noises.append(RowColumnStripingNoise(stripe_threshold=rcn_thresh))

# ------ Clipping / Saturation ------
with st.sidebar.expander("Clipping / Saturation", expanded=False):
    clip_on = st.checkbox("Enable", value=True, key="clip_on")
    clip_thresh = st.slider(
        "Clipped pixel fraction threshold",
        min_value=0.001, max_value=0.2, value=0.01, step=0.001,
        format="%.3f", key="clip_thresh",
    )
    clip_radius = st.slider(
        "Inpaint radius (px)", min_value=1, max_value=10, value=3, key="clip_radius",
    )
if clip_on:
    active_noises.append(
        ClippingSaturationNoise(clip_threshold=clip_thresh, inpaint_radius=clip_radius)
    )

# ------ Quantization ------
with st.sidebar.expander("Quantization", expanded=False):
    quant_on = st.checkbox("Enable", value=True, key="quant_on")
    quant_bits = st.slider(
        "Effective-bits threshold (< → noisy)",
        min_value=1, max_value=8, value=5, key="quant_bits",
    )
    quant_sigma = st.slider(
        "Smoothing σ", min_value=0.1, max_value=3.0, value=0.8, step=0.1,
        key="quant_sigma",
    )
if quant_on:
    active_noises.append(
        QuantizationNoise(
            effective_bits_threshold=quant_bits, gauss_sigma=quant_sigma
        )
    )

# ------ White Gaussian ------
with st.sidebar.expander("White Gaussian (AWGN)", expanded=False):
    wgn_on = st.checkbox("Enable", value=True, key="wgn_on")
    wgn_sigma = st.slider(
        "σ threshold (DN)", min_value=1.0, max_value=30.0, value=5.0, step=0.5,
        key="wgn_sigma",
    )
    wgn_h = st.slider(
        "NLM filter strength h", min_value=1, max_value=30, value=10, key="wgn_h",
    )
if wgn_on:
    active_noises.append(WhiteGaussianNoise(sigma_threshold=wgn_sigma, h=wgn_h))

# ------ Shot Noise ------
with st.sidebar.expander("Shot / Photon Noise", expanded=False):
    shot_on = st.checkbox("Enable", value=True, key="shot_on")
    shot_r2 = st.slider(
        "R² threshold", min_value=0.1, max_value=1.0, value=0.7, step=0.05,
        key="shot_r2",
    )
    shot_bs = st.slider(
        "Block size (px)", min_value=8, max_value=64, value=16, step=8,
        key="shot_bs",
    )
    shot_gsigma = st.slider(
        "Gaussian σ (removal)", min_value=0.1, max_value=5.0, value=1.0, step=0.1,
        key="shot_gsigma",
    )
if shot_on:
    active_noises.append(
        ShotNoise(r2_threshold=shot_r2, block_size=shot_bs, gauss_sigma=shot_gsigma)
    )

# ------ Scaled Poisson ------
with st.sidebar.expander("Scaled Poisson Noise", expanded=False):
    sp_on = st.checkbox("Enable", value=True, key="sp_on")
    sp_r2 = st.slider(
        "R² threshold", min_value=0.1, max_value=1.0, value=0.7, step=0.05,
        key="sp_r2",
    )
    sp_amin = st.slider(
        "Min gain α", min_value=0.01, max_value=5.0, value=0.1, step=0.01,
        key="sp_amin",
    )
    sp_bs = st.slider(
        "Block size (px)", min_value=8, max_value=64, value=16, step=8,
        key="sp_bs",
    )
    sp_gsigma = st.slider(
        "Gaussian σ (removal)", min_value=0.1, max_value=5.0, value=1.0, step=0.1,
        key="sp_gsigma",
    )
if sp_on:
    active_noises.append(
        ScaledPoissonNoise(
            r2_threshold=sp_r2,
            alpha_min=sp_amin,
            block_size=sp_bs,
            gauss_sigma=sp_gsigma,
        )
    )

# ------ Poisson-Gaussian ------
with st.sidebar.expander("Poisson–Gaussian Mixed Noise", expanded=False):
    pg_on = st.checkbox("Enable", value=True, key="pg_on")
    pg_r2 = st.slider(
        "R² threshold", min_value=0.1, max_value=1.0, value=0.6, step=0.05,
        key="pg_r2",
    )
    pg_bs = st.slider(
        "Block size (px)", min_value=8, max_value=64, value=16, step=8,
        key="pg_bs",
    )
    pg_gsigma = st.slider(
        "Gaussian σ (removal)", min_value=0.1, max_value=5.0, value=1.0, step=0.1,
        key="pg_gsigma",
    )
if pg_on:
    active_noises.append(
        PoissonGaussianNoise(
            r2_threshold=pg_r2, block_size=pg_bs, gauss_sigma=pg_gsigma
        )
    )

# ------ Heteroskedastic Gaussian ------
with st.sidebar.expander("Heteroskedastic Gaussian Noise", expanded=False):
    hg_on = st.checkbox("Enable", value=True, key="hg_on")
    hg_slope = st.slider(
        "Slope threshold", min_value=0.001, max_value=1.0, value=0.05, step=0.005,
        format="%.3f", key="hg_slope",
    )
    hg_bins = st.slider(
        "Intensity bins", min_value=2, max_value=32, value=8, key="hg_bins",
    )
    hg_bs = st.slider(
        "Block size (px)", min_value=8, max_value=64, value=16, step=8,
        key="hg_bs",
    )
if hg_on:
    active_noises.append(
        HeteroskedasticGaussianNoise(
            slope_threshold=hg_slope, n_bins=hg_bins, block_size=hg_bs
        )
    )

# ------ Multiplicative ------
with st.sidebar.expander("Multiplicative Noise", expanded=False):
    mul_on = st.checkbox("Enable", value=True, key="mul_on")
    mul_cv = st.slider(
        "CV ratio threshold", min_value=0.1, max_value=1.0, value=0.6, step=0.05,
        key="mul_cv",
    )
    mul_gsigma = st.slider(
        "Gaussian σ (removal)", min_value=0.1, max_value=5.0, value=1.5, step=0.1,
        key="mul_gsigma",
    )
if mul_on:
    active_noises.append(
        MultiplicativeNoise(
            cv_ratio_threshold=mul_cv, gauss_sigma=mul_gsigma
        )
    )

# ------ Thermal ------
with st.sidebar.expander("Thermal Noise (Dark Current)", expanded=False):
    thm_on = st.checkbox("Enable", value=True, key="thm_on")

if thm_on:
    active_noises.append(ThermalNoise())

# ------ Cross-Talk ------
with st.sidebar.expander("Cross-Talk Noise", expanded=False):
    ct_on = st.checkbox("Enable", value=True, key="ct_on")

if ct_on:
    active_noises.append(CrossTalkNoise())

# ------ Colored / Correlated ------
with st.sidebar.expander("Colored / Correlated Noise", expanded=False):
    cc_on = st.checkbox("Enable", value=True, key="cc_on")

if cc_on:
    active_noises.append(ColoredCorrelatedNoise())

# ------ Gain Nonuniformity ------
with st.sidebar.expander("Gain Nonuniformity", expanded=False):
    gnu_on = st.checkbox("Enable", value=True, key="gnu_on")
    gnu_grid = st.slider(
        "Grid size (tiles per side)", min_value=2, max_value=16, value=4,
        key="gnu_grid",
    )
    gnu_cv = st.slider(
        "CV threshold", min_value=0.01, max_value=0.3, value=0.05, step=0.01,
        key="gnu_cv",
    )
if gnu_on:
    active_noises.append(
        GainNonuniformityNoise(grid_size=gnu_grid, cv_threshold=gnu_cv)
    )

# ------ Lens Flare ------
with st.sidebar.expander("Lens Flare Noise", expanded=False):
    lf_on = st.checkbox("Enable", value=True, key="lf_on")

if lf_on:
    active_noises.append(LensFlareNoise())

# ------ Fixed Pattern ------
with st.sidebar.expander("Fixed-Pattern Noise (temporal)", expanded=False):
    fpn_on = st.checkbox("Enable", value=True, key="fpn_on")
    fpn_warmup = st.slider(
        "Warmup frames", min_value=2, max_value=30, value=10, key="fpn_warmup",
    )
    fpn_thresh = st.slider(
        "Pattern std threshold", min_value=0.1, max_value=20.0, value=3.0, step=0.1,
        key="fpn_thresh",
    )
    fpn_lr = st.slider(
        "Learning rate", min_value=0.01, max_value=0.5, value=0.05, step=0.01,
        key="fpn_lr",
    )
if fpn_on:
    active_noises.append(
        FixedPatternNoise(
            warmup_frames=fpn_warmup,
            pattern_threshold=fpn_thresh,
            learning_rate=fpn_lr,
        )
    )

# ------ PRNU ------
with st.sidebar.expander("PRNU (temporal)", expanded=False):
    prnu_on = st.checkbox("Enable", value=True, key="prnu_on")
    prnu_warmup = st.slider(
        "Warmup frames", min_value=2, max_value=30, value=15, key="prnu_warmup",
    )
    prnu_cv = st.slider(
        "CV threshold", min_value=0.001, max_value=0.2, value=0.02, step=0.001,
        format="%.3f", key="prnu_cv",
    )
    prnu_lr = st.slider(
        "Learning rate", min_value=0.01, max_value=0.5, value=0.05, step=0.01,
        key="prnu_lr",
    )
if prnu_on:
    active_noises.append(
        PRNUNoise(
            warmup_frames=prnu_warmup,
            cv_threshold=prnu_cv,
            learning_rate=prnu_lr,
        )
    )

# ------ Flicker ------
with st.sidebar.expander("Flicker / Pink Noise (temporal)", expanded=False):
    flk_on = st.checkbox("Enable", value=True, key="flk_on")
    flk_warmup = st.slider(
        "Warmup frames", min_value=2, max_value=20, value=5, key="flk_warmup",
    )
    flk_thresh = st.slider(
        "Brightness deviation threshold (DN)",
        min_value=0.5, max_value=30.0, value=5.0, step=0.5, key="flk_thresh",
    )
    flk_lr = st.slider(
        "Learning rate", min_value=0.01, max_value=0.5, value=0.1, step=0.01,
        key="flk_lr",
    )
if flk_on:
    active_noises.append(
        FlickerNoise(
            warmup_frames=flk_warmup,
            flicker_threshold=flk_thresh,
            learning_rate=flk_lr,
        )
    )

# ------ Real Camera Raw ------
with st.sidebar.expander("Real Camera Raw (catch-all)", expanded=False):
    rcr_on = st.checkbox("Enable", value=True, key="rcr_on")

if rcr_on:
    active_noises.append(RealCameraRawNoise())

# ------ Temporal Block Outlier ------
with st.sidebar.expander("Temporal Block-Outlier Noise", expanded=False):
    tbo_on = st.checkbox("Enable", value=True, key="tbo_on")
    tbo_bs = st.slider(
        "Block size (px)", min_value=4, max_value=64, value=8, step=4,
        key="tbo_bs",
    )
    tbo_zt = st.slider(
        "Z-score threshold", min_value=1.0, max_value=10.0, value=3.0, step=0.5,
        key="tbo_zt",
    )
    tbo_frac = st.slider(
        "Min outlier block fraction",
        min_value=0.001, max_value=0.5, value=0.01, step=0.001,
        format="%.3f", key="tbo_frac",
    )
    tbo_buf = st.slider(
        "Frame buffer size", min_value=3, max_value=60, value=20, key="tbo_buf",
    )
    tbo_warmup = st.slider(
        "Warmup frames", min_value=2, max_value=20, value=5, key="tbo_warmup",
    )
if tbo_on:
    active_noises.append(
        TemporalBlockOutlierNoise(
            block_size=tbo_bs,
            zscore_threshold=tbo_zt,
            min_outlier_fraction=tbo_frac,
            buffer_size=tbo_buf,
            warmup_frames=tbo_warmup,
        )
    )

# ---------------------------------------------------------------------------
# File upload
# ---------------------------------------------------------------------------

st.sidebar.divider()
uploaded = st.sidebar.file_uploader(
    "📂 Upload image or video",
    type=["png", "jpg", "jpeg", "bmp", "tiff", "mp4", "avi", "mov", "mkv"],
    help="Supported: PNG, JPG, BMP, TIFF, MP4, AVI, MOV, MKV",
)

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

if uploaded is None:
    st.info(
        "👈 Upload an image or video using the sidebar to get started. "
        "Then enable noise detectors and adjust their parameters."
    )
    st.stop()

is_video = uploaded.type.startswith("video/") or uploaded.name.lower().endswith(
    (".mp4", ".avi", ".mov", ".mkv")
)

# ---------------------------------------------------------------------------
# Process the uploaded file
# ---------------------------------------------------------------------------


def process_frame(frame: np.ndarray, noises: list) -> tuple[np.ndarray, list[str]]:
    """Run all active noise detectors on *frame*.

    Returns
    -------
    denoised:
        The denoised frame (BGR uint8).
    detected_names:
        Names of noise types that were detected (and removed).
    """
    result = frame.copy()
    detected: list[str] = []
    for noise in noises:
        if noise.detect(result):
            detected.append(noise.name)
            result = noise.remove(result)
    return result, detected


def get_mask_overlay(frame: np.ndarray, noises: list) -> np.ndarray:
    """Paint noise pixels red across all active detectors (union of masks)."""
    bgr = frame.copy() if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    union_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for noise in noises:
        try:
            m = noise._noise_mask(frame)
            union_mask = np.maximum(union_mask, m)
        except Exception:
            pass
    bgr[union_mask == 255] = (0, 0, 255)
    return bgr


# ---- Image path ----
if not is_video:
    file_bytes = np.frombuffer(uploaded.read(), dtype=np.uint8)
    original = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if original is None:
        st.error("Could not decode the uploaded image.")
        st.stop()

    denoised, detected_names = process_frame(original, active_noises)

    display_original = (
        get_mask_overlay(original, active_noises) if show_mask else original
    )

    col_before, col_after = st.columns(2)
    with col_before:
        st.subheader("Before")
        st.image(bgr_to_rgb(display_original), use_container_width=True)
    with col_after:
        st.subheader("After (denoised)")
        st.image(bgr_to_rgb(denoised), use_container_width=True)

    if detected_names:
        st.success("**Noise detected and removed:** " + ", ".join(detected_names))
    else:
        st.info("No noise detected with the current settings.")

# ---- Video path ----
else:
    suffix = os.path.splitext(uploaded.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    with st.spinner("Reading video frames…"):
        frames = read_video_frames(tmp_path)

    if not frames:
        st.error("Could not read any frames from the uploaded video.")
        st.stop()

    total = len(frames)
    st.caption(f"Video: **{total}** frames — use the slider to select a frame.")

    frame_idx = st.slider(
        "Frame index", min_value=0, max_value=total - 1, value=0, step=1,
    )

    # Feed all frames up to (and including) the selected one through the
    # pipeline so that temporal detectors accumulate history correctly.
    # We cache this to avoid re-processing on every slider interaction.
    @st.cache_data(show_spinner="Processing frames…")
    def process_up_to(
        frame_bytes_list: list[bytes],
        noise_key: str,  # used as cache key; changes when params change
        target_idx: int,
    ) -> tuple[list[bytes], list[list[str]]]:
        """Process frames 0..target_idx and return encoded results."""
        import pickle  # noqa: PLC0415

        noises = pickle.loads(noise_key)  # noqa: S301
        results: list[bytes] = []
        det_lists: list[list[str]] = []
        for raw in frame_bytes_list[: target_idx + 1]:
            arr = np.frombuffer(raw, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            denoised, dets = process_frame(frame, noises)
            _, enc = cv2.imencode(".png", denoised)
            results.append(enc.tobytes())
            det_lists.append(dets)
        return results, det_lists

    # Encode frames as PNG bytes for caching
    @st.cache_data(show_spinner="Encoding frames…")
    def encode_frames(path: str) -> list[bytes]:
        caps = read_video_frames(path)
        out = []
        for f in caps:
            _, enc = cv2.imencode(".png", f)
            out.append(enc.tobytes())
        return out

    import pickle  # noqa: PLC0415

    encoded_frames = encode_frames(tmp_path)
    noise_key = pickle.dumps(active_noises)

    processed_encoded, det_lists = process_up_to(
        encoded_frames, noise_key, frame_idx
    )

    # Decode original and denoised for the selected frame
    orig_arr = np.frombuffer(encoded_frames[frame_idx], dtype=np.uint8)
    original = cv2.imdecode(orig_arr, cv2.IMREAD_COLOR)

    den_arr = np.frombuffer(processed_encoded[-1], dtype=np.uint8)
    denoised = cv2.imdecode(den_arr, cv2.IMREAD_COLOR)
    detected_names = det_lists[-1]

    display_original = (
        get_mask_overlay(original, active_noises) if show_mask else original
    )

    col_before, col_after = st.columns(2)
    with col_before:
        st.subheader(f"Before — frame {frame_idx}")
        st.image(bgr_to_rgb(display_original), use_container_width=True)
    with col_after:
        st.subheader(f"After (denoised) — frame {frame_idx}")
        st.image(bgr_to_rgb(denoised), use_container_width=True)

    if detected_names:
        st.success("**Noise detected and removed:** " + ", ".join(detected_names))
    else:
        st.info("No noise detected with the current settings.")

# ---------------------------------------------------------------------------
# Detection status table (all active detectors)
# ---------------------------------------------------------------------------

with st.expander("📊 Active detectors", expanded=False):
    if active_noises:
        rows = []
        for n in active_noises:
            rows.append({"Noise type": n.name})
        st.table(rows)
    else:
        st.write("No detectors enabled.")
