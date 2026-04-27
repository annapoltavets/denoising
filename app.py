"""Streamlit app — Interactive noise detection and denoising.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import os
import sys
import tempfile

import cv2
import numpy as np
import streamlit as st

# ---------------------------------------------------------------------------
# Ensure the repo src package is importable when the app is run from any CWD
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))

from src.noise import (  # noqa: E402
    ClippingSaturationNoise,
    ColoredCorrelatedNoise,
    CrossTalkNoise,
    FixedPatternNoise,
    FlickerNoise,
    GainNonuniformityNoise,
    HeteroskedasticGaussianNoise,
    LensFlareNoise,
    MultiplicativeNoise,
    PoissonGaussianNoise,
    PRNUNoise,
    QuantizationNoise,
    RealCameraRawNoise,
    RowColumnStripingNoise,
    ScaledPoissonNoise,
    ShotNoise,
    TemporalBlockOutlierNoise,
    ThermalNoise,
    WhiteGaussianNoise,
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


def build_noises_from_config(cfg: dict) -> list:
    """Reconstruct noise instances from a plain-dict configuration.

    Using a plain dict (rather than serialising live objects) as the cache
    key is safe and avoids any deserialization risks.
    """
    noises = []
    p = cfg  # shorthand

    if p["rcn_on"]:
        noises.append(RowColumnStripingNoise(stripe_threshold=p["rcn_thresh"]))
    if p["clip_on"]:
        noises.append(
            ClippingSaturationNoise(
                clip_threshold=p["clip_thresh"], inpaint_radius=p["clip_radius"]
            )
        )
    if p["quant_on"]:
        noises.append(
            QuantizationNoise(
                effective_bits_threshold=p["quant_bits"],
                gauss_sigma=p["quant_sigma"],
            )
        )
    if p["wgn_on"]:
        noises.append(
            WhiteGaussianNoise(sigma_threshold=p["wgn_sigma"], h=p["wgn_h"])
        )
    if p["shot_on"]:
        noises.append(
            ShotNoise(
                r2_threshold=p["shot_r2"],
                block_size=p["shot_bs"],
                gauss_sigma=p["shot_gsigma"],
            )
        )
    if p["sp_on"]:
        noises.append(
            ScaledPoissonNoise(
                r2_threshold=p["sp_r2"],
                alpha_min=p["sp_amin"],
                block_size=p["sp_bs"],
                gauss_sigma=p["sp_gsigma"],
            )
        )
    if p["pg_on"]:
        noises.append(
            PoissonGaussianNoise(
                r2_threshold=p["pg_r2"],
                block_size=p["pg_bs"],
                gauss_sigma=p["pg_gsigma"],
            )
        )
    if p["hg_on"]:
        noises.append(
            HeteroskedasticGaussianNoise(
                slope_threshold=p["hg_slope"],
                n_bins=p["hg_bins"],
                block_size=p["hg_bs"],
            )
        )
    if p["mul_on"]:
        noises.append(
            MultiplicativeNoise(
                cv_ratio_threshold=p["mul_cv"], gauss_sigma=p["mul_gsigma"]
            )
        )
    if p["thm_on"]:
        noises.append(
            ThermalNoise(
                dark_threshold=p["thm_dark"],
                sigma_threshold=p["thm_sigma"],
                gauss_ksize=p["thm_ksize"],
            )
        )
    if p["ct_on"]:
        noises.append(CrossTalkNoise(corr_threshold=p["ct_corr"]))
    if p["cc_on"]:
        noises.append(ColoredCorrelatedNoise(flatness_threshold=p["cc_flat"]))
    if p["gnu_on"]:
        noises.append(
            GainNonuniformityNoise(
                grid_size=p["gnu_grid"], cv_threshold=p["gnu_cv"]
            )
        )
    if p["lf_on"]:
        noises.append(
            LensFlareNoise(
                bright_threshold=p["lf_bright"],
                area_threshold=p["lf_area"],
                dilate_ksize=p["lf_dilate"],
                inpaint_radius=p["lf_inpaint"],
            )
        )
    if p["fpn_on"]:
        noises.append(
            FixedPatternNoise(
                warmup_frames=p["fpn_warmup"],
                pattern_threshold=p["fpn_thresh"],
                learning_rate=p["fpn_lr"],
            )
        )
    if p["prnu_on"]:
        noises.append(
            PRNUNoise(
                warmup_frames=p["prnu_warmup"],
                cv_threshold=p["prnu_cv"],
                learning_rate=p["prnu_lr"],
            )
        )
    if p["flk_on"]:
        noises.append(
            FlickerNoise(
                warmup_frames=p["flk_warmup"],
                flicker_threshold=p["flk_thresh"],
                learning_rate=p["flk_lr"],
            )
        )
    if p["rcr_on"]:
        noises.append(
            RealCameraRawNoise(
                noise_threshold=p["rcr_thresh"],
                block_size=p["rcr_bs"],
                h_luminance=p["rcr_hlum"],
                h_color=p["rcr_hcol"],
            )
        )
    if p["tbo_on"]:
        noises.append(
            TemporalBlockOutlierNoise(
                block_size=p["tbo_bs"],
                zscore_threshold=p["tbo_zt"],
                min_outlier_fraction=p["tbo_frac"],
                buffer_size=p["tbo_buf"],
                warmup_frames=p["tbo_warmup"],
            )
        )
    return noises


# ---------------------------------------------------------------------------
# Sidebar — noise type toggles and parameter controls
# ---------------------------------------------------------------------------

st.sidebar.header("⚙️ Noise detectors")
st.sidebar.markdown("Enable detectors and tune their parameters.")

show_mask = st.sidebar.toggle("🔴 Overlay noise mask", value=False)

st.sidebar.divider()

# Collect all user-chosen values into a plain dict used as a cache key and to
# reconstruct noise instances without any serialisation / deserialisation.
cfg: dict = {}

# ------ Row/Column Striping ------
with st.sidebar.expander("Row / Column Striping", expanded=False):
    cfg["rcn_on"] = st.checkbox("Enable", value=True, key="rcn_on")
    cfg["rcn_thresh"] = st.slider(
        "Stripe threshold (σ of row/col means)",
        min_value=0.5, max_value=20.0, value=2.0, step=0.5, key="rcn_thresh",
    )

# ------ Clipping / Saturation ------
with st.sidebar.expander("Clipping / Saturation", expanded=False):
    cfg["clip_on"] = st.checkbox("Enable", value=True, key="clip_on")
    cfg["clip_thresh"] = st.slider(
        "Clipped pixel fraction threshold",
        min_value=0.001, max_value=0.2, value=0.01, step=0.001,
        format="%.3f", key="clip_thresh",
    )
    cfg["clip_radius"] = st.slider(
        "Inpaint radius (px)", min_value=1, max_value=10, value=3, key="clip_radius",
    )

# ------ Quantization ------
with st.sidebar.expander("Quantization", expanded=False):
    cfg["quant_on"] = st.checkbox("Enable", value=True, key="quant_on")
    cfg["quant_bits"] = st.slider(
        "Effective-bits threshold (< → noisy)",
        min_value=1, max_value=8, value=5, key="quant_bits",
    )
    cfg["quant_sigma"] = st.slider(
        "Smoothing σ", min_value=0.1, max_value=3.0, value=0.8, step=0.1,
        key="quant_sigma",
    )

# ------ White Gaussian ------
with st.sidebar.expander("White Gaussian (AWGN)", expanded=False):
    cfg["wgn_on"] = st.checkbox("Enable", value=True, key="wgn_on")
    cfg["wgn_sigma"] = st.slider(
        "σ threshold (DN)", min_value=1.0, max_value=30.0, value=5.0, step=0.5,
        key="wgn_sigma",
    )
    cfg["wgn_h"] = st.slider(
        "NLM filter strength h", min_value=1, max_value=30, value=10, key="wgn_h",
    )

# ------ Shot Noise ------
with st.sidebar.expander("Shot / Photon Noise", expanded=False):
    cfg["shot_on"] = st.checkbox("Enable", value=True, key="shot_on")
    cfg["shot_r2"] = st.slider(
        "R² threshold", min_value=0.1, max_value=1.0, value=0.7, step=0.05,
        key="shot_r2",
    )
    cfg["shot_bs"] = st.slider(
        "Block size (px)", min_value=8, max_value=64, value=16, step=8,
        key="shot_bs",
    )
    cfg["shot_gsigma"] = st.slider(
        "Gaussian σ (removal)", min_value=0.1, max_value=5.0, value=1.0, step=0.1,
        key="shot_gsigma",
    )

# ------ Scaled Poisson ------
with st.sidebar.expander("Scaled Poisson Noise", expanded=False):
    cfg["sp_on"] = st.checkbox("Enable", value=True, key="sp_on")
    cfg["sp_r2"] = st.slider(
        "R² threshold", min_value=0.1, max_value=1.0, value=0.7, step=0.05,
        key="sp_r2",
    )
    cfg["sp_amin"] = st.slider(
        "Min gain α", min_value=0.01, max_value=5.0, value=0.1, step=0.01,
        key="sp_amin",
    )
    cfg["sp_bs"] = st.slider(
        "Block size (px)", min_value=8, max_value=64, value=16, step=8,
        key="sp_bs",
    )
    cfg["sp_gsigma"] = st.slider(
        "Gaussian σ (removal)", min_value=0.1, max_value=5.0, value=1.0, step=0.1,
        key="sp_gsigma",
    )

# ------ Poisson-Gaussian ------
with st.sidebar.expander("Poisson–Gaussian Mixed Noise", expanded=False):
    cfg["pg_on"] = st.checkbox("Enable", value=True, key="pg_on")
    cfg["pg_r2"] = st.slider(
        "R² threshold", min_value=0.1, max_value=1.0, value=0.6, step=0.05,
        key="pg_r2",
    )
    cfg["pg_bs"] = st.slider(
        "Block size (px)", min_value=8, max_value=64, value=16, step=8,
        key="pg_bs",
    )
    cfg["pg_gsigma"] = st.slider(
        "Gaussian σ (removal)", min_value=0.1, max_value=5.0, value=1.0, step=0.1,
        key="pg_gsigma",
    )

# ------ Heteroskedastic Gaussian ------
with st.sidebar.expander("Heteroskedastic Gaussian Noise", expanded=False):
    cfg["hg_on"] = st.checkbox("Enable", value=True, key="hg_on")
    cfg["hg_slope"] = st.slider(
        "Slope threshold", min_value=0.001, max_value=1.0, value=0.05, step=0.005,
        format="%.3f", key="hg_slope",
    )
    cfg["hg_bins"] = st.slider(
        "Intensity bins", min_value=2, max_value=32, value=8, key="hg_bins",
    )
    cfg["hg_bs"] = st.slider(
        "Block size (px)", min_value=8, max_value=64, value=16, step=8,
        key="hg_bs",
    )

# ------ Multiplicative ------
with st.sidebar.expander("Multiplicative Noise", expanded=False):
    cfg["mul_on"] = st.checkbox("Enable", value=True, key="mul_on")
    cfg["mul_cv"] = st.slider(
        "CV ratio threshold", min_value=0.1, max_value=1.0, value=0.6, step=0.05,
        key="mul_cv",
    )
    cfg["mul_gsigma"] = st.slider(
        "Gaussian σ (removal)", min_value=0.1, max_value=5.0, value=1.5, step=0.1,
        key="mul_gsigma",
    )

# ------ Thermal ------
with st.sidebar.expander("Thermal Noise (Dark Current)", expanded=False):
    cfg["thm_on"] = st.checkbox("Enable", value=True, key="thm_on")
    cfg["thm_dark"] = st.slider(
        "Dark pixel threshold (DN)", min_value=10, max_value=100, value=50,
        key="thm_dark",
    )
    cfg["thm_sigma"] = st.slider(
        "Noise σ threshold (DN)", min_value=0.5, max_value=20.0, value=4.0, step=0.5,
        key="thm_sigma",
    )
    cfg["thm_ksize"] = st.slider(
        "Gaussian kernel size (odd px)", min_value=3, max_value=15, value=5, step=2,
        key="thm_ksize",
    )

# ------ Cross-Talk ------
with st.sidebar.expander("Cross-Talk Noise", expanded=False):
    cfg["ct_on"] = st.checkbox("Enable", value=True, key="ct_on")
    cfg["ct_corr"] = st.slider(
        "Lag-1 correlation threshold", min_value=0.01, max_value=0.9, value=0.15,
        step=0.01, key="ct_corr",
    )

# ------ Colored / Correlated ------
with st.sidebar.expander("Colored / Correlated Noise", expanded=False):
    cfg["cc_on"] = st.checkbox("Enable", value=True, key="cc_on")
    cfg["cc_flat"] = st.slider(
        "Spectral flatness threshold (< → noisy)",
        min_value=0.05, max_value=0.95, value=0.3, step=0.05, key="cc_flat",
    )

# ------ Gain Nonuniformity ------
with st.sidebar.expander("Gain Nonuniformity", expanded=False):
    cfg["gnu_on"] = st.checkbox("Enable", value=True, key="gnu_on")
    cfg["gnu_grid"] = st.slider(
        "Grid size (tiles per side)", min_value=2, max_value=16, value=4,
        key="gnu_grid",
    )
    cfg["gnu_cv"] = st.slider(
        "CV threshold", min_value=0.01, max_value=0.3, value=0.05, step=0.01,
        key="gnu_cv",
    )

# ------ Lens Flare ------
with st.sidebar.expander("Lens Flare Noise", expanded=False):
    cfg["lf_on"] = st.checkbox("Enable", value=True, key="lf_on")
    cfg["lf_bright"] = st.slider(
        "Bright pixel threshold (DN)", min_value=150, max_value=255, value=240,
        key="lf_bright",
    )
    cfg["lf_area"] = st.slider(
        "Min flare area (px²)", min_value=10, max_value=2000, value=200,
        key="lf_area",
    )
    cfg["lf_dilate"] = st.slider(
        "Dilation kernel size (px)", min_value=1, max_value=15, value=5, step=2,
        key="lf_dilate",
    )
    cfg["lf_inpaint"] = st.slider(
        "Inpaint radius (px)", min_value=1, max_value=20, value=5, key="lf_inpaint",
    )

# ------ Fixed Pattern ------
with st.sidebar.expander("Fixed-Pattern Noise (temporal)", expanded=False):
    cfg["fpn_on"] = st.checkbox("Enable", value=True, key="fpn_on")
    cfg["fpn_warmup"] = st.slider(
        "Warmup frames", min_value=2, max_value=30, value=10, key="fpn_warmup",
    )
    cfg["fpn_thresh"] = st.slider(
        "Pattern std threshold", min_value=0.1, max_value=20.0, value=3.0, step=0.1,
        key="fpn_thresh",
    )
    cfg["fpn_lr"] = st.slider(
        "Learning rate", min_value=0.01, max_value=0.5, value=0.05, step=0.01,
        key="fpn_lr",
    )

# ------ PRNU ------
with st.sidebar.expander("PRNU (temporal)", expanded=False):
    cfg["prnu_on"] = st.checkbox("Enable", value=True, key="prnu_on")
    cfg["prnu_warmup"] = st.slider(
        "Warmup frames", min_value=2, max_value=30, value=15, key="prnu_warmup",
    )
    cfg["prnu_cv"] = st.slider(
        "CV threshold", min_value=0.001, max_value=0.2, value=0.02, step=0.001,
        format="%.3f", key="prnu_cv",
    )
    cfg["prnu_lr"] = st.slider(
        "Learning rate", min_value=0.01, max_value=0.5, value=0.05, step=0.01,
        key="prnu_lr",
    )

# ------ Flicker ------
with st.sidebar.expander("Flicker / Pink Noise (temporal)", expanded=False):
    cfg["flk_on"] = st.checkbox("Enable", value=True, key="flk_on")
    cfg["flk_warmup"] = st.slider(
        "Warmup frames", min_value=2, max_value=20, value=5, key="flk_warmup",
    )
    cfg["flk_thresh"] = st.slider(
        "Brightness deviation threshold (DN)",
        min_value=0.5, max_value=30.0, value=5.0, step=0.5, key="flk_thresh",
    )
    cfg["flk_lr"] = st.slider(
        "Learning rate", min_value=0.01, max_value=0.5, value=0.1, step=0.01,
        key="flk_lr",
    )

# ------ Real Camera Raw ------
with st.sidebar.expander("Real Camera Raw (catch-all)", expanded=False):
    cfg["rcr_on"] = st.checkbox("Enable", value=True, key="rcr_on")
    cfg["rcr_thresh"] = st.slider(
        "Noise σ threshold (DN)", min_value=1.0, max_value=30.0, value=5.0, step=0.5,
        key="rcr_thresh",
    )
    cfg["rcr_bs"] = st.slider(
        "Block size (px)", min_value=8, max_value=64, value=16, step=8,
        key="rcr_bs",
    )
    cfg["rcr_hlum"] = st.slider(
        "NLM luminance filter strength h", min_value=1, max_value=30, value=10,
        key="rcr_hlum",
    )
    cfg["rcr_hcol"] = st.slider(
        "NLM colour filter strength h", min_value=1, max_value=30, value=10,
        key="rcr_hcol",
    )

# ------ Temporal Block Outlier ------
with st.sidebar.expander("Temporal Block-Outlier Noise", expanded=False):
    cfg["tbo_on"] = st.checkbox("Enable", value=True, key="tbo_on")
    cfg["tbo_bs"] = st.slider(
        "Block size (px)", min_value=4, max_value=64, value=8, step=4,
        key="tbo_bs",
    )
    cfg["tbo_zt"] = st.slider(
        "Z-score threshold", min_value=1.0, max_value=10.0, value=3.0, step=0.5,
        key="tbo_zt",
    )
    cfg["tbo_frac"] = st.slider(
        "Min outlier block fraction",
        min_value=0.001, max_value=0.5, value=0.01, step=0.001,
        format="%.3f", key="tbo_frac",
    )
    cfg["tbo_buf"] = st.slider(
        "Frame buffer size", min_value=3, max_value=60, value=20, key="tbo_buf",
    )
    cfg["tbo_warmup"] = st.slider(
        "Warmup frames", min_value=2, max_value=20, value=5, key="tbo_warmup",
    )

# Build the active noise list from the collected configuration
active_noises = build_noises_from_config(cfg)

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

    # Encode frames as PNG bytes for caching (avoids storing large numpy arrays)
    @st.cache_data(show_spinner="Encoding frames…")
    def encode_frames(path: str) -> list[bytes]:
        caps = read_video_frames(path)
        out = []
        for f in caps:
            _, enc = cv2.imencode(".png", f)
            out.append(enc.tobytes())
        return out

    # Process frames 0..target_idx and return encoded denoised results.
    # The noise configuration is passed as a plain hashable dict so that
    # Streamlit can safely cache and invalidate results when params change.
    @st.cache_data(show_spinner="Processing frames…")
    def process_up_to(
        frame_bytes_list: list[bytes],
        noise_cfg: dict,
        target_idx: int,
    ) -> tuple[list[bytes], list[list[str]]]:
        """Process frames 0..target_idx; return encoded denoised frames + detection lists."""
        noises = build_noises_from_config(noise_cfg)
        results: list[bytes] = []
        det_lists: list[list[str]] = []
        for raw in frame_bytes_list[: target_idx + 1]:
            arr = np.frombuffer(raw, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            denoised_f, dets = process_frame(frame, noises)
            _, enc = cv2.imencode(".png", denoised_f)
            results.append(enc.tobytes())
            det_lists.append(dets)
        return results, det_lists

    encoded_frames = encode_frames(tmp_path)
    processed_encoded, det_lists = process_up_to(encoded_frames, cfg, frame_idx)

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
        st.table([{"Noise type": n.name} for n in active_noises])
    else:
        st.write("No detectors enabled.")
