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
    "Upload an image, then use the sidebar detectors to highlight noise and "
    "denoise step by step."
)

# ---------------------------------------------------------------------------
# Colour palette  (name → BGR tuple for OpenCV)
# ---------------------------------------------------------------------------

COLORS: dict[str, tuple[int, int, int]] = {  # values are BGR tuples (OpenCV convention)
    "Red":     (0,   0,   255),
    "Green":   (0,   255, 0),
    "Blue":    (255, 0,   0),
    "Yellow":  (0,   255, 255),
    "Cyan":    (255, 255, 0),
    "Magenta": (255, 0,   255),
    "Orange":  (0,   165, 255),
    "White":   (255, 255, 255),
}
COLOR_NAMES = list(COLORS.keys())

# ---------------------------------------------------------------------------
# Session-state defaults
# ---------------------------------------------------------------------------

for _key, _default in [
    ("original",       None),   # BGR ndarray — never mutated after load
    ("noise_frame",    None),   # BGR ndarray — noise pixels coloured cumulatively
    ("denoised_frame", None),   # BGR ndarray — progressively denoised
    ("upload_name",    None),   # str — tracks which file is currently loaded
    ("status_msg",     ""),     # str — feedback for the last action
    ("status_ok",      True),   # bool — True → success, False → info
    ("video_frames",   None),   # list[np.ndarray] | None — all video frames (BGR)
    ("frame_idx",      0),      # int — index of the currently displayed frame
    ("video_fps",      30.0),   # float — frames per second of the loaded video
]:
    if _key not in st.session_state:
        st.session_state[_key] = _default

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    """Convert a BGR uint8 image to RGB for display in Streamlit."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def apply_mask_color(
    base: np.ndarray,
    mask: np.ndarray,
    bgr_color: tuple[int, int, int],
) -> np.ndarray:
    """Return a copy of *base* with pixels where mask==255 set to *bgr_color*."""
    out = base.copy()
    out[mask == 255] = bgr_color
    return out


def safe_noise_mask(noise_obj, frame: np.ndarray) -> np.ndarray:
    """Return the noise mask from *noise_obj* or an all-zero mask on failure."""
    try:
        return noise_obj._noise_mask(frame)
    except Exception:
        return np.zeros(frame.shape[:2], dtype=np.uint8)


def read_video_frames(path: str) -> list[np.ndarray]:
    """Read all frames from a video file as BGR arrays."""
    cap = cv2.VideoCapture(path)
    frames: list[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def read_video(path: str) -> tuple[list[np.ndarray], float]:
    """Read all frames and FPS from a video file."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames: list[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames, fps


def frame_timestamp(idx: int, fps: float) -> str:
    """Return a human-readable timestamp string (MM:SS.f) for a frame index."""
    total_secs = idx / max(fps, 1e-6)
    minutes = int(total_secs // 60)
    seconds = total_secs % 60
    return f"{minutes:02d}:{seconds:05.2f}"


def set_status(msg: str, ok: bool = True) -> None:
    """Update the status message in session state."""
    st.session_state.status_msg = msg
    st.session_state.status_ok = ok


# ---------------------------------------------------------------------------
# Sidebar — file upload
# ---------------------------------------------------------------------------

st.sidebar.header("📂 Upload image")
uploaded = st.sidebar.file_uploader(
    "Image or video file",
    type=["png", "jpg", "jpeg", "bmp", "tiff", "mp4", "avi", "mov", "mkv"],
    label_visibility="collapsed",
)

if uploaded is not None:
    _is_video = uploaded.type.startswith("video/") or uploaded.name.lower().endswith(
        (".mp4", ".avi", ".mov", ".mkv")
    )
    # Reinitialise only when a different file is uploaded
    if uploaded.name != st.session_state.upload_name:
        st.session_state.upload_name = uploaded.name
        st.session_state.video_frames = None
        st.session_state.frame_idx    = 0
        if not _is_video:
            _raw = np.frombuffer(uploaded.read(), dtype=np.uint8)
            _img = cv2.imdecode(_raw, cv2.IMREAD_COLOR)
        else:
            _suffix = os.path.splitext(uploaded.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=_suffix) as _tmp:
                _tmp.write(uploaded.read())
                _tmp_path = _tmp.name
            _all_frames, _fps = read_video(_tmp_path)
            if _all_frames:
                st.session_state.video_frames = _all_frames
                st.session_state.video_fps    = _fps
            _img = _all_frames[0] if _all_frames else None

        if _img is not None:
            st.session_state.original       = _img
            st.session_state.noise_frame    = _img.copy()
            st.session_state.denoised_frame = _img.copy()
            set_status("")

if st.session_state.original is not None:
    if st.sidebar.button(
        "🔄 Reset frames",
        help="Clear all noise highlights and denoising — restore original",
    ):
        st.session_state.noise_frame    = st.session_state.original.copy()
        st.session_state.denoised_frame = st.session_state.original.copy()
        set_status("Frames reset to original.", ok=True)

# ---------------------------------------------------------------------------
# Sidebar — per-noise controls
# ---------------------------------------------------------------------------

st.sidebar.divider()
st.sidebar.header("⚙️ Noise detectors")
st.sidebar.caption(
    "Expand a detector, pick a **highlight colour**, tune parameters, then:\n"
    "- **Detect** — paint noise pixels on the *Noise* frame\n"
    "- **Denoise** — apply removal to the *Denoised* frame"
)

# Each entry collected below: (detect_clicked, denoise_clicked, noise_instance, color_bgr)
_pending: list[tuple[bool, bool, object, tuple[int, int, int]]] = []


def _color_and_buttons(key: str, default_color: str = "Red"):
    """Render colour selector + Detect/Denoise buttons. Returns (detect, denoise, color_bgr)."""
    color_name = st.selectbox(
        "Highlight colour",
        COLOR_NAMES,
        index=COLOR_NAMES.index(default_color),
        key=f"{key}_color",
        label_visibility="collapsed",
    )
    st.caption(f"🎨 **{color_name}**")
    _c1, _c2 = st.columns(2)
    _det = _c1.button("🔍 Detect",  key=f"{key}_detect",  use_container_width=True)
    _den = _c2.button("🔧 Denoise", key=f"{key}_denoise", use_container_width=True)
    return _det, _den, COLORS[color_name]


# ------ Row / Column Striping ------
with st.sidebar.expander("Row / Column Striping", expanded=False):
    _det, _den, _col = _color_and_buttons("rcn")
    _rcn_thresh = st.slider(
        "Stripe threshold (σ of row/col means)",
        0.5, 20.0, 2.0, 0.5, key="rcn_thresh",
    )
    _pending.append((_det, _den, RowColumnStripingNoise(stripe_threshold=_rcn_thresh), _col))

# ------ Clipping / Saturation ------
with st.sidebar.expander("Clipping / Saturation", expanded=False):
    _det, _den, _col = _color_and_buttons("clip")
    _clip_thresh  = st.slider("Clipped pixel fraction", 0.001, 0.2,  0.01,  0.001, format="%.3f", key="clip_thresh")
    _clip_radius  = st.slider("Inpaint radius (px)",    1,     10,   3,            key="clip_radius")
    _pending.append((_det, _den, ClippingSaturationNoise(clip_threshold=_clip_thresh, inpaint_radius=_clip_radius), _col))

# ------ Quantization ------
with st.sidebar.expander("Quantization", expanded=False):
    _det, _den, _col = _color_and_buttons("quant")
    _quant_bits  = st.slider("Effective-bits threshold (< → noisy)", 1,   8,   5,         key="quant_bits")
    _quant_sigma = st.slider("Smoothing σ",                           0.1, 3.0, 0.8, 0.1, key="quant_sigma")
    _pending.append((_det, _den, QuantizationNoise(effective_bits_threshold=_quant_bits, gauss_sigma=_quant_sigma), _col))

# ------ White Gaussian ------
with st.sidebar.expander("White Gaussian (AWGN)", expanded=False):
    _det, _den, _col = _color_and_buttons("wgn")
    _wgn_sigma = st.slider("σ threshold (DN)",      1.0, 30.0, 5.0, 0.5, key="wgn_sigma")
    _wgn_h     = st.slider("NLM filter strength h", 1,   30,   10,       key="wgn_h")
    _pending.append((_det, _den, WhiteGaussianNoise(sigma_threshold=_wgn_sigma, h=_wgn_h), _col))

# ------ Shot Noise ------
with st.sidebar.expander("Shot / Photon Noise", expanded=False):
    _det, _den, _col = _color_and_buttons("shot")
    _shot_r2     = st.slider("R² threshold",         0.1, 1.0,  0.7, 0.05, key="shot_r2")
    _shot_bs     = st.slider("Block size (px)",       8,   64,   16,  8,    key="shot_bs")
    _shot_gsigma = st.slider("Gaussian σ (removal)",  0.1, 5.0,  1.0, 0.1,  key="shot_gsigma")
    _pending.append((_det, _den, ShotNoise(r2_threshold=_shot_r2, block_size=_shot_bs, gauss_sigma=_shot_gsigma), _col))

# ------ Scaled Poisson ------
with st.sidebar.expander("Scaled Poisson Noise", expanded=False):
    _det, _den, _col = _color_and_buttons("sp")
    _sp_r2     = st.slider("R² threshold",        0.1,  1.0, 0.7, 0.05, key="sp_r2")
    _sp_amin   = st.slider("Min gain α",           0.01, 5.0, 0.1, 0.01, key="sp_amin")
    _sp_bs     = st.slider("Block size (px)",      8,    64,  16,  8,    key="sp_bs")
    _sp_gsigma = st.slider("Gaussian σ (removal)", 0.1,  5.0, 1.0, 0.1,  key="sp_gsigma")
    _pending.append((_det, _den, ScaledPoissonNoise(r2_threshold=_sp_r2, alpha_min=_sp_amin, block_size=_sp_bs, gauss_sigma=_sp_gsigma), _col))

# ------ Poisson–Gaussian ------
with st.sidebar.expander("Poisson–Gaussian Mixed Noise", expanded=False):
    _det, _den, _col = _color_and_buttons("pg")
    _pg_r2     = st.slider("R² threshold",        0.1, 1.0, 0.6, 0.05, key="pg_r2")
    _pg_bs     = st.slider("Block size (px)",      8,   64,  16,  8,    key="pg_bs")
    _pg_gsigma = st.slider("Gaussian σ (removal)", 0.1, 5.0, 1.0, 0.1,  key="pg_gsigma")
    _pending.append((_det, _den, PoissonGaussianNoise(r2_threshold=_pg_r2, block_size=_pg_bs, gauss_sigma=_pg_gsigma), _col))

# ------ Heteroskedastic Gaussian ------
with st.sidebar.expander("Heteroskedastic Gaussian Noise", expanded=False):
    _det, _den, _col = _color_and_buttons("hg")
    _hg_slope = st.slider("Slope threshold",  0.001, 1.0, 0.05, 0.005, format="%.3f", key="hg_slope")
    _hg_bins  = st.slider("Intensity bins",   2,     32,  8,            key="hg_bins")
    _hg_bs    = st.slider("Block size (px)",  8,     64,  16,  8,       key="hg_bs")
    _pending.append((_det, _den, HeteroskedasticGaussianNoise(slope_threshold=_hg_slope, n_bins=_hg_bins, block_size=_hg_bs), _col))

# ------ Multiplicative ------
with st.sidebar.expander("Multiplicative Noise", expanded=False):
    _det, _den, _col = _color_and_buttons("mul")
    _mul_cv     = st.slider("CV ratio threshold",    0.1, 1.0, 0.6, 0.05, key="mul_cv")
    _mul_gsigma = st.slider("Gaussian σ (removal)",  0.1, 5.0, 1.5, 0.1,  key="mul_gsigma")
    _pending.append((_det, _den, MultiplicativeNoise(cv_ratio_threshold=_mul_cv, gauss_sigma=_mul_gsigma), _col))

# ------ Thermal ------
with st.sidebar.expander("Thermal Noise (Dark Current)", expanded=False):
    _det, _den, _col = _color_and_buttons("thm")
    _thm_dark  = st.slider("Dark pixel threshold (DN)",       10,  100, 50,       key="thm_dark")
    _thm_sigma = st.slider("Noise σ threshold (DN)",          0.5, 20.0, 4.0, 0.5, key="thm_sigma")
    _thm_ksize = st.slider("Gaussian kernel size (odd px)",   3,   15,  5,  2,    key="thm_ksize")
    _pending.append((_det, _den, ThermalNoise(dark_threshold=_thm_dark, sigma_threshold=_thm_sigma, gauss_ksize=_thm_ksize), _col))

# ------ Cross-Talk ------
with st.sidebar.expander("Cross-Talk Noise", expanded=False):
    _det, _den, _col = _color_and_buttons("ct")
    _ct_corr = st.slider("Lag-1 correlation threshold", 0.01, 0.9, 0.15, 0.01, key="ct_corr")
    _pending.append((_det, _den, CrossTalkNoise(corr_threshold=_ct_corr), _col))

# ------ Colored / Correlated ------
with st.sidebar.expander("Colored / Correlated Noise", expanded=False):
    _det, _den, _col = _color_and_buttons("cc")
    _cc_flat = st.slider("Spectral flatness threshold (< → noisy)", 0.05, 0.95, 0.3, 0.05, key="cc_flat")
    _pending.append((_det, _den, ColoredCorrelatedNoise(flatness_threshold=_cc_flat), _col))

# ------ Gain Nonuniformity ------
with st.sidebar.expander("Gain Nonuniformity", expanded=False):
    _det, _den, _col = _color_and_buttons("gnu")
    _gnu_grid = st.slider("Grid size (tiles per side)", 2,    16,  4,          key="gnu_grid")
    _gnu_cv   = st.slider("CV threshold",               0.01, 0.3, 0.05, 0.01, key="gnu_cv")
    _pending.append((_det, _den, GainNonuniformityNoise(grid_size=_gnu_grid, cv_threshold=_gnu_cv), _col))

# ------ Lens Flare ------
with st.sidebar.expander("Lens Flare Noise", expanded=False):
    _det, _den, _col = _color_and_buttons("lf")
    _lf_bright  = st.slider("Bright pixel threshold (DN)", 150, 255,  240,       key="lf_bright")
    _lf_area    = st.slider("Min flare area (px²)",         10, 2000, 200,       key="lf_area")
    _lf_dilate  = st.slider("Dilation kernel size (px)",    1,  15,   5,   2,    key="lf_dilate")
    _lf_inpaint = st.slider("Inpaint radius (px)",          1,  20,   5,         key="lf_inpaint")
    _pending.append((_det, _den, LensFlareNoise(bright_threshold=_lf_bright, area_threshold=_lf_area, dilate_ksize=_lf_dilate, inpaint_radius=_lf_inpaint), _col))

# ------ Fixed Pattern ------
with st.sidebar.expander("Fixed-Pattern Noise (temporal)", expanded=False):
    _det, _den, _col = _color_and_buttons("fpn")
    _fpn_warmup = st.slider("Warmup frames",         2,   30,  10,          key="fpn_warmup")
    _fpn_thresh = st.slider("Pattern std threshold", 0.1, 20.0, 3.0, 0.1,  key="fpn_thresh")
    _fpn_lr     = st.slider("Learning rate",         0.01, 0.5, 0.05, 0.01, key="fpn_lr")
    _pending.append((_det, _den, FixedPatternNoise(warmup_frames=_fpn_warmup, pattern_threshold=_fpn_thresh, learning_rate=_fpn_lr), _col))

# ------ PRNU ------
with st.sidebar.expander("PRNU (temporal)", expanded=False):
    _det, _den, _col = _color_and_buttons("prnu")
    _prnu_warmup = st.slider("Warmup frames",  2,     30,  15,             key="prnu_warmup")
    _prnu_cv     = st.slider("CV threshold",   0.001, 0.2, 0.02,  0.001, format="%.3f", key="prnu_cv")
    _prnu_lr     = st.slider("Learning rate",  0.01,  0.5, 0.05,  0.01,  key="prnu_lr")
    _pending.append((_det, _den, PRNUNoise(warmup_frames=_prnu_warmup, cv_threshold=_prnu_cv, learning_rate=_prnu_lr), _col))

# ------ Flicker ------
with st.sidebar.expander("Flicker / Pink Noise (temporal)", expanded=False):
    _det, _den, _col = _color_and_buttons("flk")
    _flk_warmup = st.slider("Warmup frames",                    2,   20,  5,          key="flk_warmup")
    _flk_thresh = st.slider("Brightness deviation threshold (DN)", 0.5, 30.0, 5.0, 0.5, key="flk_thresh")
    _flk_lr     = st.slider("Learning rate",                    0.01, 0.5, 0.1, 0.01, key="flk_lr")
    _pending.append((_det, _den, FlickerNoise(warmup_frames=_flk_warmup, flicker_threshold=_flk_thresh, learning_rate=_flk_lr), _col))

# ------ Real Camera Raw ------
with st.sidebar.expander("Real Camera Raw (catch-all)", expanded=False):
    _det, _den, _col = _color_and_buttons("rcr")
    _rcr_thresh = st.slider("Noise σ threshold (DN)",        1.0, 30.0, 5.0, 0.5, key="rcr_thresh")
    _rcr_bs     = st.slider("Block size (px)",               8,   64,   16,  8,    key="rcr_bs")
    _rcr_hlum   = st.slider("NLM luminance filter h",        1,   30,   10,        key="rcr_hlum")
    _rcr_hcol   = st.slider("NLM colour filter h",           1,   30,   10,        key="rcr_hcol")
    _pending.append((_det, _den, RealCameraRawNoise(noise_threshold=_rcr_thresh, block_size=_rcr_bs, h_luminance=_rcr_hlum, h_color=_rcr_hcol), _col))

# ------ Temporal Block Outlier ------
with st.sidebar.expander("Temporal Block-Outlier Noise", expanded=False):
    _det, _den, _col = _color_and_buttons("tbo")
    _tbo_bs     = st.slider("Block size (px)",             4,     64,   8,    4,    key="tbo_bs")
    _tbo_zt     = st.slider("Z-score threshold",           1.0,   10.0, 3.0,  0.5,  key="tbo_zt")
    _tbo_frac   = st.slider("Min outlier block fraction",  0.001, 0.5,  0.01, 0.001, format="%.3f", key="tbo_frac")
    _tbo_buf    = st.slider("Frame buffer size",           3,     60,   20,          key="tbo_buf")
    _tbo_warmup = st.slider("Warmup frames",               2,     20,   5,           key="tbo_warmup")
    _pending.append((_det, _den, TemporalBlockOutlierNoise(block_size=_tbo_bs, zscore_threshold=_tbo_zt, min_outlier_fraction=_tbo_frac, buffer_size=_tbo_buf, warmup_frames=_tbo_warmup), _col))

# ---------------------------------------------------------------------------
# Main area — guard
# ---------------------------------------------------------------------------

if st.session_state.original is None:
    st.info("👈 Upload an image using the sidebar to get started.")
    st.stop()

# ---------------------------------------------------------------------------
# Frame timeline — visible only when a video is loaded
# ---------------------------------------------------------------------------

_vf = st.session_state.video_frames
if _vf is not None and len(_vf) > 1:
    _total   = len(_vf)
    _fps     = st.session_state.video_fps
    _cur_idx = st.session_state.frame_idx

    st.markdown("### 🎞️ Frame timeline")

    # Navigation row: buttons + slider on one line
    _nav_prev, _nav_slider, _nav_next = st.columns([1, 12, 1])
    with _nav_prev:
        _go_prev = st.button("◀", help="Previous frame", use_container_width=True)
    with _nav_next:
        _go_next = st.button("▶", help="Next frame",     use_container_width=True)

    # Resolve button presses before reading slider (buttons take priority)
    if _go_prev:
        _cur_idx = max(0, _cur_idx - 1)
    elif _go_next:
        _cur_idx = min(_total - 1, _cur_idx + 1)

    with _nav_slider:
        _slider_val = st.slider(
            "Frame",
            min_value=0,
            max_value=_total - 1,
            value=_cur_idx,
            step=1,
            label_visibility="collapsed",
            key="frame_timeline_slider",
        )
    # Slider overrides button if both somehow triggered (shouldn't happen)
    if not _go_prev and not _go_next:
        _cur_idx = _slider_val

    st.caption(
        f"Frame **{_cur_idx + 1} / {_total}** — "
        f"{frame_timestamp(_cur_idx, _fps)} "
        f"({_fps:.2f} fps)"
    )

    # If frame changed, update the three session-state images
    if _cur_idx != st.session_state.frame_idx:
        st.session_state.frame_idx      = _cur_idx
        _new_frame = _vf[_cur_idx]
        st.session_state.original       = _new_frame
        st.session_state.noise_frame    = _new_frame.copy()
        st.session_state.denoised_frame = _new_frame.copy()
        set_status(f"Jumped to frame {_cur_idx + 1}.", ok=True)
        st.rerun()

    # Thumbnail filmstrip — up to 12 evenly-sampled frames
    _N_THUMBS = min(12, _total)
    _thumb_indices = [round(i * (_total - 1) / (_N_THUMBS - 1)) for i in range(_N_THUMBS)] \
        if _N_THUMBS > 1 else [0]

    _thumb_cols = st.columns(_N_THUMBS)
    for _ti, (_tcol, _tidx) in enumerate(zip(_thumb_cols, _thumb_indices)):
        _thumb_frame = _vf[_tidx]
        # Resize thumbnail to fixed height (96 px) for speed
        _th, _tw = _thumb_frame.shape[:2]
        _target_h = 96
        _target_w = max(1, int(_tw * _target_h / _th))
        _thumb_small = cv2.resize(_thumb_frame, (_target_w, _target_h))
        _is_current = (_tidx == st.session_state.frame_idx)
        with _tcol:
            st.image(bgr_to_rgb(_thumb_small), use_container_width=True)
            _label = f"**{_tidx + 1}**" if _is_current else str(_tidx + 1)
            st.caption(_label)

    st.divider()

# ---------------------------------------------------------------------------
# Handle button actions — process only the first clicked button per rerun
# ---------------------------------------------------------------------------

_original = st.session_state.original
if _original is not None:
    for _detect_clicked, _denoise_clicked, _noise_obj, _color_bgr in _pending:
        if _detect_clicked:
            # Run detect on the original (unmodified) image
            _was_detected = _noise_obj.detect(_original)
            if _was_detected:
                _mask = safe_noise_mask(_noise_obj, _original)
                if _mask.any():
                    st.session_state.noise_frame = apply_mask_color(
                        st.session_state.noise_frame, _mask, _color_bgr
                    )
                    set_status(f"✅ **{_noise_obj.name}** detected — noise pixels highlighted.", ok=True)
                else:
                    set_status(f"ℹ️ **{_noise_obj.name}** detected but produced an empty mask.", ok=False)
            else:
                set_status(f"ℹ️ **{_noise_obj.name}** not detected in this image.", ok=False)
            break

        if _denoise_clicked:
            # Run detect + remove on the current denoised frame (cumulative)
            _cur = st.session_state.denoised_frame
            if _noise_obj.detect(_cur):
                st.session_state.denoised_frame = _noise_obj.remove(_cur)
                set_status(f"✅ **{_noise_obj.name}** removed from the denoised frame.", ok=True)
            else:
                set_status(f"ℹ️ **{_noise_obj.name}** not detected in the denoised frame — no change.", ok=False)
            break

# ---------------------------------------------------------------------------
# Status feedback
# ---------------------------------------------------------------------------

if st.session_state.status_msg:
    if st.session_state.status_ok:
        st.success(st.session_state.status_msg)
    else:
        st.info(st.session_state.status_msg)

# ---------------------------------------------------------------------------
# 3-frame display: Initial | Noise | Denoised
# ---------------------------------------------------------------------------

col_init, col_noise, col_den = st.columns(3)

with col_init:
    st.subheader("🖼️ Initial")
    st.image(bgr_to_rgb(st.session_state.original), use_container_width=True)

with col_noise:
    st.subheader("🔴 Noise")
    st.image(bgr_to_rgb(st.session_state.noise_frame), use_container_width=True)

with col_den:
    st.subheader("✨ Denoised")
    st.image(bgr_to_rgb(st.session_state.denoised_frame), use_container_width=True)

