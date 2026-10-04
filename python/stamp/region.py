"""Crop-around-mask and feathered paste-back, so inpainting runs on a small
padded window (fast) instead of the whole frame, with no visible seam."""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def mask_bbox(mask: np.ndarray, pad: int, min_size: int = 256):
    """Return (x0, y0, x1, y1) covering the painted mask plus padding,
    clamped to the image, and at least min_size on each side."""
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return 0, 0, w, h
    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    half_w = max((x1 - x0) // 2 + pad, min_size // 2)
    half_h = max((y1 - y0) // 2 + pad, min_size // 2)
    return (
        max(0, cx - half_w),
        max(0, cy - half_h),
        min(w, cx + half_w),
        min(h, cy + half_h),
    )


def _highpass_sigma(x: np.ndarray, ref_mask: np.ndarray, blur_sigma: float = 3.0) -> float:
    """Robust std of the fine-grain (high-frequency) residual of x, measured
    only over ref_mask pixels. Used to compare how much real texture/noise
    is present in two regions regardless of their average brightness."""
    smooth = ndimage.gaussian_filter(x, sigma=blur_sigma)
    residual = x - smooth
    ref = residual[ref_mask] if ref_mask.any() else residual.ravel()
    return float(1.4826 * np.median(np.abs(ref - np.median(ref))))


def correlated_noise_field(shape: tuple, sigma: float, rng: np.random.Generator, corr_sigma: float = 0.7) -> np.ndarray:
    """Spatially-correlated noise field, ported from HaloNet's
    halo_anomaly_mask.py (round 55, 2026-10-04 -- same grain-matching
    problem, found and fixed there first), then refined here (IA stamp,
    same day) after a real "effet passoire" (sieve-like, pareidolic
    texture) complaint on the first version.

    A plain `rng.normal(0, sigma, ...)` is spatially INDEPENDENT white
    noise, but real sky background isn't: measured directly on genuine
    empty sky, adjacent-pixel correlation was 0.587 (likely from
    drizzle/debayer resampling) versus -0.003 (noise floor) for i.i.d.
    noise.

    HaloNet's own fix used an NxN uniform BOX blur (N=2 -> exact 0.5
    correlation at lag 1), which matched the target amplitude but has a
    sharp cutoff: lag-1 correlation ~0.5, lag-2 correlation already back
    down near 0. That abrupt cutoff gives the noise a quasi-periodic,
    tiled 2x2-block structure -- invisible in a single correlation number,
    but visually reads as a repeating "sieve"/cellular motif once your eye
    latches onto it (pareidolia). A real optical/sampling PSF has no such
    hard edge.

    Fix here: a GAUSSIAN blur instead of a box -- same idea (blur
    unit-variance white noise, renormalize to unit std, scale by sigma),
    but its autocorrelation decays smoothly with no fixed length scale
    (lag-1 ~0.58, lag-2 ~0.13, lag-3 ~0.01 at corr_sigma=0.7, vs the box's
    0.50/~0/~0) -- no periodic motif for the eye to lock onto.
    corr_sigma=0.7 was solved numerically to hit the same real lag-1
    target (0.587) the box version was tuned to, via
    correlation(1px) = exp(-1/(4*corr_sigma**2)) for a gaussian kernel."""
    fine = rng.normal(0.0, 1.0, size=shape)
    blurred = ndimage.gaussian_filter(fine, sigma=corr_sigma)
    cur_std = float(np.std(blurred))
    if cur_std <= 0:
        return np.zeros(shape)
    return (blurred / cur_std) * sigma


def match_noise(crop: np.ndarray, result: np.ndarray, mask: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Top up `result` with synthetic noise so its fine-grain texture inside
    the masked region matches the real texture measured outside it.

    LaMa's global (FFT-based) branch is good at coarse structure/gradients
    but tends to smooth away fine sensor noise on large holes, leaving a
    flat, blobby, texture-less patch ("mamelonnage") that stands out from a
    genuinely noisy sky background. Only ever ADDS the missing amount
    (variances add for independent noise sources), never removes texture
    LaMa already produced."""
    is_mono = crop.ndim == 2
    channels_crop = [crop] if is_mono else [crop[..., c] for c in range(crop.shape[-1])]
    channels_res = [result] if is_mono else [result[..., c] for c in range(result.shape[-1])]
    mask_bool = mask > 0

    out_channels = []
    for c_crop, c_res in zip(channels_crop, channels_res):
        real_sigma = _highpass_sigma(c_crop, valid_mask)
        fill_sigma = _highpass_sigma(c_res, mask_bool)
        deficit = np.sqrt(max(0.0, real_sigma**2 - fill_sigma**2))
        if deficit > 0:
            noise = correlated_noise_field(c_res.shape, deficit, np.random.default_rng())
            c_res = c_res + noise
        out_channels.append(c_res)

    return out_channels[0] if is_mono else np.stack(out_channels, axis=-1)


def feather_paste(
    base: np.ndarray,
    patch: np.ndarray,
    mask: np.ndarray,
    feather_px: float = 6,
    opacity: float = 1.0,
    clip_range: tuple[float, float] | None = (0, 255),
    out_dtype=np.uint8,
) -> np.ndarray:
    """Paste patch into base, blended over `mask` with a soft edge so the
    stamped region doesn't leave a hard boundary. `opacity` (0-1) scales the
    blend strength everywhere -- 1.0 is a full replacement inside the mask,
    lower values let the original show through (same idea as a brush tool's
    opacity slider). Works on 8-bit RGB (defaults: clip to [0,255], cast to
    uint8) or linear float32 data (pass clip_range=None, out_dtype=np.float32)."""
    alpha = (mask > 0).astype(np.float32)
    if feather_px > 0:
        alpha = ndimage.gaussian_filter(alpha, sigma=feather_px)
    alpha = alpha * opacity
    if base.ndim == 3:
        alpha = alpha[..., None]
    out = base.astype(np.float32) * (1 - alpha) + patch.astype(np.float32) * alpha
    if clip_range is not None:
        out = np.clip(out, clip_range[0], clip_range[1])
    return out.astype(out_dtype)
