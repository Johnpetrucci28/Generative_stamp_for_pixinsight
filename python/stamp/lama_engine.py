"""Generative inpainting engine backed by big-lama (LaMa), Apache 2.0.

Weights auto-download on first use from the Apache-2.0-licensed
enesmsahin/simple-lama-inpainting GitHub release (a torchscript trace of
the original saic-mdal/lama big-lama checkpoint) into the user's local
torch hub cache. See ../LICENSES.md.
"""
from __future__ import annotations

import os

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from simple_lama_inpainting.utils import download_model, prepare_img_and_mask

LAMA_MODEL_URL = (
    "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt"
)

# Inference time scales worse than linearly with crop area (measured: 256px
# ~1.4s, 600px ~2.9s/3.7s at 12/6 threads, 900px ~6.6s/7.4s). Capping the
# longest side fed to the network and upscaling the result back keeps large
# brush strokes fast -- cli_stamp.py's match_noise() already re-adds the
# real, correctly-correlated grain on top afterward, so the coarse
# structure LaMa predicts at a lower resolution loses little that
# match_noise wasn't already going to touch up anyway. 768 matches the
# value the dev-only Tkinter harness (app/stamp_gui.py) already used.
LAMA_MAX_SIDE = 768


class LamaEngine:
    """Loads big-lama once and runs inpainting on demand.

    Reimplements simple_lama_inpainting.SimpleLama's __init__: that class
    calls torch.jit.load() without map_location, which raises
    NotImplementedError on a CPU-only torch build because the traced graph
    carries CUDA-tagged buffers.
    """

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        if device == "cpu":
            # torch defaults to half the logical core count on this
            # machine (6 of 12) -- measured ~20% faster inference at 8-12
            # threads on a real 600px crop (3.65s -> 2.96-3.00s), no
            # measurable downside (this call blocks PixInsight's own
            # ExternalProcess wait loop anyway, so there's nothing else
            # competing for the cores during inference).
            torch.set_num_threads(os.cpu_count() or 4)
        model_path = download_model(LAMA_MODEL_URL)
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()
        self.model.to(self.device)

    def inpaint(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        """image: RGB PIL image. mask: single-channel PIL image, 255 = area to fill."""
        orig_w, orig_h = image.size
        scale = min(1.0, LAMA_MAX_SIDE / max(orig_w, orig_h))
        if scale < 1.0:
            small_size = (round(orig_w * scale), round(orig_h * scale))
            run_image = image.resize(small_size, Image.BILINEAR)
            run_mask = mask.resize(small_size, Image.NEAREST)
        else:
            run_image, run_mask = image, mask

        img_t, mask_t = prepare_img_and_mask(run_image, run_mask, self.device)
        with torch.inference_mode():
            out = self.model(img_t, mask_t)
        result = out[0].permute(1, 2, 0).detach().cpu().numpy()
        # prepare_img_and_mask pads to a multiple of 8 (symmetric padding) but
        # never crops back -- the model output keeps that padded size.
        result = result[:run_image.height, :run_image.width]
        # big-lama's decoder uses strided transposed convolutions, a
        # well-known source of a faint periodic checkerboard artifact
        # (Odena et al. 2016) -- normally masked by texture/clutter in the
        # natural photos it was trained on, but clearly visible on the
        # smooth/flat astro backgrounds this tool targets. A very light
        # blur kills that fixed ~2px-period pattern without touching real
        # structure; cli_stamp.py's match_noise() step re-adds the correct
        # amount of real (random, non-periodic) grain on top afterward, so
        # texture statistics still end up matching the surrounding sky.
        result = ndimage.gaussian_filter(result, sigma=(0.7, 0.7, 0))
        result = np.clip(result * 255, 0, 255).astype(np.uint8)
        result_img = Image.fromarray(result)
        if scale < 1.0:
            # Upscale LaMa's coarse structure/gradient back to the real crop
            # size -- the fine grain lost in the round trip is exactly what
            # cli_stamp.py's match_noise() re-adds right after this, using
            # the real crop's own measured texture, not anything LaMa ever
            # produced at full res.
            result_img = result_img.resize((orig_w, orig_h), Image.BILINEAR)
        return result_img
