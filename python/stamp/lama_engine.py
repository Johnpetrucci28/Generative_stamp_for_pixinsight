"""Generative inpainting engine backed by big-lama (LaMa), Apache 2.0.

Weights auto-download on first use from the Apache-2.0-licensed
enesmsahin/simple-lama-inpainting GitHub release (a torchscript trace of
the original saic-mdal/lama big-lama checkpoint) into the user's local
torch hub cache. See ../LICENSES.md.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from simple_lama_inpainting.utils import download_model, prepare_img_and_mask

LAMA_MODEL_URL = (
    "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt"
)


class LamaEngine:
    """Loads big-lama once and runs inpainting on demand.

    Reimplements simple_lama_inpainting.SimpleLama's __init__: that class
    calls torch.jit.load() without map_location, which raises
    NotImplementedError on a CPU-only torch build because the traced graph
    carries CUDA-tagged buffers.
    """

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        model_path = download_model(LAMA_MODEL_URL)
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()
        self.model.to(self.device)

    def inpaint(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        """image: RGB PIL image. mask: single-channel PIL image, 255 = area to fill."""
        orig_w, orig_h = image.size
        img_t, mask_t = prepare_img_and_mask(image, mask, self.device)
        with torch.inference_mode():
            out = self.model(img_t, mask_t)
        result = out[0].permute(1, 2, 0).detach().cpu().numpy()
        # prepare_img_and_mask pads to a multiple of 8 (symmetric padding) but
        # never crops back -- the model output keeps that padded size.
        result = result[:orig_h, :orig_w]
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
        return Image.fromarray(result)
