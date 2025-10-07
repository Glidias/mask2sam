import torch
import numpy as np
import json
from skimage import measure


class MaskToBBoxes:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("BBOX",)  # Use "BBOX" to match Kijai's expected type
    RETURN_NAMES = ("bboxes",)
    FUNCTION = "mask_to_bboxes"
    CATEGORY = "SAM2/utils"

    def mask_to_bboxes(self, mask):
        """
        Convert a batch of masks into bounding boxes.
        Input: mask (B, H, W) — black (0) = foreground (your Krita convention)
        Output: list of lists of [x1, y1, x2, y2] per batch item, as required by Kijai's SAM2 node.
        """
        # mask is (B, H, W)
        B, H, W = mask.shape
        all_bboxes = []

        for b in range(B):
            mask_np = mask[b].cpu().numpy()
            # Foreground = black (0)
            foreground = (mask_np == 0)
            labels = measure.label(foreground, connectivity=2)
            regions = measure.regionprops(labels)

            bboxes_this_batch = []
            for region in regions:
                min_row, min_col, max_row, max_col = region.bbox
                # Convert to (x1, y1, x2, y2) = (col, row, col, row)
                bbox = [float(min_col), float(min_row), float(max_col), float(max_row)]
                bboxes_this_batch.append(bbox)

            # Even if no objects, append empty list (Kijai handles it)
            all_bboxes.append(bboxes_this_batch)

        # Kijai's node expects: bboxes = [[bbox1, bbox2, ...], [bbox1, ...], ...]
        # This is a Python list of lists — ComfyUI will pass it as "BBOX" type
        return (all_bboxes,)