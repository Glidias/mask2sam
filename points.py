import torch
import numpy as np
import json
from skimage import measure
from .poly_decomp import polygonQuickDecomp as bayazit_decomp
from shapely.geometry import Polygon

class MaskToPosNegPoints:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "individual_objects": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("coordinates_positive", "coordinates_negative")
    FUNCTION = "mask_to_points"
    CATEGORY = "SAM2/utils"

    def get_reference_point_from_contour(self, contour):
        """Use bayazit_decomp to get centroid of largest convex piece."""
        if len(contour) < 3:
            # Fallback to centroid
            cx = np.mean(contour[:, 0])
            cy = np.mean(contour[:, 1])
            return (float(cx), float(cy))

        # Close contour if needed
        if not np.array_equal(contour[0], contour[-1]):
            contour = np.vstack([contour, contour[0]])

        # Convert to list of [x, y]
        poly_coords = contour[:-1].tolist()

        try:
            convex_pieces = bayazit_decomp(poly_coords)
            if not convex_pieces:
                raise ValueError("Decomposition failed")

            # Find largest by area
            max_area = -1
            best_centroid = None
            for piece in convex_pieces:
                if len(piece) < 3:
                    continue
                poly = Polygon(piece)
                if poly.is_valid and poly.area > max_area:
                    max_area = poly.area
                    centroid = poly.centroid
                    best_centroid = (float(centroid.x), float(centroid.y))

            if best_centroid is None:
                # Fallback
                cx = np.mean(contour[:, 0])
                cy = np.mean(contour[:, 1])
                return (float(cx), float(cy))
            return best_centroid

        except Exception as e:
            # Fallback to simple centroid
            cx = np.mean(contour[:, 0])
            cy = np.mean(contour[:, 1])
            return (float(cx), float(cy))

    def mask_to_points(self, mask, individual_objects=True):
        """
        Convert batch of masks to positive/negative points.
        - mask: (B, H, W)
        - Black (0) = positive
        - Gray (0 < val < 255) = negative
        - White (255) = ignored (optional: enforce via neg_mask = (0 < mask < 255))
        """
        B, H, W = mask.shape
        pos_points = []
        neg_points = []

        for b in range(B):
            mask_np = mask[b].cpu().numpy()

            # Positive: black (0)
            pos_mask = (mask_np == 0)
            # Negative: gray only (0 < x < 255), exclude white
            neg_mask = (mask_np > 0) & (mask_np < 255)

            # --- Process positive regions ---
            pos_labels = measure.label(pos_mask, connectivity=2)
            regionprops = measure.regionprops(pos_labels)
            print(f"Found {len(regionprops)} positive regions in batch item {b}")
            for region in regionprops:
                region_mask = (pos_labels == region.label)
                contours = measure.find_contours(region_mask, 0.5)
                if contours:
                    # Use largest contour by length (or area if you implement it)
                    main_contour = max(contours, key=lambda c: len(c))
                    contour_xy = np.fliplr(main_contour)  # (col, row)
                    ref_pt = self.get_reference_point_from_contour(contour_xy)
                    pos_points.append(ref_pt)

            # --- Process negative regions ---
            neg_labels = measure.label(neg_mask, connectivity=2)
            for region in measure.regionprops(neg_labels):
                cy, cx = region.centroid
                neg_points.append((float(cx), float(cy)))

        # --- Handle individual_objects mode ---
        if individual_objects:
            if not neg_points:
                neg_points = []
            else:
                # Pair each positive with closest negative (reuse allowed)
                paired_neg = []
                for p in pos_points:
                    distances = [(p[0]-n[0])**2 + (p[1]-n[1])**2 for n in neg_points]
                    closest_neg = neg_points[int(np.argmin(distances))]
                    paired_neg.append(closest_neg)
                neg_points = paired_neg

        # --- Format for Kijai ---
        pos_dicts = [{"x": x, "y": y} for (x, y) in pos_points]
        neg_dicts = [{"x": x, "y": y} for (x, y) in neg_points]

        pos_str = json.dumps(pos_dicts)
        neg_str = json.dumps(neg_dicts)

        print("Positive points:", pos_str)
        print("Negative points:", neg_str)

        return (pos_str, neg_str)
