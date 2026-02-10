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

    def extract_points_from_mask(self, binary_mask, use_contours=True):
        """Unified point extraction for both positive/negative regions."""
        labels = measure.label(binary_mask.astype(np.uint8), connectivity=2)
        points = []
        
        for region in measure.regionprops(labels):
            if use_contours and region.area >= 9:
                region_mask = (labels == region.label)
                contours = measure.find_contours(region_mask.astype(np.float32), 0.5)
                if contours:
                    main_contour = max(contours, key=lambda c: len(c))
                    contour_xy = np.fliplr(main_contour)
                    pt = self.get_reference_point_from_contour(contour_xy)
                    points.append(pt)
                    continue
            
            cy, cx = region.centroid
            points.append((float(cx), float(cy)))
        
        return points

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
        EXACT REVERSE SEMANTICS (your specification):
          uint8:   0 = POSITIVE (black/object to keep)
                   1-254 = NEGATIVE (gray/background to exclude)
                   255 = IGNORED (white/neutral)
                   
          float:   0.0 = POSITIVE
                   0.0 < x < 1.0 = NEGATIVE
                   1.0 = IGNORED
        """
        B, H, W = mask.shape
        all_pos_points = []
        all_neg_points = []

        for b in range(B):
            mask_np = mask[b].cpu().numpy()
            
            # === DETECT MASK TYPE ===
            is_uint8 = mask_np.max() > 1.5
            
            if is_uint8:
                pos_mask = (mask_np == 0)
                neg_mask = (mask_np > 2) & (mask_np < 253)
            else:
                pos_mask = (mask_np <= 0.0)      # Pure black only
                neg_mask = (mask_np > 0.01) & (mask_np < 0.99)  # Strict mid-values only

            pos_points = self.extract_points_from_mask(pos_mask, use_contours=True)
            neg_points = self.extract_points_from_mask(
                neg_mask, 
                use_contours=True
            )

            all_pos_points.extend(pos_points)
            all_neg_points.extend(neg_points)

        # === STRICT PAIRING (no fallbacks) ===
        if individual_objects and all_pos_points and all_neg_points:
            # Only pair if negatives exist
            paired_neg = []
            for p in all_pos_points:
                distances = [(p[0]-n[0])**2 + (p[1]-n[1])**2 for n in all_neg_points]
                closest_neg = all_neg_points[int(np.argmin(distances))]
                paired_neg.append(closest_neg)
            all_neg_points = paired_neg

        # Format output
        pos_str = json.dumps([{"x": round(x, 2), "y": round(y, 2)} for x, y in all_pos_points])
        neg_str = json.dumps([{"x": round(x, 2), "y": round(y, 2)} for x, y in all_neg_points])

        return (pos_str, neg_str)
