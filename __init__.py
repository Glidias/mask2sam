from .bbox import MaskToBBoxes
from .points import MaskToPosNegPoints

__all__ = [
    "MaskToBBoxes",
    "MaskToPoints",
]

# Add node mappings
NODE_CLASS_MAPPINGS = {
    "MaskToBBoxes": MaskToBBoxes,
    "MaskToPoints": MaskToPosNegPoints,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskToBBoxes": "📦 Mask to BBoxes",
    "MaskToPoints": "📍 Mask to Points"
}