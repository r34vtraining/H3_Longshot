"""MiniMax H3 Long Shot — split one long prompt into N stitched H3 generations."""
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

try:        # 1.8.0: Face Refine (its own module, so a problem there can't take the pack down)
    from .face_node import NODE_CLASS_MAPPINGS as _FACE_NODES, NODE_DISPLAY_NAME_MAPPINGS as _FACE_NAMES
    NODE_CLASS_MAPPINGS.update(_FACE_NODES)
    NODE_DISPLAY_NAME_MAPPINGS.update(_FACE_NAMES)
except Exception:
    import logging
    logging.getLogger("MiniMaxH3LongShot").exception("MiniMax H3 Face Refine failed to load")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
