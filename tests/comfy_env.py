"""Boot a real ComfyUI source tree in CPU mode, so the tests exercise upstream
H3 code rather than a reimplementation. Set COMFYUI_ROOT to your ComfyUI folder."""
import os
import sys


def boot():
    root = os.environ.get("COMFYUI_ROOT")
    if not root or not os.path.isdir(os.path.join(root, "comfy")):
        raise RuntimeError("Set COMFYUI_ROOT to your ComfyUI folder (the one containing 'comfy').")
    if root not in sys.path:
        sys.path.insert(0, root)
    import comfy.options
    comfy.options.enable_args_parsing(True)
    saved, sys.argv = sys.argv, ["main.py", "--cpu"]
    try:
        import comfy.cli_args  # noqa: F401
    finally:
        sys.argv = saved
    return root
