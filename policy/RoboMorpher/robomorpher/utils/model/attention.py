"""Backend availability helper adapted from the pinned upstream source.

See docs/provenance/starvla/UPSTREAM.json and LICENSE for provenance.
"""

def has_flash_attn():
    """Check whether a flash-attention backend is available.

    Returns True if either ``torch_npu`` (Ascend NPU) or ``flash_attn`` (GPU)
    can be imported, otherwise False.  This allows VLM modules to gracefully
    fall back to SDPA when no compatible flash-attention implementation is found.
    """
    try:
        import torch_npu  # NPU flash-attention backend
        return True
    except ImportError:
        pass
    try:
        import flash_attn  # GPU flash-attention backend
        return True
    except ImportError:
        return False
