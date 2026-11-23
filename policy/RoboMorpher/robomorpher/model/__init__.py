"""Project model packages: policies, backbones, tokenizer and shared components.

The action codec implementation lives under robomorpher.model.tokenizer. These module
aliases keep old imports working without retaining duplicate top-level files.
"""
import sys
from .tokenizer import action_codec
from .tokenizer import gripper_tokenizer
from .tokenizer import state_tokenizer

sys.modules[__name__ + ".action_codec"] = action_codec
sys.modules[__name__ + ".gripper_tokenizer"] = gripper_tokenizer
sys.modules[__name__ + ".state_tokenizer"] = state_tokenizer
