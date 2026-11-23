"""The config-driven policy constructor shared by training and inference.

Architecture constructors own tokenizer plugins, statistics and model options.
Callers own optimizer/resume or weight loading, device placement and mode.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def build_policy(config, **kwargs):
    os.environ['USE_TF'] = '0'
    os.environ['USE_FLAX'] = '0'
    name = str(config['framework'].get('name', 'QwenDiscreteForcing'))
    if name in {'QwenDiscreteForcing', 'QwenDiscreteDiffusion'}:
        from robomorpher.model.architectures.qwen_discrete_forcing import QwenDiscreteForcing
        constructor = QwenDiscreteForcing
    elif name == 'QwenPI_v3':
        from robomorpher.model.architectures.qwen_pi_v3 import QwenPI_v3
        constructor = QwenPI_v3
    else:
        raise ValueError(f'Unsupported framework.name={name!r}')
    # Training launchers and saved configs use project-relative dependencies.
    # Resolve them under the same root without rewriting contract-visible config.
    previous = os.getcwd()
    os.chdir(ROOT)
    try:
        return constructor(config, **kwargs)
    finally:
        os.chdir(previous)
