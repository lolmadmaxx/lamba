from .value import Value
from .mlp import MLP, Layer
from .engine import Engine, SGD, Adam, LangevinLandauOptimizer
from .wavepacket import WavePacketLayer, WavePacketMLP

__version__ = "0.0.1"
__author__ = "Mehmet Batuhan Duman"

# Initialize memory management constants
def _initialize_constants():
    """Initialize shared constants for memory optimization"""
    try:
        from ._core import get_value_lib
        value_lib = get_value_lib()
        if hasattr(value_lib, 'value_init_constants'):
            value_lib.value_init_constants()
    except (AttributeError, ImportError):
        pass  # Constants functions not available

def _cleanup_constants():
    """Cleanup shared constants"""
    try:
        from ._core import get_value_lib
        value_lib = get_value_lib()
        if hasattr(value_lib, 'value_cleanup_constants'):
            value_lib.value_cleanup_constants()
    except (AttributeError, ImportError):
        pass  # Constants functions not available

# Initialize constants when module is imported
_initialize_constants()

# Register cleanup function to run on exit
import atexit
atexit.register(_cleanup_constants)

__all__ = [
    "Value",
    "MLP",
    "Layer",
    "Engine",
    "SGD",
    "Adam",
    "LangevinLandauOptimizer",
    "WavePacketLayer",
    "WavePacketMLP",
]
