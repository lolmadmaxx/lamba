import ctypes
from ctypes import c_float, c_int, c_void_p, c_bool, POINTER, Structure
import os
from pathlib import Path
from typing import Optional

_package_dir = Path(__file__).parent
_lib_dir = _package_dir / "lib"

class Value(Structure):
    pass

Value._fields_ = [
    ('data',     c_float),
    ('grad',     c_float),
    ('left',     POINTER(Value)),
    ('right',    POINTER(Value)),
    ('backward', ctypes.CFUNCTYPE(None, POINTER(Value))),
    ('op',       ctypes.c_char),
    ('_exponent',c_float),
    ('persistent', c_bool),
]

class LibraryLoader:
    """Manages loading and caching of C libraries"""
    
    _value_lib: Optional[ctypes.CDLL] = None
    _mlp_lib: Optional[ctypes.CDLL] = None
    
    @classmethod
    def get_value_lib(cls) -> ctypes.CDLL:
        """Load and return the value library"""
        if cls._value_lib is None:
            lib_path = _lib_dir / "value.dll"
            if not lib_path.exists():
                raise FileNotFoundError(f"Value library not found at {lib_path}")
            
            cls._value_lib = ctypes.CDLL(str(lib_path))
            cls._setup_value_prototypes()
            
        return cls._value_lib
    @classmethod
    def get_mlp_lib(cls) -> ctypes.CDLL:
        """Load and return the MLP library"""
        if cls._mlp_lib is None:
            lib_path = _lib_dir / "mlp.dll"
            if not lib_path.exists():
                raise FileNotFoundError(f"MLP library not found at {lib_path}")
            
            cls._mlp_lib = ctypes.CDLL(str(lib_path))
            cls._setup_mlp_prototypes()
            
        return cls._mlp_lib
    
    @classmethod
    def _setup_value_prototypes(cls):
        """Set up function prototypes for value library"""
        lib = cls._value_lib
        
          # Value operations
        lib.value_new.argtypes = [c_float]
        lib.value_new.restype = POINTER(Value)
        
        # NEW: Persistent value constructor
        lib.value_new_persistent.argtypes = [c_float]
        lib.value_new_persistent.restype = POINTER(Value)
        
        lib.value_add.argtypes = [POINTER(Value), POINTER(Value)]
        lib.value_add.restype = POINTER(Value)
        
        lib.value_mul.argtypes = [POINTER(Value), POINTER(Value)]
        lib.value_mul.restype = POINTER(Value)
        
        lib.value_sub.argtypes = [POINTER(Value), POINTER(Value)]
        lib.value_sub.restype = POINTER(Value)
        
        lib.value_neg.argtypes = [POINTER(Value)]
        lib.value_neg.restype = POINTER(Value)
        
        lib.value_pow.argtypes = [POINTER(Value), c_float]
        lib.value_pow.restype = POINTER(Value)
        
        lib.value_div.argtypes = [POINTER(Value), POINTER(Value)]
        lib.value_div.restype = POINTER(Value)
        
        lib.value_relu.argtypes = [POINTER(Value)]
        lib.value_relu.restype = POINTER(Value)
        
        lib.value_exp.argtypes = [POINTER(Value)]
        lib.value_exp.restype = POINTER(Value)
        lib.value_log.argtypes = [POINTER(Value)]
        lib.value_log.restype = POINTER(Value)
        
        lib.value_tanh.argtypes = [POINTER(Value)]
        lib.value_tanh.restype = POINTER(Value)
        
        lib.value_softmax.argtypes = [POINTER(Value)]
        lib.value_softmax.restype = POINTER(Value)
        
        lib.value_backward.argtypes = [POINTER(Value)]
        lib.value_backward.restype = None
        
        # NEW: Backward + free in one call
        lib.value_backward_and_free.argtypes = [POINTER(Value)]
        lib.value_backward_and_free.restype = None
        
        # Safe operations
        lib.value_div_safe.argtypes = [POINTER(Value), POINTER(Value)]
        lib.value_div_safe.restype = POINTER(Value)
        
        lib.value_log_safe.argtypes = [POINTER(Value)]
        lib.value_log_safe.restype = POINTER(Value)
        
        lib.value_pow_safe.argtypes = [POINTER(Value), c_float]
        lib.value_pow_safe.restype = POINTER(Value)
        
        # Memory management
        lib.value_free.argtypes = [POINTER(Value)]
        lib.value_free.restype = None
        
        # Utility functions
        lib.value_get_data.argtypes = [POINTER(Value)]
        lib.value_get_data.restype = c_float
        
        lib.value_get_grad.argtypes = [POINTER(Value)]
        lib.value_get_grad.restype = c_float
        
        lib.value_set_data.argtypes = [POINTER(Value), c_float]
        lib.value_set_data.restype = None
        
        lib.value_set_grad.argtypes = [POINTER(Value), c_float]
        lib.value_set_grad.restype = None
        
        lib.value_zero_grad_single.argtypes = [POINTER(Value)]
        lib.value_zero_grad_single.restype = None
        
        # Debug function
        lib.debug_print_addresses.argtypes = [POINTER(Value), POINTER(POINTER(Value)), c_int]
        lib.debug_print_addresses.restype = None
        
        # Memory management functions
        try:           
            if hasattr(lib, 'value_init_constants'):
                lib.value_init_constants.argtypes = []
                lib.value_init_constants.restype = None
                
            if hasattr(lib, 'value_cleanup_constants'):
                lib.value_cleanup_constants.argtypes = []
                lib.value_cleanup_constants.restype = None
                
            if hasattr(lib, 'value_free_graph'):
                lib.value_free_graph.argtypes = [POINTER(Value)]
                lib.value_free_graph.restype = None
                
            if hasattr(lib, 'value_free_graph_safe'):
                lib.value_free_graph_safe.argtypes = [POINTER(Value), POINTER(POINTER(Value)), c_int]
                lib.value_free_graph_safe.restype = None
                
            if hasattr(lib, 'mlp_cleanup_forward_temps'):
                lib.mlp_cleanup_forward_temps.argtypes = [c_void_p]
                lib.mlp_cleanup_forward_temps.restype = None
        except AttributeError:
            pass  
    
    @classmethod
    def _setup_mlp_prototypes(cls):
        """Set up function prototypes for MLP library"""
        lib = cls._mlp_lib
        
        # MLP functions
        lib.mlp_new.argtypes = [c_int, POINTER(c_int), c_int]
        lib.mlp_new.restype = c_void_p
        
        lib.mlp_forward.argtypes = [c_void_p, POINTER(POINTER(Value))]
        lib.mlp_forward.restype = POINTER(POINTER(Value))
        
        lib.mlp_parameters.argtypes = [c_void_p, POINTER(POINTER(Value))]
        lib.mlp_parameters.restype = None
        
        lib.zero_grad.argtypes = [POINTER(POINTER(Value)), c_int]
        lib.zero_grad.restype = None
        
        # Memory management
        lib.mlp_free.argtypes = [c_void_p]
        lib.mlp_free.restype = None
        
        lib.layer_free.argtypes = [c_void_p]
        lib.layer_free.restype = None
        
        lib.neuron_free.argtypes = [c_void_p]
        lib.neuron_free.restype = None
        
        # Utility functions
        lib.mlp_parameter_count.argtypes = [c_void_p]
        lib.mlp_parameter_count.restype = c_int
        
        lib.layer_parameter_count.argtypes = [c_void_p]
        lib.layer_parameter_count.restype = c_int
        
        lib.mlp_zero_grad.argtypes = [c_void_p]
        lib.mlp_zero_grad.restype = None
        
        # Memory cleanup functions (added for safety)
        try:
            if hasattr(lib, 'mlp_free_forward_output'):
                lib.mlp_free_forward_output.argtypes = [POINTER(POINTER(Value)), c_int]
                lib.mlp_free_forward_output.restype = None
                
            if hasattr(lib, 'mlp_cleanup_forward_temps'):
                lib.mlp_cleanup_forward_temps.argtypes = [c_void_p]
                lib.mlp_cleanup_forward_temps.restype = None
        except AttributeError:
            pass  

def get_value_lib() -> ctypes.CDLL:
    """Get the loaded value library"""
    return LibraryLoader.get_value_lib()

def get_mlp_lib() -> ctypes.CDLL:
    """Get the loaded MLP library"""
    return LibraryLoader.get_mlp_lib()

# Export common ctypes types and the Value structure
CValue = Value 
__all__ = [
    'Value', 'CValue', 'get_value_lib', 'get_mlp_lib', 
    'POINTER', 'c_int', 'c_float', 'c_void_p'
]
