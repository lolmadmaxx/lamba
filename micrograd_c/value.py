from typing import Union, Optional, List
from ._core import Value as CValue, get_value_lib, POINTER

class Value:
    """
    A scalar value with automatic differentiation support.
    
    This is a Python wrapper around the C implementation that provides
    a PyTorch-like interface for scalar operations and backpropagation.
    """
    _lib = None
    
    def __init__(self, data: float, _ptr: Optional[POINTER(CValue)] = None):
        """
        Create a new Value.
        
        Args:
            data: The scalar value
            _ptr: Internal use only - pointer to existing C Value
        """
        if Value._lib is None:
            Value._lib = get_value_lib()
            
        if _ptr is not None:
            self._ptr = _ptr
        else:
            self._ptr = Value._lib.value_new(float(data))
    
    @classmethod
    def persistent(cls, data: float) -> 'Value':
        """
        Create a persistent Value that won't be automatically freed.
        
        Use this for model parameters (weights, biases) that should survive
        computation graph cleanup operations.
        
        Args:
            data: The scalar value
            
        Returns:
            A persistent Value instance
        """
        if cls._lib is None:
            cls._lib = get_value_lib()
        ptr = cls._lib.value_new_persistent(float(data))
        return cls(0.0, _ptr=ptr)
    
    @property
    def is_persistent(self) -> bool:
        """Check if this Value is persistent (won't be auto-freed)"""
        return bool(self._ptr.contents.persistent)
    
    @property
    def data(self) -> float:
        """Get the scalar data value"""
        return float(self._ptr.contents.data)
    
    @data.setter
    def data(self, value: float):
        """Set the scalar data value"""
        self._ptr.contents.data = float(value)
    
    @property
    def grad(self) -> float:
        """Get the gradient value"""
        return float(self._ptr.contents.grad)
    
    @grad.setter
    def grad(self, value: float):
        """Set the gradient value"""
        self._ptr.contents.grad = float(value)
    
    def __repr__(self) -> str:
        return f"Value(data={self.data:.6f}, grad={self.grad:.6f})"
    
    def __str__(self) -> str:
        return f"Value({self.data:.6f})"
    
    # Arithmetic operations
    def __add__(self, other: Union['Value', float, int]) -> 'Value':
        """Addition: self + other"""
        other = self._ensure_value(other)
        result_ptr = Value._lib.value_add(self._ptr, other._ptr)
        return Value(0.0, _ptr=result_ptr)
    
    def __radd__(self, other: Union[float, int]) -> 'Value':
        """Reverse addition: other + self"""
        return self.__add__(other)
    
    def __mul__(self, other: Union['Value', float, int]) -> 'Value':
        """Multiplication: self * other"""
        other = self._ensure_value(other)
        result_ptr = Value._lib.value_mul(self._ptr, other._ptr)
        return Value(0.0, _ptr=result_ptr)
    
    def __rmul__(self, other: Union[float, int]) -> 'Value':
        """Reverse multiplication: other * self"""
        return self.__mul__(other)
    
    def __sub__(self, other: Union['Value', float, int]) -> 'Value':
        """Subtraction: self - other"""
        other = self._ensure_value(other)
        result_ptr = Value._lib.value_sub(self._ptr, other._ptr)
        return Value(0.0, _ptr=result_ptr)
    
    def __rsub__(self, other: Union[float, int]) -> 'Value':
        """Reverse subtraction: other - self"""
        other = self._ensure_value(other)
        result_ptr = Value._lib.value_sub(other._ptr, self._ptr)
        return Value(0.0, _ptr=result_ptr)
    
    def __truediv__(self, other: Union['Value', float, int]) -> 'Value':
        """Division: self / other"""
        other = self._ensure_value(other)
        result_ptr = Value._lib.value_div(self._ptr, other._ptr)
        return Value(0.0, _ptr=result_ptr)
    
    def __rtruediv__(self, other: Union[float, int]) -> 'Value':
        """Reverse division: other / self"""
        other = self._ensure_value(other)
        result_ptr = Value._lib.value_div(other._ptr, self._ptr)
        return Value(0.0, _ptr=result_ptr)
    def __pow__(self, exponent: Union[float, int]) -> 'Value':
        """Power: self ** exponent"""
        result_ptr = Value._lib.value_pow(self._ptr, float(exponent))
        return Value(0.0, _ptr=result_ptr)
    
    def __rpow__(self, base: Union[float, int]) -> 'Value':
        """Reverse power: base ** self"""
        base_val = self._ensure_value(base)
        result_ptr = Value._lib.value_pow(base_val._ptr, float(self.data))
        return Value(0.0, _ptr=result_ptr)

    def __neg__(self) -> 'Value':
        """Negation: -self"""
        result_ptr = Value._lib.value_neg(self._ptr)
        return Value(0.0, _ptr=result_ptr)
    
    def relu(self) -> 'Value':
        """Apply ReLU activation function"""
        result_ptr = Value._lib.value_relu(self._ptr)
        return Value(0.0, _ptr=result_ptr)
    
    def exp(self) -> 'Value':
        """Apply exponential function: e^x"""
        result_ptr = Value._lib.value_exp(self._ptr)
        return Value(0.0, _ptr=result_ptr)
    
    def log(self) -> 'Value':
        """Apply natural logarithm: ln(x)"""
        result_ptr = Value._lib.value_log(self._ptr)
        return Value(0.0, _ptr=result_ptr)
    
    def tanh(self) -> 'Value':
        """Apply hyperbolic tangent activation function"""
        result_ptr = Value._lib.value_tanh(self._ptr)
        return Value(0.0, _ptr=result_ptr)

    def softmax(self) -> 'Value':
        """
        Apply sigmoid activation function: σ(x) = 1 / (1 + e^(-x)).

        Note: For a single scalar Value, this behaves like sigmoid, not true vector softmax.
        """
        result_ptr = Value._lib.value_softmax(self._ptr)
        return Value(0.0, _ptr=result_ptr)
    
    def backward(self):
        """
        Perform backpropagation from this value.
        
        This computes gradients for all values in the computational graph
        that led to this value.
        """
        Value._lib.value_backward(self._ptr)
    
    def backward_and_free(self):
        """
        Perform backpropagation and automatically free non-persistent nodes.
        
        This is more memory-efficient than calling backward() separately
        as it automatically cleans up temporary computation nodes while
        preserving persistent model parameters.
        """
        Value._lib.value_backward_and_free(self._ptr)
    
    def zero_grad(self):
        """Zero the gradient of this value"""
        self.grad = 0.0
    
    @classmethod
    def zero_grad_all(cls):
        """Zero gradients for all existing Value instances"""
        raise NotImplementedError(
            "zero_grad_all is unimplemented. Use zero_grad() on individual instances "
            "or MLP.zero_grad() for neural networks. Instance tracking was removed for simplicity."
        )

    def _ensure_value(self, other: Union['Value', float, int]) -> 'Value':
        """Convert scalar to Value if needed"""
        if isinstance(other, Value):
            return other
        if isinstance(other, (int, float)):
            return Value(float(other))
        raise TypeError(f"Cannot convert {type(other).__name__} to Value. Expected Value, float, or int.")
    
    # Comparison operations (for convenience, based on data value)
    def __lt__(self, other: Union['Value', float, int]) -> bool:
        other_val = other.data if isinstance(other, Value) else float(other)
        return self.data < other_val
    
    def __le__(self, other: Union['Value', float, int]) -> bool:
        other_val = other.data if isinstance(other, Value) else float(other)
        return self.data <= other_val
    
    def __gt__(self, other: Union['Value', float, int]) -> bool:
        other_val = other.data if isinstance(other, Value) else float(other)
        return self.data > other_val
    
    def __ge__(self, other: Union['Value', float, int]) -> bool:
        other_val = other.data if isinstance(other, Value) else float(other)
        return self.data >= other_val
    
    def __eq__(self, other: Union['Value', float, int]) -> bool:
        other_val = other.data if isinstance(other, Value) else float(other)
        return abs(self.data - other_val) < 1e-8
    
    def __ne__(self, other: Union['Value', float, int]) -> bool:
        return not self.__eq__(other)
    
    # Safe operations with error handling
    def div_safe(self, other: Union['Value', float, int]) -> 'Value':
        """Safe division with automatic handling of near-zero denominators"""
        other = self._ensure_value(other)
        result_ptr = Value._lib.value_div_safe(self._ptr, other._ptr)
        return Value(0.0, _ptr=result_ptr)
    
    def log_safe(self) -> 'Value':
        """Safe logarithm with automatic handling of non-positive values"""
        result_ptr = Value._lib.value_log_safe(self._ptr)
        return Value(0.0, _ptr=result_ptr)
    
    def pow_safe(self, exponent: Union[float, int]) -> 'Value':
        """Safe power operation with overflow/underflow protection"""
        result_ptr = Value._lib.value_pow_safe(self._ptr, float(exponent))
        return Value(0.0, _ptr=result_ptr)
    
    # Utility methods
    def free(self):
        """Manually free this Value (use with caution)"""
        if hasattr(self, '_ptr') and self._ptr:
            Value._lib.value_free(self._ptr)
            self._ptr = None
    
    @classmethod
    def free_graph(cls, root: 'Value', preserve: Optional[List['Value']] = None):
        """Free an entire computational graph starting from root, optionally preserving specific nodes"""
        if not root or not hasattr(root, '_ptr') or not root._ptr:
            return
            
        if preserve and hasattr(Value._lib, 'value_free_graph_safe'):
            # Use safe cleanup that preserves specific nodes
            from ctypes import POINTER, c_int
            from ._core import Value as CValue
            
            # Filter valid parameters
            valid_params = [v for v in preserve if v and hasattr(v, '_ptr') and v._ptr]
            
            if valid_params:
                param_count = len(valid_params)
                param_ptrs = (POINTER(CValue) * param_count)()
                
                for i, param in enumerate(valid_params):
                    param_ptrs[i] = param._ptr
                
                # Call the safe cleanup function
                Value._lib.value_free_graph_safe(root._ptr, param_ptrs, param_count)
            else:
                # No valid parameters to preserve, use regular cleanup
                Value._lib.value_free_graph(root._ptr)
        else:
            # Fallback to regular cleanup
            Value._lib.value_free_graph(root._ptr)

    def clone(self) -> 'Value':
        """Create a new Value with the same data and gradient but no computational history"""
        new_val = Value(self.data)
        new_val.grad = self.grad
        return new_val
    
    def detach(self) -> 'Value':
        """Create a detached copy (breaks computational graph, zeros gradient)"""
        return Value(self.data)
    
    # Advanced utility methods
    def get_data_safe(self) -> float:
        """Safely get data value"""
        return Value._lib.value_get_data(self._ptr) if self._ptr else 0.0
    
    def get_grad_safe(self) -> float:
        """Safely get gradient value"""
        return Value._lib.value_get_grad(self._ptr) if self._ptr else 0.0
    
    def set_data_safe(self, data: float):
        """Safely set data value"""
        if self._ptr:
            Value._lib.value_set_data(self._ptr, float(data))
    
    def set_grad_safe(self, grad: float):
        """Safely set gradient value"""
        if self._ptr:
            Value._lib.value_set_grad(self._ptr, float(grad))
    
    def zero_grad_safe(self):
        """Safely zero gradient"""
        if self._ptr:
            Value._lib.value_zero_grad_single(self._ptr)
