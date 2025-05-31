from typing import List, Union, Sequence
import random
from .value import Value
from ._core import get_mlp_lib, get_value_lib, POINTER, CValue, c_int, c_void_p

class Layer:
    """
    A single layer of a multi layer perceptron.
    
    This is a convenience class that represents a single dense layer
    with optional activation function.
    """
    
    def __init__(self, nin: int, nout: int, activation: str = 'linear'):
        """
        Create a new layer.
        
        Args:
            nin: Number of input features
            nout: Number of output features  
            activation: Activation function ('linear', 'relu')
        """
        self.nin = nin
        self.nout = nout
        self.activation = activation.lower()
        
        if self.activation not in ['linear', 'relu']:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def __repr__(self) -> str:
        return f"Layer({self.nin}, {self.nout}, activation='{self.activation}')"

class MLP:
    """
    Multi Layer Perceptron with automatic differentiation.

    """
    
    def __init__(self, nin: int, layer_specs: Union[List[int], List[Layer]]):
        """
        Create a new MLP.
        
        Args:
            nin: Number of input features
            layer_specs: List of layer specifications. Can be:
                - List of integers: [4, 2] creates layers with those output sizes
                - List of Layer objects: [Layer(4, 4, 'relu'), Layer(4, 2, 'linear')]
        """
        self._value_lib = get_value_lib()
        self._mlp_lib = get_mlp_lib()
        
        # Process layer specifications
        if isinstance(layer_specs[0], int):
            # Convert integer list to Layer objects with ReLU for hidden layers
            self.layers = []
            for i, nout in enumerate(layer_specs):
                # Use ReLU for all layers except the last one
                activation = 'relu' if i < len(layer_specs) - 1 else 'linear'
                nin_current = nin if i == 0 else layer_specs[i-1]
                self.layers.append(Layer(nin_current, nout, activation))
        else:
            # Use provided Layer objects
            self.layers = list(layer_specs)
            
        # Validate layer dimensions
        current_nin = nin
        for i, layer in enumerate(self.layers):
            if layer.nin != current_nin:
                raise ValueError(f"Layer {i} input size {layer.nin} doesn't match expected {current_nin}")
            current_nin = layer.nout
        
        # Extract dimensions for C library
        self.nin = nin
        nouts = [layer.nout for layer in self.layers]
        self.nouts = nouts
        self.final_nout = nouts[-1]
        
        # Create C MLP object
        L = len(nouts)
        arr = (c_int * L)(*nouts)
        self._mlp_obj = self._mlp_lib.mlp_new(nin, arr, L)
        
        # Calculate total parameter count
        ins = [nin] + nouts[:-1]
        self._param_count = sum((ins[i] + 1) * nouts[i] for i in range(L))
        
        # Get parameter pointers
        self._param_ptrs = (POINTER(CValue) * self._param_count)()
        self._mlp_lib.mlp_parameters(self._mlp_obj, self._param_ptrs)
        self._parameters = [Value(0.0, _ptr=self._param_ptrs[i]) for i in range(self._param_count)]
    
    def forward(self, inputs: Sequence[Union[float, Value]]) -> List[Value]:
        """
        Forward pass through the network.
        
        Args:
            inputs: Input values (floats or Values)
            
        Returns:
            List of output Values
        """
        if len(inputs) != self.nin:
            raise ValueError(f"Expected {self.nin} inputs, got {len(inputs)}")
        
        # Clean up previous forward pass immediately
        self.cleanup_last_forward()
        
        # Convert inputs to C Value pointers
        inp = (POINTER(CValue) * len(inputs))()
        temp_values = []  # Track temporary Values for cleanup
        
        for i, x in enumerate(inputs):
            if isinstance(x, Value):
                inp[i] = x._ptr
            else:
                temp_val = self._value_lib.value_new(float(x))
                inp[i] = temp_val
                temp_values.append(temp_val)
        
        # Perform forward pass
        outp = self._mlp_lib.mlp_forward(self._mlp_obj, inp)
        
        # Wrap outputs in Python Value objects
        outputs = [Value(0.0, _ptr=outp[i]) for i in range(self.final_nout)]
        
        # Store cleanup info for later use
        self._last_forward_outputs = outp
        self._last_temp_values = temp_values
        
        return outputs
    
    def __call__(self, inputs: Sequence[Union[float, Value]]) -> List[Value]:
        """Callable interface for forward pass"""
        return self.forward(inputs)
    
    @property
    def parameters(self) -> List[Value]:
        """Get all trainable parameters"""
        return self._parameters
    
    def zero_grad(self):
        """Zero gradients for all parameters"""
        self._mlp_lib.zero_grad(self._param_ptrs, self._param_count)
    
    def cleanup_last_forward(self):
        """Clean up memory from the last forward pass"""
        # Clean up the output array
        if hasattr(self, '_last_forward_outputs') and self._last_forward_outputs:
            try:
                if hasattr(self._mlp_lib, 'mlp_free_forward_output'):
                    self._mlp_lib.mlp_free_forward_output(self._last_forward_outputs, self.final_nout)
                else:
                    # Fallback: manually free the array if function doesn't exist
                    import ctypes
                    ctypes.pythonapi.PyMem_Free(self._last_forward_outputs)
            except (AttributeError, OSError):
                pass 
            self._last_forward_outputs = None
        
        # Clean up temporary input values
        if hasattr(self, '_last_temp_values') and self._last_temp_values:
            for temp_val in self._last_temp_values:
                if temp_val:
                    try:
                        self._value_lib.value_free(temp_val)
                    except (AttributeError, OSError):
                        pass
            self._last_temp_values = []
    
    def get_parameter_info(self) -> dict:
        """Get information about parameters organized by layer"""
        info = {
            'total_params': self._param_count,
            'layers': []
        }
        
        offset = 0
        ins = [self.nin] + [layer.nout for layer in self.layers[:-1]]
        
        for i, layer in enumerate(self.layers):
            layer_params = (ins[i] + 1) * layer.nout
            weights_start = offset
            weights_end = offset + ins[i] * layer.nout
            bias_start = weights_end
            bias_end = offset + layer_params
            
            layer_info = {
                'layer_idx': i,
                'input_size': ins[i],
                'output_size': layer.nout,
                'activation': layer.activation,
                'param_count': layer_params,
                'weight_params': ins[i] * layer.nout,
                'bias_params': layer.nout,
                'weight_indices': list(range(weights_start, weights_end)),
                'bias_indices': list(range(bias_start, bias_end))
            }
            info['layers'].append(layer_info)
            offset += layer_params
            
        return info
    
    def get_weights_and_biases(self) -> List[dict]:
        """Get weights and biases organized by layer"""
        result = []
        param_info = self.get_parameter_info()
        
        for layer_info in param_info['layers']:
            weights = [self._parameters[i] for i in layer_info['weight_indices']]
            biases = [self._parameters[i] for i in layer_info['bias_indices']]
            
            result.append({
                'weights': weights,
                'biases': biases,
                'input_size': layer_info['input_size'],
                'output_size': layer_info['output_size'],
                'activation': layer_info['activation']
            })
            
        return result
    
    def initialize_parameters(self, method: str = 'xavier', seed: int = None):
        """
        Initialize parameters using various methods.
        
        Args:
            method: Initialization method ('xavier', 'random', 'zero')
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
            
        if method == 'zero':
            for param in self._parameters:
                param.data = 0.0
        elif method == 'random':
            for param in self._parameters:
                param.data = random.uniform(-1, 1)
        elif method == 'xavier':
            param_info = self.get_parameter_info()
            param_idx = 0
            
            for layer_info in param_info['layers']:
                fan_in = layer_info['input_size']
                fan_out = layer_info['output_size']
                
                # Xavier initialization for weights
                limit = (6.0 / (fan_in + fan_out)) ** 0.5
                for _ in range(layer_info['weight_params']):
                    self._parameters[param_idx].data = random.uniform(-limit, limit)
                    param_idx += 1
                
                # Zero initialization for biases
                for _ in range(layer_info['bias_params']):
                    self._parameters[param_idx].data = 0.0
                    param_idx += 1
        else:
            raise ValueError(f"Unknown initialization method: {method}")
    
    def __repr__(self) -> str:
        layer_sizes = [self.nin] + self.nouts
        return f"MLP({layer_sizes}, {len(self._parameters)} parameters)"
      # Memory management methods
    def free(self):
        """Free the C MLP object (use with caution)"""
        if hasattr(self, '_mlp_obj') and self._mlp_obj:
            self._mlp_lib.mlp_free(self._mlp_obj)
            self._mlp_obj = None
    
    def __del__(self):
        """Destructor to automatically free resources"""
        try:
            self.cleanup_last_forward()  # Clean up any pending forward pass data
            self.free()
        except:
            pass  # Ignore errors during cleanup
    
    # Enhanced utility methods
    def parameter_count(self) -> int:
        """Get total parameter count using C implementation"""
        if hasattr(self, '_mlp_obj') and self._mlp_obj:
            return self._mlp_lib.mlp_parameter_count(self._mlp_obj)
        return len(self._parameters)
    
    def zero_grad_fast(self):
        """Fast gradient zeroing using C implementation"""
        if hasattr(self, '_mlp_obj') and self._mlp_obj:
            self._mlp_lib.mlp_zero_grad(self._mlp_obj)
        else:
            # Fallback to Python implementation
            self.zero_grad()
    
    def get_gradient_norm(self) -> float:
        """Compute L2 norm of all gradients"""
        total = 0.0
        for param in self._parameters:
            total += param.grad ** 2
        return total ** 0.5
    
    def clip_gradients(self, max_norm: float):
        """Clip gradients to prevent exploding gradients"""
        grad_norm = self.get_gradient_norm()
        if grad_norm > max_norm:
            scale = max_norm / grad_norm
            for param in self._parameters:
                param.grad *= scale
    
    def get_parameter_stats(self) -> dict:
        """Get statistics about parameters (mean, std, min, max)"""
        if not self._parameters:
            return {}
        
        data_values = [p.data for p in self._parameters]
        grad_values = [p.grad for p in self._parameters]
        
        def calc_stats(values):
            if not values:
                return {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
            mean = sum(values) / len(values)
            variance = sum((x - mean) ** 2 for x in values) / len(values)
            std = variance ** 0.5
            return {
                'mean': mean,
                'std': std,
                'min': min(values),
                'max': max(values)
            }
        
        return {
            'data': calc_stats(data_values),
            'grad': calc_stats(grad_values),
            'param_count': len(self._parameters)
        }
    
    def save_parameters(self, filepath: str):
        """Save parameters to a file"""
        import json
        data = {
            'nin': self.nin,
            'nouts': self.nouts,
            'layers': [{'nin': layer.nin, 'nout': layer.nout, 'activation': layer.activation} 
                      for layer in self.layers],
            'parameters': [param.data for param in self._parameters]
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load_parameters(cls, filepath: str) -> 'MLP':
        """Load parameters from a file"""
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Reconstruct layer specs
        layer_specs = []
        for layer_data in data['layers']:
            layer_specs.append(Layer(
                layer_data['nin'], 
                layer_data['nout'], 
                layer_data['activation']
            ))
        
        # Create MLP
        mlp = cls(data['nin'], layer_specs)
        
        # Load parameters
        for i, param_value in enumerate(data['parameters']):
            if i < len(mlp._parameters):
                mlp._parameters[i].data = param_value
        
        return mlp
