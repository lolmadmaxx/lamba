from typing import List, Callable, Union, Optional
import random
import gc
import math
from .value import Value
from .mlp import MLP

class SGD:
    """
    Stochastic Gradient Descent optimizer.
    """
    
    def __init__(self, params, lr=0.01):
        """
        SGD optimizer.
        
        Args:
            params: List of parameters to optimize (e.g., model weights)
            lr: Learning rate 
        """
        self.params = params
        self.lr = lr
    
    def step(self):
        """Perform one optimization step"""
        for param in self.params:
            param.data -= self.lr * param.grad
    
    def zero_grad(self):
        """Zero gradients for all parameters"""
        for param in self.params:
            param.grad = 0.0


class Adam:
    def __init__(self, params, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
        self.params = params
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = [0.0] * len(params)
        self.v = [0.0] * len(params)
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            g = p.grad
            # update biased moments
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            # compute bias-corrected estimates
            m_hat = self.m[i] / (1 - self.b1**self.t)
            v_hat = self.v[i] / (1 - self.b2**self.t)
            # parameter update
            p.data -= self.lr * m_hat / (v_hat**0.5 + self.eps)

    def zero_grad(self):
        for p in self.params:
            p.grad = 0.0


class LangevinLandauOptimizer:
    """
    Langevin-Landau optimizer with damping and thermal noise.
    
    This optimizer uses Langevin dynamics to add controlled noise to the optimization
    process, which can help escape local minima and improve exploration.
    
    The update rule is:
    v_t = (1 - damping) * v_t-1 + lr * force + noise
    param_t = param_t-1 + v_t
    
    where force = -gradient and noise ~ Normal(0, sqrt(2 * damping * temperature * lr))
    """
    
    def __init__(self, params, learning_rate=0.001, damping=0.1, temperature=0.01, seed=None):
        """
        Initialize Langevin-Landau optimizer.
        
        Args:
            params: List of parameters to optimize
            learning_rate: Learning rate for force updates
            damping: Damping coefficient (0-1, higher values = more damping)
            temperature: Temperature for thermal noise (higher = more exploration)
            seed: Random seed for reproducibility
        """
        self.params = params
        self.lr = learning_rate
        self.damping = damping
        self.temperature = temperature
        
        # Initialize velocities for each parameter
        self.velocities = [0.0] * len(params)
        
        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
    def step(self):
        """
        Perform one optimization step using Langevin dynamics.
        """
        for i, param in enumerate(self.params):
            # Current velocity
            velocity = self.velocities[i]
            
            # Force is negative gradient
            force = -param.grad
            
            # Update velocity with damping and force
            velocity = (1 - self.damping) * velocity + self.lr * force
            
            # Add thermal noise
            # noise ~ Normal(0, sqrt(2 * damping * temperature * lr))
            noise_std = math.sqrt(2 * self.damping * self.temperature * self.lr)
            noise = random.gauss(0, noise_std)
            velocity += noise
            
            # Store updated velocity
            self.velocities[i] = velocity
            
            # Update parameter
            param.data += velocity
    
    def zero_grad(self):
        """Zero gradients for all parameters"""
        for param in self.params:
            param.grad = 0.0
    
    def get_velocity_norm(self) -> float:
        """Get L2 norm of all velocities for monitoring"""
        total = sum(v * v for v in self.velocities)
        return math.sqrt(total)
    
    def set_temperature(self, temperature: float):
        """Dynamically adjust temperature during training"""
        self.temperature = temperature
    
    def set_damping(self, damping: float):
        """Dynamically adjust damping during training"""
        if not 0 <= damping <= 1:
            raise ValueError("Damping must be between 0 and 1")
        self.damping = damping


class Engine:
    """
    Training engine with utilities for common training patterns.
    """
    
    @staticmethod
    def mse_loss(predictions: List[Value], targets: List[Union[float, Value]]) -> Value:
        """
        Compute Mean Squared Error loss.
        
        Args:
            predictions: Model predictions
            targets: Target values
            
        Returns:
            MSE loss as a Value
        """
        if len(predictions) != len(targets):
            raise ValueError("Predictions and targets must have same length")
        
        total_loss = Value(0.0)
        n = len(predictions)
        
        for pred, target in zip(predictions, targets):
            if not isinstance(target, Value):
                target = Value(float(target))
            diff = pred - target
            total_loss = total_loss + diff * diff
        
        return total_loss * (1.0 / n)
    
    @staticmethod
    def accuracy(predictions: List[Value], targets: List[Union[float, int]]) -> float:
        """
        Compute classification accuracy.
        
        Args:
            predictions: Model predictions 
            targets: Target class indices
            
        Returns:
            Accuracy as a float between 0 and 1
        """
        if len(predictions) != len(targets):
            raise ValueError("Predictions and targets must have same length")
        
        correct = 0
        for pred_vals, target in zip(predictions, targets):
            if isinstance(pred_vals, Value):
                pred_vals = [pred_vals]
              # Find predicted class (argmax)
            pred_class = max(range(len(pred_vals)), key=lambda i: pred_vals[i].data)
            if pred_class == target:
                correct += 1
        
        return correct / len(predictions)
    
    @staticmethod
    def train_step(model: MLP, 
                   inputs: List[List[Union[float, Value]]], 
                   targets: List[Union[float, Value, List[Union[float, Value]]]],
                   loss_fn: Callable[[List[Value], List[Value]], Value],
                   optimizer: Union[SGD, Adam, LangevinLandauOptimizer]) -> float:
        """
        Perform one training step with enhanced memory management.
        
        Args:
            model: The MLP model
            inputs: Batch of input samples
            targets: Batch of target values
            loss_fn: Loss function
            optimizer: Optimizer
            
        Returns:
            Loss value as float
        """
        # Zero gradients
        optimizer.zero_grad()
        
        # Use persistent accumulator to avoid creating new Values in loop
        total_loss = Value.persistent(0.0) if hasattr(Value, 'persistent') else Value(0.0)
        batch_size = len(inputs)
        
        # Process samples one by one with immediate cleanup
        for i, (inp, target) in enumerate(zip(inputs, targets)):
            # Convert inputs to Values if needed (create minimal temporary Values)
            input_values = []
            for x in inp:
                if isinstance(x, Value):
                    input_values.append(x)
                else:
                    # Create temporary Value - will be cleaned up after backward pass
                    input_values.append(Value(float(x)))
            
            # Forward pass
            predictions = model(input_values)
            
            # Convert target to Value if needed
            if isinstance(target, list):
                target_values = []
                for t in target:
                    if isinstance(t, Value):
                        target_values.append(t)
                    else:
                        target_values.append(Value(float(t)))
            else:
                if isinstance(target, Value):
                    target_values = [target]
                else:
                    target_values = [Value(float(target))]
            
            # Compute loss for this sample
            sample_loss = loss_fn(predictions, target_values)
            
            # Add to total loss
            total_loss = total_loss + sample_loss
            
            # Immediate cleanup after every few samples to prevent accumulation
            if i % 10 == 9 or i == batch_size - 1:  # Every 10 samples or last sample
                # Force cleanup of intermediate computation graphs
                import gc
                if hasattr(model, 'cleanup_last_forward'):
                    model.cleanup_last_forward()
                gc.collect()
        
        # Average loss over batch
        batch_size_val = Value(float(batch_size))
        avg_loss = total_loss * Value(1.0 / batch_size)
        
        # Store loss value before backward pass
        loss_value = avg_loss.data
        
        # Backward pass with automatic cleanup using persistent flag system
        from ._core import get_value_lib
        value_lib = get_value_lib()
        value_lib.value_backward_and_free(avg_loss._ptr)
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()  # Zero gradients after step
        
        # Final aggressive cleanup
        if hasattr(model, 'cleanup_last_forward'):
            model.cleanup_last_forward()
        
        # Clean up persistent accumulator if we created one
        if hasattr(total_loss, 'is_persistent') and total_loss.is_persistent:
            # Reset its value for next use instead of deleting
            total_loss.data = 0.0
            total_loss.grad = 0.0
        
        import gc
        del total_loss, avg_loss, batch_size_val
        gc.collect()
        
        return loss_value
    
    @staticmethod
    def train_step_memory_efficient(model: MLP, 
                                   inputs: List[List[Union[float, Value]]], 
                                   targets: List[Union[float, Value, List[Union[float, Value]]]],
                                   loss_fn: Callable[[List[Value], List[Value]], Value],
                                   optimizer: Union[SGD, Adam, LangevinLandauOptimizer],
                                   mini_batch_size: int = 10) -> float:
        """
        Memory-efficient training step that processes data in mini-batches.
        
        This version processes large batches in smaller chunks to prevent
        memory accumulation from creating too many Value objects at once.
        
        Args:
            model: The MLP model
            inputs: Batch of input samples
            targets: Batch of target values
            loss_fn: Loss function
            optimizer: Optimizer
            mini_batch_size: Size of mini-batches for processing
            
        Returns:
            Loss value as float
        """
        # Zero gradients
        optimizer.zero_grad()
        
        total_loss = 0.0  # Use plain float to avoid Value accumulation
        batch_size = len(inputs)
        
        # Process data in mini-batches to prevent memory accumulation
        for start_idx in range(0, batch_size, mini_batch_size):
            end_idx = min(start_idx + mini_batch_size, batch_size)
            mini_batch_inputs = inputs[start_idx:end_idx]
            mini_batch_targets = targets[start_idx:end_idx]
            
            # Process mini-batch
            mini_loss = 0.0
            
            for inp, target in zip(mini_batch_inputs, mini_batch_targets):
                # Convert inputs to Values (minimal temp objects)
                input_values = []
                for x in inp:
                    if isinstance(x, Value):
                        input_values.append(x)
                    else:
                        input_values.append(Value(float(x)))
                
                # Forward pass
                predictions = model(input_values)
                
                # Convert target
                if isinstance(target, list):
                    target_values = [Value(float(t)) if not isinstance(t, Value) else t for t in target]
                else:
                    target_values = [Value(float(target)) if not isinstance(target, Value) else target]
                
                # Compute loss for this sample
                sample_loss = loss_fn(predictions, target_values)
                mini_loss += sample_loss.data
                
                # Immediate backward pass and cleanup for this sample
                sample_loss.backward_and_free()
                
                # Cleanup predictions and intermediate values
                del predictions, target_values, sample_loss
            
            total_loss += mini_loss
            
            # Force cleanup after each mini-batch
            import gc
            if hasattr(model, 'cleanup_last_forward'):
                model.cleanup_last_forward()
            gc.collect()
        
        # Calculate average loss
        avg_loss_value = total_loss / batch_size
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()
        
        # Final cleanup
        import gc
        gc.collect()
        
        return avg_loss_value

    @staticmethod
    def train(model: MLP,
              train_inputs: List[List[Union[float, Value]]],
              train_targets: List[Union[float, Value, List[Union[float, Value]]]],
              epochs: int,
              learning_rate: float = 0.01,
              batch_size: int = None,
              loss_fn: Callable[[List[Value], List[Value]], Value] = None,
              validation_data: Optional[tuple] = None,
              verbose: bool = True,
              optimizer_type: str = 'sgd',
              **optimizer_kwargs) -> dict:
        """
        Train the model.
        
        Args:
            model: The MLP model to train
            train_inputs: Training input samples
            train_targets: Training target values
            epochs: Number of training epochs
            learning_rate: Learning rate for optimization
            batch_size: Batch size (if None, use full batch)
            loss_fn: Loss function (default: MSE)
            validation_data: Optional (val_inputs, val_targets) tuple            verbose: Whether to print training progress
            
        Returns:
            Dictionary with training history
        """
        if loss_fn is None:
            loss_fn = Engine.mse_loss
        if batch_size is None:
            batch_size = len(train_inputs)
            
        # Extract our custom parameters before creating optimizer
        memory_efficient = optimizer_kwargs.pop('memory_efficient', False)
        mini_batch_size = optimizer_kwargs.pop('mini_batch_size', 10)
            
        # Extract our custom parameters before creating optimizer
        memory_efficient = optimizer_kwargs.pop('memory_efficient', False)
        mini_batch_size = optimizer_kwargs.pop('mini_batch_size', 10)
          # Create optimizer based on type
        if optimizer_type.lower() == 'adam':
            optimizer = Adam(model.parameters, lr=learning_rate, **optimizer_kwargs)
        elif optimizer_type.lower() == 'langevin' or optimizer_type.lower() == 'langevin-landau':
            # Extract Langevin-specific parameters
            damping = optimizer_kwargs.pop('damping', 0.1)
            temperature = optimizer_kwargs.pop('temperature', 0.01)
            seed = optimizer_kwargs.pop('seed', None)
            optimizer = LangevinLandauOptimizer(
                model.parameters, 
                learning_rate=learning_rate, 
                damping=damping, 
                temperature=temperature,
                seed=seed
            )
        else:
            optimizer = SGD(model.parameters, lr=learning_rate)
        
        # Initialize training history
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            # Zero gradients at start of epoch
            optimizer.zero_grad()
            
            # Shuffle training data
            combined = list(zip(train_inputs, train_targets))
            random.shuffle(combined)
            train_inputs_shuffled, train_targets_shuffled = zip(*combined)
              # Mini-batch training with automatic memory optimization
            epoch_losses = []
            for i in range(0, len(train_inputs_shuffled), batch_size):
                batch_inputs = train_inputs_shuffled[i:i+batch_size]
                batch_targets = train_targets_shuffled[i:i+batch_size]
                  # Use memory-efficient training for large batches (>50 samples)
                # or when explicitly requested
                use_memory_efficient = (
                    len(batch_inputs) > 50 or 
                    memory_efficient
                )
                
                if use_memory_efficient:
                    loss = Engine.train_step_memory_efficient(
                        model, batch_inputs, batch_targets, loss_fn, optimizer, mini_batch_size
                    )
                else:
                    loss = Engine.train_step(model, batch_inputs, batch_targets, loss_fn, optimizer)
                
                epoch_losses.append(loss)
                
                # Aggressive memory management for large datasets
                if i % batch_size == 0:  # Every batch
                    import gc
                    gc.collect()
                    
                    # Clean up MLP memory if possible
                    if hasattr(model, 'cleanup_last_forward'):
                        model.cleanup_last_forward()
                        
                # Extra cleanup every 3 batches for very large datasets
                if i % (batch_size * 3) == 0:
                    # Clean up any remaining computation graphs
                    try:
                        from ._core import get_value_lib
                        lib = get_value_lib()
                        if hasattr(lib, 'value_cleanup_constants'):
                            lib.value_cleanup_constants()
                            lib.value_init_constants()
                    except:
                        pass
                        
                    import gc
                    gc.collect()
                    gc.collect()  # Double collection to ensure cleanup
              # Zero gradients after epoch
            optimizer.zero_grad()
            
            # Aggressive memory cleanup after each epoch for long training sessions
            if epoch > 0 and epoch % 5 == 0:  # Every 5 epochs
                import gc
                # Force multiple garbage collection passes
                for _ in range(3):
                    gc.collect()
                
                # Try to cleanup any lingering C memory if available
                try:
                    from ._core import get_value_lib
                    lib = get_value_lib()
                    if hasattr(lib, 'value_cleanup_all_temporaries'):
                        lib.value_cleanup_all_temporaries()
                except:
                    pass
            
            avg_train_loss = sum(epoch_losses) / len(epoch_losses)
            history['train_loss'].append(avg_train_loss)
            
            # Validation
            val_loss = None
            if validation_data is not None:
                val_inputs, val_targets = validation_data
                val_loss = Engine.evaluate(model, val_inputs, val_targets, loss_fn)
                history['val_loss'].append(val_loss)
            
            if verbose:
                val_str = f", val_loss: {val_loss:.6f}" if val_loss is not None else ""
                print(f"Epoch {epoch+1}/{epochs}, train_loss: {avg_train_loss:.6f}{val_str}")
        
        return history
    
    @staticmethod
    def evaluate(model: MLP,
                 inputs: List[List[Union[float, Value]]],
                 targets: List[Union[float, Value, List[Union[float, Value]]]],
                 loss_fn: Callable[[List[Value], List[Value]], Value] = None) -> float:
        """
        Evaluate the model on given data.
        
        Args:
            model: The MLP model
            inputs: Input samples
            targets: Target values
            loss_fn: Loss function (default: MSE)
            
        Returns:
            Average loss as float
        """
        if loss_fn is None:
            loss_fn = Engine.mse_loss
        
        total_loss = Value(0.0)
        
        for inp, target in zip(inputs, targets):
            predictions = model(inp)
            
            # Ensure target is a list of Values
            if not isinstance(target, list):
                target = [target]
            target_values = [Value(float(t)) if not isinstance(t, Value) else t for t in target]
            
            batch_loss = loss_fn(predictions, target_values)
            total_loss = total_loss + batch_loss
        
        avg_loss = total_loss * (1.0 / len(inputs))
        return avg_loss.data
