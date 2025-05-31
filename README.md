# lamba

A C implementation of automatic differentiation with a Python interface, inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd). 

## Installation & Setup

**NOTE:** This project is developed for only Windows platforms. The Makefile automatically detects your operating system and uses the appropriate build commands.

### Requirements
- Python 3.7+
- C compiler 
- Make utility

### Quick Setup
```bash
git clone https://github.com/bthndmn12/lamba
cd lamba

# Build the C libraries
cd src
make all
make install
```

### Development Installation
```bash
git clone https://github.com/bthndmn12/lamba
cd lamba

cd src
make all
make install

cd ..
pip install -e .
```

## Building C Libraries

The project includes C implementations for performance-critical operations. Use the provided Makefile to build the shared libraries:

### Build Commands

**Build all libraries:**
```bash
cd src
make all
```

**Install libraries to correct location:**
```bash
make install
```

**Clean build artifacts:**
```bash
make clean
```

### Platform-Specific Details

The Makefile automatically detects your platform:

**On Windows:**
- Creates `.dll` files (value.dll, mlp.dll)
- Uses Windows style paths and copy commands
- Generates import libraries (.dll.a files)

**On Linux:**
- Creates `.so` files (value.so, mlp.so)  
- Uses Unix style paths and cp commands
- Includes `-fPIC` flag for position-independent code

**Example build output:**
```bash
# Windows
gcc -c -O2 -Wall -Werror -std=c99 -o value.o value.c
gcc -shared -o value.dll value.o -Wl,--out-implib,libvalue.dll.a
copy value.dll ..\micrograd_c\lib\
```

## As an Introduction

### Basic Value Operations

```python
from micrograd_c import Value

# Create values
a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)

# Perform operations
d = a * b + c  # 2 * -3 + 10 = 4
e = d.relu()   # max(0, 4) = 4

# Compute gradients
e.backward()

print(f"a.grad = {a.grad}")  # 0.0 (gradient of a)
print(f"b.grad = {b.grad}")  # 2.0 (gradient of b)
print(f"c.grad = {c.grad}")  # 1.0 (gradient of c)
```

### Building Basic Neural Nets

```python
from micrograd_c import MLP, Engine, SGD

# Create a simple neural network
model = MLP(nin=2, layer_specs=[4, 4, 1])  # 2 inputs -> 4 -> 4 -> 1 output

# Initialize parameters
model.initialize_parameters(method='xavier', seed=42)

# Create training data
inputs = [[1.0, 2.0], [2.0, 1.0], [0.5, 1.5]]
targets = [[1.0], [0.5], [0.8]]

# Create optimizer
optimizer = SGD(model.parameters, lr=0.01)

# Training loop
for epoch in range(100):
    loss = Engine.train_step(model, inputs, targets, Engine.mse_loss, optimizer)
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.6f}")
```

### Memory-Efficient Training for Large Datasets

```python
from micrograd_c import MLP, Engine, Adam

# Create model
model = MLP(8, [16, 8, 1])

# Large dataset (1000 samples)
large_inputs = [[float(i + j) for j in range(8)] for i in range(1000)]
large_targets = [float(i % 2) for i in range(1000)]

# Memory-efficient training
history = Engine.train(
    model=model,
    train_inputs=large_inputs,
    train_targets=large_targets,
    epochs=50,
    batch_size=100,          # Large batches automatically use memory-efficient mode
    learning_rate=0.001,
    optimizer_type='adam',
    memory_efficient=True,   # Explicit memory-efficient mode
    mini_batch_size=20,      # Process in chunks of 20
    verbose=True
)

print(f"Final loss: {history['train_loss'][-1]:.6f}")
```

## Core Components

### Value Class

As in Karpathy's micrograd, the names, classes and methods are mostly the left the same. The `Value` class is a scalar value with automatic differentiation capabilities.

#### Key Methods:
- `Value(data)`: Create a regular value
- `Value.persistent(data)`: Create a persistent value (for model parameters)
- `backward()`: Compute gradients via backpropagation
- `backward_and_free()`: Compute gradients and clean up temporary nodes
- Arithmetic operations: `+`, `-`, `*`, `/`, `**`
- Activation functions: `relu()`, `tanh()`, `exp()`, `log()`, `sigmoid()`

#### Persistent Flag System

Since the memory management part has evolved into an insurmountable point for me, I have resorted to LLMs a lot at this point. I will focus on this area if I continue with the project. The library implements a memory management system using persistent flags:

```python
# Regular values (temporary computation nodes)
temp = Value(1.0)                    # persistent=False (default)

# Persistent values (model parameters, constants)
weight = Value.persistent(0.5)       # persistent=True
bias = Value.persistent(0.0)         # persistent=True

# Check persistence
print(weight.is_persistent)          # True
print(temp.is_persistent)            # False
```

Persistent values survive automatic cleanup operations, making them perfect for model parameters, while temporary values are automatically freed after gradient computation.

### MLP (Multi Layer Perceptron)

The `MLP` class provides a complete neural network implementation.

#### Constructor:
```python
# create an MLP with custom layer specifications
model = MLP(nin=3, layer_specs=[16, 8, 1])

# or with Layer objects for fine control
from micrograd_c import Layer
layers = [
    Layer(nin=3, nout=16, activation='relu'),
    Layer(nin=16, nout=8, activation='relu'),
    Layer(nin=8, nout=1, activation='linear')
]
model = MLP(nin=3, layer_specs=layers)
```

#### Key Methods:
- `forward(inputs)`: Forward pass
- `parameters`: Get all trainable parameters
- `zero_grad()`: Zero all gradients
- `initialize_parameters(method, seed)`: Initialize weights and biases
- `parameter_count()`: Get total parameter count
- `get_gradient_norm()`: Compute gradient norm for monitoring
- `clip_gradients(max_norm)`: Gradient clipping
- `save_parameters(filepath)` / `load_parameters(filepath)`: Model persistence

#### Parameter Initialization Methods:
- `'xavier'`: Xavier/Glorot initialization 
- `'random'`: Random uniform initialization [-1, 1]
- `'zero'`: Zero initialization

### Engine Class

The `Engine` class provides training utilities and loss functions.

#### Loss Functions:
```python
# Mean Squared Error
loss = Engine.mse_loss(predictions, targets)

# Custom loss function
def custom_loss(pred, target):
    diff = pred[0] - target[0]
    return diff * diff

loss = custom_loss(predictions, targets)
```

#### Training Methods:

**Single Training Step:**
```python
loss = Engine.train_step(model, inputs, targets, Engine.mse_loss, optimizer)
```

**Memory-Efficient Training Step:**
```python
loss = Engine.train_step_memory_efficient(
    model, inputs, targets, Engine.mse_loss, optimizer, mini_batch_size=10
)
```

**Complete Training Loop:**
```python
history = Engine.train(
    model=model,
    train_inputs=train_inputs,
    train_targets=train_targets,
    epochs=100,
    learning_rate=0.01,
    batch_size=32,
    loss_fn=Engine.mse_loss,
    validation_data=(val_inputs, val_targets),
    optimizer_type='adam',  # 'sgd', 'adam', or 'langevin'
    memory_efficient=True,  # Enable memory optimization
    mini_batch_size=10,     # Size of processing chunks
    verbose=True,
    # Langevin-specific parameters (when optimizer_type='langevin')
    damping=0.1,           # Velocity damping coefficient
    temperature=0.01,      # Thermal noise strength
    seed=42               # Random seed for reproducibility
)
```

### Optimizers

#### SGD (Stochastic Gradient Descent)
```python
optimizer = SGD(model.parameters, lr=0.01)
```

#### Adam Optimizer
```python
optimizer = Adam(model.parameters, lr=0.001, b1=0.9, b2=0.999, eps=1e-8)
```


All optimizers support:
- `step()`: Apply gradients to parameters
- `zero_grad()`: Zero all parameter gradients


### Memory-Efficient Training Modes

The library automatically enables memory efficient training for:
- Batch sizes > 50 samples
- When explicitly requested with `memory_efficient=True`
- Long training sessions (automatic cleanup every 5 epochs)

### Gradient Management

```python
# Compute gradient norm for monitoring
grad_norm = model.get_gradient_norm()

# Clip gradients to prevent exploding gradients
model.clip_gradients(max_norm=1.0)

# Get parameter statistics
stats = model.get_parameter_stats()
print(f"Parameter mean: {stats['mean']:.6f}")
print(f"Parameter std: {stats['std']:.6f}")
```

### Model Persistence

```python
# Save model parameters
model.save_parameters('model_weights.json')

# Load model parameters
loaded_model = MLP.load_parameters('model_weights.json')
```

### Best Practices

1. **Use persistent values for model parameters:**

   ```python
   weight = Value.persistent(0.5)  # Survives cleanup
   ```

2. **Enable memory efficient training for large datasets:**

   ```python
   Engine.train(..., memory_efficient=True, mini_batch_size=20)
   ```

3. **Use `backward_and_free()` for immediate cleanup:**

   ```python
   loss.backward_and_free()  # Cleanup temporary nodes
   ```

4. **Monitor memory usage:**

   ```python
   import psutil
   memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
   ```

## Examples

### Complete Neural Network Tutorial

```python
from micrograd_c import Value, MLP, Engine, SGD, Adam
import random

# 1. Basic Value Operations
print("=== Basic Automatic Differentiation ===")
a = Value(2.0)
b = Value(-3.0)
c = a * b  # -6.0
d = c + Value(10.0)  # 4.0
e = d.relu()  # 4.0
e.backward()

print(f"a.grad = {a.grad}")  # -3.0
print(f"b.grad = {b.grad}")  # 2.0


# 2. Neural Network Construction
print("\n=== Neural Network ===")
model = MLP(nin=3, layer_specs=[4, 4, 1])
model.initialize_parameters(method='xavier', seed=42)

print(f"Model: {model}")
print(f"Parameters: {model.parameter_count()}")


# 3. Training Data
def generate_data(n_samples=100):
    """Generate synthetic dataset: y = sum(inputs) + noise"""
    data = []
    for _ in range(n_samples):
        inputs = [random.uniform(-1, 1) for _ in range(3)]
        target = sum(inputs) + random.uniform(-0.1, 0.1)
        data.append((inputs, [target]))
    return data

train_data = generate_data(200)
val_data = generate_data(50)

train_inputs = [x for x, y in train_data]
train_targets = [y for x, y in train_data]
val_inputs = [x for x, y in val_data]
val_targets = [y for x, y in val_data]


# 4. Training with Different Optimizers
print("\n=== Training with SGD ===")
sgd_model = MLP(nin=3, layer_specs=[8, 4, 1])
sgd_model.initialize_parameters(method='xavier', seed=42)

sgd_history = Engine.train(
    model=sgd_model,
    train_inputs=train_inputs,
    train_targets=train_targets,
    epochs=50,
    learning_rate=0.01,
    batch_size=32,
    validation_data=(val_inputs, val_targets),
    optimizer_type='sgd',
    verbose=True
)

print("\n=== Training with Adam ===")
adam_model = MLP(nin=3, layer_specs=[8, 4, 1])
adam_model.initialize_parameters(method='xavier', seed=42)

adam_history = Engine.train(
    model=adam_model,
    train_inputs=train_inputs,
    train_targets=train_targets,
    epochs=50,
    learning_rate=0.001,
    batch_size=32,
    validation_data=(val_inputs, val_targets),
    optimizer_type='adam',
    verbose=True
)

print("\n=== Training with Langevin-Landau ===")
langevin_model = MLP(nin=3, layer_specs=[8, 4, 1])
langevin_model.initialize_parameters(method='xavier', seed=42)

langevin_history = Engine.train(
    model=langevin_model,
    train_inputs=train_inputs,
    train_targets=train_targets,
    epochs=50,
    learning_rate=0.01,
    batch_size=32,
    validation_data=(val_inputs, val_targets),
    optimizer_type='langevin',
    damping=0.1,           # Velocity damping
    temperature=0.01,      # Thermal noise
    seed=42,
    verbose=True
)

# 5. Model Evaluation
print("\n=== Model Evaluation ===")
sgd_final_loss = Engine.evaluate(sgd_model, val_inputs, val_targets)
adam_final_loss = Engine.evaluate(adam_model, val_inputs, val_targets)
langevin_final_loss = Engine.evaluate(langevin_model, val_inputs, val_targets)

print(f"SGD final validation loss: {sgd_final_loss:.6f}")
print(f"Adam final validation loss: {adam_final_loss:.6f}")
print(f"Langevin final validation loss: {langevin_final_loss:.6f}")

# 6. Memory Efficient Training for Large Dataset
print("\n=== Memory Efficient Training ===")
large_data = generate_data(1000)
large_inputs = [x for x, y in large_data]
large_targets = [y for x, y in large_data]

memory_model = MLP(nin=3, layer_specs=[16, 8, 1])
memory_model.initialize_parameters(method='xavier', seed=42)

memory_history = Engine.train(
    model=memory_model,
    train_inputs=large_inputs,
    train_targets=large_targets,
    epochs=20,
    batch_size=100,              # Large batch triggers memory efficient mode
    learning_rate=0.001,
    optimizer_type='adam',
    memory_efficient=True,
    mini_batch_size=20,
    verbose=True
)

print(f"Memory efficient final loss: {memory_history['train_loss'][-1]:.6f}")

# 7. Model Persistence
print("\n=== Model Persistence ===")
model.save_parameters('trained_model.json')
loaded_model = MLP.load_parameters('trained_model.json')
print("Model saved and loaded successfully!")
```


## C Implementation Details

### Core Architecture

The library consists of two main components:

1. **C Backend** (`src/`):
   - `value.c`: Core automatic differentiation engine
   - `mlp.c`: Neural network implementation
   - Compiled to shared libraries (`micrograd_c/lib/`)

2. **Python Interface** (`micrograd_c/`):
   - `value.py`: Python wrapper for Value operations
   - `mlp.py`: Python wrapper for MLP operations
   - `engine.py`: Training utilities
   - `_core.py`: C library interface management

### Memory Management Architecture

The C implementation uses a sophisticated memory management system:

```c
typedef struct Value {
    float data;                             // Forward value
    float grad;                             // Gradient
    struct Value *left, *right;             // Children for backprop
    void (*backward)(struct Value *self);   // Backward function
    char op;                                // Operation type
    float _exponent;                        // For power operations
    bool persistent;                        // Persistence flag
} Value;
```

Key functions:
- `value_new(x)`: Create temporary Value
- `value_new_persistent(x)`: Create persistent Value
- `value_backward_and_free(root)`: Backprop + cleanup
- `value_free_graph_safe(root, preserve, count)`: Safe cleanup

## 🐛 Troubleshooting

### Common Issues

1. **Memory Issues on Large Datasets:**
   ```python
   # Enable memory-efficient training
   Engine.train(..., memory_efficient=True, mini_batch_size=20)
   ```

2. **Exploding Gradients:**
   ```python
   # Clip gradients
   model.clip_gradients(max_norm=1.0)
   
   # Monitor gradient norm
   grad_norm = model.get_gradient_norm()
   if grad_norm > 10.0:
       print(f"Warning: Large gradient norm {grad_norm}")
   ```

3. **Training Instability:**
   ```python
   # Use Xavier initialization
   model.initialize_parameters(method='xavier', seed=42)
   
   # Lower learning rate
   optimizer = Adam(model.parameters, lr=0.0001)
   ```


### Performance Tips

1. **Use appropriate batch sizes:**
   - Small datasets: 8-32 samples
   - Medium datasets: 32-128 samples  
   - Large datasets: 128+ samples with memory optimization

2. **Choose the right optimizer:**
   - SGD: Simple problems, when you want full control
   - Adam: Most cases, adaptive learning rate beneficial

3. **Monitor training:**
   ```python
   # Check gradient norms
   if model.get_gradient_norm() < 1e-6:
       print("Warning: Vanishing gradients")
   
   # Monitor memory usage
   import psutil
   memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
   print(f"Memory usage: {memory_mb:.2f} MB")
   ```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b [feature]`
3. Make your changes and add tests
4. Submit a pull request
