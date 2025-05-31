"""
Complete Tutorial: micrograd-c Neural Network Library
=====================================================

This tutorial introduces all features of the micrograd-c library:
1. Basic Value operations and automatic differentiation
2. Building neural networks with MLP
3. Training with different optimizers (SGD, Adam)
4. Memory-efficient training for large datasets
5. Model saving and loading
6. Real-world example with dataset

Author: batuhandumani
date: 2025-05-31
License: MIT
"""

import random
import json
from micrograd_c import Value, MLP, Engine, SGD, Adam

def section_header(title):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def subsection_header(title):
    """Print a formatted subsection header"""
    print(f"\n--- {title} ---")


# SECTION 1: Introduction to Value and Automatic Differentiation

def demo_basic_values():
    """Demonstrate basic Value operations and gradients"""
    section_header("1. Basic Value Operations & Automatic Differentiation")

    print("Creating Values:")
    a = Value(2.0)
    b = Value(3.0)
    print(f"a = {a.data}")
    print(f"b = {b.data}")
    

    print("\nBasic Operations:")
    c = a + b       # Addition
    d = a * b       # Multiplication
    e = a ** 2      # Power
    f = d.tanh()    # Activation function
    
    print(f"a + b = {c.data}")
    print(f"a * b = {d.data}")
    print(f"a^2 = {e.data}")
    print(f"tanh(a * b) = {f.data:.4f}")

    print("\nAutomatic Differentiation:")
    y = (a + b) * (a * b).tanh()
    print(f"y = (a + b) * tanh(a * b) = {y.data:.4f}")
    
    # Compute gradients
    y.backward()
    print(f"dy/da = {a.grad:.4f}")
    print(f"dy/db = {b.grad:.4f}")
    
    # Demonstrate persistent values
    subsection_header("Persistent Values (Memory Optimization)")
    param = Value.persistent(1.5)  # Won't be freed during training
    temp = Value(2.0)              # Regular value, will be cleaned up
    
    print(f"Persistent parameter: {param.data}, is_persistent: {param.is_persistent}")
    print(f"Temporary value: {temp.data}, is_persistent: {temp.is_persistent}")

# SECTION 2: Building Neural Networks

def demo_neural_networks():
    """Demonstrate building and using neural networks"""
    section_header("2. Building Neural Networks with MLP")
    
    # Create a simple neural network
    # Input: 3 features, Hidden: [4, 4], Output: 1
    print("Creating neural network:")
    model = MLP(3, [4, 4, 1])
    print(f"Model parameters: {len(model.parameters)}")

    print("\nModel structure:")
    for i, layer in enumerate(model.layers):
        print(f"Layer {i}: {layer.nin} → {layer.nout} neurons ({layer.activation})")
    
    # Forward pass
    print("\nForward pass example:")
    x = [0.5, -0.2, 0.8]  # Example input
    prediction = model(x)
    print(f"Input: {x}")
    print(f"Output: {[p.data for p in prediction]}")

    print(f"\nFirst few parameters:")
    for i, param in enumerate(model.parameters[:5]):
        print(f"param[{i}]: {param.data:.4f}")

# SECTION 3: Training with Different Optimizers

def demo_training_optimizers():
    """Demonstrate training with SGD and Adam optimizers"""
    section_header("3. Training with Different Optimizers")
    
    print("Generating synthetic dataset...")
    def generate_data(n_samples=100):
        data = []
        for _ in range(n_samples):
            x1, x2, x3 = random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)
            y = x1 + 2*x2 - x3 + random.uniform(-0.1, 0.1)  # Small noise
            data.append(([x1, x2, x3], y))
        return data
    
    train_data = generate_data(100)
    train_inputs = [x for x, y in train_data]
    train_targets = [y for x, y in train_data]
    
    print(f"Training samples: {len(train_data)}")
    print(f"Sample input: {train_inputs[0]}")
    print(f"Sample target: {train_targets[0]:.3f}")
    
    # SGD Training
    subsection_header("Training with SGD")
    model_sgd = MLP(3, [8, 1])
    
    history_sgd = Engine.train(
        model=model_sgd,
        train_inputs=train_inputs,
        train_targets=train_targets,
        epochs=20,
        learning_rate=0.01,
        batch_size=32,
        optimizer_type='sgd',
        verbose=True
    )
    
    print(f"SGD Final loss: {history_sgd['train_loss'][-1]:.6f}")
    
    # Adam Training
    subsection_header("Training with Adam")
    model_adam = MLP(3, [8, 1])
    
    history_adam = Engine.train(
        model=model_adam,
        train_inputs=train_inputs,
        train_targets=train_targets,
        epochs=20,
        learning_rate=0.001,  # Lower learning rate for Adam
        batch_size=32,
        optimizer_type='adam',
        verbose=True
    )
    
    print(f"Adam Final loss: {history_adam['train_loss'][-1]:.6f}")
    
    # Compare final performance
    test_input = [0.5, -0.3, 0.2]
    expected = 0.5 + 2*(-0.3) - 0.2  # True function
    
    pred_sgd = model_sgd(test_input)[0].data
    pred_adam = model_adam(test_input)[0].data
    
    print(f"Comparison on test input {test_input}:")
    print(f"Expected: {expected:.3f}")
    print(f"SGD prediction: {pred_sgd:.3f} (error: {abs(pred_sgd - expected):.3f})")
    print(f"Adam prediction: {pred_adam:.3f} (error: {abs(pred_adam - expected):.3f})")

# SECTION 4: Memory-Efficient Training

def demo_memory_efficient_training():
    """Demonstrate memory-efficient training for large datasets"""

    section_header("4. Memory-Efficient Training (New Feature!)")
    
    print("Memory-efficient training prevents crashes on large datasets")
    print("and reduces memory usage by processing data in mini-batches.\n")
    
    # Create a larger dataset to demonstrate memory efficiency
    def generate_large_data(n_samples=1000):
        print(f"Generating {n_samples} samples...")
        data = []
        for i in range(n_samples):
            # More complex function: y = sin(x1) + cos(x2) + x3^2
            x1, x2, x3 = random.uniform(-2, 2), random.uniform(-2, 2), random.uniform(-1, 1)
            y = 0.5 * (x1**2 + x2**2) + 0.3 * x3 + random.uniform(-0.1, 0.1)
            data.append(([x1, x2, x3], y))
        return data
    
    large_data = generate_large_data(1000)
    large_inputs = [x for x, y in large_data]
    large_targets = [y for x, y in large_data]
    
    # Regular training (will automatically use memory-efficient mode for large batches)
    subsection_header("Automatic Memory-Efficient Training")
    model = MLP(3, [16, 8, 1])
    
    print("Training with large batch (automatically uses memory-efficient mode)...")
    history = Engine.train(
        model=model,
        train_inputs=large_inputs,
        train_targets=large_targets,
        epochs=10,
        batch_size=200,  # Large batch triggers memory-efficient mode
        learning_rate=0.001,
        optimizer_type='adam',
        verbose=True
    )
    
    # Explicit memory-efficient training
    subsection_header("Explicit Memory-Efficient Training")
    model2 = MLP(3, [16, 8, 1])
    
    print("Training with explicit memory-efficient settings...")
    history2 = Engine.train(
        model=model2,
        train_inputs=large_inputs,
        train_targets=large_targets,
        epochs=10,
        batch_size=100,
        learning_rate=0.001,
        optimizer_type='adam',
        memory_efficient=True,    # Force memory efficient mode
        mini_batch_size=20,      
        verbose=True
    )
    
    print(f"Memory-efficient final loss: {history2['train_loss'][-1]:.6f}")


# SECTION 5: Model Persistence (Saving/Loading)

def demo_model_persistence():
    """Demonstrate saving and loading models"""
    section_header("5. Model Persistence (Saving & Loading)")
    
    print("Training a model to save...")
    model = MLP(2, [4, 1])
    
    # Simple dataset: XOR problem
    xor_data = [
        ([0, 0], 0),
        ([0, 1], 1),
        ([1, 0], 1),
        ([1, 1], 0)
    ]
    
    expanded_data = []
    for _ in range(200):
        x, y = random.choice(xor_data)
        # Add small noise
        x_noisy = [x[0] + random.uniform(-0.1, 0.1), x[1] + random.uniform(-0.1, 0.1)]
        expanded_data.append((x_noisy, y))
    
    inputs = [x for x, y in expanded_data]
    targets = [y for x, y in expanded_data]
    
    Engine.train(model, inputs, targets, epochs=50, learning_rate=0.1, verbose=False)
    
    # Test model before saving
    test_cases = [[0, 0], [0, 1], [1, 0], [1, 1]]
    print("Model predictions before saving:")
    for test_case in test_cases:
        pred = model(test_case)[0].data
        print(f"Input {test_case} → Output: {pred:.3f}")
    
    # Save model parameters
    print("\nSaving model parameters...")
    model_data = {
        'architecture': [2, 4, 1],
        'parameters': [p.data for p in model.parameters]
    }
    
    with open('./model_params.json', 'w') as f:
        json.dump(model_data, f, indent=2)
    print("Model saved to 'examples/model_params.json'")
    
    # Load model parameters
    print("\nLoading model parameters...")
    with open('./model_params.json', 'r') as f:
        loaded_data = json.load(f)
    
    # Create new model with same architecture
    new_model = MLP(loaded_data['architecture'][0], loaded_data['architecture'][1:])
    
    # Set parameters
    for param, value in zip(new_model.parameters, loaded_data['parameters']):
        param.data = value
    
    print("Model loaded successfully!")
    
    # Test loaded model
    print("Loaded model predictions:")
    for test_case in test_cases:
        pred = new_model(test_case)[0].data
        print(f"Input {test_case} → Output: {pred:.3f}")


# SECTION 7: Advanced Features

def demo_advanced_features():
    """Demonstrate advanced features and best practices"""
    section_header("7. Advanced Features & Best Practices")
    
    subsection_header("Custom Loss Functions")
    
    # Custom loss function example
    def custom_loss(predictions, targets):
        """Huber loss - robust to outliers"""
        total_loss = Value(0.0)
        delta = 1.0
        
        for pred, target in zip(predictions, targets):
            if not isinstance(target, Value):
                target = Value(float(target))
            
            error = pred - target
            abs_error = error.abs() if hasattr(error, 'abs') else (error * error).sqrt()
            
            # Huber loss: quadratic for small errors, linear for large errors
            if abs_error.data <= delta:
                loss = error * error * 0.5
            else:
                loss = Value(delta) * (abs_error - Value(delta * 0.5))
            
            total_loss = total_loss + loss
        
        return total_loss * (1.0 / len(predictions))
    
    print("Example: Custom Huber loss function")
    print("Huber loss is more robust to outliers than MSE")
    
    # Example with validation data
    subsection_header("Training with Validation")
    
    # Generate data with some outliers
    data = []
    for i in range(200):
        x = random.uniform(-2, 2)
        y = x * x + random.uniform(-0.2, 0.2)
        # Add some outliers
        if random.random() < 0.1:  # 10% outliers
            y += random.uniform(-2, 2)
        data.append(([x], y))
    
    # Split data
    train_split = int(0.7 * len(data))
    val_split = int(0.85 * len(data))
    
    train_data = data[:train_split]
    val_data = data[train_split:val_split]
    test_data = data[val_split:]
    
    train_inputs = [x for x, y in train_data]
    train_targets = [y for x, y in train_data]
    val_inputs = [x for x, y in val_data]
    val_targets = [y for x, y in val_data]
    
    model = MLP(1, [8, 8, 1])
    
    print("Training with validation data...")
    history = Engine.train(
        model=model,
        train_inputs=train_inputs,
        train_targets=train_targets,
        epochs=20,
        learning_rate=0.01,
        batch_size=32,
        validation_data=(val_inputs, val_targets),
        optimizer_type='adam',
        verbose=True
    )
    
    print(f"Final training loss: {history['train_loss'][-1]:.6f}")
    print(f"Final validation loss: {history['val_loss'][-1]:.6f}")

# MAIN TUTORIAL EXECUTION
def main():
    """Run the complete micrograd-c tutorial"""
    print("Welcome to the micrograd-c Neural Network Library Tutorial!")
    print("This tutorial will guide you through all features of the library.")
    print("Each section builds on the previous ones, so follow along!")
    
    try:
        demo_basic_values()
        demo_neural_networks()
        demo_training_optimizers()
        demo_memory_efficient_training()
        demo_model_persistence()
        demo_advanced_features()

    except Exception as e:
        print(f"Tutorial failed with error: {e}")
        print("Please check your micrograd-c installation.")
        raise

if __name__ == "__main__":
    main()
