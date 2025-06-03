import subprocess
import os
import shutil
import sys

# Determine repo root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT, 'src')
LIB_DIR = os.path.join(ROOT, 'micrograd_c', 'lib')

# Build C libraries
subprocess.run(['make', 'all'], cwd=SRC_DIR, check=True)

# Copy compiled libraries
for lib in ['value.dll', 'mlp.dll']:
    shutil.copy(os.path.join(SRC_DIR, lib), os.path.join(LIB_DIR, lib))

sys.path.insert(0, ROOT)
from micrograd_c import Value, MLP, Engine

def huber_loss(predictions, targets):
    """Simple Huber loss to verify operations"""
    total = Value(0.0)
    delta = 1.0
    for pred, target in zip(predictions, targets):
        if not isinstance(target, Value):
            target = Value(float(target))
        error = pred - target
        abs_error = (error * error).pow_safe(0.5)
        if abs_error.data <= delta:
            loss = error * error * 0.5
        else:
            loss = Value(delta) * (abs_error - Value(delta * 0.5))
        total = total + loss
    return total * (1.0 / len(predictions))

model = MLP(1, [2, 1])
inputs = [[1.0], [2.0]]
targets = [1.0, 2.0]

history = Engine.train(model, inputs, targets, epochs=1, learning_rate=0.01,
                       batch_size=2, loss_fn=huber_loss, verbose=False)
print('Test completed, final loss:', history['train_loss'][-1])
