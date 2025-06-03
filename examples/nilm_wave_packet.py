"""Example training script for NILM using WavePacketMLP."""
import random
import json
import math
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from micrograd_c import Value, Engine, Adam, WavePacketMLP

# Synthetic dataset generator (placeholder for real NILM data)
def generate_data(n_samples=200, seed=0):
    random.seed(seed)
    data = []
    for _ in range(n_samples):
        x = random.uniform(-1.0, 1.0)
        # simple pattern: combination of low freq and high freq signals
        y = 0.6 * (x**2) + 0.4 * math.sin(3 * x)
        data.append(([x], [y]))
    return data


def train_model():
    train = generate_data(400, seed=1)
    val = generate_data(80, seed=2)
    train_inputs = [x for x, y in train]
    train_targets = [y for x, y in train]
    val_inputs = [x for x, y in val]
    val_targets = [y for x, y in val]

    model = WavePacketMLP(wave_packets=3, mlp_layers=[6, 1], seed=42)
    optimizer = Adam(model.parameters, lr=0.001)

    for epoch in range(60):
        preds = [model.forward([Value(v[0])]) for v in train_inputs]
        loss = Engine.mse_loss([p[0] for p in preds], [Value(t[0]) for t in train_targets])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if epoch % 10 == 0:
            val_preds = [model.forward([Value(v[0])]) for v in val_inputs]
            val_loss = Engine.mse_loss([p[0] for p in val_preds], [Value(t[0]) for t in val_targets])
            print(f"Epoch {epoch}: train loss {loss.data:.6f}, val loss {val_loss.data:.6f}")

    model_path = "nilm_wavepacket_model.json"
    # save parameters
    params = [p.data for p in model.parameters]
    with open(model_path, "w") as f:
        json.dump({"params": params}, f)
    print("Model saved to", model_path)
    return model_path

if __name__ == "__main__":
    train_model()
