"""Example training script for NILM using TwoInputWavePacketMLP.

This script generates synthetic 120-minute windows of current (ampere)
and power (watt). Each window is summarised by its mean value and fed
through two separate wave packet feature extractors. The small MLP
predicts two outputs for demonstration purposes.
"""
import random
import json
import math
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from micrograd_c import Value, Engine, Adam, TwoInputWavePacketMLP

# Synthetic dataset generator (placeholder for real NILM data)
WINDOW = 120


def generate_data(n_samples: int = 200, seed: int = 0):
    """Create synthetic NILM windows with two outputs."""
    random.seed(seed)
    data = []
    for _ in range(n_samples):
        amps = [random.uniform(0.0, 1.0) for _ in range(WINDOW)]
        watts = [a * 0.5 + random.uniform(-0.1, 0.1) for a in amps]
        amp_avg = sum(amps) / WINDOW
        watt_avg = sum(watts) / WINDOW

        # simple relation for targets
        out1 = 0.3 * amp_avg + 0.7 * watt_avg
        out2 = 0.5 * amp_avg - 0.2 * watt_avg
        data.append(([amp_avg, watt_avg], [out1, out2]))
    return data


def train_model():
    train = generate_data(400, seed=1)
    val = generate_data(80, seed=2)
    train_inputs = [x for x, y in train]
    train_targets = [y for x, y in train]
    val_inputs = [x for x, y in val]
    val_targets = [y for x, y in val]

    model = TwoInputWavePacketMLP(packets_per_input=2, mlp_layers=[4, 2], seed=42)
    optimizer = Adam(model.parameters, lr=0.001)

    for epoch in range(60):
        preds = [
            model.forward([Value(v[0]), Value(v[1])]) for v in train_inputs
        ]
        loss = Engine.mse_loss(
            [p[i] for p in preds for i in range(2)],
            [Value(t[i]) for t in train_targets for i in range(2)],
        )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if epoch % 10 == 0:
            val_preds = [
                model.forward([Value(v[0]), Value(v[1])]) for v in val_inputs
            ]
            val_loss = Engine.mse_loss(
                [p[i] for p in val_preds for i in range(2)],
                [Value(t[i]) for t in val_targets for i in range(2)],
            )
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
