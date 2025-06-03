"""Run inference using a saved NILM WavePacketMLP model."""
import json
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from micrograd_c import Value, WavePacketMLP

MODEL_PATH = "nilm_wavepacket_model.json"

# Model architecture must match training
WAVE_PACKETS = 3
MLP_LAYERS = [6, 1]


def load_model(path: str) -> WavePacketMLP:
    model = WavePacketMLP(WAVE_PACKETS, MLP_LAYERS)
    with open(path, "r") as f:
        data = json.load(f)
    for p, val in zip(model.parameters, data["params"]):
        p.data = val
    return model


def predict(model: WavePacketMLP, x: float) -> float:
    return model.forward([Value(x)])[0].data


if __name__ == "__main__":
    model = load_model(MODEL_PATH)
    sample = 0.25
    output = predict(model, sample)
    print(f"Input {sample:.3f} -> prediction {output:.4f}")
