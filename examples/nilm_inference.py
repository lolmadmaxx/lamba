"""Run inference using a saved two-input wave packet NILM model."""
import json
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from micrograd_c import Value, TwoInputWavePacketMLP

MODEL_PATH = "nilm_wavepacket_model.json"

# Model architecture must match training
PACKETS_PER_INPUT = 2
MLP_LAYERS = [4, 2]


def load_model(path: str) -> TwoInputWavePacketMLP:
    model = TwoInputWavePacketMLP(PACKETS_PER_INPUT, MLP_LAYERS)
    with open(path, "r") as f:
        data = json.load(f)
    for p, val in zip(model.parameters, data["params"]):
        p.data = val
    return model


def predict(model: TwoInputWavePacketMLP, amp: float, watt: float):
    out = model.forward([Value(amp), Value(watt)])
    return [out[0].data, out[1].data]


if __name__ == "__main__":
    model = load_model(MODEL_PATH)
    amp = 0.3
    watt = 0.15
    output = predict(model, amp, watt)
    print(f"Amp {amp:.2f}, Watt {watt:.2f} -> prediction {output}")
