"""Wave packet based feature extraction layers."""

from typing import List
import random
from .value import Value
from .mlp import MLP

class WavePacketLayer:
    """Gaussian wave packet feature extractor."""
    def __init__(self, num_packets: int, input_dim: int = 1, seed: int = None):
        if seed is not None:
            random.seed(seed)
        self.num_packets = num_packets
        self.input_dim = input_dim
        self.A = [Value.persistent(random.uniform(-1.0, 1.0)) for _ in range(num_packets)]
        self.raw_sigma = [Value.persistent(random.uniform(-1.0, 1.0)) for _ in range(num_packets)]
        self.k = [Value.persistent(random.uniform(-2.0, 2.0)) for _ in range(num_packets)]
        self.omega = [Value.persistent(random.uniform(-2.0, 2.0)) for _ in range(num_packets)]
        self.x_p = [Value.persistent(random.uniform(-2.0, 2.0)) for _ in range(num_packets)]
        self.parameters = self.A + self.raw_sigma + self.k + self.omega + self.x_p

    def forward(self, inputs: List[Value]) -> List[Value]:
        if len(inputs) != self.input_dim:
            raise ValueError(f"expected {self.input_dim} inputs")
        x = inputs[0]
        outputs = []
        for i in range(self.num_packets):
            A = self.A[i]
            sigma = self.raw_sigma[i].exp()
            k = self.k[i]
            omega = self.omega[i]
            x_p = self.x_p[i]
            diff = x - x_p
            exponent = (diff * diff) / (sigma * sigma * Value(2.0))
            envelope = (Value(0.0) - exponent).exp()
            phase = k * x - omega * x_p
            cos_v = self._cos_approx(phase)
            sin_v = self._sin_approx(phase)
            base = A * envelope
            real = base * cos_v
            imag = base * sin_v
            outputs.extend([real, imag])
        return outputs

    def _cos_approx(self, x: Value, terms: int = 4) -> Value:
        result = Value(1.0)
        x2 = x * x
        power = Value(1.0)
        factorial = 1.0
        for n in range(1, terms + 1):
            power = power * x2
            factorial *= (2*n-1)*(2*n)
            term = power / Value(factorial)
            result = result - term if n % 2 == 1 else result + term
        return result

    def _sin_approx(self, x: Value, terms: int = 4) -> Value:
        result = x
        x2 = x * x
        power = x
        factorial = 1.0
        for n in range(1, terms + 1):
            power = power * x2
            factorial *= (2*n)*(2*n+1)
            term = power / Value(factorial)
            result = result - term if n % 2 == 1 else result + term
        return result

    def zero_grad(self):
        for p in self.parameters:
            p.grad = 0.0

class WavePacketMLP:
    """MLP preceded by a wave packet layer."""
    def __init__(self, wave_packets: int, mlp_layers: List[int], seed: int = None):
        self.wave = WavePacketLayer(wave_packets, input_dim=1, seed=seed)
        mlp_in = 2 * wave_packets
        self.mlp = MLP(nin=mlp_in, layer_specs=mlp_layers)
        if seed is not None:
            self.mlp.initialize_parameters(method='xavier', seed=seed)
        self.parameters = self.wave.parameters + self.mlp.parameters

    def forward(self, inputs: List[Value]) -> List[Value]:
        feats = self.wave.forward(inputs)
        return self.mlp.forward(feats)

    def zero_grad(self):
        self.wave.zero_grad()
        self.mlp.zero_grad()

    def parameter_count(self) -> int:
        return len(self.parameters)


class TwoInputWavePacketMLP:
    """Wave packet network for two scalar inputs (ampere and watt)."""

    def __init__(self, packets_per_input: int, mlp_layers: List[int], seed: int | None = None):
        self.wave_amp = WavePacketLayer(packets_per_input, input_dim=1, seed=seed)
        # use different seed for second layer to avoid identical init when provided
        self.wave_watt = WavePacketLayer(packets_per_input, input_dim=1, seed=None if seed is None else seed + 1)

        mlp_in = 4 * packets_per_input  # two layers each output 2*num_packets features
        self.mlp = MLP(nin=mlp_in, layer_specs=mlp_layers)
        if seed is not None:
            self.mlp.initialize_parameters(method='xavier', seed=seed)

        self.parameters = (
            self.wave_amp.parameters
            + self.wave_watt.parameters
            + self.mlp.parameters
        )

    def forward(self, inputs: List[Value]) -> List[Value]:
        if len(inputs) != 2:
            raise ValueError("Two inputs required: ampere and watt")

        amp_feats = self.wave_amp.forward([inputs[0]])
        watt_feats = self.wave_watt.forward([inputs[1]])
        return self.mlp.forward(amp_feats + watt_feats)

    def zero_grad(self) -> None:
        self.wave_amp.zero_grad()
        self.wave_watt.zero_grad()
        self.mlp.zero_grad()

    def parameter_count(self) -> int:
        return len(self.parameters)
