from dataclasses import dataclass, field

@dataclass
class Network_Parameters:
    """
    Network and energy consumption parameters class, used to define the characteristics of device computation and communication.
    """
    UE_flops: float = 12.0  # User equipment computing power (GFlops)
    num_node: int = 15  # Number of network nodes
    tau: int = 4  # ES computing power is a multiple of UE
    ES_flops: float = 12.0 * 4  # Edge server computing power (GFlops)
    network_type: str = '4G+5G'  # Network type
    dispersion: float = 0.1  # Random dispersion of network rate
    seed: int = 42  # Random seed
    inference_model_index: tuple = ('VGG16', 'Cifar100')  # Inference model and dataset
    UE_comp_power: float = 6.0  # User equipment computing power (W)
    UE_transmission_power: float = 20.0  # User equipment transmission power (mW)
    ES_transmission_power: float = 100.0  # Access point transmission power (mW)
    transmission_efficiency: float = 0.9  # Data transmission efficiency (0~1)
    computation_energy_per_gflop: float = 0.1  # Energy required per GFlops computation (J)

    comp_efficiency: float = field(init=False)  # Computing efficiency \mu = GFlops / W
    ES_comp_power: float = field(init=False)  # Edge server computing power (W)

    def __post_init__(self):
        self.comp_efficiency = self.UE_flops / self.UE_comp_power
        self.ES_comp_power = self.UE_flops * self.tau / self.comp_efficiency
