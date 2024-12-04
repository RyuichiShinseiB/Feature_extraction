from dataclasses import dataclass

from .autoencoder_configs import RecursiveDataclass


@dataclass
class NetworkConfig(RecursiveDataclass):
    pass
