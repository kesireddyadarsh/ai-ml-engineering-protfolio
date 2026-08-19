from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 1000
    dim: int = 256
    n_layers: int = 4
    n_heads: int = 4
    hidden_dim: int = 768
    max_seq_len: int = 128