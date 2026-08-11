from dataclasses import dataclass, asdict

@dataclass(frozen=True)
class TransformerConfig:
    d_model: int = 64
    n_heads: int = 4
    d_head: int = 16      # d_model // n_heads
    n_layers: int = 2
    d_ff: int = 128
    seq_len: int = 32
    vocab: int = 512
    eps: float = 1e-5

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0
        assert self.d_head == self.d_model // self.n_heads

    def to_dict(self):
        return asdict(self)

DEFAULT = TransformerConfig()

# Tiny config used for the sub-millisecond race
TINY = TransformerConfig(d_model=32, n_heads=2, d_head=16, n_layers=1, d_ff=64, seq_len=16, vocab=256)
