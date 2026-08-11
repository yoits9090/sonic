import numpy as np

class RNG:
    """Deterministic RNG so all impls see identical weights."""
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
    def weights(self, cfg, dtype=np.float32):
        W = {}
        g = self.rng
        scale = 1.0 / (cfg.d_model ** 0.5)
        W["emb"] = g.normal(0, scale, (cfg.vocab, cfg.d_model)).astype(dtype)
        for i in range(cfg.n_layers):
            W[f"wq{i}"] = g.normal(0, scale, (cfg.d_model, cfg.d_model)).astype(dtype)
            W[f"wk{i}"] = g.normal(0, scale, (cfg.d_model, cfg.d_model)).astype(dtype)
            W[f"wv{i}"] = g.normal(0, scale, (cfg.d_model, cfg.d_model)).astype(dtype)
            W[f"wo{i}"] = g.normal(0, scale, (cfg.d_model, cfg.d_model)).astype(dtype)
            W[f"w1{i}"] = g.normal(0, scale, (cfg.d_model, cfg.d_ff)).astype(dtype)
            W[f"w2{i}"] = g.normal(0, scale, (cfg.d_ff, cfg.d_model)).astype(dtype)
            W[f"ln1_g{i}"] = np.ones(cfg.d_model, dtype=dtype)
            W[f"ln1_b{i}"] = np.zeros(cfg.d_model, dtype=dtype)
            W[f"ln2_g{i}"] = np.ones(cfg.d_model, dtype=dtype)
            W[f"ln2_b{i}"] = np.zeros(cfg.d_model, dtype=dtype)
        W["lnf_g"] = np.ones(cfg.d_model, dtype=dtype)
        W["lnf_b"] = np.zeros(cfg.d_model, dtype=dtype)
        W["out"] = g.normal(0, scale, (cfg.d_model, cfg.vocab)).astype(dtype)
        return W

def make_input(cfg, batch=1, seed=1, dtype=np.float32):
    g = np.random.default_rng(seed)
    return g.integers(0, cfg.vocab, (batch, cfg.seq_len), dtype=np.int64)
