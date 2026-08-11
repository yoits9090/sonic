"""Transformer forward-pass implementations, all numpy. Each impl is a
callable: fn(weights: dict, x: np.ndarray[int]) -> np.ndarray[float32, (B,S,V)]
"""
import numpy as np

def _softmax(x):
    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=-1, keepdims=True)

def _layernorm(x, g, b, eps):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * g + b

def numpy_naive(W, x, cfg):
    """Baseline: everything through separate numpy ops, Python loops over heads/layers."""
    B, S = x.shape
    h = W["emb"][x]  # (B,S,d)
    n_heads, dh = cfg.n_heads, cfg.d_head
    for i in range(cfg.n_layers):
        h2 = _layernorm(h, W[f"ln1_g{i}"], W[f"ln1_b{i}"], cfg.eps)
        Q = h2 @ W[f"wq{i}"]; K = h2 @ W[f"wk{i}"]; V = h2 @ W[f"wv{i}"]
        ctx = np.zeros_like(h2)
        for hh in range(n_heads):
            q = Q[:, :, hh*dh:(hh+1)*dh]; k = K[:, :, hh*dh:(hh+1)*dh]; v = V[:, :, hh*dh:(hh+1)*dh]
            att = q @ k.swapaxes(-1, -2) / np.sqrt(dh)
            mask = np.tril(np.ones((S, S), dtype=bool))
            att = np.where(mask, att, -1e9)
            att = _softmax(att)
            ctx[:, :, hh*dh:(hh+1)*dh] = att @ v
        h = h + ctx @ W[f"wo{i}"]
        h2 = _layernorm(h, W[f"ln2_g{i}"], W[f"ln2_b{i}"], cfg.eps)
        h = h + (np.maximum(h2, 0) @ W[f"w1{i}"]) @ W[f"w2{i}"]
    h = _layernorm(h, W["lnf_g"], W["lnf_b"], cfg.eps)
    return h @ W["out"]

def numpy_vec(W, x, cfg):
    """Vectorized: all heads at once via reshape, no python head loop, fused QKV matmul."""
    B, S = x.shape
    h = W["emb"][x]
    n_heads, dh = cfg.n_heads, cfg.d_head
    for i in range(cfg.n_layers):
        h2 = _layernorm(h, W[f"ln1_g{i}"], W[f"ln1_b{i}"], cfg.eps)
        qkv = h2 @ np.concatenate([W[f"wq{i}"], W[f"wk{i}"], W[f"wv{i}"]], axis=1)
        Q, K, V = np.split(qkv, 3, axis=2)
        Q = Q.reshape(B, S, n_heads, dh).transpose(0, 2, 1, 3)
        K = K.reshape(B, S, n_heads, dh).transpose(0, 2, 1, 3)
        V = V.reshape(B, S, n_heads, dh).transpose(0, 2, 1, 3)
        att = (Q @ K.swapaxes(-1, -2)) / np.sqrt(dh)
        mask = np.tril(np.ones((S, S), dtype=bool))
        att = np.where(mask, att, -1e9)
        att = _softmax(att)
        ctx = att @ V
        ctx = ctx.transpose(0, 2, 1, 3).reshape(B, S, cfg.d_model)
        h = h + ctx @ W[f"wo{i}"]
        h2 = _layernorm(h, W[f"ln2_g{i}"], W[f"ln2_b{i}"], cfg.eps)
        h = h + (np.maximum(h2, 0) @ W[f"w1{i}"]) @ W[f"w2{i}"]
    h = _layernorm(h, W["lnf_g"], W["lnf_b"], cfg.eps)
    return h @ W["out"]

IMPLS = {"numpy_naive": numpy_naive, "numpy_vec": numpy_vec}
