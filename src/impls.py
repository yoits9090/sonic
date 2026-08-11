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

# ---------------------------------------------------------------------------
# Ground-up C implementations (ctypes thin wrapper, no numpy in the hot loop).
# Build the .so files on a Colab node with ground_up/build.sh; each eval run
# loads whatever variants are present. Missing libs degrade gracefully.
# ---------------------------------------------------------------------------
import ctypes as _ct
import os as _os

_GROUND_UP = _os.environ.get("FT_GROUND_UP_DIR",
                             _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "ground_up"))

class _FTWeights(_ct.Structure):
    """Mirror of ft_weights in ground_up/ft.h."""
    _fields_ = [
        ("emb", _ct.c_void_p),
        ("wqkv", _ct.c_void_p),
        ("wq", _ct.c_void_p), ("wk", _ct.c_void_p), ("wv", _ct.c_void_p),
        ("wo", _ct.c_void_p),
        ("w1", _ct.c_void_p), ("w2", _ct.c_void_p),
        ("ln1g", _ct.c_void_p), ("ln1b", _ct.c_void_p),
        ("ln2g", _ct.c_void_p), ("ln2b", _ct.c_void_p),
        ("lnfg", _ct.c_void_p), ("lnfb", _ct.c_void_p),
        ("out_w", _ct.c_void_p),
        ("n_layers", _ct.c_int), ("d_model", _ct.c_int), ("d_ff", _ct.c_int),
        ("n_heads", _ct.c_int), ("vocab", _ct.c_int), ("eps", _ct.c_float),
    ]

class _CImpl:
    """Callable (W, x, cfg) -> (B,S,V) float32 array, backed by libft_*.so.

    Zero-copy on the hot path: weight pointers, workspace and the output
    buffer are cached keyed on the weight arrays' memory addresses, so the
    per-call cost is one ctypes call into C. Variants v1..v4 deliberately
    re-allocate workspace/output per call (to quantify allocation cost);
    v5 reuses static buffers.
    """

    def __init__(self, so_name, reuse_workspace=False, reuse_output=False):
        self.name = so_name
        path = _os.path.join(_GROUND_UP, so_name)
        self.lib = _ct.CDLL(path)
        f = self.lib.ft_forward
        f.restype = _ct.c_int
        f.argtypes = [_ct.POINTER(_FTWeights), _ct.c_void_p, _ct.c_int, _ct.c_int,
                      _ct.c_void_p, _ct.c_void_p, _ct.c_size_t]
        s = self.lib.ft_scratch_bytes
        s.restype = _ct.c_size_t
        s.argtypes = [_ct.c_int, _ct.c_int, _ct.c_int, _ct.c_int]
        self.reuse_workspace = reuse_workspace
        self.reuse_output = reuse_output
        self._cache_key = None
        self._st = None
        self._ws = None
        self._out = None
        self._keep = None

    def _prepare(self, W, cfg, B, S):
        key = (id(W), cfg.d_model, cfg.d_ff, cfg.n_heads, cfg.vocab, cfg.n_layers, B, S)
        if key == self._cache_key:
            return
        d, dff, L = cfg.d_model, cfg.d_ff, cfg.n_layers

        def cat(names):
            arrs = [W[n] for n in names]
            if len(arrs) == 1:
                return np.ascontiguousarray(arrs[0], dtype=np.float32)
            return np.concatenate(arrs, axis=0).astype(np.float32, copy=False)

        emb = np.ascontiguousarray(W["emb"], dtype=np.float32)
        # fused [Q|K|V] weight block per layer: (L, d, 3d) — matches numpy_vec
        wqkv = np.concatenate(
            [np.concatenate([W[f"wq{i}"], W[f"wk{i}"], W[f"wv{i}"]], axis=1)
             for i in range(L)], axis=0).astype(np.float32, copy=False)
        wo = cat([f"wo{i}" for i in range(L)])
        w1 = cat([f"w1{i}" for i in range(L)])
        w2 = cat([f"w2{i}" for i in range(L)])
        ln1g = cat([f"ln1_g{i}" for i in range(L)])
        ln1b = cat([f"ln1_b{i}" for i in range(L)])
        ln2g = cat([f"ln2_g{i}" for i in range(L)])
        ln2b = cat([f"ln2_b{i}" for i in range(L)])
        lnfg = np.ascontiguousarray(W["lnf_g"], dtype=np.float32)
        lnfb = np.ascontiguousarray(W["lnf_b"], dtype=np.float32)
        outw = np.ascontiguousarray(W["out"], dtype=np.float32)
        self._keep = (emb, wqkv, wo, w1, w2, ln1g, ln1b, ln2g, ln2b, lnfg, lnfb, outw)

        st = _FTWeights()
        st.emb = emb.ctypes.data
        st.wqkv = wqkv.ctypes.data
        st.wq = wo.ctypes.data          # unused by fused path; keep non-null
        st.wk = wo.ctypes.data
        st.wv = wo.ctypes.data
        st.wo = wo.ctypes.data
        st.w1 = w1.ctypes.data
        st.w2 = w2.ctypes.data
        st.ln1g = ln1g.ctypes.data
        st.ln1b = ln1b.ctypes.data
        st.ln2g = ln2g.ctypes.data
        st.ln2b = ln2b.ctypes.data
        st.lnfg = lnfg.ctypes.data
        st.lnfb = lnfb.ctypes.data
        st.out_w = outw.ctypes.data
        st.n_layers, st.d_model, st.d_ff, st.n_heads, st.vocab = L, d, dff, cfg.n_heads, cfg.vocab
        st.eps = float(cfg.eps)
        self._st = st

        need = self.lib.ft_scratch_bytes(B, S, d, dff)
        self._ws = np.empty((need + 3) // 4, dtype=np.float32)  # bytes -> floats
        self._out = np.empty((B, S, cfg.vocab), dtype=np.float32)
        self._cache_key = key

    def __call__(self, W, x, cfg):
        B, S = x.shape
        x = np.ascontiguousarray(x)
        self._prepare(W, cfg, B, S)
        if self.reuse_output:
            out = self._out
            ws = self._ws
        else:
            out = np.empty((B, S, cfg.vocab), dtype=np.float32)
            ws = np.empty_like(self._ws)
        rc = self.lib.ft_forward(_ct.byref(self._st), x.ctypes.data, B, S,
                                 out.ctypes.data, ws.ctypes.data, ws.nbytes)
        if rc != 0:
            raise RuntimeError(f"ft_forward failed rc={rc}")
        return out


def _register_c(name, so, rw=False, ro=False):
    try:
        IMPLS[name] = _CImpl(so, rw, ro)
        return True
    except Exception as e:  # noqa: BLE001 - keep numpy impls usable if libs absent
        print(f"[ground-up-c] {name} unavailable ({so}): {e}")
        return False


_register_c("c_v1", "libft_v1.so")                 # naive matmul, unfused ops, per-call alloc
_register_c("c_v2", "libft_v2.so")                 # 8x8 blocked matmul, unfused ops
_register_c("c_v3", "libft_v3.so")                 # blocked + fused layer ops
_register_c("c_v4", "libft_v4.so")                 # blocked + fused + OpenMP
_register_c("c_v5", "libft_v5.so", rw=True, ro=True)  # + static workspace/output reuse
_register_c("c_ground_up", "libft.so", rw=True, ro=True)  # default best build


# ---------- numpy-optimizer agents' impls ----------

# per-seq-length causal mask as additive constant (float32), cached at import
_MASKADD_CACHE = {}
def _maskadd(S):
    m = _MASKADD_CACHE.get(S)
    if m is None:
        m = np.where(np.tril(np.ones((S, S), dtype=bool)), np.float32(0.0), np.float32(-1e9))
        _MASKADD_CACHE[S] = m
    return m

def _ln(x, g, b, eps):
    """Layernorm via E[x^2]-E[x]^2 (one fewer pass than mean+var), returns (x-mu)*rstd*g+b."""
    mu = x.mean(-1, keepdims=True)
    var = (x * x).mean(-1, keepdims=True) - mu * mu
    rstd = 1.0 / np.sqrt(var + eps)
    return (x - mu) * (rstd * g) + b

def numpy_opt1(W, x, cfg):
    """B=1 squeeze, fused LN stats, mask-as-add, no-max-subtraction softmax,
    scale folded into Q, single QKV matmul. No weight caching."""
    B, S = x.shape
    if B != 1:
        return numpy_vec(W, x, cfg)
    d = cfg.d_model
    h = W["emb"][x[0]]                      # (S,d)
    n_heads, dh = cfg.n_heads, cfg.d_head
    inv_scale = 1.0 / np.sqrt(dh)
    maskadd = _maskadd(S)
    for i in range(cfg.n_layers):
        h2 = _ln(h, W[f"ln1_g{i}"], W[f"ln1_b{i}"], cfg.eps)
        qkv = h2 @ np.concatenate([W[f"wq{i}"], W[f"wk{i}"], W[f"wv{i}"]], axis=1)  # (S,3d)
        Q = qkv[:, :d].reshape(S, n_heads, dh).transpose(1, 0, 2)
        K = qkv[:, d:2*d].reshape(S, n_heads, dh).transpose(1, 0, 2)
        V = qkv[:, 2*d:3*d].reshape(S, n_heads, dh).transpose(1, 0, 2)
        att = (Q * inv_scale) @ K.transpose(0, 2, 1)   # (H,S,S)
        att += maskadd
        e = np.exp(att)
        e /= e.sum(-1, keepdims=True)
        ctx = (e @ V).transpose(1, 0, 2).reshape(S, d)
        h = h + ctx @ W[f"wo{i}"]
        h2 = _ln(h, W[f"ln2_g{i}"], W[f"ln2_b{i}"], cfg.eps)
        h = h + (np.maximum(h2, 0) @ W[f"w1{i}"]) @ W[f"w2{i}"]
    h = _ln(h, W["lnf_g"], W["lnf_b"], cfg.eps)
    return (h @ W["out"])[None]             # (1,S,V)

# weight-preprocessing cache: fused QKV (with Q pre-scaled), kept alive by entry ref
_WPREP_CACHE = {}
def _prep_weights(W, cfg):
    """Fuse wq/wk/wv -> single (d,3d) matrix with Q block pre-scaled by 1/sqrt(dh)."""
    key = (id(W), cfg.d_model, cfg.n_heads, cfg.d_head, cfg.n_layers, cfg.d_ff, cfg.seq_len, cfg.vocab)
    entry = _WPREP_CACHE.get(key)
    if entry is None:
        n_heads, dh = cfg.n_heads, cfg.d_head
        d = cfg.d_model
        inv = np.float32(1.0 / np.sqrt(dh))
        qkv = {}
        for i in range(cfg.n_layers):
            wq = W[f"wq{i}"] * inv
            wk = W[f"wk{i}"]
            wv = W[f"wv{i}"]
            qkv[i] = np.concatenate([wq, wk, wv], axis=1).astype(np.float32, copy=False)
        entry = {"qkv": qkv, "ref": W}
        _WPREP_CACHE[key] = entry
    return entry

def numpy_opt2(W, x, cfg):
    """opt1 + cached fused QKV weights (Q pre-scaled); no per-call concatenate/scale."""
    B, S = x.shape
    if B != 1:
        return numpy_vec(W, x, cfg)
    d = cfg.d_model
    h = W["emb"][x[0]]
    n_heads, dh = cfg.n_heads, cfg.d_head
    prep = _prep_weights(W, cfg)
    maskadd = _maskadd(S)
    for i in range(cfg.n_layers):
        h2 = _ln(h, W[f"ln1_g{i}"], W[f"ln1_b{i}"], cfg.eps)
        qkv = h2 @ prep["qkv"][i]
        Q = qkv[:, :d].reshape(S, n_heads, dh).transpose(1, 0, 2)
        K = qkv[:, d:2*d].reshape(S, n_heads, dh).transpose(1, 0, 2)
        V = qkv[:, 2*d:3*d].reshape(S, n_heads, dh).transpose(1, 0, 2)
        att = Q @ K.transpose(0, 2, 1)
        att += maskadd
        e = np.exp(att)
        e /= e.sum(-1, keepdims=True)
        ctx = (e @ V).transpose(1, 0, 2).reshape(S, d)
        h = h + ctx @ W[f"wo{i}"]
        h2 = _ln(h, W[f"ln2_g{i}"], W[f"ln2_b{i}"], cfg.eps)
        h = h + (np.maximum(h2, 0) @ W[f"w1{i}"]) @ W[f"w2{i}"]
    h = _ln(h, W["lnf_g"], W["lnf_b"], cfg.eps)
    return (h @ W["out"])[None]

IMPLS["numpy_opt1"] = numpy_opt1
IMPLS["numpy_opt2"] = numpy_opt2

def _prep_weights_v2(W, cfg):
    """opt3 weight prep: fused QKV (Q pre-scaled) + LN-folded projection mats.
    For LN with (g,b) -> ln(x)@P = rstd*(x@A - mu*c) + db
    where A = g[:,None]*P, c = g@P, db = b@P.  (g=1,b=0 shortcuts precomputed.)"""
    key = ("v2", id(W), cfg.d_model, cfg.n_heads, cfg.d_head, cfg.n_layers, cfg.d_ff, cfg.seq_len, cfg.vocab)
    entry = _WPREP_CACHE.get(key)
    if entry is None:
        n_heads, dh, d = cfg.n_heads, cfg.d_head, cfg.d_model
        inv = np.float32(1.0 / np.sqrt(dh))
        qkv, c1, db1 = {}, {}, {}
        w1, w2, wo = {}, {}, {}
        for i in range(cfg.n_layers):
            g1 = W[f"ln1_g{i}"]; b1 = W[f"ln1_b{i}"]
            wq = W[f"wq{i}"] * inv
            wk = W[f"wk{i}"]
            wv = W[f"wv{i}"]
            P = np.concatenate([wq, wk, wv], axis=1)          # (d,3d)
            qkv[i] = P
            c1[i] = P.sum(0) if np.all(g1 == 1) else (g1 @ P)
            db1[i] = np.zeros(3 * d, dtype=np.float32) if not np.any(b1 != 0) else (b1 @ P)
            w1[i] = W[f"w1{i}"]
            w2[i] = W[f"w2{i}"]
            wo[i] = W[f"wo{i}"]
        gf = W["lnf_g"]; bf = W["lnf_b"]
        outW = W["out"]
        A3 = outW if np.all(gf == 1) else (gf[:, None] * outW)
        c3 = outW.sum(0) if np.all(gf == 1) else (gf @ outW)
        db3 = np.zeros(cfg.vocab, dtype=np.float32) if not np.any(bf != 0) else (bf @ outW)
        entry = {"qkv": qkv, "c1": c1, "db1": db1, "w1": w1, "w2": w2, "wo": wo,
                 "A3": A3, "c3": c3, "db3": db3, "ref": W}
        _WPREP_CACHE[key] = entry
    return entry

def _stats(x, eps):
    """per-row mean and rstd (E[x^2]-E[x]^2 path)."""
    mu = x.mean(-1, keepdims=True)
    var = (x * x).mean(-1, keepdims=True) - mu * mu
    rstd = 1.0 / np.sqrt(var + eps)
    return mu, rstd

def numpy_opt3(W, x, cfg):
    """opt2 + layernorm folded into downstream projections + denom-factorized ctx.
    FFN keeps ReLU BEFORE the up-projection: h += max(h-mu,0)*rstd @ w1 @ w2."""
    B, S = x.shape
    if B != 1:
        return numpy_vec(W, x, cfg)
    d = cfg.d_model
    h = W["emb"][x[0]]
    n_heads, dh = cfg.n_heads, cfg.d_head
    p = _prep_weights_v2(W, cfg)
    maskadd = _maskadd(S)
    for i in range(cfg.n_layers):
        mu, rstd = _stats(h, cfg.eps)
        qkv = (h @ p["qkv"][i] - mu * p["c1"][i]) * rstd
        Q = qkv[:, :d].reshape(S, n_heads, dh).transpose(1, 0, 2)
        K = qkv[:, d:2*d].reshape(S, n_heads, dh).transpose(1, 0, 2)
        V = qkv[:, 2*d:3*d].reshape(S, n_heads, dh).transpose(1, 0, 2)
        att = Q @ K.transpose(0, 2, 1)
        att += maskadd
        e = np.exp(att)
        ctx = (e @ V) * (1.0 / e.sum(-1, keepdims=True))
        h = h + ctx.transpose(1, 0, 2).reshape(S, d) @ p["wo"][i]
        mu, rstd = _stats(h, cfg.eps)
        ff = np.maximum(h - mu, 0) * rstd
        h = h + (ff @ p["w1"][i]) @ p["w2"][i]
    mu, rstd = _stats(h, cfg.eps)
    logits = (h @ p["A3"] - mu * p["c3"]) * rstd
    return logits[None]

def numpy_opt4(W, x, cfg):
    """opt3 with inlined stats (identical math)."""
    B, S = x.shape
    if B != 1:
        return numpy_vec(W, x, cfg)
    d = cfg.d_model
    h = W["emb"][x[0]]
    n_heads, dh = cfg.n_heads, cfg.d_head
    p = _prep_weights_v2(W, cfg)
    maskadd = _maskadd(S)
    for i in range(cfg.n_layers):
        mu = h.mean(-1, keepdims=True)
        rstd = 1.0 / np.sqrt((h * h).mean(-1, keepdims=True) - mu * mu + cfg.eps)
        qkv = (h @ p["qkv"][i] - mu * p["c1"][i]) * rstd
        Q = qkv[:, :d].reshape(S, n_heads, dh).transpose(1, 0, 2)
        K = qkv[:, d:2*d].reshape(S, n_heads, dh).transpose(1, 0, 2)
        V = qkv[:, 2*d:3*d].reshape(S, n_heads, dh).transpose(1, 0, 2)
        att = Q @ K.transpose(0, 2, 1)
        att += maskadd
        e = np.exp(att)
        ctx = (e @ V) * (1.0 / e.sum(-1, keepdims=True))
        h = h + ctx.transpose(1, 0, 2).reshape(S, d) @ p["wo"][i]
        mu = h.mean(-1, keepdims=True)
        rstd = 1.0 / np.sqrt((h * h).mean(-1, keepdims=True) - mu * mu + cfg.eps)
        ff = np.maximum(h - mu, 0) * rstd
        h = h + (ff @ p["w1"][i]) @ p["w2"][i]
    mu = h.mean(-1, keepdims=True)
    rstd = 1.0 / np.sqrt((h * h).mean(-1, keepdims=True) - mu * mu + cfg.eps)
    logits = ((h @ p["A3"] - mu * p["c3"]) * rstd)[None]
    return logits

IMPLS["numpy_opt3"] = numpy_opt3
IMPLS["numpy_opt4"] = numpy_opt4
