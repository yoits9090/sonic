/* ft.h — ground-up C transformer forward pass: public API.
 *
 * Everything is row-major float32. The Python ctypes wrapper (src/impls.py)
 * concatenates per-layer numpy weight arrays into contiguous blocks so that
 * layer i lives at offset i * rowbytes. No numpy, no BLAS, no ML libs in the
 * hot loop — this is plain C with our own matmul / softmax / layernorm /
 * attention / FFN kernels.
 */
#ifndef FT_H
#define FT_H

#include <stddef.h>

typedef struct {
    /* embedding: (vocab, d_model) */
    const float *emb;
    /* fused QKV weights: (n_layers, d_model, 3*d_model) — [Q|K|V] columns.
     * Built by the wrapper; used by the fused path. */
    const float *wqkv;
    /* separate per-kind weights: (n_layers, d_model, d_model); used by the
     * naive (unfused) path so we don't need the concatenated block there. */
    const float *wq, *wk, *wv;
    /* (n_layers, d_model, d_model) */
    const float *wo;
    /* (n_layers, d_model, d_ff) and (n_layers, d_ff, d_model) */
    const float *w1, *w2;
    /* per-layer layernorm scale/bias: (n_layers, d_model) */
    const float *ln1g, *ln1b, *ln2g, *ln2b;
    /* final layernorm: (d_model,) */
    const float *lnfg, *lnfb;
    /* output projection: (d_model, vocab) */
    const float *out_w;

    int n_layers, d_model, d_ff, n_heads, vocab;
    float eps;
} ft_weights;

/* Minimum scratch bytes needed for a forward pass over a (B, S) batch. */
size_t ft_scratch_bytes(int B, int S, int d_model, int d_ff);

/* Forward pass: out[(b*S + s)*vocab + v] = logits.
 * scratch must hold at least ft_scratch_bytes(...) bytes (64-byte aligned).
 * Returns 0 on success, -1 if scratch is too small. */
int ft_forward(const ft_weights *W, const long long *x, int B, int S,
               float *out, float *scratch, size_t scratch_bytes);

#endif /* FT_H */
