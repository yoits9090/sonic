/* transformer.c — ground-up transformer forward pass, zero ML dependencies.
 *
 * Pipeline (matches src/impls.py numpy_vec, but all in C):
 *   h = emb[x]
 *   for each layer i:
 *     h2 = layernorm(h)                       (mean/var over d_model)
 *     Q,K,V = h2 @ wqkv_i                     (one fused matmul, 3d wide)
 *     ctx  = causal_softmax(Q K^T / sqrt(dh)) @ V   (fused flash-style kernel)
 *     h   += ctx @ wo_i                       (accumulating matmul)
 *     h2   = layernorm(h)
 *     h   += relu(h2 @ w1_i) @ w2_i           (fused row-wise FFN kernel)
 *   h = layernorm(h); out = h @ out_w
 *
 * Fusion notes:
 *   - QKV is one matmul over a concatenated [Q|K|V] weight block (the wrapper
 *     builds it once per weight set) instead of three separate passes.
 *   - Attention is a single fused pass per (batch, head, query): scores for
 *     keys k <= s are computed, softmaxed in place (max-subtract + exp + sum),
 *     and immediately accumulated into the context vector. No SxS matrix is
 *     ever materialized; masked entries are never touched (equivalent to the
 *     reference's -1e9 mask because exp(-1e9) underflows to exactly 0).
 *   - The FFN is a single row-wise pass: relu is applied in registers between
 *     the two weight matrices, so w1 and w2 are each streamed once.
 *   - Layernorm is one pass over the row (mean, then var, then scale in a
 *     single re-read that stays in L1).
 *
 * The naive (FT_FUSED=0) path keeps separate Q/K/V matmuls, a materialized
 * SxS attention matrix and a full-batch FFN buffer — used as the baseline.
 *
 * All accumulation is float32 with FMA contraction (-ffast-math). Weights are
 * fp32 so an int32 path is N/A here; fp32 FMA error (~1e-6 relative) is far
 * under the 1e-3 correctness bar.
 */
#include <math.h>
#include <string.h>
#include "ft.h"

/* ft_mm from matmul.c */
void ft_mm(const float *A, const float *B, float *C,
           int M, int N, int K, int acc);

#if !defined(FT_FUSED)
#define FT_FUSED 1
#endif

/* ------------------------------------------------------------------ */
/* Embedding gather: h[(b,s),:] = emb[x[b,s],:]                         */
/* ------------------------------------------------------------------ */
static void embed(const float *restrict emb, const long long *restrict x,
                  float *restrict h, int B, int S, int d) {
    size_t n = (size_t)B * S;
    for (size_t i = 0; i < n; i++) {
        long long tok = x[i];
        memcpy(h + i * d, emb + (size_t)tok * d, (size_t)d * sizeof(float));
    }
}

/* ------------------------------------------------------------------ */
/* Layernorm: y = (x - mean)/sqrt(var + eps) * g + b, one pass per row. */
/* ------------------------------------------------------------------ */
static void layernorm(const float *restrict x, float *restrict y,
                      const float *restrict g, const float *restrict b,
                      float eps, int M, int d) {
    for (int i = 0; i < M; i++) {
        const float *xi = x + (size_t)i * d;
        float *yi = y + (size_t)i * d;
        float mean = 0.0f;
        for (int j = 0; j < d; j++) mean += xi[j];
        mean /= (float)d;
        float var = 0.0f;
        for (int j = 0; j < d; j++) {
            float t = xi[j] - mean;
            var += t * t;
        }
        var /= (float)d;
        float inv = 1.0f / sqrtf(var + eps);
        for (int j = 0; j < d; j++) yi[j] = (xi[j] - mean) * inv * g[j] + b[j];
    }
}

/* ------------------------------------------------------------------ */
/* Fused causal attention (flash-style).                                */
/* Q,K,V are (B*S, d); ctx is (B*S, d). scores[] holds S floats.        */
/* ------------------------------------------------------------------ */
static void attn_fused(const float *restrict Q, const float *restrict K,
                       const float *restrict V, float *restrict ctx,
                       int B, int S, int H, int dh, float inv,
                       float *restrict scores, int stride) {
    int d = H * dh;
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            for (int s = 0; s < S; s++) {
                const float *q = Q + ((size_t)b * S + s) * stride + h * dh;
                /* score for k=0 seeds the running max (keeps everything
                 * finite under -ffast-math, no -INFINITY needed) */
                const float *k0 = K + ((size_t)b * S) * stride + h * dh;
                float acc0 = 0.0f;
                for (int j = 0; j < dh; j++) acc0 += q[j] * k0[j];
                acc0 *= inv;
                scores[0] = acc0;
                float m = acc0;
                for (int k = 1; k <= s; k++) {
                    const float *kk = K + ((size_t)b * S + k) * stride + h * dh;
                    float acc = 0.0f;
                    for (int j = 0; j < dh; j++) acc += q[j] * kk[j];
                    acc *= inv;
                    scores[k] = acc;
                    if (acc > m) m = acc;
                }
                float sum = 0.0f;
                for (int k = 0; k <= s; k++) {
                    float e = expf(scores[k] - m);
                    scores[k] = e;
                    sum += e;
                }
                float r = 1.0f / sum;
                float *o = ctx + ((size_t)b * S + s) * d + h * dh;
                for (int j = 0; j < dh; j++) o[j] = 0.0f;
                for (int k = 0; k <= s; k++) {
                    float p = scores[k] * r;
                    const float *vv = V + ((size_t)b * S + k) * stride + h * dh;
                    for (int j = 0; j < dh; j++) o[j] += p * vv[j];
                }
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* Naive attention: materialize SxS scores per (b,h), mask, softmax,    */
/* matmul against V. scratch needs 2*S*dh + S*S + S floats.             */
/* ------------------------------------------------------------------ */
static void attn_naive(const float *restrict Q, const float *restrict K,
                       const float *restrict V, float *restrict ctx,
                       int B, int S, int H, int dh, float inv,
                       float *restrict scratch, int stride) {
    int d = H * dh;
    float *kc = scratch;            /* S x dh contiguous K copy  */
    float *vc = kc + (size_t)S * dh;/* S x dh contiguous V copy  */
    float *att = vc + (size_t)S * dh;/* S x S scores             */
    float *row = att + (size_t)S * S;/* S softmax temp           */
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            for (int k = 0; k < S; k++) {
                const float *src = K + ((size_t)b * S + k) * stride + h * dh;
                memcpy(kc + (size_t)k * dh, src, (size_t)dh * sizeof(float));
                src = V + ((size_t)b * S + k) * stride + h * dh;
                memcpy(vc + (size_t)k * dh, src, (size_t)dh * sizeof(float));
            }
            for (int s = 0; s < S; s++) {
                const float *q = Q + ((size_t)b * S + s) * stride + h * dh;
                float *a = att + (size_t)s * S;
                for (int k = 0; k < S; k++) {
                    const float *kk = kc + (size_t)k * dh;
                    float acc = 0.0f;
                    for (int j = 0; j < dh; j++) acc += q[j] * kk[j];
                    a[k] = (k <= s) ? acc * inv : -1e9f;
                }
                float m = a[0];
                for (int k = 1; k < S; k++) if (a[k] > m) m = a[k];
                float sum = 0.0f;
                for (int k = 0; k < S; k++) { float e = expf(a[k] - m); row[k] = e; sum += e; }
                float r = 1.0f / sum;
                float *o = ctx + ((size_t)b * S + s) * d + h * dh;
                for (int j = 0; j < dh; j++) o[j] = 0.0f;
                for (int k = 0; k < S; k++) {
                    float p = row[k] * r;
                    const float *vv = vc + (size_t)k * dh;
                    for (int j = 0; j < dh; j++) o[j] += p * vv[j];
                }
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* Fused FFN: h += relu(X @ w1) @ w2, row by row, single pass.          */
/* u must hold d_ff floats.                                             */
/* ------------------------------------------------------------------ */
static void ffn_fused(const float *restrict X, const float *restrict w1,
                      const float *restrict w2, float *restrict Hout,
                      int M, int d, int dff, float *restrict u) {
    for (int i = 0; i < M; i++) {
        const float *x = X + (size_t)i * d;
        /* u = x @ w1  (d -> d_ff), streaming w1 with unit stride on j */
        for (int j = 0; j < dff; j++) u[j] = 0.0f;
        for (int k = 0; k < d; k++) {
            float xk = x[k];
            const float *w1k = w1 + (size_t)k * dff;
            for (int j = 0; j < dff; j++) u[j] += xk * w1k[j];
        }
        /* relu in registers */
        for (int j = 0; j < dff; j++) u[j] = u[j] > 0.0f ? u[j] : 0.0f;
        /* h += u @ w2, outer-product form: for each j, the d-wide row of w2
         * is contiguous, so both w2 and h stream with unit stride */
        float *hi = Hout + (size_t)i * d;
        for (int j = 0; j < dff; j++) {
            float uj = u[j];
            const float *w2j = w2 + (size_t)j * d;
            for (int jj = 0; jj < d; jj++) hi[jj] += uj * w2j[jj];
        }
    }
}

/* ------------------------------------------------------------------ */
/* Scratch sizing / carving.                                            */
/* ------------------------------------------------------------------ */
size_t ft_scratch_bytes(int B, int S, int d_model, int d_ff) {
    size_t n = (size_t)B * S;
    size_t d = (size_t)d_model;
    size_t h   = n * d;
    size_t h2  = n * d;
    size_t qkv = n * 3 * d;
    size_t ctx = n * d;
    size_t ffn = n * (size_t)d_ff;                       /* naive path */
    size_t attn = 2 * (size_t)S * d + (size_t)S * S + S; /* naive path, over-provisioned */
    size_t row = (size_t)(d_ff > 3 * d_model ? d_ff : 3 * d_model); /* fused path */
    size_t tot = h + h2 + qkv + ctx + ffn + attn + row + S + 16;
    return tot * sizeof(float) + 64;
}

/* ------------------------------------------------------------------ */
/* Forward pass.                                                        */
/* ------------------------------------------------------------------ */
int ft_forward(const ft_weights *W, const long long *x, int B, int S,
               float *out, float *scratch, size_t scratch_bytes) {
    const int d = W->d_model, dff = W->d_ff, H = W->n_heads, V = W->vocab, L = W->n_layers;
    const int dh = d / H;
    const size_t n = (size_t)B * S;

    /* carve aligned scratch */
    uintptr_t cur = ((uintptr_t)scratch + 63u) & ~(uintptr_t)63u;
    float *h   = (float *)cur; cur += n * d * sizeof(float);
    float *h2  = (float *)cur; cur += n * d * sizeof(float);
    float *qkv = (float *)cur; cur += n * 3 * d * sizeof(float);
    float *ctx = (float *)cur; cur += n * d * sizeof(float);
    float *ffnbuf = (float *)cur; cur += n * dff * sizeof(float);      /* naive */
    float *attnsc = (float *)cur; cur += (2 * (size_t)S * d + (size_t)S * S + S) * sizeof(float); /* naive */
    float *urow = (float *)cur; cur += (size_t)(dff > 3 * d ? dff : 3 * d) * sizeof(float);
    float *scores = (float *)cur; cur += (size_t)S * sizeof(float);
    if (cur - (uintptr_t)scratch > scratch_bytes) return -1;

    embed(W->emb, x, h, B, S, d);
    float inv = 1.0f / sqrtf((float)dh);

    for (int i = 0; i < L; i++) {
#if FT_FUSED
        layernorm(h, h2, W->ln1g + (size_t)i * d, W->ln1b + (size_t)i * d, W->eps, (int)n, d);
        ft_mm(h2, W->wqkv + (size_t)i * d * 3 * d, qkv, (int)n, 3 * d, d, 0);
        attn_fused(qkv, qkv + d, qkv + 2 * d, ctx, B, S, H, dh, inv, scores, 3 * d);
        ft_mm(ctx, W->wo + (size_t)i * d * d, h, (int)n, d, d, 1);
        layernorm(h, h2, W->ln2g + (size_t)i * d, W->ln2b + (size_t)i * d, W->eps, (int)n, d);
        ffn_fused(h2, W->w1 + (size_t)i * d * dff, W->w2 + (size_t)i * dff * d, h, (int)n, d, dff, urow);
#else
        layernorm(h, h2, W->ln1g + (size_t)i * d, W->ln1b + (size_t)i * d, W->eps, (int)n, d);
        ft_mm(h2, W->wq + (size_t)i * d * d, qkv, (int)n, d, d, 0);
        ft_mm(h2, W->wk + (size_t)i * d * d, qkv + n * d, (int)n, d, d, 0);
        ft_mm(h2, W->wv + (size_t)i * d * d, qkv + 2 * n * d, (int)n, d, d, 0);
        attn_naive(qkv, qkv + n * d, qkv + 2 * n * d, ctx, B, S, H, dh, inv, attnsc, d);
        ft_mm(ctx, W->wo + (size_t)i * d * d, h, (int)n, d, d, 1);
        layernorm(h, h2, W->ln2g + (size_t)i * d, W->ln2b + (size_t)i * d, W->eps, (int)n, d);
        ft_mm(h2, W->w1 + (size_t)i * d * dff, ffnbuf, (int)n, dff, d, 0);
        for (size_t j = 0; j < n * dff; j++) ffnbuf[j] = ffnbuf[j] > 0.0f ? ffnbuf[j] : 0.0f;
        ft_mm(ffnbuf, W->w2 + (size_t)i * dff * d, h, (int)n, d, dff, 1);
#endif
    }
    layernorm(h, h2, W->lnfg, W->lnfb, W->eps, (int)n, d);
    ft_mm(h2, W->out_w, out, (int)n, V, d, 0);
    return 0;
}
