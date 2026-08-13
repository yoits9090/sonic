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

#if defined(FT_OPENMP)
#include <omp.h>
#endif

#if !defined(FT_ATTN_OMP_MIN)
#define FT_ATTN_OMP_MIN 2048
#endif

/* ft_mm from matmul.c */
void ft_mm(const float *A, const float *B, float *C,
           int M, int N, int K, int acc);

#if !defined(FT_FUSED)
#define FT_FUSED 1
#endif

#if !defined(FT_FASTEXP)
#define FT_FASTEXP 0
#endif

#if !defined(FT_ATTN_BLOCK)
#define FT_ATTN_BLOCK 0
#endif

#if !defined(FT_FFN_MM)
#define FT_FFN_MM 0
#endif

#if !defined(FT_INT8)
#define FT_INT8 0
#endif
#if !defined(FT_INT8_ATTN)
#define FT_INT8_ATTN 0
#endif
#if !defined(FT_BF16)
#define FT_BF16 0
#endif

/* Precision-knob dispatch: MM_W = weight matmuls (static B, cacheable),
 * MM_A = attention matmuls (scratch B, quantized per call).
 * All knobs are no-ops at their default (0) values: the fp32 champion
 * path stays bit-identical. */
void ft_mm_i8(const float *A, const float *B, float *C, int M, int N, int K, int acc, int cache_b);
void ft_mm_bf16(const float *A, const float *B, float *C, int M, int N, int K, int acc);
#if FT_INT8
#define MM_W(A,B,C,M,N,K,acc) ft_mm_i8((A),(B),(C),(M),(N),(K),(acc),1)
#define MM_A(A,B,C,M,N,K,acc) ft_mm_i8((A),(B),(C),(M),(N),(K),(acc),0)
#elif FT_BF16
#define MM_W(A,B,C,M,N,K,acc) ft_mm_bf16((A),(B),(C),(M),(N),(K),(acc))
#define MM_A(A,B,C,M,N,K,acc) ft_mm_bf16((A),(B),(C),(M),(N),(K),(acc))
#elif FT_INT8_ATTN
#define MM_W(A,B,C,M,N,K,acc) ft_mm((A),(B),(C),(M),(N),(K),(acc))
#define MM_A(A,B,C,M,N,K,acc) ft_mm_i8((A),(B),(C),(M),(N),(K),(acc),0)
#else
#define MM_W(A,B,C,M,N,K,acc) ft_mm((A),(B),(C),(M),(N),(K),(acc))
#define MM_A(A,B,C,M,N,K,acc) ft_mm((A),(B),(C),(M),(N),(K),(acc))
#endif

/* exp(x) for x <= 0: Chebyshev-node LS fit of 2^f on [0,1), degree
 * FT_EXP_DEG (3/4/5/6). exp(x) = 2^(x*log2e); the integer part is folded
 * into the float exponent bits (branch-free, SIMD-friendly).
 * Max rel errors: deg3 ~1e-4, deg4 ~3.5e-6, deg5 ~1e-7, deg6 ~2.5e-9
 * (deg6 coefficients are the original v14 champion set, kept bit-identical).
 * ~3-4x faster than glibc expf. */
#if !defined(FT_EXP_DEG)
#define FT_EXP_DEG 6
#endif
#if FT_EXP_DEG == 6
#define FT_EXP_POLY(f) (1.0000000025791764f + (f) * (0.6931469288712638f + (f) * (0.24023050092090045f \
              + (f) * (0.055480429145105356f + (f) * (0.009684577477983502f \
              + (f) * (0.0012387831477474515f + (f) * 0.0002187751645542967f))))))
#elif FT_EXP_DEG == 5
#define FT_EXP_POLY(f) (9.999998983500e-01f + (f) * (6.931544896632e-01f + (f) * (2.401418182014e-01f \
              + (f) * (5.586033707741e-02f + (f) * (8.949590423122e-03f + (f) * 1.893754058301e-03f)))))
#elif FT_EXP_DEG == 4
#define FT_EXP_POLY(f) (1.000003492908e+00f + (f) * (6.929729221730e-01f + (f) * (2.416043572701e-01f \
              + (f) * (5.174499776409e-02f + (f) * 1.367030945336e-02f))))
#elif FT_EXP_DEG == 3
#define FT_EXP_POLY(f) (9.999002881728e-01f + (f) * (6.963247710916e-01f + (f) * (2.246931557640e-01f \
              + (f) * 7.896725704217e-02f)))
#else
#error "FT_EXP_DEG must be 3, 4, 5 or 6"
#endif
static inline float fast_exp(float x) {
    if (x < -88.0f) return 0.0f;   /* underflow guard (masked entries) */
    float t = x * 1.4426950408889634f;
    float n = floorf(t);
    float f = t - n;
    float p = FT_EXP_POLY(f);
    union { float f; int32_t i; } u;
    u.f = p;
    u.i += (int32_t)n << 23;   /* scale by 2^n via exponent field */
    return u.f;
}
#define EXP(x) (FT_FASTEXP ? fast_exp(x) : expf(x))

/* FT_PROFILE: per-stage timing (debug builds only). */
#if defined(FT_PROFILE)
#include <stdio.h>
#include <time.h>
static double _now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
#define TIC(name) double _t_##name = _now()
#define TOC(name, what) do { \
    double _d = (_now() - _t_##name) * 1e6; \
    FILE *_fp = fopen("/tmp/ft_prof.log", "a"); \
    if (_fp) { fprintf(_fp, "[prof] %-8s %10.1f us\n", what, _d); fclose(_fp); } \
} while (0)
#else
#define TIC(name)
#define TOC(name, what)
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
static inline float _dot_generic(const float *restrict a, const float *restrict b, int n) {
    float s = 0.0f;
    for (int j = 0; j < n; j++) s += a[j] * b[j];
    return s;
}

/* 16-wide dot product with two independent 8-wide accumulator chains so the
 * compiler emits back-to-back vector FMAs instead of one serial chain. */
static inline float dot16(const float *restrict a, const float *restrict b) {
    float s0 = 0.0f, s1 = 0.0f;
#pragma GCC unroll 8
    for (int j = 0; j < 8; j++) { s0 += a[j] * b[j]; s1 += a[8 + j] * b[8 + j]; }
    return s0 + s1;
}

/* Blocked attention: per (b,h), scores = Q@K^T and ctx = att@V via the fast
 * matmul kernel, with a vectorized row softmax on contiguous (S,S) blocks.
 * Matches numpy_vec's math exactly (mask -1e9, softmax over all S). */
static void attn_blocked(const float *restrict Q, const float *restrict K,
                         const float *restrict V, float *restrict ctx,
                         int B, int S, int H, int dh, float inv,
                         float *restrict scratch, int stride) {
    int d = H * dh;
#if defined(FT_OPENMP)
#pragma omp parallel for schedule(static) collapse(2) if ((long)B * H * S * S >= 2048)
#endif
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
#if defined(FT_OPENMP)
            float *sc = scratch + (size_t)omp_get_thread_num() * (4 * S * dh + S * S);
#else
            float *sc = scratch;
#endif
            float *qc = sc;                      /* S*dh */
            float *kc = qc + (size_t)S * dh;     /* S*dh */
            float *vc = kc + (size_t)S * dh;     /* S*dh */
            float *kt = vc + (size_t)S * dh;     /* dh*S (same size, separate from att!) */
            float *att = kt + (size_t)S * dh;    /* S*S */
            for (int k = 0; k < S; k++) {
                const float *src = Q + ((size_t)b * S + k) * stride + h * dh;
                memcpy(qc + (size_t)k * dh, src, (size_t)dh * sizeof(float));
                src = K + ((size_t)b * S + k) * stride + h * dh;
                memcpy(kc + (size_t)k * dh, src, (size_t)dh * sizeof(float));
                src = V + ((size_t)b * S + k) * stride + h * dh;
                memcpy(vc + (size_t)k * dh, src, (size_t)dh * sizeof(float));
            }
            /* K^T (dh x S) from kc (S x dh) */
            for (int k = 0; k < S; k++)
                for (int j = 0; j < dh; j++) kt[j * S + k] = kc[k * dh + j];
            MM_A(qc, kt, att, S, S, dh, 0);   /* scores (S x S) */
            for (int i = 0; i < S * S; i++) att[i] *= inv;   /* 1/sqrt(dh) like the ref */
            /* causal mask + row softmax (vectorized over the contiguous row) */
            for (int s = 0; s < S; s++) {
                float *row = att + (size_t)s * S;
                for (int k = s + 1; k < S; k++) row[k] = -1e9f;
                float m = row[0];
                for (int k = 1; k < S; k++) if (row[k] > m) m = row[k];
                float sum = 0.0f;
#pragma omp simd reduction(+:sum)
                for (int k = 0; k < S; k++) {
                    float e = EXP(row[k] - m);
                    row[k] = e;
                    sum += e;
                }
                float r = 1.0f / sum;
                for (int k = 0; k < S; k++) row[k] *= r;
            }
            MM_A(att, vc, qc, S, dh, S, 0);    /* ctx (S x dh) into qc (reused) */
            for (int k = 0; k < S; k++) {
                float *dst = ctx + ((size_t)b * S + k) * d + h * dh;
                memcpy(dst, qc + (size_t)k * dh, (size_t)dh * sizeof(float));
            }
        }
    }
}

static void attn_fused(const float *restrict Q, const float *restrict K,
                       const float *restrict V, float *restrict ctx,
                       int B, int S, int H, int dh, float inv,
                       float *restrict scores, int stride) {
    int d = H * dh;
#if defined(FT_OPENMP)
#pragma omp parallel for schedule(static) collapse(2) if ((long)B * H * S * S >= FT_ATTN_OMP_MIN)
#endif
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
#if defined(FT_OPENMP)
            float *sc = scores + (size_t)omp_get_thread_num() * S; /* thread-private */
            /* contiguous K/V copies for this (b,h): keys are 64B apart in the
             * qkv buffer (row stride 3d), which wastes a cache line per key;
             * a packed (S,dh) copy makes every dot and ctx pass stream L1.
             * Per-thread pads: 2*S*dh floats after the 8*S score slots. */
            float *kc = scores + (size_t)S * 8 + (size_t)omp_get_thread_num() * (2 * S * dh);
            float *vc = kc + (size_t)S * dh;
#else
            float *sc = scores;
            float *kc = scores + (size_t)S * 8;
            float *vc = kc + (size_t)S * dh;
#endif
            for (int k = 0; k < S; k++) {
                const float *src = K + ((size_t)b * S + k) * stride + h * dh;
                memcpy(kc + (size_t)k * dh, src, (size_t)dh * sizeof(float));
                src = V + ((size_t)b * S + k) * stride + h * dh;
                memcpy(vc + (size_t)k * dh, src, (size_t)dh * sizeof(float));
            }
            const float *qbase = Q + ((size_t)b * S) * stride + h * dh;
            for (int s = 0; s < S; s++) {
                const float *q = qbase + (size_t)s * stride;
                /* score for k=0 seeds the running max (keeps everything
                 * finite under -ffast-math, no -INFINITY needed) */
                float acc0 = (dh == 16) ? dot16(q, kc) : _dot_generic(q, kc, dh);
                acc0 *= inv;
                sc[0] = acc0;
                float m = acc0;
                for (int k = 1; k <= s; k++) {
                    float acc = (dh == 16) ? dot16(q, kc + (size_t)k * dh)
                                           : _dot_generic(q, kc + (size_t)k * dh, dh);
                    acc *= inv;
                    sc[k] = acc;
                    if (acc > m) m = acc;
                }
                /* fused exp + ctx accumulation: o is scaled by 1/sum at the
                 * end, so the exp and context passes share one loop (no es[]). */
                float *o = ctx + ((size_t)b * S + s) * d + h * dh;
                for (int j = 0; j < dh; j++) o[j] = 0.0f;
                float sum = 0.0f;
                for (int k = 0; k <= s; k++) {
                    float e = EXP(sc[k] - m);
                    sum += e;
                    const float *vv = vc + (size_t)k * dh;
                    for (int j = 0; j < dh; j++) o[j] += e * vv[j];
                }
                float r = 1.0f / sum;
                for (int j = 0; j < dh; j++) o[j] *= r;
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
                for (int k = 0; k < S; k++) { float e = EXP(a[k] - m); row[k] = e; sum += e; }
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
/* This benchmark's FFN applies ReLU to the layernorm OUTPUT before the first
 * matmul:  h += relu(x) @ w1 @ w2   (matches the f64 reference and numpy_vec).
 * So the fused kernel folds relu into the x-streaming pass; the w2 pass is an
 * outer-product update (unit stride on both w2 rows and h). */
static void ffn_fused(const float *restrict X, const float *restrict w1,
                      const float *restrict w2, float *restrict Hout,
                      int M, int d, int dff, float *restrict u) {
    for (int i = 0; i < M; i++) {
        const float *x = X + (size_t)i * d;
        /* u = relu(x) @ w1  (d -> d_ff), streaming w1 with unit stride on j */
        for (int j = 0; j < dff; j++) u[j] = 0.0f;
        for (int k = 0; k < d; k++) {
            float xk = x[k] > 0.0f ? x[k] : 0.0f;   /* ReLU on the input */
            const float *w1k = w1 + (size_t)k * dff;
            for (int j = 0; j < dff; j++) u[j] += xk * w1k[j];
        }
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
#if FT_ATTN_BLOCK
    size_t attn_scratch = 8 * (4 * (size_t)S * d_model + (size_t)S * S);
#else
    size_t attn_scratch = 8 * S + 16 * (size_t)S * d_model;
#endif
    size_t tot = h + h2 + qkv + ctx + ffn + attn + row + attn_scratch + 16;
    return tot * sizeof(float) + 64;
}

/* ------------------------------------------------------------------ */
/* Forward pass.                                                        */
/* ------------------------------------------------------------------ */
/* If dbg != NULL, each stage output is copied into it in order:
 *   [h_embed | h2_ln1 | qkv | ctx | h_after_wo | h2_ln2 | h_after_ffn |
 *    h2_final | out]   sizes: n*d, n*d, n*3d, n*d, n*d, n*d, n*d, n*d, n*V
 */
static int forward_impl(const ft_weights *W, const long long *x, int B, int S,
                        float *out, float *scratch, size_t scratch_bytes,
                        float *dbg) {
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
    size_t attn_need = (8 * (size_t)S + 16 * (size_t)S * dh);
#if FT_ATTN_BLOCK
    attn_need = 8 * (4 * (size_t)S * dh + (size_t)S * S);
#endif
    float *scores = (float *)cur; cur += attn_need * sizeof(float);  /* per-thread (up to 8) scratch */
    if (cur - (uintptr_t)scratch > scratch_bytes) return -1;

    embed(W->emb, x, h, B, S, d);
    if (dbg) { memcpy(dbg, h, n * d * sizeof(float)); dbg += n * d; }
    float inv = 1.0f / sqrtf((float)dh);

    for (int i = 0; i < L; i++) {
#if FT_FUSED
        TIC(ln1);
        layernorm(h, h2, W->ln1g + (size_t)i * d, W->ln1b + (size_t)i * d, W->eps, (int)n, d);
        TOC(ln1, "ln1");
        if (dbg) { memcpy(dbg, h2, n * d * sizeof(float)); dbg += n * d; }
        TIC(qkv);
        MM_W(h2, W->wqkv + (size_t)i * d * 3 * d, qkv, (int)n, 3 * d, d, 0);
        TOC(qkv, "qkv");
        if (dbg) { memcpy(dbg, qkv, n * 3 * d * sizeof(float)); dbg += n * 3 * d; }
        TIC(attn);
#if FT_ATTN_BLOCK
        attn_blocked(qkv, qkv + d, qkv + 2 * d, ctx, B, S, H, dh, inv, scores, 3 * d);
#else
        attn_fused(qkv, qkv + d, qkv + 2 * d, ctx, B, S, H, dh, inv, scores, 3 * d);
#endif
        TOC(attn, "attn");
        if (dbg) { memcpy(dbg, ctx, n * d * sizeof(float)); dbg += n * d; }
        TIC(wo);
        MM_W(ctx, W->wo + (size_t)i * d * d, h, (int)n, d, d, 1);
        TOC(wo, "wo");
        if (dbg) { memcpy(dbg, h, n * d * sizeof(float)); dbg += n * d; }
        TIC(ln2);
        layernorm(h, h2, W->ln2g + (size_t)i * d, W->ln2b + (size_t)i * d, W->eps, (int)n, d);
        TOC(ln2, "ln2");
        if (dbg) { memcpy(dbg, h2, n * d * sizeof(float)); dbg += n * d; }
        TIC(ffn);
#if FT_FFN_MM
        for (size_t j = 0; j < n * d; j++) h2[j] = h2[j] > 0.0f ? h2[j] : 0.0f; /* ReLU on ln2 output */
        MM_W(h2, W->w1 + (size_t)i * d * dff, ffnbuf, (int)n, dff, d, 0);
        MM_W(ffnbuf, W->w2 + (size_t)i * dff * d, h, (int)n, d, dff, 1);
#else
        ffn_fused(h2, W->w1 + (size_t)i * d * dff, W->w2 + (size_t)i * dff * d, h, (int)n, d, dff, urow);
#endif
        TOC(ffn, "ffn");
        if (dbg) { memcpy(dbg, h, n * d * sizeof(float)); dbg += n * d; }
#else
        layernorm(h, h2, W->ln1g + (size_t)i * d, W->ln1b + (size_t)i * d, W->eps, (int)n, d);
        if (dbg) { memcpy(dbg, h2, n * d * sizeof(float)); dbg += n * d; }
        ft_mm(h2, W->wq + (size_t)i * d * d, qkv, (int)n, d, d, 0);
        ft_mm(h2, W->wk + (size_t)i * d * d, qkv + n * d, (int)n, d, d, 0);
        ft_mm(h2, W->wv + (size_t)i * d * d, qkv + 2 * n * d, (int)n, d, d, 0);
        if (dbg) { memcpy(dbg, qkv, n * 3 * d * sizeof(float)); dbg += n * 3 * d; }
        attn_naive(qkv, qkv + n * d, qkv + 2 * n * d, ctx, B, S, H, dh, inv, attnsc, d);
        if (dbg) { memcpy(dbg, ctx, n * d * sizeof(float)); dbg += n * d; }
        ft_mm(ctx, W->wo + (size_t)i * d * d, h, (int)n, d, d, 1);
        if (dbg) { memcpy(dbg, h, n * d * sizeof(float)); dbg += n * d; }
        layernorm(h, h2, W->ln2g + (size_t)i * d, W->ln2b + (size_t)i * d, W->eps, (int)n, d);
        if (dbg) { memcpy(dbg, h2, n * d * sizeof(float)); dbg += n * d; }
        for (size_t j = 0; j < n * d; j++) h2[j] = h2[j] > 0.0f ? h2[j] : 0.0f; /* ReLU on ln2 output */
        ft_mm(h2, W->w1 + (size_t)i * d * dff, ffnbuf, (int)n, dff, d, 0);
        ft_mm(ffnbuf, W->w2 + (size_t)i * dff * d, h, (int)n, d, dff, 1);
        if (dbg) { memcpy(dbg, h, n * d * sizeof(float)); dbg += n * d; }
#endif
    }
    TIC(lnf);
    layernorm(h, h2, W->lnfg, W->lnfb, W->eps, (int)n, d);
    TOC(lnf, "lnf");
    if (dbg) { memcpy(dbg, h2, n * d * sizeof(float)); dbg += n * d; }
    TIC(outm);
    MM_W(h2, W->out_w, out, (int)n, V, d, 0);
    TOC(outm, "out");
    if (dbg) { memcpy(dbg, out, n * V * sizeof(float)); dbg += n * V; }
    return 0;
}

int ft_forward(const ft_weights *W, const long long *x, int B, int S,
               float *out, float *scratch, size_t scratch_bytes) {
    return forward_impl(W, x, B, S, out, scratch, scratch_bytes, NULL);
}

/* Debug: same forward, dumps stage outputs into dbg (see forward_impl). */
int ft_forward_debug(const ft_weights *W, const long long *x, int B, int S,
                     float *out, float *scratch, size_t scratch_bytes,
                     float *dbg) {
    return forward_impl(W, x, B, S, out, scratch, scratch_bytes, dbg);
}
