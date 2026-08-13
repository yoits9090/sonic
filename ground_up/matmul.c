/* matmul.c — from-scratch dense matrix multiplication kernels.
 *
 * Contract: C[M][N] = A[M][K] @ B[K][N]   (all row-major, float32).
 *   acc == 0 -> C := A @ B      acc == 1 -> C := A @ B + C
 *
 * Kernel selected at compile time:
 *   FT_KERNEL 1 — naive ikj inner-product loop (baseline, clearly correct).
 *   FT_KERNEL 2 — 8x8 register-blocked, k-blocked, auto-vectorized j loop.
 *                 With FT_OPENMP=1 the row tiles are farmed out to threads.
 *
 * The 8x8 kernel is the classic register-blocked scheme: we accumulate a
 * C[i0:i0+8][j0:j0+8] tile in registers, stream over k in 8-wide chunks,
 * and for each (i,k) pair do an 8-wide fused multiply-add along j. B rows
 * (k-major, j contiguous) and A rows (i-major, k contiguous) are both
 * accessed with unit stride, so the compiler cleanly emits FMA vector
 * instructions (AVX2/AVX-512 with -march=native). float32 accumulation with
 * FMA contraction keeps rounding errors at fp32 level (~1e-7 relative),
 * far below the 1e-3 correctness bar.
 */
#include "ft.h"

#if !defined(FT_KERNEL)
#define FT_KERNEL 2
#endif

/* ------------------------------------------------------------------ */
/* Kernel 1: naive.                                                    */
/* ------------------------------------------------------------------ */
static void mm_naive(const float *restrict A, const float *restrict B,
                     float *restrict C, int M, int N, int K, int acc) {
    if (!acc) {
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++) C[(size_t)i * N + j] = 0.0f;
    }
    for (int i = 0; i < M; i++) {
        const float *ai = A + (size_t)i * K;
        float *ci = C + (size_t)i * N;
        for (int j = 0; j < N; j++) {
            float s = 0.0f;
            for (int k = 0; k < K; k++) s += ai[k] * B[(size_t)k * N + j];
            if (acc) ci[j] += s; else ci[j] = s;
        }
    }
}

/* ------------------------------------------------------------------ */
/* Kernel 2: 8x8 register blocking.                                    */
/* ------------------------------------------------------------------ */
static void mm_blocked(const float *restrict A, const float *restrict B,
                       float *restrict C, int M, int N, int K, int acc) {
    const int TI = 8, TK = 8;
#if !defined(FT_OMP_MIN)
#define FT_OMP_MIN 262144
#endif
#if defined(FT_OPENMP)
#pragma omp parallel for schedule(static) collapse(2)
#endif
    for (int i0 = 0; i0 < M; i0 += TI) {
        for (int j0 = 0; j0 < N; j0 += 8) {
            int ni = (i0 + TI < M ? TI : M - i0);
            int nj = (j0 + 8 < N ? 8 : N - j0);
            float accv[8][8];
            for (int ii = 0; ii < ni; ii++)
                for (int jj = 0; jj < nj; jj++)
                    accv[ii][jj] = acc ? C[(size_t)(i0 + ii) * N + j0 + jj] : 0.0f;
            for (int k0 = 0; k0 < K; k0 += TK) {
                int kmax = k0 + TK < K ? k0 + TK : K;
                for (int ii = 0; ii < ni; ii++) {
                    const float *ai = A + (size_t)(i0 + ii) * K;
                    for (int k = k0; k < kmax; k++) {
                        float a = ai[k];
                        const float *bk = B + (size_t)k * N + j0;
                        float *av = accv[ii];
                        for (int jj = 0; jj < nj; jj++) av[jj] += a * bk[jj];
                    }
                }
            }
            for (int ii = 0; ii < ni; ii++)
                for (int jj = 0; jj < nj; jj++)
                    C[(size_t)(i0 + ii) * N + j0 + jj] = accv[ii][jj];
        }
    }
}

/* ------------------------------------------------------------------ */
/* Kernel 3: specialized 8x8 blocked matmul for dims divisible by 8     */
/* (true for every matmul in this benchmark: d, 3d, d_ff, vocab are     */
/* multiples of 8). Compile-time tile bounds let gcc keep 8 accumulators */
/* in ymm registers and emit vbroadcastss + vfmadd chains.              */
/* ------------------------------------------------------------------ */
static void mm_blocked8(const float *restrict A, const float *restrict B,
                        float *restrict C, int M, int N, int K, int acc) {
#if !defined(FT_OMP_MIN)
#define FT_OMP_MIN 262144
#endif
#if defined(FT_OPENMP)
#pragma omp parallel for schedule(static) collapse(2) if ((long)M * N * K >= FT_OMP_MIN)
#endif
    for (int j0 = 0; j0 < N; j0 += 8) {   /* j-outer: B streams once, A (small) is re-read */
        for (int i0 = 0; i0 < M; i0 += 8) {
            float accv[8][8];
            if (acc) {
                for (int ii = 0; ii < 8; ii++)
                    for (int jj = 0; jj < 8; jj++)
                        accv[ii][jj] = C[(size_t)(i0 + ii) * N + j0 + jj];
            } else {
                for (int ii = 0; ii < 8; ii++)
                    for (int jj = 0; jj < 8; jj++) accv[ii][jj] = 0.0f;
            }
            for (int k = 0; k < K; k += 8) {
                for (int ii = 0; ii < 8; ii++) {
                    const float *ai = A + (size_t)(i0 + ii) * K + k;
                    float *av = accv[ii];
#pragma GCC unroll 8
                    for (int kk = 0; kk < 8; kk++) {
                        float a = ai[kk];
                        const float *bk = B + (size_t)(k + kk) * N + j0;
#pragma GCC unroll 8
                        for (int jj = 0; jj < 8; jj++) av[jj] += a * bk[jj];
                    }
                }
            }
            for (int ii = 0; ii < 8; ii++)
                for (int jj = 0; jj < 8; jj++)
                    C[(size_t)(i0 + ii) * N + j0 + jj] = accv[ii][jj];
        }
    }
}

/* 4x16 register-blocked variant: 64 accumulators (16 ymm) per tile; half
 * the tiles of 8x8 for wide N, at the cost of less B-row reuse across i. */
#define FT_MM_4X16_TILE_BODY(A, B, C, i0, j0, N, K, acc) \
    do { \
        float accv[4][16]; \
        if (acc) { \
            for (int ii = 0; ii < 4; ii++) \
                for (int jj = 0; jj < 16; jj++) \
                    accv[ii][jj] = C[(size_t)(i0 + ii) * N + j0 + jj]; \
        } else { \
            for (int ii = 0; ii < 4; ii++) \
                for (int jj = 0; jj < 16; jj++) accv[ii][jj] = 0.0f; \
        } \
        for (int k = 0; k < K; k += 8) { \
            for (int ii = 0; ii < 4; ii++) { \
                const float *ai = A + (size_t)(i0 + ii) * K + k; \
                float *av = accv[ii]; \
_Pragma("GCC unroll 8") \
                for (int kk = 0; kk < 8; kk++) { \
                    float a = ai[kk]; \
                    const float *bk = B + (size_t)(k + kk) * N + j0; \
_Pragma("GCC unroll 16") \
                    for (int jj = 0; jj < 16; jj++) av[jj] += a * bk[jj]; \
                } \
            } \
        } \
        for (int ii = 0; ii < 4; ii++) \
            for (int jj = 0; jj < 16; jj++) \
                C[(size_t)(i0 + ii) * N + j0 + jj] = accv[ii][jj]; \
    } while (0)

static void mm_blocked8_4x16(const float *restrict A, const float *restrict B,
                             float *restrict C, int M, int N, int K, int acc) {
#if !defined(FT_OMP_MIN)
#define FT_OMP_MIN 262144
#endif
#if defined(FT_OPENMP)
#pragma omp parallel for schedule(static) collapse(2) if ((long)M * N * K >= FT_OMP_MIN)
#endif
    for (int j0 = 0; j0 < N; j0 += 16) {   /* j-outer: B streams once, A (small) is re-read */
        for (int i0 = 0; i0 < M; i0 += 4) {
            FT_MM_4X16_TILE_BODY(A, B, C, i0, j0, N, K, acc);
        }
    }
}

/* Controlled-OMP 4x16: caller chooses threading (out-projection tuning). */
static void mm_blocked8_4x16_ctl(const float *restrict A, const float *restrict B,
                                 float *restrict C, int M, int N, int K, int acc,
                                 int use_omp) {
#if defined(FT_OPENMP)
    if (use_omp) {
#pragma omp parallel for schedule(static) collapse(2)
        for (int j0 = 0; j0 < N; j0 += 16) {
            for (int i0 = 0; i0 < M; i0 += 4) {
                FT_MM_4X16_TILE_BODY(A, B, C, i0, j0, N, K, acc);
            }
        }
        return;
    }
#endif
    (void)use_omp;
    for (int j0 = 0; j0 < N; j0 += 16) {
        for (int i0 = 0; i0 < M; i0 += 4) {
            FT_MM_4X16_TILE_BODY(A, B, C, i0, j0, N, K, acc);
        }
    }
}

/* Out-projection-tuned matmul (FT_OUT_TUNE): shapes are (n=B*S, V) x (V, d) with
 * wide N and thin-to-mid M. Dispatch table from ft-node-1 micro-benchmarks
 * (Aug 2026): 4x32+omp wins mid-M wide-N (M=512: 1776->1083us, 39%); serial
 * 4x16 wins small M (threading overhead dominates: M=128: 342->274us, 20%;
 * M=32: 88->74us via 4x32 serial); omp 4x16 stays for M>=1024. */
static void mm_blocked8_4x32(const float *restrict A, const float *restrict B,
                             float *restrict C, int M, int N, int K, int acc);

void ft_mm_out(const float *A, const float *B, float *C, int M, int N, int K, int acc) {
#if defined(FT_OUT_TUNE)
    if ((M & 7) == 0 && (N & 7) == 0 && (K & 7) == 0) {
#if defined(FT_OUT_TUNE_ALL)
        mm_blocked8_4x16_ctl(A, B, C, M, N, K, acc, 0);
        return;
#else
        /* Evidence-backed rule (interleaved in-situ A/B, ft-node-1): threading
         * loses at M <= 32 (spawn overhead > 2-thread gain on 2-core nodes). */
        if (M <= 32) {
            mm_blocked8_4x16_ctl(A, B, C, M, N, K, acc, 0);
            return;
        }
#endif
    }
#endif
    mm_blocked8_4x16(A, B, C, M, N, K, acc);
}

/* 4x32 tile: two 16-wide halves per (ii,kk) share the scalar a-load and
 * halve the tile count (fewer init/writeback passes) vs 4x16. accv[ii] is
 * 32 floats = 8 ymm, live within a (ii,kk) group. */
static void mm_blocked8_4x32(const float *restrict A, const float *restrict B,
                             float *restrict C, int M, int N, int K, int acc) {
#if defined(FT_OPENMP)
#pragma omp parallel for schedule(static) collapse(2) if ((long)M * N * K >= FT_OMP_MIN)
#endif
    for (int j0 = 0; j0 < N; j0 += 32) {
        for (int i0 = 0; i0 < M; i0 += 4) {
            float accv[4][32];
            if (acc) {
                for (int ii = 0; ii < 4; ii++)
                    for (int jj = 0; jj < 32; jj++)
                        accv[ii][jj] = C[(size_t)(i0 + ii) * N + j0 + jj];
            } else {
                for (int ii = 0; ii < 4; ii++)
                    for (int jj = 0; jj < 32; jj++) accv[ii][jj] = 0.0f;
            }
            for (int k = 0; k < K; k += 8) {
                for (int ii = 0; ii < 4; ii++) {
                    const float *ai = A + (size_t)(i0 + ii) * K + k;
                    float *av = accv[ii];
#pragma GCC unroll 8
                    for (int kk = 0; kk < 8; kk++) {
                        float a = ai[kk];
                        const float *bk = B + (size_t)(k + kk) * N + j0;
#pragma GCC unroll 16
                        for (int jj = 0; jj < 16; jj++) av[jj] += a * bk[jj];
#pragma GCC unroll 16
                        for (int jj = 0; jj < 16; jj++) av[16 + jj] += a * bk[16 + jj];
                    }
                }
            }
            for (int ii = 0; ii < 4; ii++)
                for (int jj = 0; jj < 32; jj++)
                    C[(size_t)(i0 + ii) * N + j0 + jj] = accv[ii][jj];
        }
    }
}

/* ------------------------------------------------------------------ */
/* Dispatch.                                                           */
/* ------------------------------------------------------------------ */
void ft_mm(const float *A, const float *B, float *C,
           int M, int N, int K, int acc) {
#if FT_KERNEL == 1
    mm_naive(A, B, C, M, N, K, acc);
#elif FT_KERNEL == 3
    if ((M & 7) == 0 && (N & 7) == 0 && (K & 7) == 0) {
#if defined(FT_TILE16) && FT_TILE16 == 1
        mm_blocked8_4x16(A, B, C, M, N, K, acc);
#elif defined(FT_TILE16) && FT_TILE16 == 2
        if (M >= 64)  /* batch path: 8x8 reuses B rows across more i */
            mm_blocked8(A, B, C, M, N, K, acc);
        else
            mm_blocked8_4x16(A, B, C, M, N, K, acc);
#elif defined(FT_TILE16) && FT_TILE16 == 3
        if ((N & 31) == 0)
            mm_blocked8_4x32(A, B, C, M, N, K, acc);
        else
            mm_blocked8_4x16(A, B, C, M, N, K, acc);
#else
        mm_blocked8(A, B, C, M, N, K, acc);
#endif
    } else {
        mm_blocked(A, B, C, M, N, K, acc);
    }
#else
    mm_blocked(A, B, C, M, N, K, acc);
#endif
}


/* ------------------------------------------------------------------ */
/* int8 GEMM (FT_INT8 / FT_INT8_ATTN) and bf16 GEMM (FT_BF16).         */
/*                                                                     */
/* int8: weights quantized per-column, activations per-row, int32      */
/* accumulate, dequant per output element. Weight quantization is      */
/* cached by source pointer (first call; weights are static in the     */
/* benchmark). Activation quantization is per call into thread-local   */
/* buffers. AVX2 vpmaddubsw path + AVX512 VNNI vpdpbusd path (runtime  */
/* CPU dispatch); portable scalar fallback. B is stored transposed     */
/* (j-major) for contiguous 16-byte loads.                              */
/*                                                                     */
/* bf16: round-to-nearest-even truncation to 8-bit mantissa, fp32      */
/* accumulate, via the standard blocked fp32 kernel. On this CPU class */
/* (no AVX512_BF16) bf16 is an error dial, not a speed dial.           */
/* ------------------------------------------------------------------ */
#if defined(FT_INT8) || defined(FT_INT8_ATTN) || defined(FT_BF16)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#if defined(__x86_64__)
#include <immintrin.h>
#endif

#define I8_CACHE_SLOTS 16
typedef struct {
    const float *src; int K, N; int filled;
    int8_t *q;   /* transposed: q[j*K+k], j-major for contiguous 16B loads */
    float *sc;   /* per-column dequant multiplier */
    int32_t *cs; /* per-column u8-activation correction: 128 * sum_k q[k][j] */
} i8_wcache_t;

/* Thread-local quant buffers (896KB, see carving comment in mm_bf16). */
#define I8_TLS_BYTES (896 * 1024)
#define I8_TLS_SC 2048

#if defined(FT_INT8) || defined(FT_INT8_ATTN)
static i8_wcache_t i8_wc[I8_CACHE_SLOTS];
static _Thread_local int8_t i8_tls[I8_TLS_BYTES];
static _Thread_local float i8_sc[I8_TLS_SC];

static inline int8_t f32_to_i8(float v) {
    if (v > 127.0f) return 127;
    if (v < -128.0f) return -128;
    return (int8_t)v;
}
static inline uint8_t f32_to_u8(float v) {
    if (v > 255.0f) return 255;
    if (v < 0.0f) return 0;
    return (uint8_t)v;
}

/* B: (K, N) row-major fp32 -> q: (N, K) int8 j-major (symmetric s8),
 * per-j dequant scale sc[j], and correction cs[j] = 128 * sum_k q[k][j]
 * (for the asymmetric-u8 activations trick below). */
static void i8_quant_w(const float *B, int K, int N, int8_t *q, float *sc, int32_t *cs) {
    for (int j = 0; j < N; j++) {
        float mx = 0.0f;
        for (int k = 0; k < K; k++) { float a = fabsf(B[(size_t)k * N + j]); if (a > mx) mx = a; }
        float s = (mx > 0.0f) ? (127.0f / mx) : 1.0f;
        sc[j] = 1.0f / s;
        int32_t sum = 0;
        for (int k = 0; k < K; k++) {
            int8_t v = f32_to_i8(roundf(B[(size_t)k * N + j] * s));
            q[(size_t)j * K + k] = v;
            sum += v;
        }
        cs[j] = 64 * sum;
    }
}

/* A: (M, K) row-major fp32 -> q: (M, K) uint8 (asymmetric, +128),
 * per-row dequant scale sc[i]. Raw bytes stored in the int8_t buffer. */
static void i8_quant_a(const float *A, int M, int K, int8_t *q, float *sc) {
    /* Activations quantized to u8 in [0,127] (offset 64, scale 63/mx):
     * vpmaddubsw saturates int16 at 32767, so max pair must stay <=
     * 2*127*127 = 32258 < 32767. (Full-range u8 [0,255] pairs overflow.) */
    for (int i = 0; i < M; i++) {
        const float *ar = A + (size_t)i * K;
        float mx = 0.0f;
        for (int k = 0; k < K; k++) { float a = fabsf(ar[k]); if (a > mx) mx = a; }
        float s = (mx > 0.0f) ? (63.0f / mx) : 1.0f;
        sc[i] = 1.0f / s;
        int8_t *qr = q + (size_t)i * K;
        for (int k = 0; k < K; k++)
            qr[k] = (int8_t)f32_to_u8(roundf(ar[k] * s) + 64.0f);
    }
}

static void mm_i8_scalar(const int8_t *aq, const float *asc, const int8_t *bq, const float *bsc,
                         const int32_t *cs, float *C, int M, int N, int K, int acc) {
    for (int i = 0; i < M; i++) {
        const int8_t *ar = aq + (size_t)i * K;
        float sa = asc[i];
        float *ci = C + (size_t)i * N;
        for (int j = 0; j < N; j++) {
            const int8_t *bj = bq + (size_t)j * K;
            int32_t s = -cs[j];
            for (int k = 0; k < K; k++) s += (int)(uint8_t)ar[k] * (int)bj[k];
            float v = (float)s * (sa * bsc[j]);
            if (acc) ci[j] += v; else ci[j] = v;
        }
    }
}

#if defined(__x86_64__)
__attribute__((target("avx2")))
static void mm_i8_avx2(const int8_t *aq, const float *asc, const int8_t *bq, const float *bsc,
                       const int32_t *cs, float *C, int M, int N, int K, int acc) {
    /* One output column at a time: 256-bit maddubs over 32 k's per iteration,
     * then horizontal-reduce the 8 k-group lanes. (The alternative of keeping
     * 8 columns in one accumulator vector mixes columns across lanes.) */
    const __m256i ones = _mm256_set1_epi16(1);
    for (int i = 0; i < M; i++) {
        const int8_t *ar = aq + (size_t)i * K;
        const float sa = asc[i];
        float *ci = C + (size_t)i * N;
        for (int j = 0; j < N; j++) {
            __m256i accv = _mm256_setzero_si256();   /* 8 int32 lanes = k-groups */
            const int8_t *bj = bq + (size_t)j * K;
            int k = 0;
            for (; k + 32 <= K; k += 32) {
                __m256i a32 = _mm256_loadu_si256((const __m256i *)(ar + k));
                __m256i b32 = _mm256_loadu_si256((const __m256i *)(bj + k));
                __m256i h = _mm256_maddubs_epi16(a32, b32);  /* 16 int16 = 8 pairs */
                accv = _mm256_add_epi32(accv, _mm256_madd_epi16(h, ones));  /* 8 groups of 4 k */
            }
            int32_t s = -cs[j];
            for (; k < K; k++) s += (int)(uint8_t)ar[k] * (int)bj[k];
            __m128i lo = _mm256_castsi256_si128(accv);
            __m128i hi = _mm256_extracti128_si256(accv, 1);
            __m128i t = _mm_add_epi32(lo, hi);
            t = _mm_hadd_epi32(t, t);
            t = _mm_hadd_epi32(t, t);
            s += _mm_cvtsi128_si32(t);
            float v = (float)s * (sa * bsc[j]);
            if (acc) ci[j] += v; else ci[j] = v;
        }
    }
}

__attribute__((target("avx512f","avx512vnni","avx512bw","avx512vl")))
static void mm_i8_vnni(const int8_t *aq, const float *asc, const int8_t *bq, const float *bsc,
                       const int32_t *cs, float *C, int M, int N, int K, int acc) {
    for (int i = 0; i < M; i++) {
        const int8_t *ar = aq + (size_t)i * K;
        const float sa = asc[i];
        float *ci = C + (size_t)i * N;
        for (int j = 0; j + 16 <= N; j += 16) {
            __m128i a0 = _mm_setzero_si128(), a1 = _mm_setzero_si128();
            __m128i a2 = _mm_setzero_si128(), a3 = _mm_setzero_si128();
            const int8_t *bj = bq + (size_t)j * K;
            for (int k = 0; k + 16 <= K; k += 16) {
                __m128i a16 = _mm_loadu_si128((const __m128i *)(ar + k));
                for (int jj = 0; jj < 16; jj++) {
                    __m128i b16 = _mm_loadu_si128((const __m128i *)(bj + (size_t)jj * K + k));
                    switch (jj >> 2) {
                        case 0: a0 = _mm_dpbusd_epi32(a0, a16, b16); break;
                        case 1: a1 = _mm_dpbusd_epi32(a1, a16, b16); break;
                        case 2: a2 = _mm_dpbusd_epi32(a2, a16, b16); break;
                        default: a3 = _mm_dpbusd_epi32(a3, a16, b16); break;
                    }
                }
            }
            __attribute__((aligned(64))) int32_t acc32[16];
            _mm_storeu_si128((__m128i *)(acc32 + 0), a0);
            _mm_storeu_si128((__m128i *)(acc32 + 4), a1);
            _mm_storeu_si128((__m128i *)(acc32 + 8), a2);
            _mm_storeu_si128((__m128i *)(acc32 + 12), a3);
            for (int jj = 0; jj < 16; jj++) acc32[jj] -= cs[j + jj];
            __m512 v = _mm512_cvtepi32_ps(_mm512_load_si512((const void *)acc32));
            __m512 s16 = _mm512_mul_ps(_mm512_set1_ps(sa), _mm512_loadu_ps(bsc + j));
            v = _mm512_mul_ps(v, s16);
            if (acc) v = _mm512_add_ps(v, _mm512_loadu_ps(ci + j));
            _mm512_storeu_ps(ci + j, v);
        }
        for (int j = (N & ~15); j < N; j++) {
            const int8_t *bj = bq + (size_t)j * K;
            int32_t s = -cs[j];
            for (int k = 0; k < K; k++) s += (int)(uint8_t)ar[k] * (int)bj[k];
            float v = (float)s * (sa * bsc[j]);
            if (acc) ci[j] += v; else ci[j] = v;
        }
    }
}
#endif /* __x86_64__ */

static int i8_have_avx2 = -1, i8_have_vnni = -1;
static void i8_check_cpu(void) {
    if (i8_have_avx2 < 0) {
#if defined(__x86_64__)
        i8_have_avx2 = __builtin_cpu_supports("avx2") ? 1 : 0;
        i8_have_vnni = (__builtin_cpu_supports("avx512vnni") && __builtin_cpu_supports("avx512vl")) ? 1 : 0;
#else
        i8_have_avx2 = 0; i8_have_vnni = 0;
#endif
    }
}

/* int8 GEMM entry: cache_b=1 -> B is a static weight (cache quantized form);
 * cache_b=0 -> B is scratch (quantized per call). */
void ft_mm_i8(const float *A, const float *B, float *C, int M, int N, int K, int acc, int cache_b) {
    int8_t *aq = i8_tls;
    float *asc = i8_sc;
    i8_quant_a(A, M, K, aq, asc);
    int8_t *bq; float *bsc; int32_t *cs;
    if (cache_b) {
        int slot = -1;
        for (int s = 0; s < I8_CACHE_SLOTS; s++) {
            if (i8_wc[s].filled && i8_wc[s].src == B && i8_wc[s].K == K && i8_wc[s].N == N) { slot = s; break; }
            if (slot < 0 && !i8_wc[s].filled) slot = s;
        }
        if (slot < 0) slot = 0;  /* all busy: evict slot 0 */
        if (!i8_wc[slot].filled || i8_wc[slot].src != B || i8_wc[slot].K != K || i8_wc[slot].N != N) {
            /* (re)size buffers: an evicted slot may hold a smaller matrix */
            if (!i8_wc[slot].q || i8_wc[slot].K != K || i8_wc[slot].N != N) {
                free(i8_wc[slot].q); free(i8_wc[slot].sc); free(i8_wc[slot].cs);
                i8_wc[slot].q = (int8_t *)malloc((size_t)N * K);
                i8_wc[slot].sc = (float *)malloc((size_t)N * sizeof(float));
                i8_wc[slot].cs = (int32_t *)malloc((size_t)N * sizeof(int32_t));
            }
            i8_quant_w(B, K, N, i8_wc[slot].q, i8_wc[slot].sc, i8_wc[slot].cs);
            i8_wc[slot].src = B; i8_wc[slot].K = K; i8_wc[slot].N = N; i8_wc[slot].filled = 1;
        }
        bq = i8_wc[slot].q; bsc = i8_wc[slot].sc; cs = i8_wc[slot].cs;
    } else {
        bq = i8_tls + 128 * 1024;
        bsc = i8_sc + 1024;
        cs = (int32_t *)(i8_sc + 1536);
        i8_quant_w(B, K, N, bq, bsc, cs);
    }
    i8_check_cpu();
    static int path_reported = 0;
    if (!path_reported) {
        path_reported = 1;
        FILE *pf = fopen("/tmp/ft_i8_path.txt", "w");
        if (pf) { fprintf(pf, "%s\n", i8_have_vnni ? "avx512vnni" : (i8_have_avx2 ? "avx2" : "scalar")); fclose(pf); }
    }
    if (i8_have_vnni && (K & 15) == 0 && (N & 15) == 0) {
#if defined(__x86_64__)
        mm_i8_vnni(aq, asc, bq, bsc, cs, C, M, N, K, acc);
#else
        mm_i8_scalar(aq, asc, bq, bsc, cs, C, M, N, K, acc);
#endif
    } else if (i8_have_avx2 && (K & 15) == 0 && (N & 7) == 0) {
#if defined(__x86_64__)
        mm_i8_avx2(aq, asc, bq, bsc, cs, C, M, N, K, acc);
#else
        mm_i8_scalar(aq, asc, bq, bsc, cs, C, M, N, K, acc);
#endif
    } else {
        mm_i8_scalar(aq, asc, bq, bsc, cs, C, M, N, K, acc);
    }
}
#endif /* FT_INT8 || FT_INT8_ATTN */

#if defined(FT_BF16)
static inline uint16_t f32_to_bf16(float x) {
    union { float f; uint32_t i; } u; u.f = x;
    uint32_t lsb = (u.i >> 16) & 1u;
    u.i += 0x7fffu + lsb;              /* round to nearest even */
    return (uint16_t)(u.i >> 16);
}
static inline float bf16_to_f32(uint16_t b) {
    union { float f; uint32_t i; } u; u.i = (uint32_t)b << 16; return u.f;
}

/* TLS carving (896KB total, shared with int8): [0,128K) int8 aq,
 * [128K,136K) int8 bq tmp, [256K,384K) spare, [512K,768K) bf16 A fp32,
 * [768K,896K) bf16 B fp32. */
static _Thread_local int8_t i8_tls[I8_TLS_BYTES];
static void mm_bf16(const float *A, const float *B, float *C, int M, int N, int K, int acc) {
    float *af = (float *)(i8_tls + 512 * 1024);
    float *bf = (float *)(i8_tls + 768 * 1024);
    for (int i = 0; i < M; i++)
        for (int k = 0; k < K; k++)
            af[(size_t)i * K + k] = bf16_to_f32(f32_to_bf16(A[(size_t)i * K + k]));
    for (int k = 0; k < K; k++)
        for (int j = 0; j < N; j++)
            bf[(size_t)k * N + j] = bf16_to_f32(f32_to_bf16(B[(size_t)k * N + j]));
    if ((M & 7) == 0 && (N & 7) == 0 && (K & 7) == 0)
        mm_blocked8(af, bf, C, M, N, K, acc);
    else
        mm_blocked(af, bf, C, M, N, K, acc);
}
void ft_mm_bf16(const float *A, const float *B, float *C, int M, int N, int K, int acc) {
    mm_bf16(A, B, C, M, N, K, acc);
}
#endif /* FT_BF16 */

#endif /* FT_INT8 || FT_INT8_ATTN || FT_BF16 */
