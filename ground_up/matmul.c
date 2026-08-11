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
/* Dispatch.                                                           */
/* ------------------------------------------------------------------ */
void ft_mm(const float *A, const float *B, float *C,
           int M, int N, int K, int acc) {
#if FT_KERNEL == 1
    mm_naive(A, B, C, M, N, K, acc);
#else
    mm_blocked(A, B, C, M, N, K, acc);
#endif
}
