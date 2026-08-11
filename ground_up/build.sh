#!/usr/bin/env bash
# build.sh — build all libft_*.so variants on the Colab node.
# Usage: bash ground_up/build.sh   (works from any cwd)
set -euo pipefail
cd "$(dirname "$0")"
CC="${CC:-cc}"
CFLAGS="-O3 -march=native -ffast-math -funroll-loops -fopenmp-simd -flto -fPIC -shared -Wall -Wextra -Wno-unused-function -Wno-unused-variable"
echo "== compiler: $($CC --version | head -1)"
echo "== nproc: $(nproc)"
echo "== build libft_v1.so (naive matmul, unfused layer ops)"
$CC $CFLAGS -DFT_KERNEL=1 -DFT_FUSED=0 -o libft_v1.so matmul.c transformer.c -lm
echo "== build libft_v2.so (8x8 blocked matmul, unfused layer ops)"
$CC $CFLAGS -DFT_KERNEL=2 -DFT_FUSED=0 -o libft_v2.so matmul.c transformer.c -lm
echo "== build libft_v3.so (8x8 blocked, fused layer ops)"
$CC $CFLAGS -DFT_KERNEL=2 -DFT_FUSED=1 -o libft_v3.so matmul.c transformer.c -lm
echo "== build libft_v4.so (blocked + OpenMP threaded matmul, fused ops)"
$CC $CFLAGS -fopenmp -DFT_KERNEL=2 -DFT_FUSED=1 -DFT_OPENMP=1 -o libft_v4.so matmul.c transformer.c -lm
echo "== build libft_v5.so (specialized 8x8 unrolled matmul, fused ops)"
$CC $CFLAGS -DFT_KERNEL=3 -DFT_FUSED=1 -o libft_v5.so matmul.c transformer.c -lm
echo "== build libft_v6.so (specialized 8x8 + OpenMP, fused ops)"
$CC $CFLAGS -fopenmp -DFT_KERNEL=3 -DFT_FUSED=1 -DFT_OPENMP=1 -o libft_v6.so matmul.c transformer.c -lm
echo "== build libft_v7.so (spec8x8 + fast-exp + ffn-via-matmul)"
$CC $CFLAGS -DFT_KERNEL=3 -DFT_FUSED=1 -DFT_FASTEXP=1 -DFT_FFN_MM=1 -o libft_v7.so matmul.c transformer.c -lm
echo "== build libft_v8.so (v7 + 4x16 tiles)"
$CC $CFLAGS -DFT_KERNEL=3 -DFT_FUSED=1 -DFT_FASTEXP=1 -DFT_FFN_MM=1 -DFT_TILE16=1 -o libft_v8.so matmul.c transformer.c -lm
echo "== build libft_v9.so (v8 + OpenMP)"
$CC $CFLAGS -fopenmp -DFT_KERNEL=3 -DFT_FUSED=1 -DFT_FASTEXP=1 -DFT_FFN_MM=1 -DFT_TILE16=1 -DFT_OPENMP=1 -o libft_v9.so matmul.c transformer.c -lm
echo "== build libft_v10.so (v9 + vectorizable fast-exp + auto-tile by M, +omp)"
$CC $CFLAGS -fopenmp -DFT_KERNEL=3 -DFT_FUSED=1 -DFT_FASTEXP=1 -DFT_FFN_MM=1 -DFT_TILE16=2 -DFT_OPENMP=1 -o libft_v10.so matmul.c transformer.c -lm
echo "== build libft_v11.so (v10 + 4x16 tiles always, +omp)"
$CC $CFLAGS -fopenmp -DFT_KERNEL=3 -DFT_FUSED=1 -DFT_FASTEXP=1 -DFT_FFN_MM=1 -DFT_TILE16=1 -DFT_OPENMP=1 -o libft_v11.so matmul.c transformer.c -lm
echo "== build libft_v12.so (v10 + 8x8 tiles always, +omp)"
$CC $CFLAGS -fopenmp -DFT_KERNEL=3 -DFT_FUSED=1 -DFT_FASTEXP=1 -DFT_FFN_MM=1 -DFT_TILE16=0 -DFT_OPENMP=1 -o libft_v12.so matmul.c transformer.c -lm
echo "== build libft_v14.so (v9 + blocked attention, +omp)"
$CC $CFLAGS -fopenmp -DFT_KERNEL=3 -DFT_FUSED=1 -DFT_FASTEXP=1 -DFT_FFN_MM=1 -DFT_TILE16=1 -DFT_ATTN_BLOCK=1 -DFT_OPENMP=1 -o libft_v14.so matmul.c transformer.c -lm
echo "== build libft.so (default champion: v14 config; wrapper reuses workspace/output)"
cp libft_v14.so libft.so
ls -la libft*.so
