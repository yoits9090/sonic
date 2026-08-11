#!/usr/bin/env bash
# build.sh — build all libft_*.so variants on the Colab node.
# Usage: bash ground_up/build.sh   (works from any cwd)
set -euo pipefail
cd "$(dirname "$0")"
CC="${CC:-cc}"
CFLAGS="-O3 -march=native -ffast-math -flto -fPIC -shared -Wall -Wextra -Wno-unused-function -Wno-unused-variable"
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
echo "== build libft_v5.so (same kernels as v3; wrapper reuses workspace/output)"
$CC $CFLAGS -DFT_KERNEL=2 -DFT_FUSED=1 -o libft_v5.so matmul.c transformer.c -lm
echo "== build libft.so (default alias of v5)"
cp libft_v5.so libft.so
ls -la libft*.so
