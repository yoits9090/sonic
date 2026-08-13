# Pareto frontier — latency vs error (node: ft-node-2)

Generated: 2026-08-13 02:37:08 by evals/pareto_analysis.py
Champion impl: `c_fp32_e6` (fp32, exp-degree 6; err ~1.1e-6). Budgets: max_abs_err < 0.001, 0.01 vs f64 ref.
Impls with data: ['c_bf16_e6', 'c_fp32_e3', 'c_fp32_e4', 'c_fp32_e6', 'c_int8_attn_e6', 'c_int8_e6']
Convention: min median_us across attempts per (impl, cell).

## Budget 0.001 — fastest passing impl per cell

| cell | champion us | champion err | winner | winner us | winner err | speedup vs champ | sub-ms |
|---|---|---|---|---|---|---|---|
| tiny_b1_s16 | 34.3 | 1.12e-06 | `c_fp32_e6` | 34.3 | 1.12e-06 | 1.00x | yes |
| tiny_b1_s32 | 64.9 | 1.17e-06 | `c_fp32_e6` | 64.9 | 1.17e-06 | 1.00x | yes |
| tiny_b1_s64 | 124.5 | 1.63e-06 | `c_fp32_e6` | 124.5 | 1.63e-06 | 1.00x | yes |
| tiny_b4_s16 | 115.1 | 1.12e-06 | `c_fp32_e6` | 115.1 | 1.12e-06 | 1.00x | yes |
| tiny_b4_s32 | 194.9 | 1.86e-06 | `c_fp32_e6` | 194.9 | 1.86e-06 | 1.00x | yes |
| tiny_b4_s64 | 405.7 | 1.71e-06 | `c_fp32_e6` | 405.7 | 1.71e-06 | 1.00x | yes |
| tiny_b16_s16 | 371.1 | 1.76e-06 | `c_fp32_e6` | 371.1 | 1.76e-06 | 1.00x | yes |
| tiny_b16_s32 | 729.8 | 2.28e-06 | `c_fp32_e6` | 729.8 | 2.28e-06 | 1.00x | yes |
| tiny_b16_s64 | 1602.1 | 2.28e-06 | `c_fp32_e6` | 1602.1 | 2.28e-06 | 1.00x | no |
| default_b1_s16 | 178.6 | 1.17e-06 | `c_fp32_e3` | 177.2 | 1.67e-04 | 1.01x | yes |
| default_b1_s32 | 266.4 | 1.17e-06 | `c_fp32_e3` | 264.4 | 1.67e-04 | 1.01x | yes |
| default_b1_s64 | 553.7 | 1.37e-06 | `c_fp32_e3` | 528.6 | 2.35e-04 | 1.05x | yes |
| default_b4_s16 | 493.1 | 1.71e-06 | `c_fp32_e6` | 493.1 | 1.71e-06 | 1.00x | yes |
| default_b4_s32 | 966.4 | 1.47e-06 | `c_fp32_e6` | 966.4 | 1.47e-06 | 1.00x | yes |
| default_b4_s64 | 2048.3 | 1.55e-06 | `c_fp32_e6` | 2048.3 | 1.55e-06 | 1.00x | no |
| default_b16_s16 | 1919.3 | 1.71e-06 | `c_fp32_e6` | 1919.3 | 1.71e-06 | 1.00x | no |
| default_b16_s32 | 3975.0 | 1.68e-06 | `c_fp32_e3` | 3953.8 | 3.40e-04 | 1.01x | no |
| default_b16_s64 | 8550.1 | 1.74e-06 | `c_fp32_e6` | 8550.1 | 1.74e-06 | 1.00x | no |

**Budget 0.001 score: 13/18 cells sub-ms with a passing impl.**

## Budget 0.01 — fastest passing impl per cell

| cell | champion us | champion err | winner | winner us | winner err | speedup vs champ | sub-ms |
|---|---|---|---|---|---|---|---|
| tiny_b1_s16 | 34.3 | 1.12e-06 | `c_fp32_e6` | 34.3 | 1.12e-06 | 1.00x | yes |
| tiny_b1_s32 | 64.9 | 1.17e-06 | `c_fp32_e6` | 64.9 | 1.17e-06 | 1.00x | yes |
| tiny_b1_s64 | 124.5 | 1.63e-06 | `c_fp32_e6` | 124.5 | 1.63e-06 | 1.00x | yes |
| tiny_b4_s16 | 115.1 | 1.12e-06 | `c_fp32_e6` | 115.1 | 1.12e-06 | 1.00x | yes |
| tiny_b4_s32 | 194.9 | 1.86e-06 | `c_fp32_e6` | 194.9 | 1.86e-06 | 1.00x | yes |
| tiny_b4_s64 | 405.7 | 1.71e-06 | `c_fp32_e6` | 405.7 | 1.71e-06 | 1.00x | yes |
| tiny_b16_s16 | 371.1 | 1.76e-06 | `c_fp32_e6` | 371.1 | 1.76e-06 | 1.00x | yes |
| tiny_b16_s32 | 729.8 | 2.28e-06 | `c_fp32_e6` | 729.8 | 2.28e-06 | 1.00x | yes |
| tiny_b16_s64 | 1602.1 | 2.28e-06 | `c_fp32_e6` | 1602.1 | 2.28e-06 | 1.00x | no |
| default_b1_s16 | 178.6 | 1.17e-06 | `c_fp32_e3` | 177.2 | 1.67e-04 | 1.01x | yes |
| default_b1_s32 | 266.4 | 1.17e-06 | `c_fp32_e3` | 264.4 | 1.67e-04 | 1.01x | yes |
| default_b1_s64 | 553.7 | 1.37e-06 | `c_fp32_e3` | 528.6 | 2.35e-04 | 1.05x | yes |
| default_b4_s16 | 493.1 | 1.71e-06 | `c_fp32_e6` | 493.1 | 1.71e-06 | 1.00x | yes |
| default_b4_s32 | 966.4 | 1.47e-06 | `c_fp32_e6` | 966.4 | 1.47e-06 | 1.00x | yes |
| default_b4_s64 | 2048.3 | 1.55e-06 | `c_fp32_e6` | 2048.3 | 1.55e-06 | 1.00x | no |
| default_b16_s16 | 1919.3 | 1.71e-06 | `c_fp32_e6` | 1919.3 | 1.71e-06 | 1.00x | no |
| default_b16_s32 | 3975.0 | 1.68e-06 | `c_fp32_e3` | 3953.8 | 3.40e-04 | 1.01x | no |
| default_b16_s64 | 8550.1 | 1.74e-06 | `c_fp32_e6` | 8550.1 | 1.74e-06 | 1.00x | no |

**Budget 0.01 score: 13/18 cells sub-ms with a passing impl.**

## Frontier sets per cell (non-dominated points, all budgets overlaid)

| cell | frontier (err -> impl @ us) |
|---|---|
| tiny_b1_s16 | 1.1e-06 c_fp32_e6@34us |
| tiny_b1_s32 | 1.2e-06 c_fp32_e6@65us |
| tiny_b1_s64 | 1.6e-06 c_fp32_e6@125us |
| tiny_b4_s16 | 1.1e-06 c_fp32_e6@115us |
| tiny_b4_s32 | 1.9e-06 c_fp32_e6@195us |
| tiny_b4_s64 | 1.7e-06 c_fp32_e6@406us |
| tiny_b16_s16 | 1.8e-06 c_fp32_e6@371us |
| tiny_b16_s32 | 2.3e-06 c_fp32_e6@730us |
| tiny_b16_s64 | 2.3e-06 c_fp32_e6@1602us |
| default_b1_s16 | 1.2e-06 c_fp32_e6@179us, 1.7e-04 c_fp32_e3@177us |
| default_b1_s32 | 1.2e-06 c_fp32_e6@266us, 1.7e-04 c_fp32_e3@264us |
| default_b1_s64 | 1.4e-06 c_fp32_e6@554us, 2.3e-04 c_fp32_e3@529us |
| default_b4_s16 | 1.7e-06 c_fp32_e6@493us |
| default_b4_s32 | 1.5e-06 c_fp32_e6@966us |
| default_b4_s64 | 1.6e-06 c_fp32_e6@2048us |
| default_b16_s16 | 1.7e-06 c_fp32_e6@1919us |
| default_b16_s32 | 1.7e-06 c_fp32_e6@3975us, 3.4e-04 c_fp32_e3@3954us |
| default_b16_s64 | 1.7e-06 c_fp32_e6@8550us |

## Notes
- `c_fp32_e6` is the incumbent v14 champion (alias of `c_ground_up`/libft.so);
  its row is the reference point for speedups.
- Frontier excludes impls whose run errored (median_us non-finite).
- Dominated points are omitted from frontier sets (strict: p.err<=q.err AND p.med<q.med).