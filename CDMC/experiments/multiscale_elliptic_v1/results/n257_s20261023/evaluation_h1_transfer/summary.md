# Multiscale elliptic: paired mechanism experiment

HANO coefficient law; regenerated same-grid P1-FEM targets. Not the original downsampled-label benchmark.

| Method | L2 % | Gradient % | Energy % | Flux % | Online ms | L2 reduction vs NO % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NO | 1.9665 | 40.4212 | 47.2866 | 64.5144 | 1.5026 | 0.00 |
| zero_MG_K1 | 3.79973 | 12.3765 | 11.2128 | 10.4357 | 18.6275 | -93.22 |
| zero_MG_K2 | 1.50563 | 2.17334 | 2.09022 | 2.05916 | 34.8216 | 23.44 |
| fixed_MG_K1 | 0.879893 | 2.64567 | 2.84941 | 3.28395 | 20.3509 | 55.26 |
| fixed_MG_K2 | 0.290161 | 0.47073 | 0.437671 | 0.450969 | 37.9698 | 85.24 |
| actual_K1 | 0.684328 | 2.08292 | 2.30987 | 2.75172 | 20.9259 | 65.20 |
| actual_K2 | 0.199863 | 0.347333 | 0.339554 | 0.376186 | 37.1816 | 89.84 |
| generic_K1 | 1.24457 | 4.18834 | 3.57782 | 4.41872 | 20.8254 | 36.71 |
| generic_K2 | 0.356817 | 0.715347 | 0.644849 | 0.747071 | 37.1843 | 81.86 |

GPU: NVIDIA GeForce RTX 4090; n=16. batch1 synchronized wallclock, device-resident inputs, 1 warmup, median repeats; online total reruns NO/build/setup/cycles; zero_MG skips NO.
One-time mesh plan: 5.018 s.

## Paired task-matching evidence

| Comparison | Mean L2 reduction % | Per-sample wins |
| --- | ---: | ---: |
| actual_K1 vs fixed_MG_K1 | 22.23 | 16/16 |
| actual_K1 vs generic_K1 | 45.01 | 16/16 |
| actual_K2 vs fixed_MG_K2 | 31.12 | 16/16 |
| actual_K2 vs generic_K2 | 43.99 | 16/16 |
