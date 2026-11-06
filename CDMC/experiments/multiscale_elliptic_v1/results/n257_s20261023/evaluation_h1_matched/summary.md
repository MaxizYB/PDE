# Multiscale elliptic: paired mechanism experiment

HANO coefficient law; regenerated same-grid P1-FEM targets. Not the original downsampled-label benchmark.

| Method | L2 % | Gradient % | Energy % | Flux % | Online ms | L2 reduction vs NO % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NO | 1.92358 | 40.6274 | 47.2164 | 63.8289 | 1.4766 | 0.00 |
| zero_MG_K1 | 3.75145 | 12.1866 | 11.0575 | 10.2805 | 18.1723 | -95.02 |
| zero_MG_K2 | 1.47075 | 2.13011 | 2.05523 | 2.02569 | 33.9580 | 23.54 |
| zero_MG_K4 | 0.103176 | 0.133901 | 0.13079 | 0.131925 | 65.5062 | 94.64 |
| fixed_MG_K1 | 0.809786 | 2.596 | 2.80666 | 3.25014 | 19.8340 | 57.90 |
| fixed_MG_K2 | 0.270543 | 0.453815 | 0.420661 | 0.429338 | 35.6756 | 85.94 |
| fixed_MG_K4 | 0.0191828 | 0.0282655 | 0.0256406 | 0.0259883 | 67.3678 | 99.00 |
| h1_actual_K1 | 0.667307 | 2.2565 | 2.46591 | 3.01972 | 20.3244 | 65.31 |
| h1_actual_K2 | 0.227513 | 0.616262 | 0.571777 | 0.646739 | 36.2247 | 88.17 |
| h1_actual_K4 | 0.0560653 | 0.318895 | 0.287234 | 0.343075 | 67.9217 | 97.09 |

GPU: NVIDIA GeForce RTX 4090; n=32. batch1 synchronized wallclock, device-resident inputs, 1 warmup, median repeats; online total reruns NO/build/setup/cycles; zero_MG skips NO.
One-time mesh plan: 5.164 s.

## Paired task-matching evidence

| Comparison | Mean L2 reduction % | Per-sample wins |
| --- | ---: | ---: |
| h1_actual_K1 vs fixed_MG_K1 | 17.59 | 30/32 |
| h1_actual_K2 vs fixed_MG_K2 | 15.90 | 30/32 |
| h1_actual_K4 vs fixed_MG_K4 | -192.27 | 29/32 |

## Discretization audit

Coarse truth versus restricted 513-grid truth (first held-out parameter vectors): 9.4286%, 8.9571%, 9.5097%.
The table measures algebraic accuracy on the stored P1 system; continuum refinement error is separate.
