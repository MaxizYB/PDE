# Multiscale elliptic: paired mechanism experiment

HANO coefficient law; regenerated same-grid P1-FEM targets. Not the original downsampled-label benchmark.

| Method | L2 % | Gradient % | Energy % | Flux % | Online ms | L2 reduction vs NO % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NO | 1.9711 | 44.3256 | 53.7502 | 76.5044 | 1.4832 | 0.00 |
| zero_MG_K1 | 3.75146 | 12.1866 | 11.0575 | 10.2805 | 18.5268 | -90.32 |
| zero_MG_K2 | 1.47078 | 2.13013 | 2.05526 | 2.02572 | 34.7338 | 25.38 |
| zero_MG_K4 | 0.103147 | 0.133891 | 0.130766 | 0.131901 | 67.0491 | 94.77 |
| fixed_MG_K1 | 0.884498 | 2.93897 | 3.12153 | 3.68408 | 20.2184 | 55.13 |
| fixed_MG_K2 | 0.306826 | 0.499562 | 0.466051 | 0.495349 | 36.4740 | 84.43 |
| fixed_MG_K4 | 0.0216914 | 0.0311911 | 0.0288047 | 0.0309966 | 68.9314 | 98.90 |
| actual_K1 | 0.711625 | 2.46779 | 2.69055 | 3.30162 | 20.7192 | 63.90 |
| actual_K2 | 0.242211 | 0.590009 | 0.5827 | 0.718383 | 36.9609 | 87.71 |
| actual_K4 | 0.0480358 | 0.263812 | 0.249084 | 0.307382 | 69.4441 | 97.56 |
| generic_K1 | 1.15276 | 4.43401 | 3.77385 | 4.97204 | 20.7475 | 41.52 |
| generic_K2 | 0.341506 | 0.726499 | 0.67717 | 0.853406 | 37.0624 | 82.67 |
| generic_K4 | 0.0208215 | 0.0549277 | 0.0831025 | 0.129712 | 69.4329 | 98.94 |

GPU: NVIDIA GeForce RTX 4090; n=32. batch1 synchronized wallclock, device-resident inputs, 1 warmup, median repeats; online total reruns NO/build/setup/cycles; zero_MG skips NO.
One-time mesh plan: 5.155 s.

## Paired task-matching evidence

| Comparison | Mean L2 reduction % | Per-sample wins |
| --- | ---: | ---: |
| actual_K1 vs fixed_MG_K1 | 19.54 | 30/32 |
| actual_K1 vs generic_K1 | 38.27 | 30/32 |
| actual_K2 vs fixed_MG_K2 | 21.06 | 30/32 |
| actual_K2 vs generic_K2 | 29.08 | 31/32 |
| actual_K4 vs fixed_MG_K4 | -121.45 | 30/32 |
| actual_K4 vs generic_K4 | -130.70 | 30/32 |

## Mature numerical baseline (CPU)

CPU PyAMG 5.3.0, smoothed aggregation; symmetric strength; default symmetric smoothing

| Method | L2 % | Gradient % | Solve ms | Full CPU ms |
| --- | ---: | ---: | ---: | ---: |
| zero_AMG_K1 | 55.277 | 60.4948 | 7.0412 | 250.2708 |
| zero_AMG_K2 | 33.79 | 39.3553 | 13.5313 | 256.7608 |
| zero_AMG_K4 | 13.924 | 17.8781 | 26.4508 | 269.6804 |
| zero_AMG_CG_tol1e-10 | 4.02785e-06 | 0.000228597 | 213.4897 | 456.7193 |
| NO_AMG_K1 | 1.29293 | 8.54759 | 7.1018 | 250.3314 |
| NO_AMG_K2 | 0.892946 | 3.16786 | 13.5863 | 256.8159 |
| NO_AMG_K4 | 0.379908 | 0.816716 | 26.3890 | 269.6185 |
| NO_AMG_CG_tol1e-10 | 4.02785e-06 | 0.000228597 | 201.0310 | 444.2605 |

CPU baseline, not a controlled same-hardware GPU speedup comparison; full coefficient-dependent AMG setup charged separately

## Mature numerical baseline (CPU)

CPU PyAMG 5.3.0, adaptive smoothed aggregation; 2 candidates, 5 candidate iterations, 1 improvement; classical strength theta=.25

| Method | L2 % | Gradient % | Solve ms | Full CPU ms |
| --- | ---: | ---: | ---: | ---: |
| zero_AMG_K1 | 8.7241 | 13.8473 | 17.4206 | 1927.2797 |
| zero_AMG_K2 | 2.59726 | 4.8738 | 34.0577 | 1943.9168 |
| zero_AMG_K4 | 0.388043 | 0.793467 | 67.3020 | 1977.1611 |
| zero_AMG_CG_tol1e-10 | 4.02785e-06 | 0.000228597 | 286.8236 | 2196.6827 |
| NO_AMG_K1 | 0.272183 | 2.40268 | 17.4189 | 1927.2780 |
| NO_AMG_K2 | 0.0841384 | 0.416552 | 34.0542 | 1943.9133 |
| NO_AMG_K4 | 0.0115832 | 0.0391897 | 67.5167 | 1977.3758 |
| NO_AMG_CG_tol1e-10 | 4.02785e-06 | 0.000228597 | 270.2904 | 2180.1495 |

CPU baseline, not a controlled same-hardware GPU speedup comparison; full coefficient-dependent AMG setup charged separately

## Mature numerical baseline (CPU)

CPU PyAMG 5.3.0, classical Ruge-Stuben, default strength theta=0.25 and symmetric Gauss-Seidel

| Method | L2 % | Gradient % | Solve ms | Full CPU ms |
| --- | ---: | ---: | ---: | ---: |
| zero_AMG_K1 | 70.4374 | 70.6856 | 7.6378 | 141.7551 |
| zero_AMG_K2 | 51.2205 | 52.0183 | 14.4400 | 148.5573 |
| zero_AMG_K4 | 28.4648 | 29.6194 | 28.0859 | 162.2031 |
| zero_AMG_CG_tol1e-10 | 4.02785e-06 | 0.000228597 | 201.3359 | 335.4532 |
| NO_AMG_K1 | 0.94709 | 2.78317 | 7.5242 | 141.6415 |
| NO_AMG_K2 | 0.643482 | 1.02262 | 14.3916 | 148.5089 |
| NO_AMG_K4 | 0.335086 | 0.415322 | 28.0944 | 162.2117 |
| NO_AMG_CG_tol1e-10 | 4.02785e-06 | 0.000228597 | 191.1075 | 325.2248 |

CPU baseline, not a controlled same-hardware GPU speedup comparison; full coefficient-dependent AMG setup charged separately

## Counterfactual error-direction diagnostic

Same held-out operators; synthetic initial errors matched to NO error amplitude. This is an offline diagnostic, not a predictor.

| Corrector | Actual NO start, K2 L2 % | Generic-probe start, K2 L2 % |
| --- | ---: | ---: |
| fixed_MG | 0.306826 | 0.0780806 |
| actual | 0.242211 | 0.0797345 |
| generic | 0.341506 | 0.055295 |

## Train-calibrated scalar relaxation control

The global gate 1.217080 is the mean network output over 48 training samples. No test tuning.

| Method | L2 % | Gradient % |
| --- | ---: | ---: |
| constant_K1 | 0.723537 | 2.35121 |
| constant_K2 | 0.233907 | 0.425482 |
| constant_K4 | 0.0257824 | 0.0900205 |

This control is more accurate than the graph setup at K2 in this run: the Actual-vs-Generic gain alone does not establish a spatial-setup advantage.

## Smoother stability diagnosis

| Test index (zero-based) | Max fixed Rayleigh | Max learned Rayleigh | Zero-step cycles |
| --- | ---: | ---: | --- |
| 0 | 1.9631 | 2.3054 | [] |
| 30 | 2.3480 | 3.0192 | [2, 3, 4] |
| 22 | 2.1641 | 2.6848 | [4] |

Rayleigh value >2 proves an expanding Jacobi eigenmode; <=2 alone is not a certified upper bound

## Discretization audit

Coarse truth versus restricted 513-grid truth (first held-out parameter vectors): 9.4286%, 8.9571%, 9.5097%.
The table measures algebraic accuracy on the stored P1 system; continuum refinement error is separate.
