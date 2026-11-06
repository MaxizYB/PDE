# CDMC preliminary validation results v1

This document consolidates the latest reproducible FNO and HPM experiments and
the core same-backbone ablations. Errors are relative L2 field errors in percent
unless stated otherwise. K is the fixed number of online post-solver updates;
K=0 is the neural-operator proposal alone.

## 1. Evaluation protocol

The benchmark factorizes the pipeline as `fixed neural proposal -> fixed
numerical/CDMC action -> fixed budget K`. Within each dataset, the backbone
checkpoint and test split are held fixed. Timing is end-to-end online latency
per sample/frame under the corresponding accepted GPU protocol. HPM full
evaluations use the normalization statistics stored in the isolated training
checkpoint. The old fast-profile folders remain historical references and are
not merged with the full rows below.

## 2. Main FNO results

### Regular Darcy

| method | K | mean (%) | P95 (%) | time (ms/sample) |
|---|---:|---:|---:|---:|
| FNO | 0 | 0.92375 | 1.55633 | 0.13606 |
| FNO + numerical | 1 | 0.22889 | 0.53412 | 0.85229 |
| FNO + CDMC | 1 | 0.15087 | 0.30359 | 1.14355 |
| FNO + numerical | 2 | 0.05570 | 0.14656 | 1.18797 |
| FNO + CDMC | 2 | **0.03640** | **0.09267** | 1.45325 |

CDMC reduces the FNO K=0 error by 96.06% at K=2 and is 34.7% lower than the
same-budget numerical branch (0.03640% versus 0.05570%).

### Navier–Stokes rollout

| method | K | mean rollout error (%) | time (ms/frame) |
|---|---:|---:|---:|
| FNO | 0 | 9.47102 | 0.04505 |
| FNO + numerical Oseen | 1 | 9.00778 | 0.22124 |
| FNO + numerical Oseen | 10 | 8.33996 | 0.54142 |
| FNO + CDMC | 2 | 8.32555 | 0.39870 |
| FNO + CDMC | 5 | **6.39253** | 0.92830 |
| FNO + CDMC | 10 | **6.39253** | 0.62512 |

The causal CDMC closure reaches a lower rollout error than the Oseen-only branch
at the reported fixed budgets.

## 3. Main HPM results

### Heat Transfer — isolated full checkpoint

Training: 5000 epochs, 1849.19 s. The proposal and all corrector rows use the
same full checkpoint and saved full-training statistics.

| method | K | mean (%) | P95 (%) | time (ms/sample) |
|---|---:|---:|---:|---:|
| HPM | 0 | 0.0182628 | 0.0271139 | 0.279519 |
| HPM + fixed LBO | 1 | 0.0321577 | 0.0693022 | 0.279436 |
| HPM + direct corrector | 1 | 0.0182578 | 0.0271598 | 0.281224 |
| HPM + CDMC | 1 | **0.0148967** | **0.0193557** | 0.282294 |
| HPM + CDMC | 2 | **0.0148853** | **0.0192907** | 0.281970 |

CDMC lowers full-HPM mean error by 18.49% with about 0.0025 ms/sample online
overhead. K=1 and K=2 are effectively saturated, so the gain is not from an
unreported long iteration.

### Irregular Darcy — isolated full checkpoint

Training: 500 epochs, 2244.50 s. The mesh has 2290 nodes, 4338 triangles, and
2076 interior nodes. The graph hierarchy uses aggregate size 8, 69 HPM LBO
modes, and a 16-mode terminal complementary LBO space.

| method | K | mean (%) | P95 (%) | gradient (%) | time (ms/sample) |
|---|---:|---:|---:|---:|---:|
| HPM | 0 | 0.680571 | 1.06328 | 8.39019 | 0.372414 |
| flat LBO | 1 | 0.475395 | 0.755790 | 6.27438 | 0.541079 |
| graph V | 1 | 0.478821 | 0.779020 | 6.05591 | 0.505979 |
| graph V + terminal LBO/CDMC | 1 | **0.461659** | **0.742931** | **6.05002** | 0.516294 |
| flat LBO | 2 | 0.444212 | 0.704534 | 5.98393 | 0.624538 |
| graph V | 2 | 0.448460 | 0.713058 | 5.95936 | 0.549586 |
| graph V + terminal LBO/CDMC | 2 | **0.441846** | **0.704748** | **5.95822** | 0.646993 |

The full HPM proposal is much better than the previous fast checkpoint (0.6806%
versus 8.8702%). CDMC reduces it by 35.08% at K=2 and is slightly better than
the historical 0.4429% irregular-Darcy CDMC result. Dynamic strict Galerkin and
reusable setup agree to a maximum absolute field difference of `1.31e-6`.

## 4. Core ablation conclusions

| claim | direct evidence |
|---|---|
| Numerical action is necessary | HPM Heat direct corrector: 0.01826% -> 0.01826%; CDMC K=1: 0.01490% |
| CDMC is not a free field head | FNO Darcy K=2 CDMC: 0.03640%; numerical K=2: 0.05570% |
| Complementary space matters on irregular meshes | graph V K=2: 0.44846% -> graph V + terminal LBO: 0.44185% |
| Fixed budget yields an auditable Pareto curve | Darcy K=1/K=2 and NS K=2/K=5/K=10 rows |
| Bridge is mesh/operator consistent | irregular dynamic-vs-reuse max difference: `1.31e-6` |

The complete component matrix is in
`results/ablation_core_v1/CORE_ABLATION_SUMMARY.md`.

## 5. Artifact map

- FNO formal results: `results/experiments/fno_formal_accurate_20260922/`
- HPM Heat full: `results/hpm_full_heat_cdmc_aligned/`
- HPM Irregular Darcy full: `results/hpm_full_irregular_cdmc_aligned/`
- Core ablation bundle: `results/ablation_core_v1/`
- HPM provenance note: `results/HPM_HEAT_IRREGULAR_CURRENT_REPORT.md`
- Historical accepted v1 results: `CDMC/methods/v1/EXPERIMENTS_V1.md`

Every headline table has matching `raw.json`, `details.json`, and `summary.csv`
in its result directory, so the markdown numbers are traceable to the exact
backend report and configuration.
