# HPM full validation: Heat Transfer and Irregular Darcy

Date: 2026-09-22. This report contains the latest isolated HPM checkpoints and
the aligned CDMC evaluations. The metrics and end-to-end timing follow the
accepted CDMC protocol; Irregular Darcy additionally records the current
mesh-consistent hierarchy configuration (69 HPM modes and aggregate size 8).

## Checkpoint provenance

The full checkpoints are isolated from the generic fast-profile directory and
are evaluated with the statistics saved inside the corresponding training
checkpoint. This prevents the earlier normalization mismatch (recomputing
statistics from the first 100 samples) from changing the reported HPM field.

| dataset | checkpoint | training budget | evaluation status |
|---|---|---:|---|
| Heat Transfer | `checkpoints_full/heat_transfer/.../checkpoint.pt` | 5000 epochs, 1849.19 s | complete |
| Irregular Darcy | `checkpoints_full/irregular_darcy/.../checkpoint.pt` | 500 epochs, 2244.50 s | complete |

## Heat Transfer

| method | K | mean error (%) | P95 (%) | time (ms/sample) | reduction vs HPM (%) |
|---|---:|---:|---:|---:|---:|
| HPM proposal | 0 | 0.0182628 | 0.0271139 | 0.279519 | 0.00 |
| numerical LBO affine | 1 | 0.4252510 | 0.5010460 | 0.002349 | -2228.50 |
| HPM + fixed LBO | 1 | 0.0321577 | 0.0693022 | 0.279436 | -76.08 |
| HPM + direct corrector | 1 | 0.0182578 | 0.0271598 | 0.281224 | 0.03 |
| HPM + CDMC | 1 | **0.0148967** | **0.0193557** | 0.282294 | **18.43** |
| HPM + CDMC | 2 | **0.0148853** | **0.0192907** | 0.281970 | **18.49** |

The aligned full HPM/CDMC result is in the same error range as the reported
LRSA Heat Transfer result (about 0.0149%). CDMC improves the HPM proposal by
18.5% while adding only about 0.0025 ms/sample.

## Irregular Darcy

This experiment uses the current irregular mesh (2290 nodes, 4338 triangles,
2076 interior nodes), a strict Galerkin graph hierarchy with aggregate size 8,
and 69 HPM LBO modes. The terminal complementary space has 16 LBO modes.

| method | K | mean error (%) | P95 (%) | gradient error (%) | end-to-end time (ms/sample) | reduction vs HPM (%) |
|---|---:|---:|---:|---:|---:|---:|
| HPM proposal | 0 | 0.680571 | 1.06328 | 8.39019 | 0.372414 | 0.00 |
| flat LBO | 1 | 0.475395 | 0.755790 | 6.27438 | 0.541079 | 30.15 |
| graph V | 1 | 0.478821 | 0.779020 | 6.05591 | 0.505979 | 29.64 |
| graph V + terminal LBO/CDMC | 1 | **0.461659** | **0.742931** | **6.05002** | 0.516294 | **32.17** |
| flat LBO | 2 | 0.444212 | 0.704534 | 5.98393 | 0.624538 | 34.73 |
| graph V | 2 | 0.448460 | 0.713058 | 5.95936 | 0.549586 | 34.11 |
| graph V + terminal LBO/CDMC | 2 | **0.441846** | **0.704748** | **5.95822** | 0.646993 | **35.08** |

The strict-Galerkin dynamic path agrees with the reusable-setup path to a
maximum absolute field difference of `1.31e-6`; therefore the reported graph
and CDMC rows are not an approximation caused by a stale coarse operator. The
terminal LBO complement lowers K=2 error from 0.44846% (graph V) to 0.44185%
and lowers K=1 error from 0.47882% to 0.46166%.

## Artifact locations

- `hpm_full_heat_cdmc_aligned/summary.md` and `details.json`
- `hpm_full_irregular_cdmc_aligned/summary.md` and `details.json`
- server-only full checkpoints:
  `artifacts/checkpoints_full/<task>/hpm/seed_20260921/`
- timing fix: `experiments/irregular_multilevel_unification_v1/fixed_hierarchy_preview.py`

The older `hpm_*_fast` folders remain as historical fast-budget references;
they are not mixed with the full rows above.
