# inhomogeneous-lgcp-cnn

Code, data and figures for *Neural estimation of the latent-field parameters
of inhomogeneous log-Gaussian Cox processes with multiscale intensity
descriptors* (Romero & Melo), submitted to the *Journal of Statistical
Computation and Simulation*.

The method estimates the standard deviation `σ` and the correlation scale `ξ`
of the latent field of an inhomogeneous LGCP with covariates. It is evaluated
by simulation, compared with two-step INLA-SPDE and applied to the 2020
Colombian seismic catalogue.

## Model and method

The log-intensity is written in centred form

```
log Λ(s) = β₀ + Σ βₖ Zₖ(s) + Y(s) − σ²/2
```

with `Y` a zero-mean Gaussian field with Matérn covariance (ν = 1), standard
deviation `σ` and scale `ξ`.

Estimation has two steps:

1. An inhomogeneous Poisson fit (`spatstat::ppm`) gives the trend
   coefficients `β̂` and the trend intensity `λ̂(s)`.
2. A CNN maps `(D(r), β̂, N, f)` to `(σ, ξ)`. `D(r) = L(r) − r` is the
   centred inhomogeneous L curve and `f` is a bank of 43 intensity
   descriptors: 3 quadrat-dispersion statistics on 6 grids (k = 2 to 64) and
   5 kernel-density statistics at 5 bandwidths (10 to 160 km).

The network has four branches joined in a dense head (79 186 parameters):

| Input | Shape | Branch |
|---|---|---|
| `D(r)`, `dD/dr`, `r/rmax` | 513 × 3 | three 1D convolution blocks (dilation 1, 2, 4) with average pooling, dense 32 |
| `β̂` | 5 | dense 32, 16 |
| `N` | 1 | passed through |
| descriptors `f` | 43 | dense 64, 32 |

The targets are `sqrt(σ)` and `log(ξ)`, standardised. Predictions are the
average of 5 training replicates.

## Repository layout

```
scripts/       the pipeline, 01 to 10 (05 is not included)
data/          window, catalogue and covariate rasters
simulations/   the 200 simulation chunks, design_meta.rds, observed_inputs.rds
figures/       the figures
outputs/       everything the pipeline computes (not versioned)
```

## Pipeline

Run every script from the repository root, in order:

```bash
Rscript scripts/01_simulate_training_set.R
```

| Script | Does | Main outputs | Time |
|---|---|---|---|
| `01_simulate_training_set.R` | draws 20 000 designs, simulates one realisation each and computes the two-step inputs | `simulations/` | ~3.3 h, 8 cores |
| `02_train_cnn.R` | trains the CNN (5 seeds), test metrics, catalogue estimate, DCLF envelope, parametric bootstrap | `metrics.csv`, `estimates_observed.csv`, `bootstrap_ic.csv`, `cnn_pred_test.csv` | GPU |
| `03_fit_inla_spde.R` | builds and caches the INLA mesh used by the later scripts; also fits the catalogue with trend and field together | `inla_mesh_*.rds`, `inla_catalog_*.csv` | a few minutes |
| `04_intensity_map.R` | CNN plug-in intensity map of the catalogue | `cnn_map_*` | ~2 min |
| `06_sensitivity_mc.R` | repeats the inference on the events with Mw ≥ 1.92 (magnitude of completeness) and saves the 5-seed ensemble | `sens_mc_*`, `sens_mc_model_s*/` | ~17 min |
| `07_ablation_bank.R` | retrains the network without descriptors and with the 8 descriptors of a single scale (same split, scalers, head and seeds) | `ablation_*.csv` | ~18 min |
| `08_inla_two_step.R` | two-step INLA-SPDE of the catalogue (trend fixed at the ppm estimate), its map, the CNN map on the same grid and the Mw ≥ 1.92 subset | `inla2s_catalog_*.csv`, `inla2s_map_*.csv` | ~5 min |
| `09_inla_two_step_simulations.R` | two-step INLA-SPDE on 500 random test realisations, regenerated from their seeds, and comparison with the CNN variants of 07 | `inla2s_sim_*.csv` | ~67 min, 8 workers |
| `10_fixed_value_evaluation.R` | CNN and two-step INLA-SPDE at nine fixed values of `(σ, ξ)`, 50 realisations each | `fixed_*.csv` | ~70 min, 8 workers |

The 200 `sim_*.rds` chunks are included, so the pipeline can start at script
02. Scripts 09 and 10 cache each INLA fit and can be resumed.

## Figures

| File | Shows |
|---|---|
| `fig1_training_pipeline.pdf` | generation of one training realisation |
| `fig2_count_residuals.pdf` | Pearson count residuals of the first-order fit |
| `fig3_cnn_architecture.pdf` | network architecture |
| `fig4_observed_vs_simulated.pdf` | observed catalogue next to one simulated realisation (not in the manuscript) |
| `fig5_parameter_recovery.pdf` | true against estimated `(σ, ξ)` on the test set |
| `fig6_parametric_bootstrap.pdf` | bootstrap distribution of the catalogue estimate |
| `fig7_intensity_maps.pdf` | expected intensity, INLA-SPDE and CNN |
| `fig8_k_envelope.pdf` | simulation envelope of the inhomogeneous K function |
| `fig9_estimator_comparison.pdf` | CNN variants and INLA-SPDE on the same 420 test realisations |
| `fig10_fixed_value_boxplots.pdf` | CNN and INLA-SPDE at the nine fixed values |

## Main results

Test set, 2926 realisations (scripts 02 and 07):

| Estimator | σ bias | σ R² | ξ bias | ξ RMSE | ξ R² |
|---|---|---|---|---|---|
| CNN, no descriptors | −0.044 | 0.828 | −6.4 km | 28.4 km | 0.515 |
| CNN, single scale (8 descriptors) | −0.022 | 0.879 | −4.2 km | 27.3 km | 0.553 |
| CNN, multiscale bank (43 descriptors) | −0.028 | 0.881 | −3.4 km | 24.9 km | 0.628 |

Comparison with two-step INLA-SPDE (script 09): 500 random test
realisations; the 80 INLA-SPDE estimates outside the design support are
excluded. On the remaining 420, the R² is 0.895 (CNN) and 0.803 (INLA-SPDE)
for `σ`, and 0.638 and 0.149 for `ξ`. INLA-SPDE is more accurate for short
scales (ξ < 55 km) and underestimates long ones. At the nine fixed values
(script 10) the RMSE over the 450 realisations is 0.26 against 1.00 for `σ`
and 21.3 km against 32.7 km for `ξ`.

2020 catalogue (17 857 declustered events):

| Method | σ̂ [95 %] | ξ̂ km [95 %] |
|---|---|---|
| CNN, multiscale bank | 2.28 [1.90, 2.64] | 22.6 [15.8, 33.2] |
| Two-step INLA-SPDE | 1.95 [1.73, 2.21] | 37.3 [32.6, 42.9] |

CNN intervals: parametric bootstrap, 200 replicates. INLA-SPDE: posterior
credible intervals. The DCLF test does not reject the CNN model (p = 0.40).

## Requirements

R 4.5.2 with `spatstat` 3.3.3, `R-INLA` 24.11.25 and `keras` 2.16.0 on a
TensorFlow GPU backend, plus `sf`, `tidyverse`, `patchwork`, `pbmcapply` and
`Matrix`. Script 02 needs Keras 2.x in the virtualenv
`~/.virtualenvs/r-tensorflow` and stops if it finds Keras 3.

Reference machine: Ryzen 9 7950X (16 cores), 64 GB RAM, RTX 4090, Ubuntu
22.04.

## Seeds

| Seed | Used for |
|---|---|
| `42 + i` | training design, realisation `i` |
| `123` | train/validation/test split |
| `1000 + i` | the 5 training replicates |
| `1` to `99` | realisations from the fitted model |
| `10000 + i` | DCLF envelope |
| `20000 + i` | parametric bootstrap |
| `20260803`, `20261008` | the 500 test realisations of script 09 |
| `70000 + r`, `80000 + 1000k + 10r + attempt` | trend and pattern of replicate `r` at point `k` in script 10 |

Script 02 also sets `tensorflow::set_random_seed()` and
`enable_op_determinism()`.

## Data

The 2020 seismic catalogue is public and distributed by the Servicio
Geológico Colombiano (SGC). The covariate rasters (distance to reverse
faults, isostatic anomaly, topography, distance to volcanoes) are scaled to
[−1, 1].

## Citation

```bibtex
@unpublished{romero_melo_lgcp_cnn,
  author = {Romero, Jason and Melo, Carlos},
  title  = {Neural estimation of the latent-field parameters of inhomogeneous
            log-{G}aussian {C}ox processes with multiscale intensity descriptors},
  note   = {Submitted to the Journal of Statistical Computation and Simulation},
  year   = {2026}
}
```

## License

MIT (see `LICENSE`). The seismic catalogue is redistributed under the terms
of its publisher, the Servicio Geológico Colombiano.
