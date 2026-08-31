# CNN with multiscale intensity features, inhomogeneous univariate LGCP,
# Colombian seismicity 2020.
#
# Consumes the training set built by 01_simulate_training_set.R and fits one
# model: a 1D convolutional branch on the centred inhomogeneous L curve, plus
# dense branches on the stage-1 ppm coefficients, the point count, and the 43
# multiscale intensity descriptors. Target: theta = (sigma, xi).
#
# Inputs per realisation:
#   S2     = L(r) - r        513 x 3 channels [D(r), dD/dr, r/rmax]
#   S1     = beta_hat        5 ppm coefficients
#   N      = point count     1
#   I-feat = descriptor bank 43 (3 quadrat metrics x 6 sizes + 5 KDE metrics x 5 bandwidths)
#
# The intercept is not sampled: script 01 solves it from the expected count, so
# META$beta_range$lo is a scalar covering the slopes only, and the number of
# coefficients comes from META$beta_names. Units are metres over a
# 1410 x 1899 km window; xi is reported in km where that helps.
#
# Reproducibility: every source of randomness is seeded explicitly - the split
# (123), the five training replicates (1000+i), the resimulation of section 9
# (1..99), the envelope (10000+i) and the bootstrap (20000+i) - and TensorFlow
# op determinism is enabled below. From a clean R session two runs give the same
# numbers.
#
# Outputs, in figures_sismos_multiescala_matern/: metrics.csv, cnn_pred_test.csv,
# estimates_observed.csv, bootstrap_ic.csv, envolvente.csv, dclf_pvalue.csv,
# the LaTeX tables, the figures, and model_CNN_Ifeat/ with scalers.rds.
#
# Usage:  Rscript scripts/02_train_cnn.R   (from the repository root)


library(reticulate)
use_virtualenv("~/.virtualenvs/r-tensorflow", required = TRUE)  # before library(keras)

library(keras)
Sys.setenv(TF_USE_LEGACY_KERAS = "0")

library(tidyverse)
library(patchwork)
suppressMessages(library(spatstat))   # sections 9-11 only

local({
  kv <- as.character(reticulate::py_get_attr(reticulate::import("keras"), "__version__"))
  if (!startsWith(kv, "2.")) stop("Keras ", kv, " detectado; este script necesita Keras 2.x.")
  message("Keras backend OK: ", kv)
})

set.seed(123)
tensorflow::set_random_seed(123)

# set_random_seed() seeds the generators but is not enough on its own:
# convolutions and reductions can accumulate in nondeterministic order, and two
# runs with the same seed then differ in the last digits, which after 200 epochs
# are no longer the last. Available from TF 2.9.
local({
  ok <- tryCatch({ tensorflow::tf$config$experimental$enable_op_determinism(); TRUE },
                 error = function(e) FALSE)
  message(if (ok) "Determinismo de operaciones TF: ACTIVADO"
          else "AVISO: enable_op_determinism() no disponible; la corrida puede NO ser reproducible bit a bit.")
})

stopifnot(dir.exists("data"), dir.exists("Results_simulation_sismos_multiescala_matern"))

SIM_DIR <- "Results_simulation_sismos_multiescala_matern"
fig_dir <- "figures_sismos_multiescala_matern"
dir.create(fig_dir, showWarnings = FALSE)

# Script 01 generates the field with rGRFmatern(nu = 1). Sections 9, 10 and 13
# resimulate from theta_hat and must use the same covariance; with an
# exponential the fitted realisations, the DCLF envelope and the bootstrap
# interval would measure a different model, and xi would not even mean the same
# thing. Checked against META$nu_matern when the design is loaded.
NU_MATERN <- 1

# ---- 0. Configuration -------------------------------------------------------

FRAC_TRAIN <- 0.70
FRAC_VAL   <- 0.15          # el resto va a test

# beta_hat is not clipped: the covariates are continuous so there is no
# separation, the intercept has no sampling box to clip against, and its spread
# under clustering is itself informative about sigma. Standardisation and
# CLIP_SD already control the tails.

# Dropping realisations with a degenerate first-order fit is not a neutral
# filter (it removes high sigma and low counts), so it is off by default.
DROP_DEGENERATE <- FALSE

# Safety clip on the standardised inputs, in standard deviations.
CLIP_SD <- 10

# A second evaluation on a narrow band of xi, for two reasons. Methodological:
# a MAPE computed on a narrow range does not distinguish a model with skill from
# a constant predictor. Substantive: the band brackets the plausible value, the
# minimum-contrast reference at 39.2 km. The band is not excluded from training;
# it is taken from the test split itself. Note it is not centred on 39.2 km,
# which falls at the 21st percentile of the band.
XI_NARROW <- c(35000, 55000)   # m

# "asinh" compresses the tails without destroying the relative amplitude that
# encodes sigma; "none" standardises the raw curve.
S2_TRANSFORM <- "none"   # "asinh" | "none"

# S2 branch, final architecture:
#   input (513, 3) -> conv 64 k=7 d=1 -> BN -> ReLU -> avgpool 5
#                  -> conv 64 k=7 d=2 -> BN -> ReLU -> avgpool 5
#                  -> conv 32 k=7 d=4 -> BN -> ReLU
#                  -> flatten (20 x 32 = 640) -> dense 32 -> BN
# Receptive-field details are in the header of build_S2_branch().
S2_CHANNELS     <- 3L   # [D(r), dD/dr, r/rmax]
S2_LAST_FILTERS <- 32L
S2_PROJECT      <- 32L  # closing dense projection, balances the concatenate

EPOCHS     <- 200
BATCH_SIZE <- 64
LR         <- 1e-3
DROPOUT    <- 0.2

# A single run is not a stable result; predictions are averaged over N_SEEDS.
N_SEEDS <- 5

PAR_COLS  <- c("sigma", "xi")

# Which descriptors feed the network. Names come from design_meta, not from a
# hard-coded list, so a change to the bank in script 01 propagates here.
# Script 01 extracts only the 43 bank descriptors, so "bank" is the only option
# for this design; the guard below rejects the others explicitly.
#   "bank"   43 multiscale descriptors
#   "legacy"  8 single-scale descriptors
#   "all"    51
FEAT_SET <- "bank"

# Units of the I-feat dense branch. c(32, 16) was sized for 8 inputs; with 43 it
# compresses instead of expanding, which would confound "the bank adds nothing"
# with "the branch does not fit". Use c(32, 16) for the ablation.
FEAT_UNITS <- c(64, 32)

M_FEAT <- "CNN + I-feat multiescala"

# ---- 1. Load ----------------------------------------------------------------

META <- readRDS(file.path(SIM_DIR, "design_meta.rds"))
NR   <- length(META$r)
# Not length(META$beta_range$lo): that field is a scalar covering the slopes,
# because the intercept is solved rather than sampled.
NB   <- length(META$beta_names)

# Guards. Pointing SIM_DIR at an older design would otherwise fail 200 chunks
# later inside vapply with an unhelpful message about lengths.
if (is.null(META$feat_bank))
  stop(sprintf(paste0("%s no tiene banco multiescala (design_meta sin feat_bank).\n",
                      "  Run 01_simulate_training_set.R first,\n",
                      "  or use the single-scale design if that is what you wanted."),
               SIM_DIR))

# Section 9 resimulates from theta_hat and must use the same covariance that
# generated the training set, or the fitted realisations, the DCLF test and the
# bootstrap interval all measure a different model.
if (is.null(META$field_model) || is.null(META$nu_matern))
  stop(sprintf(paste0("%s/design_meta.rds no declara el campo latente.\n",
                      "  Regenerate it with 01_simulate_training_set.R."),
               SIM_DIR))
if (!identical(META$field_model, "matern") || META$nu_matern != NU_MATERN)
  stop(sprintf("Campo latente incompatible: el diseño es %s(nu = %g) y este script asume matern(nu = %g).",
               META$field_model, META$nu_matern, NU_MATERN))

if (FEAT_SET != "bank" && is.null(META$feat_legacy))
  stop(sprintf(paste0("FEAT_SET = '%s' pero %s no tiene features legacy.\n",
                      "  Script 01 extracts only the 43 bank descriptors.\n",
                      "  Use FEAT_SET = 'bank', or the single-scale design for the ablation."),
               FEAT_SET, SIM_DIR))

FEAT_COLS <- switch(FEAT_SET,
                    bank   = META$feat_bank,
                    legacy = META$feat_legacy,
                    all    = c(META$feat_legacy, META$feat_bank),
                    stop("FEAT_SET debe ser 'bank', 'legacy' o 'all': ", FEAT_SET))

message(sprintf("Diseño: %d valores de r (rmax %.0f km) | %d coeficientes beta | %d features (%s)",
                NR, max(META$r) / 1000, NB, length(FEAT_COLS), FEAT_SET))
message(sprintf("Banco: cuadrantes k = %s | KDE bw = %s km (malla %s)",
                paste(META$nquad_bank, collapse = ", "),
                paste(META$kde_bw_bank / 1000, collapse = ", "),
                paste(META$kde_dimyx_bank, collapse = "x")))
message(sprintf("Prior: sigma U[%.1f, %.1f] | xi U[%.0f, %.0f] km | pendientes U[%g, %g]",
                META$sigma_range[1], META$sigma_range[2],
                META$xi_range[1] / 1000, META$xi_range[2] / 1000,
                META$beta_range$lo, META$beta_range$hi))

# Reads only what the network consumes from the chunk files.
load_sims <- function(dir) {
  files <- list.files(dir, pattern = "^sim_[0-9]+\\.rds$", full.names = TRUE)
  if (length(files) == 0) stop("No hay simulaciones en ", dir)
  message(sprintf("Leyendo %d bloques de %s ...", length(files), dir))

  parts <- lapply(files, function(f) {
    s <- readRDS(f)
    s <- s[!vapply(s, is.null, logical(1))]
    if (length(s) == 0) return(NULL)
    list(
      L   = do.call(rbind, lapply(s, `[[`, "L")),
      N   = matrix(vapply(s, `[[`, numeric(1), "N"), ncol = 1),
      B   = t(vapply(s, function(z) z$beta_hat, numeric(NB))),
      F   = t(vapply(s, function(z) unlist(z[FEAT_COLS]), numeric(length(FEAT_COLS)))),
      Y   = t(vapply(s, function(z) unlist(z[PAR_COLS]), numeric(length(PAR_COLS)))),
      hit = vapply(s, `[[`, numeric(1), "floor_hit")
    )
  })
  parts <- parts[!vapply(parts, is.null, logical(1))]

  d <- lapply(c("L", "N", "B", "F", "Y"),
              function(nm) do.call(rbind, lapply(parts, `[[`, nm)))
  names(d) <- c("L", "N", "B", "F", "Y")
  d$hit <- unlist(lapply(parts, `[[`, "hit"))
  colnames(d$N) <- "N"
  colnames(d$F) <- FEAT_COLS
  colnames(d$Y) <- PAR_COLS
  colnames(d$B) <- META$beta_names
  d
}

DAT <- load_sims(SIM_DIR)
message(sprintf("Realizaciones cargadas: %d", nrow(DAT$Y)))

ok <- is.finite(rowSums(DAT$L)) & is.finite(rowSums(DAT$F)) &
      is.finite(rowSums(DAT$B)) & is.finite(rowSums(DAT$Y)) & is.finite(DAT$N[, 1])
if (DROP_DEGENERATE) ok <- ok & (DAT$hit == 0)
message(sprintf("Descartadas por NA/Inf%s: %d (%.2f%%)",
                if (DROP_DEGENERATE) " o ajuste degenerado" else "",
                sum(!ok), 100 * mean(!ok)))

DAT <- c(lapply(DAT[c("L", "N", "B", "F", "Y")], function(m) m[ok, , drop = FALSE]),
         list(hit = DAT$hit[ok]))
n <- nrow(DAT$Y)
message(sprintf("Realizaciones utilizables: %d", n))

IS_NARROW <- DAT$Y[, "xi"] >= XI_NARROW[1] & DAT$Y[, "xi"] <= XI_NARROW[2]
message(sprintf("En la caja estrecha xi en [%g, %g] km: %d (%.1f%%)",
                XI_NARROW[1] / 1000, XI_NARROW[2] / 1000,
                sum(IS_NARROW), 100 * mean(IS_NARROW)))

# ---- 2. Stratified train / validation / test split --------------------------
# A random split is balanced in expectation but not exactly, and MAPE divides by
# the true value, so the small-xi tail would dominate the metric. Strata are the
# quantiles of both parameters crossed with the degenerate-fit indicator.
STRAT_BINS      <- NULL   # NULL = automatic
MIN_PER_STRATUM <- 20

qbin <- function(x, k) {
  if (k <= 1) return(rep(1L, length(x)))
  br <- unique(quantile(x, seq(0, 1, length.out = k + 1), na.rm = TRUE))
  if (length(br) < 2) return(rep(1L, length(x)))
  cut(x, breaks = br, labels = FALSE, include.lowest = TRUE)
}

if (is.null(STRAT_BINS)) {
  STRAT_BINS <- 1L
  for (k in 8:1) {
    if (n / (k^length(PAR_COLS) * 2) >= MIN_PER_STRATUM) { STRAT_BINS <- k; break }
  }
  message(sprintf("STRAT_BINS automático: %d (~%.0f realizaciones por estrato)",
                  STRAT_BINS, n / (STRAT_BINS^length(PAR_COLS) * 2)))
}

strata <- do.call(interaction,
                  c(lapply(PAR_COLS, function(p) qbin(DAT$Y[, p], STRAT_BINS)),
                    list(DAT$hit > 0), list(drop = TRUE)))
strata_sizes <- as.integer(table(strata))
message(sprintf("Estratos no vacíos: %d | tamaño mediano %.1f (mínimo %d)",
                nlevels(strata), median(strata_sizes), min(strata_sizes)))

# Midpoint rule within each stratum: after permuting, position (i - 0.5)/n picks
# the split. Distributes remainders without bias and handles tiny strata, where
# a single realisation goes to train rather than test.
stratified_split <- function(strata, fracs, seed = 123) {
  set.seed(seed)
  brk <- c(0, head(cumsum(fracs), -1), 1)
  grp <- integer(length(strata))
  for (lv in levels(strata)) {
    idx <- sample(which(strata == lv))
    pos <- (seq_along(idx) - 0.5) / length(idx)
    grp[idx] <- cut(pos, breaks = brk, labels = FALSE, include.lowest = TRUE)
  }
  grp
}

FRACS <- c(train = FRAC_TRAIN, val = FRAC_VAL, test = 1 - FRAC_TRAIN - FRAC_VAL)
stopifnot(all(FRACS > 0))
grp    <- stratified_split(strata, FRACS)
idx_tr <- which(grp == 1L); idx_va <- which(grp == 2L); idx_te <- which(grp == 3L)

message(sprintf("Train: %d (%.1f%%) | Validación: %d (%.1f%%) | Test: %d (%.1f%%)",
                length(idx_tr), 100 * length(idx_tr) / n,
                length(idx_va), 100 * length(idx_va) / n,
                length(idx_te), 100 * length(idx_te) / n))

balance <- purrr::imap_dfr(list(train = idx_tr, val = idx_va, test = idx_te),
                           function(ix, nm) {
  c(list(split = nm, n = length(ix)),
    as.list(round(colMeans(DAT$Y[ix, , drop = FALSE]), 3)),
    list(degenerados_pct = round(100 * mean(DAT$hit[ix] > 0), 2)))
})
message("\nBalance de la partición:")
print(balance)

# ---- 3. Normalisation, with statistics fitted on train only -----------------

fit_scaler <- function(m) {
  mu <- colMeans(m); sdv <- apply(m, 2, sd); sdv[sdv == 0 | !is.finite(sdv)] <- 1
  list(mu = mu, sd = sdv)
}
apply_scaler <- function(m, sc) {
  z <- scale(m, center = sc$mu, scale = sc$sd)
  z[!is.finite(z)] <- 0
  pmax(pmin(z, CLIP_SD), -CLIP_SD)
}

# s0 is the typical curve amplitude: the median over realisations of the maximum
# absolute value, so extreme realisations do not move it.
local({
  M  <- DAT$L[idx_tr, , drop = FALSE]
  s0 <- median(apply(abs(M), 1, max, na.rm = TRUE), na.rm = TRUE)
  if (!is.finite(s0) || s0 <= 0) s0 <- 1
  v  <- if (S2_TRANSFORM == "asinh") asinh(as.vector(M) / s0) else as.vector(M)
  s  <- sd(v, na.rm = TRUE); if (s == 0 || !is.finite(s)) s <- 1
  sc_S2 <<- list(s0 = s0, mu = mean(v, na.rm = TRUE), sd = s)
})

transform_S2 <- function(M) {
  z <- if (S2_TRANSFORM == "asinh") asinh(M / sc_S2$s0) else M
  z <- (z - sc_S2$mu) / sc_S2$sd
  z[!is.finite(z)] <- 0
  pmax(pmin(z, CLIP_SD), -CLIP_SD)
}

# Row-wise numerical derivative on the r grid. diff gives NR-1 values, so the
# first is repeated to keep the channel aligned with r[1].
raw_derivative <- function(M) {
  d <- t(apply(M, 1, diff)) / diff(META$r)[1]
  cbind(d[, 1, drop = FALSE], d)
}

# The derivative lives on a different scale from L and needs its own statistic.
sc_dL <- local({
  s <- sd(as.vector(raw_derivative(DAT$L[idx_tr, , drop = FALSE])), na.rm = TRUE)
  if (!is.finite(s) || s == 0) 1 else s
})

transform_dL <- function(M) {
  z <- raw_derivative(M) / sc_dL
  z[!is.finite(z)] <- 0
  pmax(pmin(z, CLIP_SD), -CLIP_SD)
}

# Coordinate channel: r normalised to [0, 1]. Constant across realisations, but
# it gives the network explicit access to position in r, which is what encodes
# xi; a convolution on its own is translation-equivariant.
R_CHANNEL <- META$r / max(META$r)

# Used for the training sets and for the observed pattern alike, so both go
# through exactly the same chain.
build_S2_array <- function(L) {
  a <- array(NA_real_, c(nrow(L), NR, S2_CHANNELS))
  a[, , 1] <- transform_S2(L)                 # D(r)   -> amplitude (sigma)
  a[, , 2] <- transform_dL(L)                 # dD/dr  -> growth rate
  a[, , 3] <- rep(R_CHANNEL, each = nrow(L))  # r/rmax -> position (xi)
  a
}

make_S2 <- function(idx) build_S2_array(DAT$L[idx, , drop = FALSE])
message(sprintf("S2 (%s, %d canales): amplitud típica de la curva = %.0f m (%.1f km)",
                S2_TRANSFORM, S2_CHANNELS, sc_S2$s0, sc_S2$s0 / 1000))

# Diagnostic: if the clipped fraction grows with sigma, CLIP_SD is saturating
# the very information that encodes sigma and S2_TRANSFORM should be "asinh".
local({
  z    <- transform_S2(DAT$L[idx_tr, , drop = FALSE])
  frac <- rowMeans(abs(z) >= CLIP_SD - 1e-9)
  sg   <- DAT$Y[idx_tr, "sigma"]
  tab  <- tapply(frac, cut(sg, breaks = quantile(sg, seq(0, 1, 0.2)),
                           include.lowest = TRUE), mean)
  message(sprintf("\nSaturación de S2 por CLIP_SD = %g (fracción de puntos recortados):",
                  CLIP_SD))
  print(round(100 * tab, 2))
  if (max(tab) > 0.01)
    message("  AVISO: hay recorte apreciable. Si crece con sigma, usar S2_TRANSFORM = \"asinh\".")
})

sc_N <- fit_scaler(DAT$N[idx_tr, , drop = FALSE])
sc_B <- fit_scaler(DAT$B[idx_tr, , drop = FALSE])
sc_F <- fit_scaler(DAT$F[idx_tr, , drop = FALSE])

X_S2 <- list(tr = make_S2(idx_tr), va = make_S2(idx_va), te = make_S2(idx_te))
X_N  <- list(tr = apply_scaler(DAT$N[idx_tr, , drop = FALSE], sc_N),
             va = apply_scaler(DAT$N[idx_va, , drop = FALSE], sc_N),
             te = apply_scaler(DAT$N[idx_te, , drop = FALSE], sc_N))
X_B  <- list(tr = apply_scaler(DAT$B[idx_tr, ], sc_B), va = apply_scaler(DAT$B[idx_va, ], sc_B),
             te = apply_scaler(DAT$B[idx_te, ], sc_B))
X_F  <- list(tr = apply_scaler(DAT$F[idx_tr, ], sc_F), va = apply_scaler(DAT$F[idx_va, ], sc_F),
             te = apply_scaler(DAT$F[idx_te, ], sc_F))

# Both links guarantee positivity when inverted, but they are not
# interchangeable. sigma spans a narrow additive range, so sqrt is enough. xi is
# a scale parameter spanning a factor of 15 and is reported as MAPE, a relative
# error, so log makes the training MSE optimise that quantity directly instead
# of over-weighting large xi. The standardisation that follows is essential:
# without it the xi term would dominate the MSE and sigma would not be learned.
TARGET_LINK <- c(sigma = "sqrt", xi = "log")[PAR_COLS]

apply_link <- function(m) {
  out <- m
  for (j in seq_along(PAR_COLS))
    out[, j] <- switch(TARGET_LINK[j], sqrt = sqrt(m[, j]), log = log(m[, j]))
  out
}
invert_link <- function(m) {
  out <- m
  for (j in seq_along(PAR_COLS))
    out[, j] <- switch(TARGET_LINK[j], sqrt = m[, j]^2, log = exp(m[, j]))
  out
}

Y_link   <- apply_link(DAT$Y)
sc_Y     <- fit_scaler(Y_link[idx_tr, , drop = FALSE])
Y_tr     <- scale(Y_link[idx_tr, ], center = sc_Y$mu, scale = sc_Y$sd)
Y_va     <- scale(Y_link[idx_va, ], center = sc_Y$mu, scale = sc_Y$sd)
TRUE_PAR <- DAT$Y[idx_te, , drop = FALSE]      # original scale, for the metrics

invert_target <- function(z) {
  m   <- sweep(sweep(z, 2, sc_Y$sd, `*`), 2, sc_Y$mu, `+`)
  out <- invert_link(m)
  colnames(out) <- PAR_COLS
  out
}

# ---- 4. Architecture --------------------------------------------------------

# Linear convolution -> BN -> ReLU, the canonical order. With the activation
# inside the convolution, BN would normalise an already rectified distribution.
conv_block <- function(x, filters, dilation = 1L, padding = "same") {
  x %>%
    layer_conv_1d(filters = filters, kernel_size = 7, dilation_rate = dilation,
                  padding = padding, activation = NULL) %>%
    layer_batch_normalization() %>%
    layer_activation("relu")
}

# Average, not max. D(r) is smooth and nearly increasing, so the maximum of a
# window is always its right edge: max pooling degenerates into biased
# decimation, exactly where the border-corrected estimator is noisiest.
pool_layer <- function(x, size = 5L) layer_average_pooling_1d(x, pool_size = size)

# 1D convolutional branch for S2 (NR x S2_CHANNELS). Three k = 7 blocks with
# dilation 1/2/4 and "same" padding. Receptive field over the 513 r values
# (dr = 0.39 km, rmax = 200 km):
#
#   conv d=1  (effective kernel  7)  ->  RF   7,  stride  1
#   avgpool 5                        ->  RF  11,  stride  5   (513 -> 102)
#   conv d=2  (effective kernel 13)  ->  RF  71,  stride  5
#   avgpool 5                        ->  RF  91,  stride 25   (102 ->  20)
#   conv d=4  (effective kernel 25)  ->  RF 691,  stride 25
#
# 691 x 0.39 km = 270 km > rmax, so the whole curve, at no extra parameter cost:
# dilation only spreads the filter taps. Undilated, the same three blocks reach
# 82 km, about half the xi prior. "same" padding is required, since with an
# effective kernel of 25 the length collapses before the third block.
build_S2_branch <- function(input_layer) {
  x <- input_layer %>%
    conv_block(64L, dilation = 1L) %>%
    pool_layer() %>%
    conv_block(64L, dilation = 2L) %>%
    pool_layer() %>%
    conv_block(S2_LAST_FILTERS, dilation = 4L)

  # Flatten, not GAP/GMP: global aggregates are translation-invariant and would
  # erase the positional information that encodes xi.
  x <- layer_flatten(x)

  # Closing projection, to balance branch dimensionality before the concatenate.
  # Without it S2 contributes 640 dimensions against 16, 1 and 32, and the dense
  # layer that follows splits its capacity in that proportion, burying the
  # descriptors. With it the concatenate is 32 + 16 + 1 + 32 = 81.
  x %>% layer_dense(units = S2_PROJECT, activation = "relu") %>%
    layer_batch_normalization()
}
build_dense_branch <- function(input_layer, units = c(32, 16)) {
  input_layer %>%
    layer_dense(units = units[1], activation = "relu") %>%
    layer_batch_normalization() %>%
    layer_dense(units = units[2], activation = "relu")
}

make_callbacks <- function(es_patience = 15, lr_patience = 7) {
  list(
    callback_early_stopping(monitor = "val_loss", patience = es_patience,
                            restore_best_weights = TRUE, verbose = 0),
    callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.5,
                                  patience = lr_patience, min_lr = 1e-6, verbose = 0)
  )
}

train_once <- function(tag) {
  in_S2 <- layer_input(shape = c(NR, S2_CHANNELS),    name = paste0("S2",    tag))
  in_B  <- layer_input(shape = c(ncol(DAT$B)),        name = paste0("S1",    tag))
  in_N  <- layer_input(shape = c(1),                  name = paste0("N",     tag))
  in_F  <- layer_input(shape = c(length(FEAT_COLS)),  name = paste0("Ifeat", tag))

  inputs   <- list(in_S2, in_B, in_N, in_F)
  branches <- list(build_S2_branch(in_S2), build_dense_branch(in_B, c(32, 16)),
                   # S1 stays at 32-16, five coefficients; I-feat uses
                   # FEAT_UNITS because it now takes 43 inputs, not 8.
                   in_N, build_dense_branch(in_F, FEAT_UNITS))

  x_tr <- list(X_S2$tr, X_B$tr, X_N$tr, X_F$tr)
  x_va <- list(X_S2$va, X_B$va, X_N$va, X_F$va)
  x_te <- list(X_S2$te, X_B$te, X_N$te, X_F$te)

  output <- layer_concatenate(branches) %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dropout(rate = DROPOUT) %>%
    layer_dense(units = 32, activation = "relu") %>%
    layer_dropout(rate = DROPOUT) %>%
    layer_dense(units = length(PAR_COLS), activation = "linear")

  model <- keras_model(inputs = inputs, outputs = output)
  model %>% compile(loss = "mse", optimizer = optimizer_adam(learning_rate = LR),
                    metrics = list("mae"))

  history <- model %>% fit(
    x = x_tr, y = Y_tr, epochs = EPOCHS, batch_size = BATCH_SIZE,
    validation_data = list(x_va, Y_va), callbacks = make_callbacks(), verbose = 2
  )
  list(model = model, history = history, pred = invert_target(predict(model, x_te, verbose = 0)))
}

# Trains N_SEEDS replicates and averages their predictions.
train_seeds <- function(label) {
  message(sprintf("\n=== Entrenando %s (%d semillas) ===", label, N_SEEDS))
  runs <- lapply(seq_len(N_SEEDS), function(i) {
    tensorflow::set_random_seed(1000 + i)
    train_once(sprintf("_s%d", i))
  })
  pred <- Reduce(`+`, lapply(runs, `[[`, "pred")) / N_SEEDS
  colnames(pred) <- PAR_COLS
  list(name = label, runs = runs, model = runs[[1]]$model,
       history = runs[[1]]$history,
       n_params = as.numeric(runs[[1]]$model$count_params()),
       pred = pred)
}

FIT <- train_seeds(M_FEAT)
PRED <- FIT$pred

# ---- 5. Metrics -------------------------------------------------------------

r2_score <- function(true_vec, pred_vec) {
  1 - sum((pred_vec - true_vec)^2) / sum((true_vec - mean(true_vec))^2)
}

# RMSE_rel is the RMSE divided by the standard deviation of the evaluated set:
# 1 means no better than predicting the mean. On a narrow test range MAPE can
# look good while being worse than a constant predictor.
metrics_on <- function(pred, rows, label, model_label = M_FEAT) {
  purrr::map_dfr(PAR_COLS, function(p) {
    tv <- TRUE_PAR[rows, p]; pv <- pred[rows, p]; e <- pv - tv
    tibble(modelo = model_label, conjunto = label, n = length(rows), param = p,
           Bias = mean(e), RMSE = sqrt(mean(e^2)),
           RMSE_rel = sqrt(mean(e^2)) / sd(tv),
           MAPE = 100 * mean(abs(e / tv)),
           R2 = r2_score(tv, pv))
  })
}

narrow_te  <- which(IS_NARROW[idx_te])
has_narrow <- length(narrow_te) >= 50
if (!has_narrow)
  message(sprintf("Sólo %d realizaciones del test en la caja estrecha: se omite.",
                  length(narrow_te)))

all_metrics <- bind_rows(
  metrics_on(PRED, seq_len(nrow(TRUE_PAR)), "amplio"),
  if (has_narrow) metrics_on(PRED, narrow_te, "estrecho")
)

message("\nMétricas en el conjunto de test (escala original, xi en metros):")
print(all_metrics %>% mutate(across(Bias:R2, ~ round(.x, 4))), n = Inf)
write_csv(all_metrics, file.path(fig_dir, "metrics.csv"))

# Per-realisation test predictions, so the true-vs-estimated figures can be
# redrawn without retraining. gid indexes the full set.
write_csv(tibble(gid = idx_te,
                 sigma_true = TRUE_PAR[, "sigma"], xi_true = TRUE_PAR[, "xi"],
                 sigma_cnn  = PRED[, "sigma"],     xi_cnn  = PRED[, "xi"]),
          file.path(fig_dir, "cnn_pred_test.csv"))

# Anchor against the trivial predictor, the train mean, so the claim is that the
# model adds something and not merely that it reaches some MAPE.
const_ref <- purrr::map_dfr(PAR_COLS, function(p) {
  cte <- mean(DAT$Y[idx_tr, p])
  purrr::map_dfr(list(amplio = seq_len(nrow(TRUE_PAR)),
                      estrecho = if (has_narrow) narrow_te else integer(0)),
                 function(rows) {
    if (length(rows) == 0) return(NULL)
    tv <- TRUE_PAR[rows, p]; e <- cte - tv
    tibble(param = p, n = length(rows), MAPE_constante = 100 * mean(abs(e / tv)),
           RMSE_constante = sqrt(mean(e^2)))
  }, .id = "conjunto")
})
comparacion <- all_metrics %>%
  select(conjunto, param, MAPE_modelo = MAPE, RMSE_modelo = RMSE, RMSE_rel, R2) %>%
  left_join(const_ref, by = c("conjunto", "param")) %>%
  mutate(mejor_que_constante = RMSE_modelo < RMSE_constante)
message("\nModelo frente al predictor constante (media del train):")
print(comparacion %>% mutate(across(where(is.numeric), ~ round(.x, 4))), n = Inf)
write_csv(comparacion, file.path(fig_dir, "vs_constante.csv"))

coste <- tibble(modelo = M_FEAT, parametros_red = FIT$n_params,
                entradas_extra = length(FEAT_COLS))
message("\nCoste:")
print(coste)
write_csv(coste, file.path(fig_dir, "coste.csv"))

# ---- 6. LaTeX table ---------------------------------------------------------
param_latex <- c(sigma = "$\\sigma$", xi = "$\\xi$")

build_tex_table <- function(cj) {
  rows_cj <- all_metrics %>% filter(conjunto == cj)
  c("\\begin{table}[ht!]", "\\centering",
    sprintf("\\caption{Recuperación de los parámetros del campo latente (test %s, n=%d).",
            cj, rows_cj$n[1]),
    if (cj == "estrecho")
      sprintf("$\\xi$ restringido a $[%g, %g]$ km.", XI_NARROW[1] / 1000, XI_NARROW[2] / 1000)
    else "Todo el conjunto de test.",
    "RMSE$_{rel}$ es la RMSE dividida por la desviación típica del conjunto evaluado:",
    "1 equivale a predecir la media, y por encima de 1 el modelo es peor que esa constante.}",
    sprintf("\\label{tab:metrics_sismos_%s}", cj),
    "\\begin{tabular}{llrrrrr}", "\\hline",
    "Modelo & Parámetro & Bias & RMSE & RMSE$_{rel}$ & MAPE (\\%) & $R^2$ \\\\", "\\hline",
    vapply(seq_len(nrow(rows_cj)), function(i) {
      row <- rows_cj[i, ]
      sprintf("%s & %s & %.3f & %.3f & %.3f & %.2f & %.4f \\\\",
              if (i == 1) as.character(row$modelo) else "",
              param_latex[row$param], row$Bias, row$RMSE, row$RMSE_rel,
              row$MAPE, row$R2)
    }, character(1)),
    "\\hline", "\\end{tabular}", "\\end{table}")
}
writeLines(unlist(lapply(unique(as.character(all_metrics$conjunto)),
                         function(cj) c(build_tex_table(cj), ""))),
           file.path(fig_dir, "metrics_table.tex"))
cat("\nTabla LaTeX guardada\n")

# ---- 7. Figures -------------------------------------------------------------
# xi is plotted in km; in metres the axes come out in scientific notation.
param_labels <- c(sigma = "sigma", xi = "xi (km)")
par_scale    <- c(sigma = 1, xi = 1 / 1000)

plots <- lapply(PAR_COLS, function(p) {
  df  <- tibble(true = TRUE_PAR[, p] * par_scale[[p]], pred = PRED[, p] * par_scale[[p]])
  r2  <- round(r2_score(df$true, df$pred), 3)
  rng <- range(c(df$true, df$pred), finite = TRUE)
  ggplot(df, aes(true, pred)) +
    geom_point(alpha = 0.2, size = 0.5, stroke = 0, colour = "steelblue4") +
    geom_abline(slope = 1, intercept = 0, colour = "grey40", linetype = "dashed") +
    annotate("text", x = rng[1], y = rng[2], label = paste0("R² = ", r2),
             hjust = 0, vjust = 1, size = 3.5) +
    labs(x = "Verdadero", y = "Estimado", title = param_labels[[p]]) +
    theme_bw(base_size = 10) +
    theme(panel.grid = element_blank(), plot.title = element_text(hjust = 0.5, size = 11))
})
ggsave(file.path(fig_dir, "scatter_CNN_Ifeat.pdf"),
       wrap_plots(plots, nrow = 1) +
         plot_annotation(title = M_FEAT,
                         theme = theme(plot.title = element_text(hjust = 0.5, face = "bold"))),
       width = 8, height = 3.8, device = cairo_pdf)

curve_df <- bind_rows(
  tibble(epoch = seq_along(FIT$history$metrics$loss), Entrenamiento = FIT$history$metrics$loss,
         Validacion = FIT$history$metrics$val_loss, Métrica = "MSE"),
  tibble(epoch = seq_along(FIT$history$metrics$mae), Entrenamiento = FIT$history$metrics$mae,
         Validacion = FIT$history$metrics$val_mae, Métrica = "MAE")
) %>%
  pivot_longer(c(Entrenamiento, Validacion), names_to = "Conjunto", values_to = "value")

ggsave(file.path(fig_dir, "loss.pdf"),
       ggplot(curve_df, aes(epoch, value, colour = Conjunto)) +
         geom_line(linewidth = 0.6) +
         facet_wrap(~ Métrica, scales = "free_y") +
         scale_colour_manual(values = c(Entrenamiento = "steelblue", Validacion = "tomato")) +
         labs(x = "Época", y = NULL) +
         theme_bw(base_size = 10) +
         theme(panel.grid.minor = element_blank(), legend.position = "bottom"),
       width = 8, height = 3.5, device = cairo_pdf)

p_r2 <- all_metrics %>%
  mutate(param = factor(param, levels = PAR_COLS), vj = ifelse(R2 >= 0, -0.3, 1.2)) %>%
  ggplot(aes(param, R2, fill = conjunto)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.6) +
  geom_hline(yintercept = 0, colour = "grey40", linewidth = 0.3) +
  geom_text(aes(label = sprintf("%.3f", R2), vjust = vj),
            position = position_dodge(width = 0.8), size = 3) +
  scale_fill_manual(values = c(amplio = "darkorange", estrecho = "grey65")) +
  labs(x = "Parámetro", y = expression(R^2), fill = "Conjunto de test", title = M_FEAT) +
  theme_bw(base_size = 11) +
  theme(panel.grid = element_blank(), legend.position = "bottom")
ggsave(file.path(fig_dir, "r2.pdf"), p_r2, width = 5.5, height = 4, device = cairo_pdf)
cat("Figuras guardadas\n")

# ---- 8. Estimation on the real catalogue ------------------------------------
# The observed inputs go through exactly the same transformation chain as the
# training ones.

OBS <- readRDS(file.path(SIM_DIR, "observed_inputs.rds"))

obs_S2 <- build_S2_array(matrix(OBS$L, nrow = 1))
obs_B  <- apply_scaler(matrix(OBS$beta_hat, nrow = 1), sc_B)
obs_N  <- apply_scaler(matrix(OBS$N, nrow = 1), sc_N)
obs_F  <- apply_scaler(matrix(unlist(OBS[FEAT_COLS]), nrow = 1), sc_F)
obs_in <- list(obs_S2, obs_B, obs_N, obs_F)

P <- Reduce(`+`, lapply(FIT$runs, function(r)
  invert_target(predict(r$model, obs_in, verbose = 0)))) / N_SEEDS
obs_pred <- tibble(Modelo = M_FEAT, !!!as.list(setNames(as.numeric(P), PAR_COLS)))

# Classical anchor: minimum contrast on the real data, independent of the
# network. Only the K-based reference was stored.
MC <- META$mc_reference
obs_pred <- bind_rows(obs_pred,
  tibble(Modelo = "Mín. contraste (K)", sigma = MC$K[["sigma"]], xi = MC$K[["xi"]]),
  if (!is.null(MC$pcf))
    tibble(Modelo = "Mín. contraste (pcf)", sigma = MC$pcf[["sigma"]], xi = MC$pcf[["xi"]]))
obs_pred <- obs_pred %>% mutate(xi_km = xi / 1000)

cat("\n--- Parámetros estimados para el catálogo sísmico real ---\n")
print(obs_pred %>% mutate(across(where(is.numeric), ~ round(.x, 3))))
write_csv(obs_pred, file.path(fig_dir, "estimates_observed.csv"))

writeLines(c(
  "\\begin{table}[ht!]", "\\centering",
  "\\caption{Parámetros estimados del campo latente para el catálogo sísmico de Colombia (2020).}",
  "\\label{tab:params_obs_sismos}",
  "\\begin{tabular}{lrr}", "\\hline",
  "Modelo & $\\hat{\\sigma}$ & $\\hat{\\xi}$ (km) \\\\", "\\hline",
  vapply(seq_len(nrow(obs_pred)), function(i) {
    r <- obs_pred[i, ]
    sprintf("%s & %.3f & %.1f \\\\", r$Modelo, r$sigma, r$xi_km)
  }, character(1)),
  "\\hline", "\\end{tabular}", "\\end{table}"),
  file.path(fig_dir, "params_table_obs.tex"))

# ---- 9. Realisations from the fitted model ----------------------------------
# Under the centred form the fitted trend estimates exp(Z beta), so simulating
# from theta_hat is direct: Lambda(s) = lambda_hat(s) * exp(Y(s) - sigma^2/2).
# Qualitative adequacy check: these realisations should resemble the observed
# catalogue in point count and clustering scale.

N_REALISATIONS <- 99

th    <- as.list(obs_pred %>% filter(Modelo == M_FEAT))
W_OBS <- Window(OBS$X)
# eval.im() finds images by looking for symbols, so OBS$lambda has to be bound
# to a variable first.
LAM_OBS <- OBS$lambda

simulate_fitted <- function(seed) {
  set.seed(seed)
  # Matern nu = 1, the same generator as script 01.
  Y   <- spatstat.random::rGRFmatern(W = W_OBS, mu = 0, var = th$sigma^2,
                                     scale = th$xi, nu = NU_MATERN,
                                     dimyx = META$dim)
  cst <- 0.5 * th$sigma^2
  rpoispp(eval.im(LAM_OBS * exp(Y - cst)))[W_OBS]
}

message(sprintf("\nSimulando %d realizaciones con theta_hat (sigma = %.3f, xi = %.1f km)...",
                N_REALISATIONS, th$sigma, th$xi / 1000))
sims_fit <- lapply(seq_len(N_REALISATIONS), simulate_fitted)

# The window is taller than wide, so 3 x 4 keeps the figure nearly square.
png(file.path(fig_dir, "realizaciones_ajustadas.png"),
    width = 1600, height = 1500, res = 130)
op <- par(mfrow = c(3, 4), mar = c(0.5, 0.5, 2.0, 0.5))
plot(OBS$X, main = sprintf("OBSERVADO (%d)", OBS$N), pch = 16, cex = 0.1)
for (i in seq_along(sims_fit))
  plot(sims_fit[[i]], main = sprintf("Sim %d (%d)", i, npoints(sims_fit[[i]])),
       pch = 16, cex = 0.1)
par(op); dev.off()

counts <- vapply(sims_fit, npoints, numeric(1))
resumen_sim <- tibble(observado = OBS$N, sim_media = round(mean(counts), 1),
                      sim_min = min(counts), sim_max = max(counts),
                      dentro = OBS$N >= min(counts) & OBS$N <= max(counts))
cat("\n--- Conteos de las realizaciones ajustadas ---\n")
print(resumen_sim)
write_csv(resumen_sim, file.path(fig_dir, "realizaciones_conteos.csv"))

# ---- 10. Simulation envelope and DCLF test ----------------------------------
# Goodness of fit on the real data, where no truth is known.
#   H0: the observed pattern comes from the fitted model.
# The DCLF statistic is T = int (K(r) - Kbar(r))^2 dr. For the simulations the
# mean leaves out the realisation itself; including it would make each
# simulation part of its own average and the test conservative.
# lambda_hat is rescaled to each pattern's count, otherwise realisations with
# fewer points than the observed one would sit artificially low.

N_ENVELOPE <- 99
R_ENV      <- seq(0, max(META$r), length.out = 257)

rescale_lambda <- function(lam, pp) {
  tot <- integral.im(lam)
  if (!is.finite(tot) || tot <= 0) return(lam)
  eval.im(lam * (npoints(pp) / tot))
}

k_curve <- function(X)
  Kinhom(X, lambda = rescale_lambda(LAM_OBS, X), correction = "border", r = R_ENV)$border

message(sprintf("Envolvente: simulando %d realizaciones bajo el modelo ajustado...",
                N_ENVELOPE))
K_obs  <- k_curve(OBS$X)
K_sims <- t(vapply(seq_len(N_ENVELOPE),
                   function(i) k_curve(simulate_fitted(10000 + i)), numeric(length(K_obs))))

dclf_p <- function(obs, sims) {
  dr    <- diff(R_ENV)[1]
  nsim  <- nrow(sims)
  T_obs <- sum((obs - colMeans(sims))^2) * dr
  T_sim <- vapply(seq_len(nsim), function(i)
    sum((sims[i, ] - colMeans(sims[-i, , drop = FALSE]))^2) * dr, numeric(1))
  (1 + sum(T_sim >= T_obs)) / (1 + nsim)
}

p_dclf <- dclf_p(K_obs, K_sims)
env_df <- tibble(r = R_ENV, obs = K_obs, med = colMeans(K_sims),
                 lo = apply(K_sims, 2, min), hi = apply(K_sims, 2, max))
cat(sprintf("\n--- Test DCLF: p = %.3f -> %s\n", p_dclf,
            if (p_dclf > 0.05) "no se rechaza H0" else "SE RECHAZA H0"))
write_csv(env_df, file.path(fig_dir, "envolvente.csv"))
write_csv(tibble(p_DCLF = p_dclf), file.path(fig_dir, "dclf_pvalue.csv"))

draw_envelope <- function() {
  op <- par(mar = c(4.4, 5.2, 3.2, 1.2))
  rkm <- env_df$r / 1000
  plot(range(rkm), range(c(env_df$lo, env_df$hi, env_df$obs)), type = "n",
       xlab = "r (km)", ylab = expression(K[inhom](r)),
       main = sprintf("Envolvente bajo el modelo ajustado (%s)", M_FEAT))
  polygon(c(rkm, rev(rkm)), c(env_df$lo, rev(env_df$hi)), col = "grey80", border = NA)
  lines(rkm, env_df$med, col = "red",   lty = 2, lwd = 1.2)
  lines(rkm, env_df$obs, col = "black", lty = 1, lwd = 1.6)
  legend("topleft",
         legend = c(expression(hat(K)[inhom]^obs * (r)), expression(bar(K)[inhom](r)),
                    expression(hat(K)[inhom]^hi * (r)),  expression(hat(K)[inhom]^lo * (r))),
         col = c("black", "red", "grey80", "grey80"), lty = c(1, 2, 1, 1),
         lwd = c(1.6, 1.2, 4, 4), bty = "o", bg = "white", cex = 0.85, y.intersp = 1.35)
  mtext(sprintf("p (DCLF) = %.2f", p_dclf), side = 3, line = 0.2, cex = 0.8)
  par(op)
}
png(file.path(fig_dir, "envolvente.png"), width = 800, height = 620, res = 130)
draw_envelope(); dev.off()
cairo_pdf(file.path(fig_dir, "envolvente.pdf"), width = 6.5, height = 5)
draw_envelope(); dev.off()

# ---- 11. Count-residual image -----------------------------------------------
# Pearson residual of the first-order fit on the coarse grid,
# I(g) = (N(g) - E(g)) / sqrt(E(g)) with E(g) = int_g lambda_hat. Positive values
# are clustering the covariate trend does not explain.
#
# The fitted trend has extreme contrast, so some cells have E ~ 0 and the
# residual diverges there: one event over E = 1e-10 would give 1e5 and flatten
# the colour scale. Cells without enough exposure are masked instead.
E_MIN_RES <- 0.5     # minimum expected counts for a legible residual

WMASK_RES   <- as.mask(W_OBS, dimyx = OBS$dim_res)
PIXAREA_RES <- with(WMASK_RES, xstep * ystep)
LAM_RES     <- OBS$lambda_res

Nim <- pixellate(OBS$X, W = WMASK_RES)
Eim <- eval.im(LAM_RES * PIXAREA_RES)
RES_IM <- eval.im(ifelse(Eim >= E_MIN_RES, (Nim - Eim) / sqrt(Eim), NA_real_))

res_df <- expand.grid(x = RES_IM$xcol, y = RES_IM$yrow) %>%
  mutate(valor = as.vector(t(RES_IM$v))) %>%
  filter(is.finite(valor))

# Asymmetric limits, not +/- q99(|residual|). The asymmetry is structural: since
# N >= 0 the residual is bounded below by -sqrt(E), around -8.7 here, while
# clustering takes the upper tail to +197. A symmetric scale would spend most of
# the colour range on values that do not occur.
lim_lo <- as.numeric(quantile(res_df$valor, 0.01, na.rm = TRUE))
lim_hi <- as.numeric(quantile(res_df$valor, 0.99, na.rm = TRUE))
res_df <- res_df %>% mutate(valor_plot = pmax(pmin(valor, lim_hi), lim_lo))

p_res <- ggplot(res_df, aes(x / 1000, y / 1000, fill = valor_plot)) +
  geom_raster() +
  scale_fill_gradient2(low = "#2166AC", mid = "grey95", high = "#B2182B",
                       midpoint = 0, limits = c(lim_lo, lim_hi), name = "Residuo") +
  coord_equal(expand = FALSE) +
  labs(x = "x (km)", y = "y (km)",
       title = "Residuos de conteo del ajuste de primer orden",
       subtitle = sprintf("celdas con E >= %.1f conteos esperados; escala recortada a [%.1f, %.1f]",
                          E_MIN_RES, lim_lo, lim_hi)) +
  theme_bw(base_size = 10) +
  theme(panel.grid = element_blank(),
        plot.title = element_text(hjust = 0.5, size = 10),
        plot.subtitle = element_text(hjust = 0.5, size = 8))

ggsave(file.path(fig_dir, "residuos_conteo.pdf"), p_res,
       width = 6, height = 7, device = cairo_pdf)
ggsave(file.path(fig_dir, "residuos_conteo.png"), p_res,
       width = 6, height = 7, dpi = 130)

cat(sprintf("\n--- Residuos de conteo: %d celdas con E >= %.1f (%.0f%% de la ventana), rango [%.2f, %.2f], %.1f%% positivos ---\n",
            nrow(res_df), E_MIN_RES,
            100 * nrow(res_df) / sum(!is.na(as.vector(Eim$v))),
            min(res_df$valor), max(res_df$valor), 100 * mean(res_df$valor > 0)))

# ---- 12. Save model and scalers ---------------------------------------------
# The scalers are part of the model. Saved before the bootstrap on purpose: if
# section 13 fails, the expensive part is already on disk.
save_model_tf(FIT$model, file.path(fig_dir, "model_CNN_Ifeat"))
saveRDS(list(sc_S2 = sc_S2, sc_N = sc_N, sc_B = sc_B, sc_F = sc_F, sc_Y = sc_Y,
             clip_sd = CLIP_SD, s2_transform = S2_TRANSFORM,
             # Recorded so an artefact on disk identifies the network that
             # produced it without re-reading this script.
             arq_S2 = list(canales = S2_CHANNELS, kernel = 7L,
                           filtros = c(64L, 64L, S2_LAST_FILTERS),
                           dilataciones = c(1L, 2L, 4L), padding = "same",
                           pooling = "avg", pool_size = 5L,
                           agregacion = "flatten", proyeccion = S2_PROJECT),
             xi_narrow = XI_NARROW, n_seeds = N_SEEDS,
             par_cols = PAR_COLS, feat_cols = FEAT_COLS,
             # Without these, two runs with different FEAT_SET or FEAT_UNITS
             # leave indistinguishable artefacts on disk.
             feat_set = FEAT_SET, feat_units = FEAT_UNITS,
             nquad_bank = META$nquad_bank, kde_bw_bank = META$kde_bw_bank,
             beta_names = META$beta_names, modelo = M_FEAT),
        file.path(fig_dir, "scalers.rds"))
cat("\nModelo y escaladores guardados en:", fig_dir, "\n")

# ---- 13. Parametric bootstrap of the estimator ------------------------------
# M_BOOT replicates of the catalogue are simulated under theta_hat with the same
# generator as section 9 (trend fixed at stage 1, latent field at theta_hat),
# each replicate goes through the complete estimation pipeline (ppm -> L curve
# -> descriptor bank -> CNN, averaged over the N_SEEDS replicates, exactly as
# the observed pattern), and the quantiles of the re-estimates give the interval.
# It therefore measures the sampling variability of the whole two-step
# estimator, conditional on the first-order trend.
#
# The re-estimation machinery is copied from script 01, parameterised from META,
# because a replicate must follow exactly the same path as the training
# simulations and the catalogue.
#
# Parallelism: the spatstat part runs under mclapply; the keras prediction comes
# afterwards, in the main process and in a single batch, because TensorFlow is
# not safe inside forked workers.

M_BOOT          <- 200     # enough for stable 2.5/97.5 quantiles
BOOT_CORES      <- 16
BOOT_SEED_BASE  <- 20000   # clear of sections 9 (1..99) and 10 (10000+i)

COVS_BOOT <- lapply(META$cov_names, function(nm)
  readRDS(file.path("data", paste0(nm, "_im_scaled.rds"))))
names(COVS_BOOT) <- META$cov_names
COVS_BOOT <- lapply(COVS_BOOT, function(z) as.im(z, xy = COVS_BOOT[[1]]))
COVS_BOOT <- lapply(COVS_BOOT, function(z) z[W_OBS, drop = FALSE])
names(COVS_BOOT) <- META$cov_names

fit_first_order_boot <- function(pp) {
  fo <- as.formula(paste("pp ~", META$rhs))
  environment(fo) <- environment()
  ppm(fo, covariates = COVS_BOOT, nd = META$nquad_ppm)
}

WMASK_SIM <- as.mask(W_OBS, dimyx = META$dim)

quad_stats <- function(qc) {
  if (is.null(qc) || length(qc) < 4)
    return(c(var = 0, VMR = 1, range_ratio = 1))
  v <- var(qc); m <- mean(qc)
  c(var = v,
    VMR = if (m > 0) v / m else 1,
    range_ratio = max(qc) / (min(qc) + 1))
}

kde_stats <- function(pp, bw, dimyx) {
  kde <- tryCatch(density.ppp(pp, sigma = bw, dimyx = dimyx),
                  error = function(e) NULL)
  if (is.null(kde))
    return(c(var = 0, skew = 0, kurt = 0, entropy = 0, cv = 0))
  vals <- as.vector(kde$v)
  vals <- vals[!is.na(vals)]
  vals <- pmax(vals, 0)
  m <- mean(vals); s <- sd(vals)
  if (s > 0) {
    z <- (vals - m) / s
    sk <- mean(z^3); ku <- mean(z^4) - 3
  } else {
    sk <- 0; ku <- 0
  }
  tot <- sum(vals)
  ent <- if (tot > 0) {
    p <- vals / tot; p <- p[p > 0]
    -sum(p * log(p)) / log(length(p))
  } else 0
  c(var = s^2, skew = sk, kurt = ku, entropy = ent,
    cv = if (m > 0) s / m else 0)
}

# No border-tile filter, matching script 01, which counts every tile including
# the slivers at the edge of W. An earlier version filtered on a design_meta
# field that no longer exists, which silently left 18 of the 43 bootstrap
# descriptors constant.
QUAD_TESS_BOOT <- lapply(META$nquad_bank, function(k) {
  list(k = k, tess = quadrats(W_OBS, nx = k, ny = k))
})

# No legacy block: script 01 extracts only the 43 bank descriptors.
boot_features <- function(pp) {
  quad_bank <- do.call(c, unname(lapply(QUAD_TESS_BOOT, function(z) {
    qc <- tryCatch(as.vector(quadratcount(pp, tess = z$tess)),
                   error = function(e) NULL)
    st <- quad_stats(qc)
    setNames(as.list(st), paste0("quad_", names(st), "_k", z$k))
  })))
  kde_bank <- do.call(c, unname(lapply(META$kde_bw_bank, function(b) {
    st <- kde_stats(pp, b, META$kde_dimyx_bank)
    setNames(as.list(st), paste0("kde_", names(st), "_bw", round(b / 1000)))
  })))
  c(quad_bank, kde_bank)
}

# One replicate: simulate at theta_hat and re-extract every input. The 100/80000
# thresholds are MIN_POINTS/MAX_POINTS of script 01, so a replicate the training
# design would have discarded does not enter the interval.
boot_one <- function(i) {
  X <- tryCatch(simulate_fitted(BOOT_SEED_BASE + i), error = function(e) NULL)
  if (is.null(X)) return(NULL)
  n <- npoints(X)
  if (n < 100 || n > 80000) return(NULL)

  fit <- tryCatch(fit_first_order_boot(X), error = function(e) NULL)
  if (is.null(fit)) return(NULL)
  bhat <- unname(coef(fit))
  if (anyNA(bhat) || length(bhat) != length(META$beta_names)) return(NULL)

  lam <- tryCatch(predict(fit, type = "trend", locations = WMASK_SIM),
                  error = function(e) NULL)
  if (is.null(lam)) return(NULL)
  flv <- META$lambda_floor * n / area.owin(W_OBS)
  lam <- eval.im(pmax(lam, flv))

  L <- tryCatch(Linhom(X, lambda = lam, correction = "border", r = META$r)$border,
                error = function(e) NULL)
  if (is.null(L) || anyNA(L)) return(NULL)

  ft <- tryCatch(boot_features(X), error = function(e) NULL)
  if (is.null(ft)) return(NULL)
  fv <- unlist(ft[FEAT_COLS])
  if (anyNA(fv)) return(NULL)

  list(L = L - META$r, bhat = bhat, N = n, F = fv)
}

message(sprintf("\n=== Bootstrap paramétrico: %d réplicas en theta_hat (sigma = %.3f, xi = %.1f km), %d núcleos ===",
                M_BOOT, th$sigma, th$xi / 1000, BOOT_CORES))
t_boot <- system.time({
  BOOT_RAW <- parallel::mclapply(seq_len(M_BOOT), boot_one,
                                 mc.cores = BOOT_CORES, mc.preschedule = FALSE)
})
BOOT_OK <- Filter(function(o) is.list(o) && !is.null(o$L), BOOT_RAW)
n_ok <- length(BOOT_OK)
message(sprintf("Réplicas válidas: %d/%d (%.1f min)", n_ok, M_BOOT,
                t_boot[["elapsed"]] / 60))
if (n_ok < 100)
  warning(sprintf("Sólo %d réplicas válidas: los cuantiles 2.5/97.5%% son inestables por debajo de ~100.",
                  n_ok))

# Batch prediction with the same ensemble used for the catalogue.
boot_L <- do.call(rbind, lapply(BOOT_OK, `[[`, "L"))
boot_in <- list(build_S2_array(boot_L),
                apply_scaler(do.call(rbind, lapply(BOOT_OK, `[[`, "bhat")), sc_B),
                apply_scaler(matrix(vapply(BOOT_OK, `[[`, numeric(1), "N"), ncol = 1), sc_N),
                apply_scaler(do.call(rbind, lapply(BOOT_OK, `[[`, "F")), sc_F))
BOOT_PRED <- Reduce(`+`, lapply(FIT$runs, function(r)
  invert_target(predict(r$model, boot_in, verbose = 0)))) / N_SEEDS
colnames(BOOT_PRED) <- PAR_COLS

# The percentile interval is what gets reported. The bootstrap bias is kept as a
# diagnostic: if it is large relative to the interval width there is appreciable
# shrinkage. The basic interval is stored for completeness.
theta_hat <- setNames(c(th$sigma, th$xi), PAR_COLS)
boot_ic <- purrr::map_dfr(PAR_COLS, function(p) {
  v <- BOOT_PRED[, p]
  q <- quantile(v, c(0.025, 0.5, 0.975))
  # `hat`, not `theta_hat`: inside tibble() the new column of that name would
  # mask the outer variable.
  hat <- theta_hat[[p]]
  tibble(param = p, theta_hat = hat,
         n_boot = n_ok,
         media_boot = mean(v), sesgo_boot = mean(v) - hat,
         sd_boot = sd(v),
         ic_lo = q[[1]], mediana = q[[2]], ic_hi = q[[3]],
         ic_basico_lo = 2 * hat - q[[3]],
         ic_basico_hi = 2 * hat - q[[1]])
})
cat("\n--- Bootstrap paramétrico (IC 95%, percentil) ---\n")
print(boot_ic %>% mutate(across(where(is.numeric), ~ signif(.x, 4))))
write_csv(as_tibble(BOOT_PRED) %>% mutate(replica = seq_len(n_ok), .before = 1),
          file.path(fig_dir, "bootstrap_theta.csv"))
write_csv(boot_ic, file.path(fig_dir, "bootstrap_ic.csv"))

writeLines(c(
  "\\begin{table}[ht!]", "\\centering",
  sprintf("\\caption{Intervalos bootstrap paramétrico (percentil, 95\\%%, %d réplicas válidas de %d)",
          n_ok, M_BOOT),
  "para la estimación de la CNN sobre el catálogo sísmico de Colombia (2020). Cada réplica se",
  "simula bajo $\\hat{\\theta}$ con la tendencia de primer orden fija (misma etapa 1 que el resto",
  "del pipeline) y se re-estima con el pipeline completo, de modo que el intervalo refleja la",
  "variabilidad muestral del estimador en dos pasos.}",
  "\\label{tab:bootstrap_ic_sismos}",
  "\\begin{tabular}{lrc}", "\\hline",
  "Parámetro & $\\hat{\\theta}$ & IC 95\\% \\\\", "\\hline",
  sprintf("$\\sigma$ & %.3f & [%.3f, %.3f] \\\\",
          boot_ic$theta_hat[boot_ic$param == "sigma"],
          boot_ic$ic_lo[boot_ic$param == "sigma"],
          boot_ic$ic_hi[boot_ic$param == "sigma"]),
  sprintf("$\\xi$ (km) & %.1f & [%.1f, %.1f] \\\\",
          boot_ic$theta_hat[boot_ic$param == "xi"] / 1000,
          boot_ic$ic_lo[boot_ic$param == "xi"] / 1000,
          boot_ic$ic_hi[boot_ic$param == "xi"] / 1000),
  "\\hline", "\\end{tabular}", "\\end{table}"),
  file.path(fig_dir, "bootstrap_ic.tex"))

# The figure answers whether the minimum-contrast reference falls inside the
# CNN interval.
boot_hist_df <- bind_rows(
  tibble(param = "sigma", valor = BOOT_PRED[, "sigma"]),
  tibble(param = "xi (km)", valor = BOOT_PRED[, "xi"] / 1000))
boot_ref_df <- bind_rows(
  tibble(param = "sigma", ref = c(theta_hat[["sigma"]], MC$K[["sigma"]]),
         tipo = c("CNN", "Mín. contraste")),
  tibble(param = "xi (km)", ref = c(theta_hat[["xi"]], MC$K[["xi"]]) / 1000,
         tipo = c("CNN", "Mín. contraste")))
p_boot <- ggplot(boot_hist_df, aes(valor)) +
  geom_histogram(bins = 30, fill = "grey70", colour = "grey35", linewidth = 0.2) +
  geom_vline(data = boot_ref_df, aes(xintercept = ref, colour = tipo),
             linetype = "dashed", linewidth = 0.6) +
  scale_colour_manual(values = c("CNN" = "darkorange", "Mín. contraste" = "black"),
                      name = NULL) +
  facet_wrap(~param, scales = "free") +
  labs(x = "Re-estimación bootstrap", y = "Réplicas",
       title = sprintf("Bootstrap paramétrico (%s, %d réplicas)", M_FEAT, n_ok)) +
  theme_bw(base_size = 11) +
  theme(panel.grid = element_blank(), legend.position = "bottom",
        plot.title = element_text(hjust = 0.5, size = 11))
ggsave(file.path(fig_dir, "bootstrap_hist.pdf"), p_boot,
       width = 7.5, height = 3.8, device = cairo_pdf)

cat("\nListo. Modelo, tablas y figuras en:", fig_dir, "\n")

