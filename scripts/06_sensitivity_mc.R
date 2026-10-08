# Sensitivity of the catalogue estimates to the completeness magnitude.
#
# The paper analyses every declustered event of 2020 (N = 17857). Detection
# below the completeness magnitude is spatially uneven (denser networks record
# smaller events), so part of the residual clustering could be instrumental.
# This script re-estimates the latent-field parameters on the subset with
# Mw >= MC_THRESHOLD (the completeness magnitude of the 2018-2020 network stage)
# with the complete two-step chain and the same CNN ensemble, and refits the
# INLA-SPDE model on the same subset. Nothing here modifies the outputs of
# scripts 01-04; every file is written with the prefix sens_mc_.
#
# The ensemble is retrained with the seeds of script 02 (the SavedModel on disk
# holds a single replicate, and the paper reports the five-seed average). The
# five models are saved on the first run and reloaded afterwards, so a failure
# in a later section does not cost the training again.
#
# Usage:  Rscript scripts/06_sensitivity_mc.R   (from the repository root)
# Requires simulations/ (script 01), outputs/inla_mesh_*.rds (script 03) and
# outputs/bootstrap_ic.csv, dclf_pvalue.csv, inla_catalog_observed.csv (scripts
# 02-03) for the full-catalogue column of the summary table.

library(reticulate)
use_virtualenv("~/.virtualenvs/r-tensorflow", required = TRUE)  # before library(keras)

library(keras)
Sys.setenv(TF_USE_LEGACY_KERAS = "0")

suppressMessages({
  library(tidyverse)
  library(spatstat)
  library(sf)
})

local({
  kv <- as.character(reticulate::py_get_attr(reticulate::import("keras"), "__version__"))
  if (!startsWith(kv, "2.")) stop("Keras ", kv, " detected; this script needs Keras 2.x.")
  message("Keras backend OK: ", kv)
})

set.seed(123)
tensorflow::set_random_seed(123)
local({
  ok <- tryCatch({ tensorflow::tf$config$experimental$enable_op_determinism(); TRUE },
                 error = function(e) FALSE)
  message(if (ok) "TF op determinism: ON" else "WARNING: enable_op_determinism() unavailable.")
})

stopifnot(dir.exists("data"), dir.exists("simulations"), dir.exists("outputs"))

SIM_DIR <- "simulations"
OUT_DIR <- "outputs"
PREFIX  <- "sens_mc"
out_f   <- function(name) file.path(OUT_DIR, paste0(PREFIX, "_", name))

# ---- 0. Configuration (identical to script 02 where it matters) -------------

MC_THRESHOLD <- 1.92     # Mw; completeness of the 2018-2020 network stage
CATALOGUE    <- "data/gdf_espacial_2020_paper.gpkg"

FRAC_TRAIN <- 0.70; FRAC_VAL <- 0.15
DROP_DEGENERATE <- FALSE
CLIP_SD <- 10
S2_TRANSFORM <- "none"; S2_CHANNELS <- 3L; S2_LAST_FILTERS <- 32L; S2_PROJECT <- 32L
EPOCHS <- 200; BATCH_SIZE <- 64; LR <- 1e-3; DROPOUT <- 0.2
N_SEEDS <- 5
PAR_COLS <- c("sigma", "xi")
FEAT_UNITS <- c(64, 32)
NU_MATERN <- 1

M_BOOT <- 200; BOOT_CORES <- 16; BOOT_SEED_BASE <- 30000
N_REALISATIONS <- 99; N_ENVELOPE <- 99; ENV_SEED_BASE <- 40000

MESH_CUTOFF <- 5000; MESH_OFFSET <- c(100, 20000); MESH_EDGE_DIV <- 240
INT_STRATEGY <- "ccd"

# ---- 1. Design and the two catalogues ---------------------------------------

META <- readRDS(file.path(SIM_DIR, "design_meta.rds"))
if (!identical(META$field_model, "matern") || META$nu_matern != NU_MATERN)
  stop("Incompatible latent field in design_meta.rds.")
NR <- length(META$r); NB <- length(META$beta_names); FEAT_COLS <- META$feat_bank

OBS_FULL <- readRDS(file.path(SIM_DIR, "observed_inputs.rds"))
W        <- Window(OBS_FULL$X)
WMASK    <- as.mask(W, dimyx = META$dim)

G  <- st_read(CATALOGUE, quiet = TRUE)
XY <- st_coordinates(G)
IN <- inside.owin(XY[, 1], XY[, 2], W)
keep <- IN & !is.na(G$MAGNITUD_MW) & G$MAGNITUD_MW >= MC_THRESHOLD
X_MC <- unique(ppp(XY[keep, 1], XY[keep, 2], window = W, check = FALSE))
message(sprintf("Catalogue: %d events in W | Mw >= %.2f: %d (%.1f%% kept)",
                OBS_FULL$N, MC_THRESHOLD, npoints(X_MC), 100 * npoints(X_MC) / OBS_FULL$N))

COVS <- lapply(META$cov_names, function(n) readRDS(file.path("data", paste0(n, "_im_scaled.rds"))))
names(COVS) <- META$cov_names
COVS <- lapply(COVS, function(z) as.im(z, xy = COVS[[1]]))
COVS <- lapply(COVS, function(z) z[W, drop = FALSE])
names(COVS) <- META$cov_names

fit_first_order <- function(pp) {
  fo <- as.formula(paste("pp ~", META$rhs))
  environment(fo) <- environment()
  ppm(fo, covariates = COVS, nd = META$nquad_ppm)
}

QUAD_TESS <- lapply(META$nquad_bank, function(k) list(k = k, tess = quadrats(W, nx = k, ny = k)))

quad_stats <- function(qc) {
  if (is.null(qc) || length(qc) < 4) return(c(var = 0, VMR = 1, range_ratio = 1))
  v <- var(qc); m <- mean(qc)
  c(var = v, VMR = if (m > 0) v / m else 1, range_ratio = max(qc) / (min(qc) + 1))
}
kde_stats <- function(pp, bw, dimyx) {
  kde <- tryCatch(density.ppp(pp, sigma = bw, dimyx = dimyx), error = function(e) NULL)
  if (is.null(kde)) return(c(var = 0, skew = 0, kurt = 0, entropy = 0, cv = 0))
  vals <- as.vector(kde$v); vals <- vals[!is.na(vals)]; vals <- pmax(vals, 0)
  m <- mean(vals); s <- sd(vals)
  if (s > 0) { z <- (vals - m) / s; sk <- mean(z^3); ku <- mean(z^4) - 3 } else { sk <- 0; ku <- 0 }
  tot <- sum(vals)
  ent <- if (tot > 0) { p <- vals / tot; p <- p[p > 0]; -sum(p * log(p)) / log(length(p)) } else 0
  c(var = s^2, skew = sk, kurt = ku, entropy = ent, cv = if (m > 0) s / m else 0)
}
extract_features <- function(pp) {
  quad_bank <- do.call(c, unname(lapply(QUAD_TESS, function(z) {
    qc <- tryCatch(as.vector(quadratcount(pp, tess = z$tess)), error = function(e) NULL)
    st <- quad_stats(qc); setNames(as.list(st), paste0("quad_", names(st), "_k", z$k))
  })))
  kde_bank <- do.call(c, unname(lapply(META$kde_bw_bank, function(b) {
    st <- kde_stats(pp, b, META$kde_dimyx_bank)
    setNames(as.list(st), paste0("kde_", names(st), "_bw", round(b / 1000)))
  })))
  c(quad_bank, kde_bank)
}

# The same two-step chain as script 01: ppm trend, floored intensity, centred
# inhomogeneous L curve and the 43 descriptors.
observed_inputs <- function(pp) {
  fit  <- fit_first_order(pp)
  lam  <- predict(fit, type = "trend", locations = WMASK)
  flv  <- META$lambda_floor * npoints(pp) / area.owin(W)
  lamf <- eval.im(pmax(lam, flv))
  L    <- Linhom(pp, lambda = lamf, correction = "border", r = META$r)$border
  stopifnot(!anyNA(L))
  c(list(X = pp, N = npoints(pp), beta_hat = unname(coef(fit)), lambda = lam,
         floor_hit = mean(lam[pp] < flv), L = L - META$r),
    extract_features(pp))
}

message("Building the two-step inputs of both catalogues...")
IN_FULL <- observed_inputs(OBS_FULL$X)
IN_MC   <- observed_inputs(X_MC)
saveRDS(IN_MC, out_f("inputs_mc.rds"))

# The reimplementation must reproduce the stored inputs of the full catalogue.
chk <- c(beta = max(abs(IN_FULL$beta_hat - OBS_FULL$beta_hat)),
         L    = max(abs(IN_FULL$L - OBS_FULL$L)),
         F    = max(abs(unlist(IN_FULL[FEAT_COLS]) - unlist(OBS_FULL[FEAT_COLS]))))
message(sprintf("Check vs observed_inputs.rds: max |diff| beta %.2e | L %.2e m | features %.2e",
                chk[["beta"]], chk[["L"]], chk[["F"]]))
if (chk[["beta"]] > 1e-4) warning("The first-order fit differs from the stored one.")

cat("\n--- ppm coefficients ---\n")
print(round(rbind(all = IN_FULL$beta_hat, mc = IN_MC$beta_hat) %>%
              `colnames<-`(META$beta_names), 3))

# ---- 2. Training set, split and scalers (as in script 02) -------------------

load_sims <- function(dir) {
  files <- list.files(dir, pattern = "^sim_[0-9]+\\.rds$", full.names = TRUE)
  message(sprintf("Reading %d chunks from %s ...", length(files), dir))
  parts <- lapply(files, function(f) {
    s <- readRDS(f); s <- s[!vapply(s, is.null, logical(1))]
    if (length(s) == 0) return(NULL)
    list(L = do.call(rbind, lapply(s, `[[`, "L")),
         N = matrix(vapply(s, `[[`, numeric(1), "N"), ncol = 1),
         B = t(vapply(s, function(z) z$beta_hat, numeric(NB))),
         F = t(vapply(s, function(z) unlist(z[FEAT_COLS]), numeric(length(FEAT_COLS)))),
         Y = t(vapply(s, function(z) unlist(z[PAR_COLS]), numeric(length(PAR_COLS)))),
         hit = vapply(s, `[[`, numeric(1), "floor_hit"))
  })
  parts <- parts[!vapply(parts, is.null, logical(1))]
  d <- lapply(c("L", "N", "B", "F", "Y"), function(nm) do.call(rbind, lapply(parts, `[[`, nm)))
  names(d) <- c("L", "N", "B", "F", "Y")
  d$hit <- unlist(lapply(parts, `[[`, "hit"))
  colnames(d$N) <- "N"; colnames(d$F) <- FEAT_COLS; colnames(d$Y) <- PAR_COLS; colnames(d$B) <- META$beta_names
  d
}
DAT <- load_sims(SIM_DIR)
ok <- is.finite(rowSums(DAT$L)) & is.finite(rowSums(DAT$F)) & is.finite(rowSums(DAT$B)) &
      is.finite(rowSums(DAT$Y)) & is.finite(DAT$N[, 1])
if (DROP_DEGENERATE) ok <- ok & (DAT$hit == 0)
DAT <- c(lapply(DAT[c("L", "N", "B", "F", "Y")], function(m) m[ok, , drop = FALSE]), list(hit = DAT$hit[ok]))
n <- nrow(DAT$Y)
message(sprintf("Usable realisations: %d", n))

qbin <- function(x, k) {
  if (k <= 1) return(rep(1L, length(x)))
  br <- unique(quantile(x, seq(0, 1, length.out = k + 1), na.rm = TRUE))
  if (length(br) < 2) return(rep(1L, length(x)))
  cut(x, breaks = br, labels = FALSE, include.lowest = TRUE)
}
STRAT_BINS <- 1L
for (k in 8:1) if (n / (k^length(PAR_COLS) * 2) >= 20) { STRAT_BINS <- k; break }
strata <- do.call(interaction, c(lapply(PAR_COLS, function(p) qbin(DAT$Y[, p], STRAT_BINS)),
                                 list(DAT$hit > 0), list(drop = TRUE)))
stratified_split <- function(strata, fracs, seed = 123) {
  set.seed(seed)
  brk <- c(0, head(cumsum(fracs), -1), 1); grp <- integer(length(strata))
  for (lv in levels(strata)) {
    idx <- sample(which(strata == lv)); pos <- (seq_along(idx) - 0.5) / length(idx)
    grp[idx] <- cut(pos, breaks = brk, labels = FALSE, include.lowest = TRUE)
  }
  grp
}
grp <- stratified_split(strata, c(FRAC_TRAIN, FRAC_VAL, 1 - FRAC_TRAIN - FRAC_VAL))
idx_tr <- which(grp == 1L); idx_va <- which(grp == 2L); idx_te <- which(grp == 3L)
message(sprintf("Split: %d / %d / %d", length(idx_tr), length(idx_va), length(idx_te)))

fit_scaler <- function(m) {
  mu <- colMeans(m); sdv <- apply(m, 2, sd); sdv[sdv == 0 | !is.finite(sdv)] <- 1
  list(mu = mu, sd = sdv)
}
apply_scaler <- function(m, sc) {
  z <- scale(m, center = sc$mu, scale = sc$sd); z[!is.finite(z)] <- 0
  pmax(pmin(z, CLIP_SD), -CLIP_SD)
}
sc_S2 <- local({
  M <- DAT$L[idx_tr, , drop = FALSE]
  s0 <- median(apply(abs(M), 1, max, na.rm = TRUE), na.rm = TRUE); if (!is.finite(s0) || s0 <= 0) s0 <- 1
  v <- if (S2_TRANSFORM == "asinh") asinh(as.vector(M) / s0) else as.vector(M)
  s <- sd(v, na.rm = TRUE); if (s == 0 || !is.finite(s)) s <- 1
  list(s0 = s0, mu = mean(v, na.rm = TRUE), sd = s)
})
transform_S2 <- function(M) {
  z <- if (S2_TRANSFORM == "asinh") asinh(M / sc_S2$s0) else M
  z <- (z - sc_S2$mu) / sc_S2$sd; z[!is.finite(z)] <- 0
  pmax(pmin(z, CLIP_SD), -CLIP_SD)
}
raw_derivative <- function(M) { d <- t(apply(M, 1, diff)) / diff(META$r)[1]; cbind(d[, 1, drop = FALSE], d) }
sc_dL <- local({ s <- sd(as.vector(raw_derivative(DAT$L[idx_tr, , drop = FALSE])), na.rm = TRUE)
                 if (!is.finite(s) || s == 0) 1 else s })
transform_dL <- function(M) { z <- raw_derivative(M) / sc_dL; z[!is.finite(z)] <- 0; pmax(pmin(z, CLIP_SD), -CLIP_SD) }
R_CHANNEL <- META$r / max(META$r)
build_S2_array <- function(L) {
  a <- array(NA_real_, c(nrow(L), NR, S2_CHANNELS))
  a[, , 1] <- transform_S2(L); a[, , 2] <- transform_dL(L); a[, , 3] <- rep(R_CHANNEL, each = nrow(L))
  a
}
sc_N <- fit_scaler(DAT$N[idx_tr, , drop = FALSE]); sc_B <- fit_scaler(DAT$B[idx_tr, , drop = FALSE])
sc_F <- fit_scaler(DAT$F[idx_tr, , drop = FALSE])
make_inputs <- function(idx) list(build_S2_array(DAT$L[idx, , drop = FALSE]),
                                  apply_scaler(DAT$B[idx, , drop = FALSE], sc_B),
                                  apply_scaler(DAT$N[idx, , drop = FALSE], sc_N),
                                  apply_scaler(DAT$F[idx, , drop = FALSE], sc_F))
TARGET_LINK <- c(sigma = "sqrt", xi = "log")[PAR_COLS]
apply_link  <- function(m) { o <- m; for (j in seq_along(PAR_COLS)) o[, j] <- switch(TARGET_LINK[j], sqrt = sqrt(m[, j]), log = log(m[, j])); o }
invert_link <- function(m) { o <- m; for (j in seq_along(PAR_COLS)) o[, j] <- switch(TARGET_LINK[j], sqrt = m[, j]^2, log = exp(m[, j])); o }
Y_link <- apply_link(DAT$Y); sc_Y <- fit_scaler(Y_link[idx_tr, , drop = FALSE])
Y_tr <- scale(Y_link[idx_tr, ], center = sc_Y$mu, scale = sc_Y$sd)
Y_va <- scale(Y_link[idx_va, ], center = sc_Y$mu, scale = sc_Y$sd)
invert_target <- function(z) {
  m <- sweep(sweep(z, 2, sc_Y$sd, `*`), 2, sc_Y$mu, `+`); out <- invert_link(m); colnames(out) <- PAR_COLS; out
}
X_tr <- make_inputs(idx_tr); X_va <- make_inputs(idx_va); X_te <- make_inputs(idx_te)

# ---- 3. Ensemble: retrain with the seeds of script 02, or reload ------------

conv_block <- function(x, filters, dilation = 1L, padding = "same") {
  x %>% layer_conv_1d(filters = filters, kernel_size = 7, dilation_rate = dilation,
                      padding = padding, activation = NULL) %>%
    layer_batch_normalization() %>% layer_activation("relu")
}
pool_layer <- function(x, size = 5L) layer_average_pooling_1d(x, pool_size = size)
build_S2_branch <- function(input_layer) {
  x <- input_layer %>% conv_block(64L, dilation = 1L) %>% pool_layer() %>%
    conv_block(64L, dilation = 2L) %>% pool_layer() %>% conv_block(S2_LAST_FILTERS, dilation = 4L)
  x <- layer_flatten(x)
  x %>% layer_dense(units = S2_PROJECT, activation = "relu") %>% layer_batch_normalization()
}
build_dense_branch <- function(input_layer, units = c(32, 16)) {
  input_layer %>% layer_dense(units = units[1], activation = "relu") %>%
    layer_batch_normalization() %>% layer_dense(units = units[2], activation = "relu")
}
make_callbacks <- function(es_patience = 15, lr_patience = 7) list(
  callback_early_stopping(monitor = "val_loss", patience = es_patience, restore_best_weights = TRUE, verbose = 0),
  callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.5, patience = lr_patience, min_lr = 1e-6, verbose = 0))
train_once <- function(tag) {
  in_S2 <- layer_input(shape = c(NR, S2_CHANNELS), name = paste0("S2", tag))
  in_B  <- layer_input(shape = c(ncol(DAT$B)), name = paste0("S1", tag))
  in_N  <- layer_input(shape = c(1), name = paste0("N", tag))
  in_F  <- layer_input(shape = c(length(FEAT_COLS)), name = paste0("Ifeat", tag))
  branches <- list(build_S2_branch(in_S2), build_dense_branch(in_B, c(32, 16)), in_N, build_dense_branch(in_F, FEAT_UNITS))
  output <- layer_concatenate(branches) %>%
    layer_dense(units = 64, activation = "relu") %>% layer_dropout(rate = DROPOUT) %>%
    layer_dense(units = 32, activation = "relu") %>% layer_dropout(rate = DROPOUT) %>%
    layer_dense(units = length(PAR_COLS), activation = "linear")
  model <- keras_model(inputs = list(in_S2, in_B, in_N, in_F), outputs = output)
  model %>% compile(loss = "mse", optimizer = optimizer_adam(learning_rate = LR), metrics = list("mae"))
  model %>% fit(x = X_tr, y = Y_tr, epochs = EPOCHS, batch_size = BATCH_SIZE,
                validation_data = list(X_va, Y_va), callbacks = make_callbacks(), verbose = 0)
  model
}

model_dir <- function(i) out_f(sprintf("model_s%d", i))
if (all(dir.exists(vapply(seq_len(N_SEEDS), model_dir, character(1))))) {
  message("Loading the saved ensemble...")
  MODELS <- lapply(seq_len(N_SEEDS), function(i) load_model_tf(model_dir(i)))
} else {
  message(sprintf("Training %d seeds (seeds 1000+i, as in script 02)...", N_SEEDS))
  t_train <- system.time({
    MODELS <- lapply(seq_len(N_SEEDS), function(i) {
      tensorflow::set_random_seed(1000 + i)
      m <- train_once(sprintf("_s%d", i)); message(sprintf("  seed %d done", i)); m
    })
  })
  message(sprintf("Training time: %.1f min", t_train[["elapsed"]] / 60))
  for (i in seq_len(N_SEEDS)) save_model_tf(MODELS[[i]], model_dir(i))
}

ensemble_predict <- function(inputs)
  Reduce(`+`, lapply(MODELS, function(m) invert_target(predict(m, inputs, verbose = 0)))) / N_SEEDS

# Reproduction check on the test split (paper: R2 0.881 for sigma, 0.628 for xi).
P_te <- ensemble_predict(X_te)
r2 <- function(a, b) 1 - sum((b - a)^2) / sum((a - mean(a))^2)
test_metrics <- tibble(param = PAR_COLS,
                       R2 = c(r2(DAT$Y[idx_te, "sigma"], P_te[, "sigma"]), r2(DAT$Y[idx_te, "xi"], P_te[, "xi"])))
cat("\n--- Retrained ensemble, test split ---\n"); print(test_metrics)
write_csv(test_metrics, out_f("test_metrics.csv"))

# ---- 4. Estimates for both catalogues ---------------------------------------

catalogue_inputs <- function(o) list(build_S2_array(matrix(o$L, nrow = 1)),
                                     apply_scaler(matrix(o$beta_hat, nrow = 1), sc_B),
                                     apply_scaler(matrix(o$N, nrow = 1), sc_N),
                                     apply_scaler(matrix(unlist(o[FEAT_COLS]), nrow = 1), sc_F))
EST <- bind_rows(
  tibble(catalogue = "all", N = IN_FULL$N, !!!as.list(setNames(as.numeric(ensemble_predict(catalogue_inputs(IN_FULL))), PAR_COLS))),
  tibble(catalogue = sprintf("Mw>=%.2f", MC_THRESHOLD), N = IN_MC$N,
         !!!as.list(setNames(as.numeric(ensemble_predict(catalogue_inputs(IN_MC))), PAR_COLS)))) %>%
  mutate(xi_km = xi / 1000)
cat("\n--- CNN estimates ---\n"); print(EST %>% mutate(across(where(is.numeric), ~ round(.x, 3))))
write_csv(EST, out_f("estimates.csv"))
th <- as.list(EST[2, ])

# ---- 5. Parametric bootstrap on the filtered catalogue ----------------------

LAM_MC <- IN_MC$lambda
simulate_fitted <- function(seed, theta = th) {
  set.seed(seed)
  Y <- spatstat.random::rGRFmatern(W = W, mu = 0, var = theta$sigma^2, scale = theta$xi,
                                   nu = NU_MATERN, dimyx = META$dim)
  cst <- 0.5 * theta$sigma^2
  rpoispp(eval.im(LAM_MC * exp(Y - cst)))[W]
}
boot_one <- function(i) {
  X <- tryCatch(simulate_fitted(BOOT_SEED_BASE + i), error = function(e) NULL)
  if (is.null(X)) return(NULL)
  nn <- npoints(X); if (nn < 100 || nn > 80000) return(NULL)
  o <- tryCatch(observed_inputs(X), error = function(e) NULL)
  if (is.null(o)) return(NULL)
  fv <- unlist(o[FEAT_COLS]); if (anyNA(fv)) return(NULL)
  list(L = o$L, bhat = o$beta_hat, N = nn, F = fv)
}
message(sprintf("\nBootstrap: %d replicates at (sigma = %.3f, xi = %.1f km), %d cores",
                M_BOOT, th$sigma, th$xi / 1000, BOOT_CORES))
t_boot <- system.time(BOOT_RAW <- parallel::mclapply(seq_len(M_BOOT), boot_one,
                                                     mc.cores = BOOT_CORES, mc.preschedule = FALSE))
BOOT_OK <- Filter(function(o) is.list(o) && !is.null(o$L), BOOT_RAW); n_ok <- length(BOOT_OK)
message(sprintf("Valid replicates: %d/%d (%.1f min)", n_ok, M_BOOT, t_boot[["elapsed"]] / 60))
boot_in <- list(build_S2_array(do.call(rbind, lapply(BOOT_OK, `[[`, "L"))),
                apply_scaler(do.call(rbind, lapply(BOOT_OK, `[[`, "bhat")), sc_B),
                apply_scaler(matrix(vapply(BOOT_OK, `[[`, numeric(1), "N"), ncol = 1), sc_N),
                apply_scaler(do.call(rbind, lapply(BOOT_OK, `[[`, "F")), sc_F))
BOOT_PRED <- ensemble_predict(boot_in)
theta_hat <- setNames(c(th$sigma, th$xi), PAR_COLS)
boot_ic <- map_dfr(PAR_COLS, function(p) {
  v <- BOOT_PRED[, p]; q <- quantile(v, c(0.025, 0.5, 0.975)); hat <- theta_hat[[p]]
  tibble(param = p, theta_hat = hat, n_boot = n_ok, boot_mean = mean(v), boot_bias = mean(v) - hat,
         boot_sd = sd(v), ic_lo = q[[1]], median = q[[2]], ic_hi = q[[3]])
})
cat("\n--- Bootstrap (95% percentile), Mw-filtered catalogue ---\n")
print(boot_ic %>% mutate(across(where(is.numeric), ~ signif(.x, 4))))
write_csv(as_tibble(BOOT_PRED) %>% mutate(replicate = seq_len(n_ok), .before = 1), out_f("bootstrap_theta.csv"))
write_csv(boot_ic, out_f("bootstrap_ic.csv"))

# ---- 6. Fitted counts, envelope and DCLF on the filtered catalogue ----------

counts <- vapply(seq_len(N_REALISATIONS), function(i) npoints(simulate_fitted(i)), numeric(1))
fitted_counts <- tibble(observed = IN_MC$N, sim_mean = mean(counts), sim_min = min(counts), sim_max = max(counts),
                        inside = IN_MC$N >= min(counts) & IN_MC$N <= max(counts))
cat("\n--- Counts of the fitted realisations ---\n"); print(fitted_counts)
write_csv(fitted_counts, out_f("fitted_counts.csv"))

R_ENV <- seq(0, max(META$r), length.out = 257)
rescale_lambda <- function(lam, pp) { tot <- integral.im(lam); if (!is.finite(tot) || tot <= 0) return(lam); eval.im(lam * (npoints(pp) / tot)) }
k_curve <- function(X) Kinhom(X, lambda = rescale_lambda(LAM_MC, X), correction = "border", r = R_ENV)$border
message(sprintf("Envelope: %d realisations...", N_ENVELOPE))
K_obs  <- k_curve(X_MC)
K_sims <- t(vapply(seq_len(N_ENVELOPE), function(i) k_curve(simulate_fitted(ENV_SEED_BASE + i)), numeric(length(K_obs))))
dclf_p <- function(obs, sims) {
  dr <- diff(R_ENV)[1]; nsim <- nrow(sims)
  T_obs <- sum((obs - colMeans(sims))^2) * dr
  T_sim <- vapply(seq_len(nsim), function(i) sum((sims[i, ] - colMeans(sims[-i, , drop = FALSE]))^2) * dr, numeric(1))
  (1 + sum(T_sim >= T_obs)) / (1 + nsim)
}
p_dclf <- dclf_p(K_obs, K_sims)
cat(sprintf("\n--- DCLF test (Mw-filtered): p = %.3f\n", p_dclf))
write_csv(tibble(r = R_ENV, obs = K_obs, med = colMeans(K_sims), lo = apply(K_sims, 2, min), hi = apply(K_sims, 2, max)),
          out_f("envelope.csv"))
write_csv(tibble(p_DCLF = p_dclf), out_f("dclf_pvalue.csv"))

# ---- 7. INLA-SPDE on the filtered catalogue (as script 03, without the map) -

suppressMessages({ library(INLA); library(Matrix) })
mesh_cache <- file.path(OUT_DIR, sprintf("inla_mesh_d%d_c%d_o%d.rds", MESH_EDGE_DIV, MESH_CUTOFF, MESH_OFFSET[2]))
if (!file.exists(mesh_cache)) stop("Mesh cache not found; run scripts/03_fit_inla_spde.R first: ", mesh_cache)
.mc <- readRDS(mesh_cache); mesh <- .mc$mesh; w <- .mc$w; n_mesh <- mesh$n
COV_FULL <- lapply(META$cov_names, function(n) readRDS(file.path("data", paste0(n, "_im_scaled.rds"))))
names(COV_FULL) <- META$cov_names
COV_FULL <- lapply(COV_FULL, function(z) as.im(z, xy = COV_FULL[[1]])); names(COV_FULL) <- META$cov_names
COV_MAT <- lapply(COV_FULL, as.matrix)
eval_covar <- function(coords) {
  coords <- as.matrix(coords)
  out <- matrix(NA_real_, nrow(coords), length(META$cov_names), dimnames = list(NULL, META$cov_names))
  for (k in seq_along(COV_FULL)) {
    px <- nearest.valid.pixel(coords[, 1], coords[, 2], COV_FULL[[k]]); out[, k] <- COV_MAT[[k]][cbind(px$row, px$col)]
  }
  out
}
XI_TO_RANGE <- sqrt(8 * NU_MATERN)
spde <- inla.spde2.pcmatern(mesh = mesh, alpha = NU_MATERN + 1,
                            prior.range = c(XI_TO_RANGE * mean(META$xi_range), 0.5),
                            prior.sigma = c(mean(META$sigma_range), 0.5))
idx <- inla.spde.make.index("spatial.field", spde$n.spde)
pts <- cbind(X_MC$x, X_MC$y); nX <- nrow(pts)
A <- rbind(Diagonal(n_mesh, rep(1, n_mesh)), inla.spde.make.A(mesh, pts))
y <- rep(0:1, c(n_mesh, nX)); e <- c(w, rep(0, nX))
covs <- as.data.frame(rbind(eval_covar(mesh$loc[, 1:2]), eval_covar(pts))); covs$b0 <- 1
stopifnot(!anyNA(covs))
stk <- inla.stack(data = list(y = y, e = e), A = list(A, 1), effects = list(idx, as.list(covs)), tag = "pp")
formula <- as.formula(paste("y ~ 0 + b0 +", paste(META$cov_names, collapse = " + "), "+ f(spatial.field, model = spde)"))
beta_lo <- META$beta_range$lo; beta_hi <- META$beta_range$hi
ctrl_fixed <- list(mean = list(b0 = 0, default = (beta_lo + beta_hi) / 2),
                   prec = list(b0 = 0.001, default = 12 / (beta_hi - beta_lo)^2))
message(sprintf("\nINLA-SPDE on the Mw-filtered catalogue (%d events, %s)...", nX, INT_STRATEGY))
t_inla <- system.time({
  res <- inla(formula, family = "poisson", data = inla.stack.data(stk), E = inla.stack.data(stk)$e,
              control.inla = list(int.strategy = INT_STRATEGY, cmin = 0), control.fixed = ctrl_fixed,
              control.predictor = list(A = inla.stack.A(stk), compute = FALSE),
              control.compute = list(config = FALSE, cpo = FALSE, dic = FALSE, waic = FALSE))
})
hp <- res$summary.hyperpar; ir <- grep("^Range", rownames(hp))[1]; isd <- grep("^Stdev", rownames(hp))[1]
INLA_OUT <- tibble(method = sprintf("INLA-SPDE (%s)", INT_STRATEGY), N = nX, time_s = t_inla[["elapsed"]],
                   sigma = hp[isd, "mean"], sigma_lo = hp[isd, "0.025quant"], sigma_hi = hp[isd, "0.975quant"],
                   range_km = hp[ir, "mean"] / 1000, range_lo_km = hp[ir, "0.025quant"] / 1000, range_hi_km = hp[ir, "0.975quant"] / 1000,
                   xi_km = hp[ir, "mean"] / XI_TO_RANGE / 1000, xi_lo_km = hp[ir, "0.025quant"] / XI_TO_RANGE / 1000,
                   xi_hi_km = hp[ir, "0.975quant"] / XI_TO_RANGE / 1000, mode_status = res$mode$mode.status)
cat("\n--- INLA-SPDE (Mw-filtered) ---\n"); print(INLA_OUT %>% mutate(across(where(is.numeric), ~ signif(.x, 4))))
write_csv(INLA_OUT, out_f("inla_observed.csv"))
write_csv(as_tibble(res$summary.fixed, rownames = "coef") %>% select(coef, mean, sd, `0.025quant`, `0.975quant`),
          out_f("inla_betas.csv"))

# ---- 8. Summary: full catalogue (scripts 02-03) vs Mw-filtered (this run) ---

read_first <- function(cands) { f <- cands[file.exists(cands)][1]; if (is.na(f)) NULL else read_csv(f, show_col_types = FALSE) }
ic_full   <- read_first(file.path(OUT_DIR, "bootstrap_ic.csv"))
p_full    <- read_first(file.path(OUT_DIR, "dclf_pvalue.csv"))
inla_full <- read_first(file.path(OUT_DIR, c("inla_catalog_observed.csv", "inla_catalogo_observado.csv")))
pick <- function(df, ...) { nms <- c(...); nm <- nms[nms %in% names(df)][1]; df[[nm]] }
row_full <- tibble(
  catalogue = "all", N = IN_FULL$N, beta_inversa = IN_FULL$beta_hat[2],
  cnn_sigma = EST$sigma[1], cnn_sigma_paper = pick(ic_full, "theta_hat")[1],
  cnn_sigma_lo = pick(ic_full, "ic_lo")[1], cnn_sigma_hi = pick(ic_full, "ic_hi")[1],
  cnn_xi_km = EST$xi_km[1], cnn_xi_lo_km = pick(ic_full, "ic_lo")[2] / 1000, cnn_xi_hi_km = pick(ic_full, "ic_hi")[2] / 1000,
  p_dclf = p_full$p_DCLF[1],
  inla_sigma = inla_full$sigma, inla_sigma_lo = inla_full$sigma_lo, inla_sigma_hi = inla_full$sigma_hi,
  inla_xi_km = inla_full$xi_km, inla_xi_lo_km = inla_full$xi_lo_km, inla_xi_hi_km = inla_full$xi_hi_km)
row_mc <- tibble(
  catalogue = sprintf("Mw>=%.2f", MC_THRESHOLD), N = IN_MC$N, beta_inversa = IN_MC$beta_hat[2],
  cnn_sigma = th$sigma, cnn_sigma_paper = NA_real_,
  cnn_sigma_lo = boot_ic$ic_lo[1], cnn_sigma_hi = boot_ic$ic_hi[1],
  cnn_xi_km = th$xi / 1000, cnn_xi_lo_km = boot_ic$ic_lo[2] / 1000, cnn_xi_hi_km = boot_ic$ic_hi[2] / 1000,
  p_dclf = p_dclf,
  inla_sigma = INLA_OUT$sigma, inla_sigma_lo = INLA_OUT$sigma_lo, inla_sigma_hi = INLA_OUT$sigma_hi,
  inla_xi_km = INLA_OUT$xi_km, inla_xi_lo_km = INLA_OUT$xi_lo_km, inla_xi_hi_km = INLA_OUT$xi_hi_km)
SUMMARY <- bind_rows(row_full, row_mc)
cat("\n=== SUMMARY ===\n"); print(as.data.frame(SUMMARY %>% mutate(across(where(is.numeric), ~ signif(.x, 4)))))
write_csv(SUMMARY, out_f("summary.csv"))

fmt <- function(x, lo, hi, d = 2) sprintf(paste0("%.", d, "f [%.", d, "f, %.", d, "f]"), x, lo, hi)
writeLines(c(
  "\\begin{table}[htbp]", "\\centering",
  sprintf("\\caption{Sensitivity of the catalogue estimates to the completeness magnitude ($M_w \\ge %.2f$)}", MC_THRESHOLD),
  "\\label{tab:sensitivity}",
  "\\begin{tabular}{lrrccccc}", "\\toprule",
  "Catalogue & $N$ & $\\hat\\beta_{\\mathrm{faults}}$ & $\\hat\\sigma$ (CNN) & $\\hat\\xi$ km (CNN) & $p_{\\mathrm{DCLF}}$ & $\\hat\\sigma$ (INLA) & $\\hat\\xi$ km (INLA)\\\\",
  "\\midrule",
  vapply(seq_len(nrow(SUMMARY)), function(i) { s <- SUMMARY[i, ]
    sprintf("%s & %d & %.2f & %s & %s & %.2f & %s & %s\\\\",
            if (i == 1) "All events" else sprintf("$M_w \\ge %.2f$", MC_THRESHOLD), s$N, s$beta_inversa,
            fmt(s$cnn_sigma, s$cnn_sigma_lo, s$cnn_sigma_hi), fmt(s$cnn_xi_km, s$cnn_xi_lo_km, s$cnn_xi_hi_km, 1),
            s$p_dclf, fmt(s$inla_sigma, s$inla_sigma_lo, s$inla_sigma_hi), fmt(s$inla_xi_km, s$inla_xi_lo_km, s$inla_xi_hi_km, 1)) },
    character(1)),
  "\\bottomrule", "\\end{tabular}",
  "\\footnotetext{CNN: 95\\% percentile interval of the parametric bootstrap (200 replicates). INLA--SPDE: posterior mean and 95\\% credible interval.}",
  "\\end{table}"), out_f("summary_table.tex"))

# Bootstrap distributions of both catalogues, for a supplementary figure.
bt_full <- read_first(file.path(OUT_DIR, "bootstrap_theta.csv"))
if (!is.null(bt_full)) {
  bd <- bind_rows(bt_full %>% transmute(catalogue = "All events", sigma, xi_km = xi / 1000),
                  as_tibble(BOOT_PRED) %>% transmute(catalogue = sprintf("Mw >= %.2f", MC_THRESHOLD), sigma, xi_km = xi / 1000)) %>%
    pivot_longer(c(sigma, xi_km), names_to = "param", values_to = "value") %>%
    mutate(param = factor(param, c("sigma", "xi_km"), c("hat(sigma)", "hat(xi)~(km)")))
  p <- ggplot(bd, aes(value, colour = catalogue, fill = catalogue)) +
    geom_density(alpha = 0.25, linewidth = 0.6) +
    facet_wrap(~ param, scales = "free", labeller = label_parsed) +
    scale_colour_manual(values = c("#1f77b4", "#d62728"), name = NULL, aesthetics = c("colour", "fill")) +
    labs(x = NULL, y = "Bootstrap density") + theme_bw(base_size = 11) +
    theme(panel.grid = element_blank(), legend.position = "bottom")
  ggsave(out_f("bootstrap_compare.pdf"), p, width = 7, height = 3.4, device = cairo_pdf)
}
message(sprintf("\nDone. Outputs %s_* in %s/", PREFIX, OUT_DIR))
