# Two-step INLA-SPDE on simulated test realisations, and comparison with the
# CNN variants of script 07.
#
# As in Zou et al. (arXiv:2607.13113), INLA-SPDE runs in two stages: the trend
# is fixed at the ppm estimate beta_hat stored for each realisation (the same
# stage-1 input the CNN receives) and INLA estimates only the latent field.
# Mesh, PC priors and offset are those of the catalogue fit (scripts 04, 08);
# hyperparameters are held at their posterior mode ("eb") for cost.
#
# Realisations: 500 drawn at random from the test split of script 02. The first
# 200 are sort(sample(test, 200)) with seed 20260803;
# the other 300 are drawn from the rest of the test split with seed 20261008.
#
# Script 01 drops the point patterns before saving, so each pattern is
# regenerated bit for bit: the generator index i is recovered by matching
# (sigma, xi) against sample_params(N_SIMS, BASE_SEED), and the field and the
# Poisson draw are repeated with set.seed(BASE_SEED + i). Every regenerated
# pattern must reproduce the stored count N, or the script stops.
#
# Each fit is cached in outputs/inla2s_sim_cache/, so the run can be resumed.
# Nothing written by scripts 01-08 is modified. Outputs carry the prefix
# "inla2s_sim_".
#
# Usage:  Rscript scripts/09_inla_two_step_simulations.R   (from the repository root)
#         INLA_WORKERS=8 by default (each worker uses 2 threads).

stopifnot(file.exists("scripts/04_intensity_map.R"),
          file.exists("outputs/ablation_pred_test.csv"))

T0 <- Sys.time()

# ---- 1. Setup of script 04 (design, covariates, mesh, SPDE) ----------------
# As in script 08, the lines that read the CNN catalogue estimate are skipped.
local({
  lines     <- readLines("scripts/04_intensity_map.R")
  stop_at   <- grep("^# ---- Fit:", lines)[1]
  skip_from <- grep("^cnn <- read_csv", lines)[1]
  skip_to   <- grep("BHAT_OBS\\), collapse", lines)[1]
  stopifnot(!is.na(stop_at), !is.na(skip_from), !is.na(skip_to), skip_to > skip_from)
  keep <- setdiff(seq_len(stop_at - 1), skip_from:skip_to)
  eval(parse(text = lines[keep], keep.source = FALSE), envir = globalenv())
})
suppressMessages(library(parallel))

PREFIX     <- "inla2s_sim"
out_f      <- function(name) file.path(OUT_DIR, paste0(PREFIX, "_", name))
CACHE_DIR  <- file.path(OUT_DIR, "inla2s_sim_cache")
dir.create(CACHE_DIR, showWarnings = FALSE, recursive = TRUE)

N_WORKERS  <- as.integer(Sys.getenv("INLA_WORKERS", "8"))
N_FIRST    <- 200;  SEED_FIRST <- 20260803
N_EXTRA    <- 300;  SEED_EXTRA <- 20261008

# Generator constants of script 01.
GEN_N_SIMS <- 20000; GEN_BASE_SEED <- 42; GEN_MAX_POINTS <- 80000

# ---- 2. Test realisations and their stored scalars --------------------------
# gid indexes the usable realisations in file order, as in scripts 02 and 07.
sim_files <- sort(list.files(SIM_DIR, pattern = "^sim_[0-9]+\\.rds$", full.names = TRUE))
SC <- bind_rows(lapply(sim_files, function(f) {
  s <- readRDS(f); s <- s[!vapply(s, is.null, logical(1))]
  if (!length(s)) return(NULL)
  tibble(file = f, pos = seq_along(s),
         sigma = vapply(s, `[[`, numeric(1), "sigma"),
         xi    = vapply(s, `[[`, numeric(1), "xi"),
         N     = vapply(s, `[[`, numeric(1), "N"),
         ok    = vapply(s, function(z) all(is.finite(z$L)) &&
                          length(z$beta_hat) == N_BETA && all(is.finite(z$beta_hat)) &&
                          all(is.finite(unlist(z[META$feat_bank]))), logical(1)))
}))
SC <- SC %>% filter(ok, is.finite(sigma), is.finite(xi), is.finite(N)) %>%
  mutate(gid = row_number())

PRED <- read_csv(file.path(OUT_DIR, "ablation_pred_test.csv"), show_col_types = FALSE)
test <- sort(PRED$gid)
stopifnot(all(test %in% SC$gid),
          max(abs(SC$sigma[match(PRED$gid, SC$gid)] - PRED$sigma_true)) < 1e-12,
          max(abs(SC$xi[match(PRED$gid, SC$gid)] - PRED$xi_true)) < 1e-6)

set.seed(SEED_FIRST); g1 <- sort(sample(test, N_FIRST))
set.seed(SEED_EXTRA); g2 <- sort(sample(setdiff(test, g1), N_EXTRA))
EVAL <- SC %>% filter(gid %in% c(g1, g2)) %>%
  mutate(batch = ifelse(gid %in% g1, "first_200", "extra_300"))
stopifnot(nrow(EVAL) == N_FIRST + N_EXTRA)
message(sprintf("Test split: %d realisations | evaluated: %d (%d + %d)",
                length(test), nrow(EVAL), N_FIRST, N_EXTRA))

# beta_hat (stage 1) and the true beta of each evaluated realisation.
STORED <- list()
for (f in unique(EVAL$file)) {
  s <- readRDS(f); s <- s[!vapply(s, is.null, logical(1))]
  for (j in which(EVAL$file == f)) {
    z <- s[[EVAL$pos[j]]]
    STORED[[as.character(EVAL$gid[j])]] <- list(bhat = as.numeric(z$beta_hat),
                                                beta = as.numeric(z$beta),
                                                n_target = z$n_target)
  }
}

# ---- 3. Deterministic regeneration of the point patterns --------------------
sample_params_gen <- function(n, seed) {
  set.seed(seed)
  list(sigma    = runif(n, META$sigma_range[1], META$sigma_range[2]),
       xi       = runif(n, META$xi_range[1],    META$xi_range[2]),
       n_target = runif(n, META$n_range[1],     META$n_range[2]),
       slopes   = matrix(runif(n * length(model_covariates),
                               META$beta_range$lo, META$beta_range$hi),
                         nrow = n, ncol = length(model_covariates)))
}
P_GEN <- sample_params_gen(GEN_N_SIMS, GEN_BASE_SEED)
key   <- function(s, x) paste0(sprintf("%.17g", s), "_", sprintf("%.17g", x))
P_KEY <- key(P_GEN$sigma, P_GEN$xi)
stopifnot(!anyDuplicated(P_KEY))
EVAL$i_gen <- match(key(EVAL$sigma, EVAL$xi), P_KEY)
stopifnot(!anyNA(EVAL$i_gen))
# The stored true slopes and n_target must be those of the recovered index.
for (j in seq_len(nrow(EVAL))) {
  z <- STORED[[as.character(EVAL$gid[j])]]
  stopifnot(max(abs(z$beta[-1] - P_GEN$slopes[EVAL$i_gen[j], ])) < 1e-12,
            abs(z$n_target - P_GEN$n_target[EVAL$i_gen[j]]) < 1e-9)
}

# Covariates on the simulation mask, as COVS_SIM in script 01.
WMASK_GEN   <- as.mask(W, dimyx = META$dim)
PIXAREA_GEN <- WMASK_GEN$xstep * WMASK_GEN$ystep
COVS_GEN <- lapply(covar, function(z) nearestValue(as.im(z[W, drop = FALSE], W = WMASK_GEN)))
names(COVS_GEN) <- model_covariates

trend_image_gen <- function(beta) {
  tr <- as.im(beta[1], W = WMASK_GEN)
  for (j in seq_along(model_covariates)) {
    Zj <- COVS_GEN[[j]]; bj <- beta[j + 1]
    tr <- eval.im(tr + bj * Zj)
  }
  tr
}
solve_intercept_gen <- function(slopes, n_target) {
  eta <- trend_image_gen(c(0, slopes))
  v <- eta$v[is.finite(eta$v)]; m <- max(v)
  log(n_target) - (m + log(sum(exp(v - m))) + log(PIXAREA_GEN))
}
# Same RNG consumption as simulate_lgcp() of script 01, cut after rpoispp().
regen_X <- function(i) {
  set.seed(GEN_BASE_SEED + i)
  slopes <- P_GEN$slopes[i, ]
  beta   <- c(solve_intercept_gen(slopes, P_GEN$n_target[i]), slopes)
  Y   <- spatstat.random::rGRFmatern(W = W, mu = 0, var = P_GEN$sigma[i]^2,
                                     scale = P_GEN$xi[i], nu = NU_MATERN,
                                     dimyx = META$dim)
  tr  <- trend_image_gen(beta)
  cst <- 0.5 * P_GEN$sigma[i]^2
  lam <- eval.im(exp(tr + Y - cst))
  if (integral.im(lam) > GEN_MAX_POINTS) stop("generator mismatch at i = ", i)
  rpoispp(lam)[W]
}

# ---- 4. Two-step fit --------------------------------------------------------
# The ppm trend is one column `off` with its coefficient pinned to 1 (an exact
# offset); b0 is free and absorbs the -sigma^2/2 of the centred form.
fit_two_step <- function(X, bhat) {
  p <- cbind(X$x, X$y)
  A <- rbind(IMA, inla.spde.make.A(mesh, p))
  Z <- rbind(Z_NODES, eval_covar(p))
  covs <- data.frame(off = as.vector(cbind(1, Z) %*% bhat), b0 = 1)
  stk  <- inla.stack(data = list(y = rep(0:1, c(n_mesh, nrow(p))),
                                 e = c(w, rep(0, nrow(p)))),
                     A = list(A, 1), effects = list(spde.index, as.list(covs)),
                     tag = "pp")
  inla(y ~ 0 + b0 + off + f(spatial.field, model = spde),
       family = "poisson", data = inla.stack.data(stk), E = inla.stack.data(stk)$e,
       control.inla      = list(int.strategy = "eb", cmin = 0),
       control.fixed     = list(mean = list(off = 1, b0 = 0),
                                prec = list(off = 1e9, b0 = 0.001)),
       control.predictor = list(A = inla.stack.A(stk), compute = FALSE),
       control.compute   = list(config = FALSE, cpo = FALSE, dic = FALSE, waic = FALSE),
       num.threads       = "2:1")
}

fit_one <- function(j) {
  gid     <- EVAL$gid[j]
  cache_f <- file.path(CACHE_DIR, sprintf("fit_%05d.rds", gid))
  if (file.exists(cache_f)) return(readRDS(cache_f))
  X <- regen_X(EVAL$i_gen[j])
  if (npoints(X) != EVAL$N[j])
    stop(sprintf("gid %d: regenerated N = %d, stored N = %d", gid, npoints(X), EVAL$N[j]))
  t0  <- proc.time()[["elapsed"]]
  res <- tryCatch(fit_two_step(X, STORED[[as.character(gid)]]$bhat),
                  error = function(e) NULL)
  el  <- proc.time()[["elapsed"]] - t0
  out <- list(gid = gid, N = npoints(X), time_s = el, ok = FALSE,
              sigma = NA_real_, sigma_lo = NA_real_, sigma_hi = NA_real_,
              xi = NA_real_, xi_lo = NA_real_, xi_hi = NA_real_, mode_status = NA_real_)
  if (!is.null(res)) {
    hp <- res$summary.hyperpar
    ir <- grep("^Range", rownames(hp))[1]; isd <- grep("^Stdev", rownames(hp))[1]
    if (!is.na(ir) && !is.na(isd)) out <- modifyList(out, list(
      ok = TRUE, sigma = hp[isd, "mean"],
      sigma_lo = hp[isd, "0.025quant"], sigma_hi = hp[isd, "0.975quant"],
      xi = hp[ir, "mean"] / XI_TO_RANGE,
      xi_lo = hp[ir, "0.025quant"] / XI_TO_RANGE, xi_hi = hp[ir, "0.975quant"] / XI_TO_RANGE,
      mode_status = res$mode$mode.status))
  }
  saveRDS(out, cache_f)
  out
}

n_cached <- sum(file.exists(file.path(CACHE_DIR, sprintf("fit_%05d.rds", EVAL$gid))))
message(sprintf("\n=== Fitting %d realisations (%d cached, %d workers) ===",
                nrow(EVAL), n_cached, N_WORKERS))
t_loop <- system.time(
  RESL <- mclapply(seq_len(nrow(EVAL)), function(j)
    tryCatch(fit_one(j), error = function(e) conditionMessage(e)),
    mc.cores = N_WORKERS, mc.preschedule = FALSE))
bad <- !vapply(RESL, is.list, logical(1))
if (any(bad)) stop(sprintf("%d realisations failed outside INLA; first: %s",
                           sum(bad), RESL[[which(bad)[1]]]))
message(sprintf("Loop: %.1f min", t_loop[["elapsed"]] / 60))

FITS <- bind_rows(lapply(RESL, as_tibble)) %>%
  left_join(EVAL %>% select(gid, batch, sigma_true = sigma, xi_true = xi), by = "gid") %>%
  mutate(inla_valid = ok & sigma >= META$sigma_range[1] & sigma <= META$sigma_range[2] &
                      xi >= META$xi_range[1] & xi <= META$xi_range[2])
write_csv(FITS, out_f("fits.csv"))

# ---- 5. Comparison with the CNN variants -------------------------------------
JOINT <- inner_join(PRED, FITS %>% select(gid, batch, N, sigma_inla = sigma, xi_inla = xi,
                                          inla_ok = ok, inla_valid, inla_time_s = time_s),
                    by = "gid")
stopifnot(nrow(JOINT) == nrow(EVAL))
write_csv(JOINT, out_f("estimates.csv"))

r2 <- function(t, e) 1 - sum((e - t)^2) / sum((t - mean(t))^2)
metric_row <- function(tv, pv, method, set, param) {
  e <- pv - tv
  tibble(set = set, method = method, param = param, n = length(tv),
         Bias = mean(e), RMSE = sqrt(mean(e^2)), MAPE = 100 * mean(abs(e / tv)),
         R2 = r2(tv, pv))
}
methods <- c(no_bank = "CNN without descriptors",
             single  = "CNN, single-scale descriptors (8)",
             multi   = "CNN, multiscale bank (43)",
             inla    = "INLA-SPDE")
metrics_set <- function(d, set) purrr::map_dfr(names(methods), function(m)
  purrr::map_dfr(c("sigma", "xi"), function(p)
    metric_row(d[[paste0(p, "_true")]], d[[paste0(p, "_", m)]], methods[[m]], set, p)))
VALID <- JOINT %>% filter(inla_valid)
METRICS <- bind_rows(
  metrics_set(VALID, sprintf("INLA-SPDE in support (%d of %d)", nrow(VALID), nrow(JOINT))),
  metrics_set(JOINT %>% filter(inla_ok), sprintf("all finished fits (%d)", sum(JOINT$inla_ok)))
) %>% mutate(across(c(Bias, RMSE), ~ ifelse(param == "xi", .x / 1000, .x)))
write_csv(METRICS, out_f("metrics.csv"))
message("\nMetrics (xi bias/RMSE in km):")
print(METRICS %>% mutate(across(Bias:R2, ~ round(.x, 3))), n = Inf, width = Inf)

# Paired bootstrap of the R2 difference CNN - INLA on the same realisations.
set.seed(1)
BOOT <- t(replicate(4000, {
  d <- VALID[sample(nrow(VALID), replace = TRUE), ]
  c(sigma_multi = r2(d$sigma_true, d$sigma_multi) - r2(d$sigma_true, d$sigma_inla),
    xi_multi    = r2(d$xi_true, d$xi_multi) - r2(d$xi_true, d$xi_inla),
    xi_no_bank  = r2(d$xi_true, d$xi_no_bank) - r2(d$xi_true, d$xi_inla))
}))
R2DIFF <- tibble(contrast = colnames(BOOT),
                 median = apply(BOOT, 2, median),
                 lo = apply(BOOT, 2, quantile, 0.025), hi = apply(BOOT, 2, quantile, 0.975))
write_csv(R2DIFF, out_f("r2_difference.csv"))
message("\nR2 difference, CNN minus INLA-SPDE (paired bootstrap, 4000 replicates):")
print(R2DIFF, width = Inf)

# Accuracy of xi by tercile of the true scale.
BY_XI <- VALID %>% mutate(tercile = ntile(xi_true, 3)) %>% group_by(tercile) %>%
  summarise(n = n(), xi_min_km = min(xi_true) / 1000, xi_max_km = max(xi_true) / 1000,
            bias_inla_km = mean(xi_inla - xi_true) / 1000, bias_cnn_km = mean(xi_multi - xi_true) / 1000,
            rmse_inla_km = sqrt(mean((xi_inla - xi_true)^2)) / 1000,
            rmse_cnn_km  = sqrt(mean((xi_multi - xi_true)^2)) / 1000, .groups = "drop")
write_csv(BY_XI, out_f("by_xi_tercile.csv"))
print(BY_XI, width = Inf)

out_sup <- JOINT %>% filter(inla_ok, !inla_valid)
SUMMARY <- tibble(
  n_fits = nrow(JOINT), n_finished = sum(JOINT$inla_ok), n_valid = nrow(VALID),
  n_out_sigma_high = sum(out_sup$sigma_inla > META$sigma_range[2]),
  median_time_s = median(JOINT$inla_time_s), max_time_s = max(JOINT$inla_time_s))
message("\nTwo-step INLA-SPDE on simulations:"); print(SUMMARY, width = Inf)
write_csv(SUMMARY, out_f("summary.csv"))

message(sprintf("\nDone in %.1f min. Outputs %s_* in %s/",
                as.numeric(difftime(Sys.time(), T0, units = "mins")), PREFIX, OUT_DIR))
