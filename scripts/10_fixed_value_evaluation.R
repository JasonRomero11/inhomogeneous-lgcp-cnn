# Evaluation at fixed values of the latent-field parameters, as in the
# homogeneous simulation study of Zou et al. (arXiv:2607.13113).
#
# The main comparison (script 09) draws (sigma, xi) from the design prior the
# network was trained on, which favours an amortised estimator. Here (sigma, xi)
# is held fixed on a 3 x 3 grid inside the design support and N_REP
# realisations are simulated at each point; the trend of each realisation is
# drawn from the design as in script 01 (slopes and target count), with the
# same draws reused across the nine points. Both estimators receive the same
# ppm trend: the CNN ensemble of the paper (the five models saved by script 06,
# which reproduce the test metrics of script 02) and two-step INLA-SPDE
# (script 09 settings: mesh, PC priors, offset, "eb").
#
# Reported per point and method: bias, RMSE and MAPE of sigma and xi on all
# finished fits, plus the share of INLA-SPDE estimates outside the support.
#
# Nothing written by scripts 01-09 is modified. Outputs carry the prefix
# "fixed_"; INLA fits are cached in outputs/fixed_inla_cache/.
#
# Usage:  Rscript scripts/10_fixed_value_evaluation.R   (from the repository root)
#         SIM_CORES=8 and INLA_WORKERS=8 by default; N_REP and FIXED_PREFIX can
#         be overridden for a pilot run.

stopifnot(file.exists("scripts/06_sensitivity_mc.R"),
          all(dir.exists(sprintf("outputs/sens_mc_model_s%d", 1:5))),
          file.exists("outputs/inla_mesh_d240_c5000_o20000.rds"))

T0 <- Sys.time()

# ---- 1. Setup of script 06: design, covariates, two-step inputs, scalers and
#         the saved ensemble. The lines that compute and write the catalogue
#         inputs and the test-metrics file are skipped.
local({
  lines <- readLines("scripts/06_sensitivity_mc.R")
  stop_at <- grep("^# ---- 4\\. Estimates", lines)[1]
  s1 <- grep('^message\\("Building the two-step inputs', lines)[1]
  e1 <- grep("^# ---- 2\\. Training set", lines)[1] - 1
  s2 <- grep("^# Reproduction check on the test split", lines)[1]
  e2 <- grep('^write_csv\\(test_metrics', lines)[1]
  stopifnot(!anyNA(c(stop_at, s1, e1, s2, e2)))
  keep <- setdiff(seq_len(stop_at - 1), c(s1:e1, s2:e2))
  eval(parse(text = lines[keep], keep.source = FALSE), envir = globalenv())
})
suppressMessages({ library(INLA); library(Matrix); library(parallel) })

PREFIX <- Sys.getenv("FIXED_PREFIX", "fixed")   # a different prefix for pilot runs
out_f  <- function(name) file.path(OUT_DIR, paste0(PREFIX, "_", name))
CACHE_DIR <- file.path(OUT_DIR, paste0(PREFIX, "_inla_cache"))
dir.create(CACHE_DIR, showWarnings = FALSE, recursive = TRUE)

SIM_CORES    <- as.integer(Sys.getenv("SIM_CORES", "8"))
INLA_WORKERS <- as.integer(Sys.getenv("INLA_WORKERS", "8"))

SIGMA_FIX <- c(1, 2, 3)
XI_FIX    <- c(20, 60, 120) * 1000          # m
N_REP     <- as.integer(Sys.getenv("N_REP", "50"))
GRID <- expand.grid(sigma = SIGMA_FIX, xi = XI_FIX) %>% mutate(point = row_number(), .before = 1)
MIN_POINTS <- 100; MAX_POINTS <- 80000

# Ensemble check against the paper (R2 0.881 for sigma, 0.628 for xi).
r2 <- function(a, b) 1 - sum((b - a)^2) / sum((a - mean(a))^2)
P_te <- ensemble_predict(X_te)
message(sprintf("Ensemble on the test split: R2 sigma %.3f | xi %.3f",
                r2(DAT$Y[idx_te, "sigma"], P_te[, "sigma"]), r2(DAT$Y[idx_te, "xi"], P_te[, "xi"])))

# ---- 2. Generator of script 01 ---------------------------------------------
COVS_SIM <- lapply(COVS, function(z) nearestValue(as.im(z, W = WMASK)))
names(COVS_SIM) <- META$cov_names
PIXAREA <- WMASK$xstep * WMASK$ystep
trend_image <- function(beta) {
  tr <- as.im(beta[1], W = WMASK)
  for (j in seq_along(META$cov_names)) { Zj <- COVS_SIM[[j]]; bj <- beta[j + 1]; tr <- eval.im(tr + bj * Zj) }
  tr
}
solve_intercept <- function(slopes, n_target) {
  eta <- trend_image(c(0, slopes)); v <- eta$v[is.finite(eta$v)]; m <- max(v)
  log(n_target) - (m + log(sum(exp(v - m))) + log(PIXAREA))
}

# Trend draws, shared across the nine points: replicate r uses the same slopes
# and target count at every (sigma, xi).
TREND <- lapply(seq_len(N_REP), function(r) {
  set.seed(70000 + r)
  list(slopes = runif(length(META$cov_names), META$beta_range$lo, META$beta_range$hi),
       n_target = runif(1, META$n_range[1], META$n_range[2]))
})

# One realisation and its two-step inputs; up to 10 pattern seeds per cell.
simulate_cell <- function(k, r) {
  sg <- GRID$sigma[k]; xi <- GRID$xi[k]; tr <- TREND[[r]]
  beta <- c(solve_intercept(tr$slopes, tr$n_target), tr$slopes)
  trend <- trend_image(beta)
  for (attempt in 0:9) {
    seed <- 80000 + 1000 * k + 10 * r + attempt
    set.seed(seed)
    Y   <- spatstat.random::rGRFmatern(W = W, mu = 0, var = sg^2, scale = xi,
                                       nu = NU_MATERN, dimyx = META$dim)
    lam <- eval.im(exp(trend + Y - 0.5 * sg^2))
    if (integral.im(lam) > MAX_POINTS) next
    X <- rpoispp(lam)[W]
    nn <- npoints(X); if (nn < MIN_POINTS || nn > MAX_POINTS) next
    o <- tryCatch(observed_inputs(X), error = function(e) NULL)
    if (is.null(o)) next
    fv <- unlist(o[FEAT_COLS]); if (anyNA(fv) || anyNA(o$L)) next
    return(list(point = k, rep = r, seed = seed, attempt = attempt,
                sigma_true = sg, xi_true = xi, n_target = tr$n_target, N = nn,
                X = X, L = o$L, bhat = o$beta_hat, F = fv, floor_hit = o$floor_hit))
  }
  NULL
}

cells <- expand.grid(r = seq_len(N_REP), k = GRID$point)
rep_file <- out_f("replicates.rds")
if (file.exists(rep_file)) {
  REPS <- readRDS(rep_file)
  message(sprintf("Replicates loaded from cache: %d", length(REPS)))
} else {
  message(sprintf("\nSimulating %d realisations (%d points x %d replicates, %d cores)...",
                  nrow(cells), nrow(GRID), N_REP, SIM_CORES))
  t_sim <- system.time(
    REPS <- mclapply(seq_len(nrow(cells)), function(i)
      tryCatch(simulate_cell(cells$k[i], cells$r[i]), error = function(e) NULL),
      mc.cores = SIM_CORES, mc.preschedule = FALSE))
  REPS <- Filter(Negate(is.null), REPS)
  message(sprintf("Valid realisations: %d/%d (%.1f min)", length(REPS), nrow(cells), t_sim[["elapsed"]] / 60))
  saveRDS(REPS, rep_file)
}
stopifnot(length(REPS) >= 0.95 * nrow(cells))

# ---- 3. CNN ensemble --------------------------------------------------------
cnn_in <- list(build_S2_array(do.call(rbind, lapply(REPS, `[[`, "L"))),
               apply_scaler(do.call(rbind, lapply(REPS, `[[`, "bhat")), sc_B),
               apply_scaler(matrix(vapply(REPS, `[[`, numeric(1), "N"), ncol = 1), sc_N),
               apply_scaler(do.call(rbind, lapply(REPS, `[[`, "F")), sc_F))
CNN_PRED <- ensemble_predict(cnn_in)

# ---- 4. Two-step INLA-SPDE (settings of script 09) -------------------------
.mc <- readRDS(file.path(OUT_DIR, "inla_mesh_d240_c5000_o20000.rds"))
mesh <- .mc$mesh; w <- .mc$w; n_mesh <- mesh$n
COV_FULL <- lapply(META$cov_names, function(n) readRDS(file.path("data", paste0(n, "_im_scaled.rds"))))
COV_FULL <- lapply(COV_FULL, function(z) as.im(z, xy = COV_FULL[[1]])); names(COV_FULL) <- META$cov_names
COV_MAT  <- lapply(COV_FULL, as.matrix)
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
spde.index <- inla.spde.make.index("spatial.field", spde$n.spde)
Z_NODES <- eval_covar(mesh$loc[, 1:2]); stopifnot(!anyNA(Z_NODES))
IMA <- Diagonal(n_mesh, rep(1, n_mesh))

fit_two_step <- function(X, bhat) {
  p <- cbind(X$x, X$y)
  A <- rbind(IMA, inla.spde.make.A(mesh, p))
  Z <- rbind(Z_NODES, eval_covar(p))
  covs <- data.frame(off = as.vector(cbind(1, Z) %*% bhat), b0 = 1)
  stk  <- inla.stack(data = list(y = rep(0:1, c(n_mesh, nrow(p))), e = c(w, rep(0, nrow(p)))),
                     A = list(A, 1), effects = list(spde.index, as.list(covs)), tag = "pp")
  inla(y ~ 0 + b0 + off + f(spatial.field, model = spde),
       family = "poisson", data = inla.stack.data(stk), E = inla.stack.data(stk)$e,
       control.inla      = list(int.strategy = "eb", cmin = 0),
       control.fixed     = list(mean = list(off = 1, b0 = 0), prec = list(off = 1e9, b0 = 0.001)),
       control.predictor = list(A = inla.stack.A(stk), compute = FALSE),
       control.compute   = list(config = FALSE, cpo = FALSE, dic = FALSE, waic = FALSE),
       num.threads       = "2:1")
}
fit_one <- function(i) {
  z <- REPS[[i]]
  cache_f <- file.path(CACHE_DIR, sprintf("fit_p%d_r%03d.rds", z$point, z$rep))
  if (file.exists(cache_f)) return(readRDS(cache_f))
  t0  <- proc.time()[["elapsed"]]
  res <- tryCatch(fit_two_step(z$X, z$bhat), error = function(e) NULL)
  out <- list(point = z$point, rep = z$rep, time_s = proc.time()[["elapsed"]] - t0, ok = FALSE,
              sigma = NA_real_, xi = NA_real_, mode_status = NA_real_)
  if (!is.null(res)) {
    hp <- res$summary.hyperpar
    ir <- grep("^Range", rownames(hp))[1]; isd <- grep("^Stdev", rownames(hp))[1]
    if (!is.na(ir) && !is.na(isd))
      out <- modifyList(out, list(ok = TRUE, sigma = hp[isd, "mean"], xi = hp[ir, "mean"] / XI_TO_RANGE,
                                  mode_status = res$mode$mode.status))
  }
  saveRDS(out, cache_f)
  out
}
n_cached <- sum(vapply(REPS, function(z) file.exists(file.path(CACHE_DIR, sprintf("fit_p%d_r%03d.rds", z$point, z$rep))), logical(1)))
message(sprintf("\n=== INLA-SPDE on %d realisations (%d cached, %d workers) ===", length(REPS), n_cached, INLA_WORKERS))
t_inla <- system.time(
  INLA_L <- mclapply(seq_along(REPS), function(i) tryCatch(fit_one(i), error = function(e) conditionMessage(e)),
                     mc.cores = INLA_WORKERS, mc.preschedule = FALSE))
bad <- !vapply(INLA_L, is.list, logical(1))
if (any(bad)) stop(sprintf("%d fits failed outside INLA; first: %s", sum(bad), INLA_L[[which(bad)[1]]]))
message(sprintf("INLA loop: %.1f min", t_inla[["elapsed"]] / 60))

# ---- 5. Estimates, metrics and figure --------------------------------------
EST <- bind_rows(lapply(seq_along(REPS), function(i) {
  z <- REPS[[i]]; f <- INLA_L[[i]]
  tibble(point = z$point, rep = z$rep, seed = z$seed, N = z$N, n_target = z$n_target,
         sigma_true = z$sigma_true, xi_true = z$xi_true,
         sigma_cnn = CNN_PRED[i, "sigma"], xi_cnn = CNN_PRED[i, "xi"],
         inla_ok = f$ok, inla_time_s = f$time_s, sigma_inla = f$sigma, xi_inla = f$xi)
})) %>%
  mutate(inla_valid = inla_ok & sigma_inla >= META$sigma_range[1] & sigma_inla <= META$sigma_range[2] &
                      xi_inla >= META$xi_range[1] & xi_inla <= META$xi_range[2])
write_csv(EST, out_f("estimates.csv"))

metric_row <- function(tv, pv, method, param, set) {
  e <- pv - tv
  tibble(set = set, method = method, param = param, n = sum(is.finite(e)), true = tv[1],
         Bias = mean(e, na.rm = TRUE), RMSE = sqrt(mean(e^2, na.rm = TRUE)),
         MAPE = 100 * mean(abs(e / tv), na.rm = TRUE))
}
metrics_of <- function(d, set) bind_rows(lapply(split(d, d$point), function(x) bind_rows(
  metric_row(x$sigma_true, x$sigma_cnn,  "CNN",       "sigma", set) %>% mutate(point = x$point[1]),
  metric_row(x$xi_true,    x$xi_cnn,     "CNN",       "xi",    set) %>% mutate(point = x$point[1]),
  metric_row(x$sigma_true, x$sigma_inla, "INLA-SPDE", "sigma", set) %>% mutate(point = x$point[1]),
  metric_row(x$xi_true,    x$xi_inla,    "INLA-SPDE", "xi",    set) %>% mutate(point = x$point[1]))))
METRICS <- bind_rows(
  metrics_of(EST %>% filter(inla_ok), "all finished fits"),
  metrics_of(EST %>% filter(inla_valid), "INLA-SPDE in support")) %>%
  left_join(GRID %>% rename(sigma_fix = sigma, xi_fix = xi), by = "point") %>%
  mutate(across(c(true, Bias, RMSE), ~ ifelse(param == "xi", .x / 1000, .x)),   # xi in km
         xi_fix = xi_fix / 1000) %>%
  select(set, point, sigma_fix, xi_fix, method, param, n, true, Bias, RMSE, MAPE)
write_csv(METRICS, out_f("metrics.csv"))

VALID_RATE <- EST %>% group_by(point, sigma_true, xi_true) %>%
  summarise(n = n(), finished = sum(inla_ok), in_support = sum(inla_valid),
            median_N = median(N), median_time_s = median(inla_time_s), .groups = "drop")
write_csv(VALID_RATE, out_f("inla_validity.csv"))

message("\nMetrics on all finished fits (xi, bias and RMSE in km):")
print(METRICS %>% filter(set == "all finished fits") %>%
        mutate(across(c(Bias, RMSE, MAPE), ~ round(.x, 2))) %>%
        select(-set), n = Inf, width = Inf)
message("\nINLA-SPDE validity per point:"); print(VALID_RATE, n = Inf, width = Inf)

# Boxplots in the style of Zou et al.: one panel per fixed (sigma, xi), the
# estimates of both methods on all finished fits, the true value as a line.
long <- EST %>% filter(inla_ok) %>%
  select(point, sigma_true, xi_true, sigma_cnn, xi_cnn, sigma_inla, xi_inla) %>%
  pivot_longer(c(sigma_cnn, xi_cnn, sigma_inla, xi_inla), names_to = c("param", "method"),
               names_sep = "_", values_to = "est") %>%
  mutate(method = recode(method, cnn = "CNN", inla = "INLA-SPDE"),
         est = ifelse(param == "xi", est / 1000, est),
         true = ifelse(param == "xi", xi_true / 1000, sigma_true),
         cell = sprintf("sigma = %g, xi = %g km", sigma_true, xi_true / 1000))
p <- ggplot(long, aes(method, est, fill = method)) +
  geom_boxplot(outlier.size = 0.6, width = 0.6, show.legend = FALSE) +
  geom_hline(aes(yintercept = true), linetype = "dashed", colour = "grey30") +
  facet_grid(param ~ cell, scales = "free_y", labeller = labeller(param = c(sigma = "sigma", xi = "xi (km)"))) +
  labs(x = NULL, y = "Estimate") + theme_bw(base_size = 9) +
  theme(panel.grid = element_blank(), strip.text.x = element_text(size = 7))
ggsave(out_f("boxplot.pdf"), p, width = 12, height = 5.5, device = cairo_pdf)

message(sprintf("\nDone in %.1f min. Outputs %s_* in %s/",
                as.numeric(difftime(Sys.time(), T0, units = "mins")), PREFIX, OUT_DIR))
