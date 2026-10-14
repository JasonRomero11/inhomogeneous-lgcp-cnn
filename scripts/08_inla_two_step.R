# Two-step INLA-SPDE: the Bayesian reference of the paper.
#
# As in Zou et al. (arXiv:2607.13113), INLA-SPDE is run in two stages: the
# first-order trend is fixed at the ppm estimate beta_hat (the same stage 1 the
# CNN uses) and enters as an offset, and INLA estimates only the latent field
# (sigma, xi). The comparison therefore isolates the second-order estimation.
#
# This script
#   1. fits the 2020 catalogue in two steps ("ccd"), with the 5 km intensity
#      map, and recomputes the CNN plug-in map on the same grid;
#   2. measures how far the two maps agree beyond the shared trend (as 05);
#   3. fits the Mw >= 1.92 subset of script 06 in two steps ("ccd").
#
# The same two-step fit on simulated test realisations is script 09.
#
# Nothing written by scripts 01-07 is modified. Outputs carry the prefix
# "inla2s_".
#
# Usage:  Rscript scripts/08_inla_two_step.R   (from the repository root)

stopifnot(file.exists("scripts/04_intensity_map.R"),
          file.exists("outputs/sens_mc_inputs_mc.rds"))

T0 <- Sys.time()

# ---- 1. Setup of script 04 (design, covariates, mesh, SPDE, grid) ----------
# Evaluated up to its fits, so that mesh, priors and grid are literally the same.
# The lines that read the CNN estimate are skipped: estimates_observed.csv may
# carry the older column name "Modelo", so it is read below with either name.
local({
  lines   <- readLines("scripts/04_intensity_map.R")
  stop_at <- grep("^# ---- Fit:", lines)[1]
  skip_from <- grep("^cnn <- read_csv", lines)[1]
  skip_to   <- grep("BHAT_OBS\\), collapse", lines)[1]
  stopifnot(!is.na(stop_at), !is.na(skip_from), !is.na(skip_to), skip_to > skip_from)
  keep <- setdiff(seq_len(stop_at - 1), skip_from:skip_to)
  eval(parse(text = lines[keep], keep.source = FALSE), envir = globalenv())
})
CNN_OBS <- read_csv(file.path(OUT_DIR, "estimates_observed.csv"), show_col_types = FALSE)
names(CNN_OBS)[names(CNN_OBS) == "Modelo"] <- "model"
CNN_OBS <- CNN_OBS %>% filter(str_detect(model, "^CNN"))
stopifnot(nrow(CNN_OBS) == 1)
SIGMA_CNN <- CNN_OBS$sigma[1]; XI_CNN <- CNN_OBS$xi[1]
message(sprintf("CNN: sigma = %.3f | xi = %.1f km", SIGMA_CNN, XI_CNN / 1000))

PREFIX <- "inla2s"
out_f  <- function(name) file.path(OUT_DIR, paste0(PREFIX, "_", name))

# ---- 2. Two-step fit --------------------------------------------------------
# The ppm trend collapses into one column `off` whose coefficient is pinned to
# 1 (prior N(1, 1e-9)), i.e. an exact offset; b0 stays free and absorbs the
# -sigma^2/2 of the centred LGCP. With sigma_fix/xi_fix the hyperparameters are
# held fixed (the CNN plug-in map); otherwise INLA estimates them.
fit_two_step <- function(X, bhat, predict = FALSE, int_strategy = "ccd",
                         sigma_fix = NULL, xi_fix = NULL, x_start = NULL) {
  stopifnot(length(bhat) == N_BETA, all(is.finite(bhat)))
  p  <- cbind(X$x, X$y)
  A  <- rbind(IMA, inla.spde.make.A(mesh, p))
  Z  <- rbind(Z_NODES, eval_covar(p))
  covs <- data.frame(off = as.vector(cbind(1, Z) %*% bhat), b0 = 1)
  stk  <- inla.stack(data = list(y = rep(0:1, c(n_mesh, nrow(p))),
                                 e = c(w, rep(0, nrow(p)))),
                     A = list(A, 1), effects = list(spde.index, as.list(covs)),
                     tag = "pp")
  if (predict) {
    covs.pred <- data.frame(off = as.vector(cbind(1, Z_PRED) %*% bhat), b0 = 1)
    stk.pred  <- inla.stack(data = list(y = rep(NA_real_, nrow(pts.pred)),
                                        e = rep(0, nrow(pts.pred))),
                            A = list(A.pred, 1),
                            effects = list(spde.index, as.list(covs.pred)),
                            tag = "pred")
    stk <- inla.stack(stk, stk.pred)
  }
  ctrl_mode <- list()
  if (!is.null(sigma_fix)) {
    ctrl_mode <- list(theta = c(log(XI_TO_RANGE * xi_fix), log(sigma_fix)), fixed = TRUE)
    int_strategy <- "eb"
    if (!is.null(x_start)) { ctrl_mode$x <- x_start; ctrl_mode$restart <- TRUE }
  }
  args <- list(
    formula = y ~ 0 + b0 + off + f(spatial.field, model = spde),
    family = "poisson", data = inla.stack.data(stk), E = inla.stack.data(stk)$e,
    control.inla      = list(int.strategy = int_strategy, cmin = 0),
    control.fixed     = list(mean = list(off = 1,   b0 = 0),
                             prec = list(off = 1e9, b0 = 0.001)),
    control.predictor = list(A = inla.stack.A(stk), compute = predict, link = 1),
    control.compute   = list(config = FALSE, cpo = FALSE, dic = FALSE, waic = FALSE))
  if (length(ctrl_mode)) args$control.mode <- ctrl_mode
  stopifnot(!anyNA(args$E))

  t0  <- proc.time()[["elapsed"]]
  res <- do.call(inla, args)
  el  <- proc.time()[["elapsed"]] - t0

  out <- list(res = res, time_s = el, mode_status = res$mode$mode.status,
              off = res$summary.fixed["off", "mean"], N = nrow(p))
  hp <- res$summary.hyperpar
  if (is.null(sigma_fix)) {
    ir <- grep("^Range", rownames(hp))[1]; isd <- grep("^Stdev", rownames(hp))[1]
    out$sigma <- hp[isd, "mean"]
    out$sigma_ci <- c(hp[isd, "0.025quant"], hp[isd, "0.975quant"])
    out$xi    <- hp[ir, "mean"] / XI_TO_RANGE
    out$xi_ci <- c(hp[ir, "0.025quant"], hp[ir, "0.975quant"]) / XI_TO_RANGE
  } else {
    out$sigma <- sigma_fix; out$xi <- xi_fix
  }
  lp <- res$summary.linear.predictor
  if (predict) {
    idx_pp <- inla.stack.index(stk, "pp")$data[seq_len(n_mesh)]
    out$lambda   <- exp(lp[inla.stack.index(stk, "pred")$data, "mean"])
    out$int_mesh <- sum(w * exp(lp[idx_pp, "mean"]))
  }
  out
}

row_obs <- function(f, label) tibble(
  catalogue = label, N = f$N, time_s = f$time_s, mode_status = f$mode_status,
  off_coef = f$off,
  sigma = f$sigma, sigma_lo = f$sigma_ci[1], sigma_hi = f$sigma_ci[2],
  xi_km = f$xi / 1000, xi_lo_km = f$xi_ci[1] / 1000, xi_hi_km = f$xi_ci[2] / 1000)

# ---- 3. Catalogue: two-step INLA-SPDE and the CNN plug-in map --------------
message("\n=== Catalogue, two-step INLA-SPDE (ccd, with map) ===")
FIT_OBS <- fit_two_step(X_OBS, BHAT_OBS, predict = TRUE)
message(sprintf("  sigma = %.3f [%.3f, %.3f] | xi = %.1f km [%.1f, %.1f] | off = %.4f | %.1f min",
                FIT_OBS$sigma, FIT_OBS$sigma_ci[1], FIT_OBS$sigma_ci[2],
                FIT_OBS$xi / 1000, FIT_OBS$xi_ci[1] / 1000, FIT_OBS$xi_ci[2] / 1000,
                FIT_OBS$off, FIT_OBS$time_s / 60))

# Plug-in map with the CNN estimates. As in script 04, the latent-field mode is
# restarted until the mesh quadrature reproduces the observed count.
fit_plugin <- function(sigma_fix, xi_fix, max_it = 8) {
  f <- fit_two_step(X_OBS, BHAT_OBS, predict = TRUE, sigma_fix = sigma_fix, xi_fix = xi_fix)
  for (it in seq_len(max_it)) {
    if (abs(f$int_mesh / nX - 1) < 0.01) break
    f <- fit_two_step(X_OBS, BHAT_OBS, predict = TRUE, sigma_fix = sigma_fix,
                      xi_fix = xi_fix, x_start = as.numeric(f$res$mode$x))
  }
  if (abs(f$int_mesh / nX - 1) >= 0.01) warning("plug-in quadrature off by > 1%")
  f
}
message("\n=== Catalogue, CNN plug-in map ===")
FIT_CNN <- fit_plugin(SIGMA_CNN, XI_CNN)
message(sprintf("  mesh integral: INLA %.0f | CNN %.0f | N %d",
                FIT_OBS$int_mesh, FIT_CNN$int_mesh, nX))

MAP <- pts.pred %>% rename(x = X, y = Y) %>%
  mutate(lambda_inla = FIT_OBS$lambda, lambda_cnn = FIT_CNN$lambda)
write_csv(MAP, out_f("map_grid.csv"))

# ---- 4. Agreement between maps beyond the shared trend (as script 05) ------
eta <- as.vector(cbind(1, Z_PRED) %*% BHAT_OBS)
li  <- log(MAP$lambda_inla); lc <- log(MAP$lambda_cnn)
jaccard <- function(a, b, p) {
  A <- a >= quantile(a, 1 - p); B <- b >= quantile(b, 1 - p)
  sum(A & B) / sum(A | B)
}
AGREE <- tibble(
  comparison = c("INLA map vs CNN map", "INLA map vs trend only",
                 "CNN map vs trend only", "fields (shared trend removed)"),
  pearson_log = c(cor(li, lc), cor(li, eta), cor(lc, eta), cor(li - eta, lc - eta)),
  jaccard_top1  = c(jaccard(li, lc, .01), jaccard(li, eta, .01), jaccard(lc, eta, .01), NA),
  jaccard_top5  = c(jaccard(li, lc, .05), jaccard(li, eta, .05), jaccard(lc, eta, .05), NA),
  jaccard_top10 = c(jaccard(li, lc, .10), jaccard(li, eta, .10), jaccard(lc, eta, .10), NA))
message(sprintf("\ncells %d | log sd: trend %.2f, INLA field %.2f, CNN field %.2f",
                nrow(MAP), sd(eta), sd(li - eta), sd(lc - eta)))
print(AGREE, width = Inf)
write_csv(AGREE, out_f("map_agreement.csv"))

# ---- 5. Mw >= 1.92 subset (inputs written by script 06) -------------------
IN_MC <- readRDS(file.path(OUT_DIR, "sens_mc_inputs_mc.rds"))
message(sprintf("\n=== Mw-filtered catalogue (%d events), two-step INLA-SPDE (ccd) ===",
                npoints(IN_MC$X)))
FIT_MC <- fit_two_step(IN_MC$X, setNames(IN_MC$beta_hat, META$beta_names))
message(sprintf("  sigma = %.3f [%.3f, %.3f] | xi = %.1f km [%.1f, %.1f] | %.1f min",
                FIT_MC$sigma, FIT_MC$sigma_ci[1], FIT_MC$sigma_ci[2],
                FIT_MC$xi / 1000, FIT_MC$xi_ci[1] / 1000, FIT_MC$xi_ci[2] / 1000,
                FIT_MC$time_s / 60))

OBS_OUT <- bind_rows(row_obs(FIT_OBS, "all events"), row_obs(FIT_MC, "Mw >= 1.92"))
print(OBS_OUT %>% mutate(across(where(is.numeric), ~ signif(.x, 4))), width = Inf)
write_csv(OBS_OUT, out_f("catalog_observed.csv"))

message(sprintf("\nDone in %.1f min. Outputs %s_* in %s/",
                as.numeric(difftime(Sys.time(), T0, units = "mins")), PREFIX, OUT_DIR))
