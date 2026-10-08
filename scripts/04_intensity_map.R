# Expected intensity map of the 2020 catalogue under the CNN-fitted model.
#
# The CNN returns only (sigma, xi). The map also needs the conditional mean of
# the latent field, so it is obtained by plug-in: the stage-1 ppm trend enters
# as an offset with its coefficient pinned to 1, the hyperparameters are held at
# the CNN estimates via control.mode(fixed = TRUE), and INLA computes nothing
# but the posterior mean of the field.
#
# The joint INLA model of script 03 is refitted here too, so both surfaces come
# from the same run and share a colour scale.
#
# Usage:  Rscript scripts/04_intensity_map.R   (from the repository root)

suppressMessages({
  library(INLA)
  library(sf)
  library(spatstat)
  library(Matrix)
  library(tidyverse)
  library(patchwork)
})

set.seed(123)
stopifnot(dir.exists("data"), dir.exists("simulations"))

SIM_DIR <- "simulations"
OUT_DIR <- "outputs"
PREFIX  <- "cnn_map"
out_f   <- function(name) file.path(OUT_DIR, paste0(PREFIX, "_", name))

model_covariates <- c("inversa", "isostasia", "topografia", "volcanes")

# Mesh and SPDE settings, identical to script 03.
MESH_CUTOFF   <- 5000
MESH_OFFSET   <- c(100, 20000)
MESH_EDGE_DIV <- 240
NU_MATERN     <- 1
ALPHA_SPDE    <- NU_MATERN + 1
XI_TO_RANGE   <- sqrt(8 * NU_MATERN)
MAP_CELL      <- 5000            # map cell size, m

# ---- Design, window, pattern, covariates ------------------------------------

META <- readRDS(file.path(SIM_DIR, "design_meta.rds"))
stopifnot(identical(META$beta_names, c("(Intercept)", model_covariates)))
N_BETA <- length(META$beta_names)
stopifnot(identical(META$field_model, "matern"), META$nu_matern == NU_MATERN)

OBS   <- readRDS(file.path(SIM_DIR, "observed_inputs.rds"))
X_OBS <- OBS$X
stopifnot(npoints(X_OBS) == META$n_obs)
W <- Window(X_OBS)
zone_sf <- st_read("data/clip_zona_continental_simplificado.geojson", quiet = TRUE)

covar <- lapply(model_covariates,
                function(n) readRDS(file.path("data", paste0(n, "_im_scaled.rds"))))
names(covar) <- model_covariates
covar <- lapply(covar, function(z) as.im(z, xy = covar[[1]]))
names(covar) <- model_covariates
COV_MAT <- lapply(covar, as.matrix)

eval_covar <- function(coords) {
  coords <- as.matrix(coords)
  out <- matrix(NA_real_, nrow(coords), length(model_covariates),
                dimnames = list(NULL, model_covariates))
  for (k in seq_along(covar)) {
    px <- nearest.valid.pixel(coords[, 1], coords[, 2], covar[[k]])
    out[, k] <- COV_MAT[[k]][cbind(px$row, px$col)]
  }
  out
}

BHAT_OBS <- setNames(as.numeric(META$beta_hat_obs), META$beta_names)
cnn <- read_csv(file.path(OUT_DIR, "estimates_observed.csv"), show_col_types = FALSE) %>%
  filter(str_detect(model, "^CNN"))
stopifnot(nrow(cnn) == 1)
SIGMA_CNN <- cnn$sigma[1]; XI_CNN <- cnn$xi[1]
message(sprintf("CNN: sigma = %.3f | xi = %.1f km | ppm beta_hat: %s",
                SIGMA_CNN, XI_CNN / 1000,
                paste(sprintf("%.3f", BHAT_OBS), collapse = ", ")))

# ---- Mesh (cached by script 03) and constant objects ------------------------

mesh_cache <- file.path(OUT_DIR,
                        sprintf("inla_mesh_d%d_c%d_o%d.rds",
                                MESH_EDGE_DIV, MESH_CUTOFF, MESH_OFFSET[2]))
stopifnot(file.exists(mesh_cache))
.mc    <- readRDS(mesh_cache)
mesh   <- .mc$mesh
w      <- .mc$w
n_mesh <- mesh$n
message(sprintf("Mesh loaded from cache: %d vertices | weights/area ratio = %.4f",
                n_mesh, sum(w) / area.owin(W)))

Z_NODES <- eval_covar(mesh$loc[, 1:2])
stopifnot(!anyNA(Z_NODES))
IMA <- Diagonal(n_mesh, rep(1, n_mesh))

spde <- inla.spde2.pcmatern(
  mesh        = mesh,
  alpha       = ALPHA_SPDE,
  prior.range = c(XI_TO_RANGE * mean(META$xi_range), 0.5),
  prior.sigma = c(mean(META$sigma_range), 0.5)
)
spde.index <- inla.spde.make.index("spatial.field", spde$n.spde)

zone_inner <- st_buffer(zone_sf, -1000)
grid    <- st_make_grid(zone_inner, cellsize = MAP_CELL,
                        what = "centers", square = TRUE)
grid_in <- grid[st_within(grid, zone_inner, sparse = FALSE)]
pts.pred <- as.data.frame(st_coordinates(grid_in))
A.pred   <- inla.spde.make.A(mesh = mesh, loc = as.matrix(pts.pred))
Z_PRED   <- eval_covar(pts.pred)
message(sprintf("Prediction grid: %d cells of %.0f km", nrow(pts.pred), MAP_CELL / 1000))

pts    <- cbind(X_OBS$x, X_OBS$y)
nX     <- nrow(pts)
A.pp   <- rbind(IMA, inla.spde.make.A(mesh, pts))
y.pp   <- rep(0:1, c(n_mesh, nX))
e.pp   <- c(w, rep(0, nX))
Z_ALL  <- rbind(Z_NODES, eval_covar(pts))

# ---- Fit: "joint" estimates betas and field together, "plugin" fixes both ---

fit_model <- function(mode, bhat = NULL, sigma_fix = NULL, xi_fix = NULL,
                      strategy = "auto", x_start = NULL) {
  if (mode == "joint") {
    covs      <- as.data.frame(Z_ALL)
    covs.pred <- as.data.frame(Z_PRED)
    terms     <- model_covariates
    beta_lo <- META$beta_range$lo; beta_hi <- META$beta_range$hi
    control_fixed <- list(mean = list(b0 = 0,     default = (beta_lo + beta_hi) / 2),
                          prec = list(b0 = 0.001, default = 12 / (beta_hi - beta_lo)^2))
    control_mode <- list()
    int_strategy <- "ccd"
  } else {
    stopifnot(length(bhat) == N_BETA, is.finite(sigma_fix), is.finite(xi_fix))
    # Stage-1 trend collapsed into a single `off` column with its coefficient
    # pinned to 1; the free b0 absorbs the -sigma^2/2 centring constant.
    covs      <- data.frame(off = as.vector(cbind(1, Z_ALL)  %*% bhat))
    covs.pred <- data.frame(off = as.vector(cbind(1, Z_PRED) %*% bhat))
    terms     <- "off"
    control_fixed <- list(mean = list(off = 1,   b0 = 0,     default = 0),
                          prec = list(off = 1e9, b0 = 0.001, default = 0.001))
    # Internal PC-Matern theta is (log range, log sigma), range = sqrt(8) * xi.
    control_mode <- list(theta = c(log(XI_TO_RANGE * xi_fix), log(sigma_fix)),
                         fixed = TRUE)
    # With theta far from the marginal optimum the inner mode search can stop
    # before the score equation for b0 is satisfied; restart from a prior mode.
    if (!is.null(x_start)) { control_mode$x <- x_start; control_mode$restart <- TRUE }
    int_strategy <- "eb"
  }
  covs$b0 <- 1; covs.pred$b0 <- 1
  stopifnot(!anyNA(covs), !anyNA(covs.pred))

  st.est <- inla.stack(data = list(y = y.pp, e = e.pp),
                       A = list(A.pp, 1),
                       effects = list(spde.index, as.list(covs)), tag = "pp")
  st.pred <- inla.stack(data = list(y = rep(NA_real_, nrow(pts.pred)),
                                    e = rep(0, nrow(pts.pred))),
                        A = list(A.pred, 1),
                        effects = list(spde.index, as.list(covs.pred)), tag = "pred")
  st.all <- inla.stack(st.est, st.pred)

  formula <- as.formula(paste("y ~ 0 + b0 +", paste(terms, collapse = " + "),
                              "+ f(spatial.field, model = spde)"))
  args <- list(
    formula = formula, family = "poisson",
    data = inla.stack.data(st.all),
    control.inla      = list(int.strategy = int_strategy, strategy = strategy, cmin = 0),
    control.fixed     = control_fixed,
    control.predictor = list(A = inla.stack.A(st.all), compute = TRUE, link = 1),
    control.compute   = list(config = FALSE, cpo = FALSE, dic = FALSE, waic = FALSE),
    E                 = inla.stack.data(st.all)$e
  )
  if (length(control_mode)) args$control.mode <- control_mode
  stopifnot(!any(is.na(args$E)), all(args$E >= 0))

  t0  <- proc.time()[["elapsed"]]
  res <- do.call(inla, args)
  t1  <- proc.time()[["elapsed"]]

  idx <- inla.stack.index(st.all, "pred")$data
  # With fixed theta, INLA leaves summary.hyperpar empty, so sigma and xi are
  # read back from the requested theta and checked against what INLA reports.
  hp <- res$summary.hyperpar
  th_inla <- as.numeric(res$mode$theta)
  if (length(control_mode)) {
    th_mode <- control_mode$theta
    if (length(th_inla) == 2) stopifnot(max(abs(th_inla - th_mode)) < 1e-6)
    message(sprintf("    (fixed) res$mode$theta length %d; summary.hyperpar %d rows; mlik = %.1f",
                    length(th_inla), if (is.null(hp)) 0L else nrow(hp), res$mlik[1]))
  } else {
    stopifnot(!is.null(hp), nrow(hp) >= 2)
    th_mode <- c(log(hp[grep("^Range", rownames(hp))[1], "mean"]),
                 log(hp[grep("^Stdev", rownames(hp))[1], "mean"]))
  }
  stopifnot(length(th_mode) == 2, all(is.finite(th_mode)))
  # Calibration check: over the mesh nodes the quadrature sum(w * exp(eta))
  # must reproduce N, which is the score equation for the free intercept.
  idx_pp   <- inla.stack.index(st.all, "pp")$data[seq_len(n_mesh)]
  lp       <- res$summary.linear.predictor
  int_mesh <- sum(w * exp(lp[idx_pp, "mean"]))
  int_mesh_mode <- if ("mode" %in% colnames(lp) && all(is.finite(lp[idx_pp, "mode"])))
    sum(w * exp(lp[idx_pp, "mode"])) else NA_real_
  field <- res$summary.random$spatial.field
  list(res = res, time_s = t1 - t0,
       lambda = exp(lp[idx, "mean"]),
       sigma  = exp(th_mode[2]),
       xi     = exp(th_mode[1]) / XI_TO_RANGE,
       theta_mode = th_mode,
       b0     = res$summary.fixed["b0", "mean"],
       int_mesh = int_mesh, int_mesh_mode = int_mesh_mode,
       field_range = range(field$mean), field_sd_mean = mean(field$sd))
}

# ---- Fits -------------------------------------------------------------------

message("\n=== (1) Joint INLA-SPDE (replicates script 03, ccd) ===")
fit_inla <- fit_model("joint")
message(sprintf("  sigma = %.3f | xi = %.1f km | b0 = %.2f | %.1f min",
                fit_inla$sigma, fit_inla$xi / 1000, fit_inla$b0, fit_inla$time_s / 60))

report <- function(f, name) message(sprintf(
  "  [%s] mesh int. (mean) = %.0f | (mode) = %.0f | grid int. = %.0f | N = %d | field mean in [%.2f, %.2f], mean sd %.2f",
  name, f$int_mesh, f$int_mesh_mode, sum(f$lambda) * MAP_CELL^2, npoints(X_OBS),
  f$field_range[1], f$field_range[2], f$field_sd_mean))
report(fit_inla, "joint INLA")

message("\n=== (2) CNN plug-in: stage-1 trend + fixed theta ===")
# Restart the mode search until the mesh quadrature reproduces N within 1 %.
fit_converged <- function(sigma_fix, xi_fix, label, max_it = 8) {
  f <- fit_model("plugin", bhat = BHAT_OBS, sigma_fix = sigma_fix, xi_fix = xi_fix)
  report(f, paste(label, "it 0"))
  for (it in seq_len(max_it)) {
    if (abs(f$int_mesh / npoints(X_OBS) - 1) < 0.01) break
    x_prev <- as.numeric(f$res$mode$x)
    if (!length(x_prev)) stop("INLA returned no field mode (res$mode$x empty); cannot restart.")
    f <- fit_model("plugin", bhat = BHAT_OBS, sigma_fix = sigma_fix, xi_fix = xi_fix,
                   x_start = x_prev)
    report(f, sprintf("%s it %d", label, it))
  }
  if (abs(f$int_mesh / npoints(X_OBS) - 1) >= 0.01)
    warning(label, ": the quadrature does not converge to N after ", max_it, " restarts.")
  f
}
fit_cnn <- fit_converged(SIGMA_CNN, XI_CNN, "CNN plug-in")

# Same ppm trend but with INLA's own theta, to isolate the effect of (sigma, xi).
message("\n=== (3) Check: plug-in with the ppm trend and INLA's theta ===")
fit_chk <- fit_converged(fit_inla$sigma, fit_inla$xi, "plug-in INLA theta")
message(sprintf("  mlik: CNN theta %.1f vs INLA theta %.1f | cor log(CNN plug-in, INLA-theta plug-in) = %.4f | cor log(INLA-theta plug-in, joint INLA) = %.4f",
                fit_cnn$res$mlik[1], fit_chk$res$mlik[1],
                cor(log(fit_cnn$lambda), log(fit_chk$lambda)),
                cor(log(fit_chk$lambda), log(fit_inla$lambda))))
saveRDS(list(inla = fit_inla[setdiff(names(fit_inla), "res")],
             cnn = fit_cnn[setdiff(names(fit_cnn), "res")],
             chk_theta_inla = fit_chk[setdiff(names(fit_chk), "res")]),
        out_f("diagnostics.rds"))
message(sprintf("  theta fixed -> INLA reports sigma = %.3f | xi = %.1f km (must match the CNN)",
                fit_cnn$sigma, fit_cnn$xi / 1000))
message(sprintf("  b0 = %.3f (expected ~ -sigma^2/2 = %.3f) | off = %.4f | %.1f min",
                fit_cnn$b0, -SIGMA_CNN^2 / 2,
                fit_cnn$res$summary.fixed["off", "mean"], fit_cnn$time_s / 60))
stopifnot(length(fit_cnn$sigma) == 1, length(fit_cnn$xi) == 1,
          abs(fit_cnn$sigma - SIGMA_CNN) < 1e-3, abs(fit_cnn$xi - XI_CNN) < 1)

# Surface integrals over the 5 km grid: of order N, below it because
# exp(posterior mean) < mean of exp.
int_cnn  <- sum(fit_cnn$lambda)  * MAP_CELL^2
int_inla <- sum(fit_inla$lambda) * MAP_CELL^2
message(sprintf("  Integral of exp(posterior mean): INLA %.0f | CNN %.0f | observed N %d",
                int_inla, int_cnn, npoints(X_OBS)))

# ---- Outputs ----------------------------------------------------------------

map_df <- pts.pred %>%
  rename(x = X, y = Y) %>%
  mutate(lambda_inla = fit_inla$lambda, lambda_cnn = fit_cnn$lambda)
write_csv(map_df, out_f("grid.csv"))

r_log <- cor(log(map_df$lambda_inla), log(map_df$lambda_cnn))
r_lin <- cor(map_df$lambda_inla, map_df$lambda_cnn)
message(sprintf("\nCorrelation between surfaces: log scale %.3f | linear %.3f", r_log, r_lin))

write_csv(tibble(
  model    = c("INLA-SPDE", "CNN (plug-in)"),
  sigma    = c(fit_inla$sigma, fit_cnn$sigma),
  xi_km    = c(fit_inla$xi, fit_cnn$xi) / 1000,
  b0       = c(fit_inla$b0, fit_cnn$b0),
  time_s   = c(fit_inla$time_s, fit_cnn$time_s),
  cor_log_between_maps = r_log, cor_linear_between_maps = r_lin,
  n_cells = nrow(map_df)), out_f("summary.csv"))

common_lims <- quantile(c(map_df$lambda_inla, map_df$lambda_cnn), c(0.02, 0.98))
draw_map <- function(col, lims, label = NULL) {
  ggplot(map_df, aes(x / 1000, y / 1000, fill = .data[[col]])) +
    geom_raster() +
    scale_fill_viridis_c(name = "Expected\nintensity", limits = lims,
                         oob = scales::squish) +
    coord_equal(expand = FALSE) +
    labs(x = "x (km)", y = "y (km)", title = label) +
    theme_bw(base_size = 10) +
    theme(panel.grid = element_blank(),
          plot.title = element_text(hjust = 0.5, size = 10))
}

ggsave(out_f("intensity.pdf"),
       draw_map("lambda_cnn", quantile(map_df$lambda_cnn, c(0.02, 0.98))),
       width = 6, height = 7, device = cairo_pdf)

p_comp <- draw_map("lambda_inla", common_lims,
                   sprintf("INLA-SPDE (sigma = %.2f, xi = %.0f km)",
                           fit_inla$sigma, fit_inla$xi / 1000)) +
          draw_map("lambda_cnn", common_lims,
                   sprintf("CNN (sigma = %.2f, xi = %.0f km)",
                           fit_cnn$sigma, fit_cnn$xi / 1000)) +
          plot_layout(guides = "collect")
ggsave(out_f("comparison.pdf"), p_comp, width = 11, height = 6.5, device = cairo_pdf)

message("\nDone: ", PREFIX, "_* outputs in ", OUT_DIR, "/")
