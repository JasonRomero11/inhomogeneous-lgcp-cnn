# Expected intensity map of the 2020 catalogue under the CNN-fitted model.
#
# The CNN returns only (sigma, xi). The map also needs the conditional mean of
# the latent field, so it is obtained by plug-in: the stage-1 ppm trend enters
# as an offset with its coefficient pinned to 1, the hyperparameters are held at
# the CNN estimates via control.mode(fixed = TRUE), and INLA computes nothing
# but the posterior mean of the field.
#
# The script also refits the joint INLA model of script 03, so both surfaces
# come from the same run and share a colour scale. Its estimates must match
# estimates_observed.csv (sigma 1.957, xi 38.0 km).
#
# Outputs, in figures_sismos_multiescala_matern/:
#   mapa_cnn_intensidad.pdf   CNN map
#   mapa_cnn_comparado.pdf    two panels, INLA-SPDE | CNN, common scale
#   mapa_cnn_grid.csv         lambda per 5 km cell for both models
#   mapa_cnn_resumen.csv      theta used, b0, correlations, timings
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
stopifnot(dir.exists("data"), dir.exists("Results_simulation_sismos_multiescala_matern"))

SIM_DIR <- "Results_simulation_sismos_multiescala_matern"
fig_dir <- "figures_sismos_multiescala_matern"
PREFIJO <- "mapa_cnn"
out_f   <- function(nombre) file.path(fig_dir, paste0(PREFIJO, "_", nombre))

covariables_modelo <- c("inversa", "isostasia", "topografia", "volcanes")

# Mesh and SPDE settings, identical to script 03.
MESH_CUTOFF   <- 5000
MESH_OFFSET   <- c(100, 20000)
MESH_EDGE_DIV <- 240
NU_MATERN     <- 1
ALPHA_SPDE    <- NU_MATERN + 1
XI_TO_RANGE   <- sqrt(8 * NU_MATERN)
interval1     <- 5000            # map cell size, m

# ---- Design, window, pattern, covariates ------------------------------------

META <- readRDS(file.path(SIM_DIR, "design_meta.rds"))
stopifnot(identical(META$beta_names, c("(Intercept)", covariables_modelo)))
N_BETA <- length(META$beta_names)
stopifnot(identical(META$field_model, "matern"), META$nu_matern == NU_MATERN)

OBS   <- readRDS(file.path(SIM_DIR, "observed_inputs.rds"))
X_OBS <- OBS$X
stopifnot(npoints(X_OBS) == META$n_obs)
W <- Window(X_OBS)
shapeZona_sp <- st_read("data/clip_zona_continental_simplificado.geojson", quiet = TRUE)

covar <- lapply(covariables_modelo,
                function(n) readRDS(file.path("data", paste0(n, "_im_scaled.rds"))))
names(covar) <- covariables_modelo
covar <- lapply(covar, function(z) as.im(z, xy = covar[[1]]))
names(covar) <- covariables_modelo
COV_MAT <- lapply(covar, as.matrix)

eval_covar <- function(coords) {
  coords <- as.matrix(coords)
  out <- matrix(NA_real_, nrow(coords), length(covariables_modelo),
                dimnames = list(NULL, covariables_modelo))
  for (k in seq_along(covar)) {
    px <- nearest.valid.pixel(coords[, 1], coords[, 2], covar[[k]])
    out[, k] <- COV_MAT[[k]][cbind(px$row, px$col)]
  }
  out
}

BHAT_OBS <- setNames(as.numeric(META$beta_hat_obs), META$beta_names)
cnn <- read_csv(file.path(fig_dir, "estimates_observed.csv"), show_col_types = FALSE) %>%
  filter(str_detect(Modelo, "^CNN"))
stopifnot(nrow(cnn) == 1)
SIGMA_CNN <- cnn$sigma[1]; XI_CNN <- cnn$xi[1]
message(sprintf("CNN: sigma = %.3f | xi = %.1f km | beta_hat ppm: %s",
                SIGMA_CNN, XI_CNN / 1000,
                paste(sprintf("%.3f", BHAT_OBS), collapse = ", ")))

# ---- Mesh (cached by script 03) and constant objects ------------------------

mesh_cache <- file.path("Results_simulation_sismos",
                        sprintf("inla_mesh_d%d_c%d_o%d.rds",
                                MESH_EDGE_DIV, MESH_CUTOFF, MESH_OFFSET[2]))
stopifnot(file.exists(mesh_cache))
.mc        <- readRDS(mesh_cache)
meshSismos <- .mc$mesh
w          <- .mc$w
nvSismos   <- meshSismos$n
message(sprintf("Malla cargada de caché: %d vértices | ratio pesos/área = %.4f",
                nvSismos, sum(w) / area.owin(W)))

Z_NODOS <- eval_covar(meshSismos$loc[, 1:2])
stopifnot(!anyNA(Z_NODOS))
IMA <- Diagonal(nvSismos, rep(1, nvSismos))

spdesismos <- inla.spde2.pcmatern(
  mesh        = meshSismos,
  alpha       = ALPHA_SPDE,
  prior.range = c(XI_TO_RANGE * mean(META$xi_range), 0.5),
  prior.sigma = c(mean(META$sigma_range), 0.5)
)
spde.index <- inla.spde.make.index("spatial.field", spdesismos$n.spde)

# 5 km prediction grid over the interior of the window.
shapeZona_sp_interno <- st_buffer(shapeZona_sp, -1000)
grid    <- st_make_grid(shapeZona_sp_interno, cellsize = interval1,
                        what = "centers", square = TRUE)
grid_in <- grid[st_within(grid, shapeZona_sp_interno, sparse = FALSE)]
pts.pred <- as.data.frame(st_coordinates(grid_in))
A.pred   <- inla.spde.make.A(mesh = meshSismos, loc = as.matrix(pts.pred))
Z_PRED   <- eval_covar(pts.pred)
message(sprintf("Rejilla de predicción: %d celdas de %.0f km", nrow(pts.pred), interval1 / 1000))

ptsSismos <- cbind(X_OBS$x, X_OBS$y)
nSismos   <- nrow(ptsSismos)
A.pp   <- rbind(IMA, inla.spde.make.A(meshSismos, ptsSismos))
y.pp   <- rep(0:1, c(nvSismos, nSismos))
e.pp   <- c(w, rep(0, nSismos))
Z_TODO <- rbind(Z_NODOS, eval_covar(ptsSismos))

# ---- Fit: "conjunto" estimates betas and field jointly, "plugin" fixes both --

ajustar <- function(modo, bhat = NULL, sigma_fix = NULL, xi_fix = NULL,
                    strategy = "auto", x_start = NULL) {
  if (modo == "conjunto") {
    covs      <- as.data.frame(Z_TODO)
    covs.pred <- as.data.frame(Z_PRED)
    terminos  <- covariables_modelo
    beta_lo <- META$beta_range$lo; beta_hi <- META$beta_range$hi
    control_fixed <- list(mean = list(b0 = 0,     default = (beta_lo + beta_hi) / 2),
                          prec = list(b0 = 0.001, default = 12 / (beta_hi - beta_lo)^2))
    control_mode <- list()
    estrategia   <- "ccd"
  } else {
    stopifnot(length(bhat) == N_BETA, is.finite(sigma_fix), is.finite(xi_fix))
    # Stage-1 trend collapsed into a single `off` column with its coefficient
    # pinned to 1; the free b0 absorbs the -sigma^2/2 centring constant.
    covs      <- data.frame(off = as.vector(cbind(1, Z_TODO) %*% bhat))
    covs.pred <- data.frame(off = as.vector(cbind(1, Z_PRED) %*% bhat))
    terminos  <- "off"
    control_fixed <- list(mean = list(off = 1,   b0 = 0,     default = 0),
                          prec = list(off = 1e9, b0 = 0.001, default = 0.001))
    # Internal PC-Matern theta is (log range, log sigma), range = sqrt(8) * xi.
    control_mode <- list(theta = c(log(XI_TO_RANGE * xi_fix), log(sigma_fix)),
                         fixed = TRUE)
    # With theta far from the marginal optimum the inner mode search can stop
    # before the score equation for b0 is satisfied; restart from a prior mode.
    if (!is.null(x_start)) { control_mode$x <- x_start; control_mode$restart <- TRUE }
    estrategia   <- "eb"
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

  formula <- as.formula(paste("y ~ 0 + b0 +", paste(terminos, collapse = " + "),
                              "+ f(spatial.field, model = spdesismos)"))
  args <- list(
    formula = formula, family = "poisson",
    data = inla.stack.data(st.all),
    control.inla      = list(int.strategy = estrategia, strategy = strategy, cmin = 0),
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
  # With fixed theta, INLA leaves summary.hyperpar and res$mode$theta empty, so
  # sigma and xi are read back from the requested theta and checked against
  # whatever INLA reports.
  hp <- res$summary.hyperpar
  th_inla <- as.numeric(res$mode$theta)
  if (length(control_mode)) {
    th_mode <- control_mode$theta
    if (length(th_inla) == 2) stopifnot(max(abs(th_inla - th_mode)) < 1e-6)
    message(sprintf("    (fixed) res$mode$theta tiene longitud %d; summary.hyperpar %d filas; mlik = %.1f",
                    length(th_inla), if (is.null(hp)) 0L else nrow(hp), res$mlik[1]))
  } else {
    stopifnot(!is.null(hp), nrow(hp) >= 2)
    th_mode <- c(log(hp[grep("^Range", rownames(hp))[1], "mean"]),
                 log(hp[grep("^Stdev", rownames(hp))[1], "mean"]))
  }
  stopifnot(length(th_mode) == 2, all(is.finite(th_mode)))
  # Calibration check: over the mesh nodes the quadrature sum(w * exp(eta))
  # must reproduce N, which is the score equation for the free intercept.
  idx_pp    <- inla.stack.index(st.all, "pp")$data[seq_len(nvSismos)]
  lp        <- res$summary.linear.predictor
  int_malla <- sum(w * exp(lp[idx_pp, "mean"]))
  int_malla_modo <- if ("mode" %in% colnames(lp) && all(is.finite(lp[idx_pp, "mode"])))
    sum(w * exp(lp[idx_pp, "mode"])) else NA_real_
  campo <- res$summary.random$spatial.field
  list(res = res, tiempo_s = t1 - t0,
       lambda = exp(lp[idx, "mean"]),
       sigma  = exp(th_mode[2]),
       xi     = exp(th_mode[1]) / XI_TO_RANGE,
       theta_mode = th_mode,
       b0     = res$summary.fixed["b0", "mean"],
       int_malla = int_malla, int_malla_modo = int_malla_modo,
       campo_rango = range(campo$mean), campo_sd_media = mean(campo$sd))
}

# ---- Fits -------------------------------------------------------------------

message("\n=== (1) INLA-SPDE conjunto (réplica del catálogo del 03, ccd) ===")
fit_inla <- ajustar("conjunto")
message(sprintf("  sigma = %.3f | xi = %.1f km | b0 = %.2f | %.1f min",
                fit_inla$sigma, fit_inla$xi / 1000, fit_inla$b0, fit_inla$tiempo_s / 60))

diag <- function(f, nombre) message(sprintf(
  "  [%s] int. malla (media) = %.0f | (modo) = %.0f | int. rejilla = %.0f | N = %d | campo media en [%.2f, %.2f], sd media %.2f",
  nombre, f$int_malla, f$int_malla_modo, sum(f$lambda) * interval1^2, npoints(X_OBS),
  f$campo_rango[1], f$campo_rango[2], f$campo_sd_media))
diag(fit_inla, "INLA conjunto")

message("\n=== (2) Plug-in de la CNN: tendencia de la etapa 1 + theta fijo ===")
# Restart the mode search until the mesh quadrature reproduces N within 1 %.
ajustar_convergido <- function(sigma_fix, xi_fix, etiqueta, max_it = 8) {
  f <- ajustar("plugin", bhat = BHAT_OBS, sigma_fix = sigma_fix, xi_fix = xi_fix)
  diag(f, paste(etiqueta, "it 0"))
  for (it in seq_len(max_it)) {
    if (abs(f$int_malla / npoints(X_OBS) - 1) < 0.01) break
    x_prev <- as.numeric(f$res$mode$x)
    if (!length(x_prev)) stop("INLA no devolvió el modo del campo (res$mode$x vacío); no se puede reiniciar.")
    f <- ajustar("plugin", bhat = BHAT_OBS, sigma_fix = sigma_fix, xi_fix = xi_fix,
                 x_start = x_prev)
    diag(f, sprintf("%s it %d", etiqueta, it))
  }
  if (abs(f$int_malla / npoints(X_OBS) - 1) >= 0.01)
    warning(etiqueta, ": la cuadratura no converge a N tras ", max_it, " reinicios.")
  f
}
fit_cnn <- ajustar_convergido(SIGMA_CNN, XI_CNN, "CNN plug-in")

# Same ppm trend but with INLA's own theta, to isolate the effect of (sigma, xi).
message("\n=== (3) Verificación: plug-in con la tendencia ppm y el theta de INLA ===")
fit_chk <- ajustar_convergido(fit_inla$sigma, fit_inla$xi, "plug-in theta INLA")
message(sprintf("  mlik: theta CNN %.1f vs theta INLA %.1f | cor log(CNN plug-in, INLA-theta plug-in) = %.4f | cor log(INLA-theta plug-in, INLA conjunto) = %.4f",
                fit_cnn$res$mlik[1], fit_chk$res$mlik[1],
                cor(log(fit_cnn$lambda), log(fit_chk$lambda)),
                cor(log(fit_chk$lambda), log(fit_inla$lambda))))
saveRDS(list(inla = fit_inla[setdiff(names(fit_inla), "res")],
             cnn = fit_cnn[setdiff(names(fit_cnn), "res")],
             chk_theta_inla = fit_chk[setdiff(names(fit_chk), "res")]),
        file.path(SIM_DIR, "mapa_cnn_diagnostico.rds"))
message(sprintf("  theta fijado -> INLA reporta sigma = %.3f | xi = %.1f km (deben coincidir con la CNN)",
                fit_cnn$sigma, fit_cnn$xi / 1000))
message(sprintf("  b0 = %.3f (esperado ~ -sigma^2/2 = %.3f) | off = %.4f | %.1f min",
                fit_cnn$b0, -SIGMA_CNN^2 / 2,
                fit_cnn$res$summary.fixed["off", "mean"], fit_cnn$tiempo_s / 60))
stopifnot(length(fit_cnn$sigma) == 1, length(fit_cnn$xi) == 1,
          abs(fit_cnn$sigma - SIGMA_CNN) < 1e-3, abs(fit_cnn$xi - XI_CNN) < 1)
# Surface integrals over the 5 km grid: of order N, below it because
# exp(posterior mean) < mean of exp.
int_cnn  <- sum(fit_cnn$lambda)  * interval1^2
int_inla <- sum(fit_inla$lambda) * interval1^2
message(sprintf("  Integral exp(media posterior): INLA %.0f | CNN %.0f | N observado %d",
                int_inla, int_cnn, npoints(X_OBS)))

# ---- Outputs ----------------------------------------------------------------

mapa_df <- pts.pred %>%
  rename(x = X, y = Y) %>%
  mutate(lambda_inla = fit_inla$lambda, lambda_cnn = fit_cnn$lambda)
write_csv(mapa_df, out_f("grid.csv"))

r_log <- cor(log(mapa_df$lambda_inla), log(mapa_df$lambda_cnn))
r_lin <- cor(mapa_df$lambda_inla, mapa_df$lambda_cnn)
message(sprintf("\nCorrelación entre superficies: log-escala %.3f | lineal %.3f", r_log, r_lin))

write_csv(tibble(
  modelo   = c("INLA-SPDE", "CNN (plug-in)"),
  sigma    = c(fit_inla$sigma, fit_cnn$sigma),
  xi_km    = c(fit_inla$xi, fit_cnn$xi) / 1000,
  b0       = c(fit_inla$b0, fit_cnn$b0),
  tiempo_s = c(fit_inla$tiempo_s, fit_cnn$tiempo_s),
  cor_log_entre_mapas = r_log, cor_lineal_entre_mapas = r_lin,
  n_celdas = nrow(mapa_df)), out_f("resumen.csv"))

lims_comun <- quantile(c(mapa_df$lambda_inla, mapa_df$lambda_cnn), c(0.02, 0.98))
mapa <- function(col, lims, etiqueta = NULL) {
  ggplot(mapa_df, aes(x / 1000, y / 1000, fill = .data[[col]])) +
    geom_raster() +
    scale_fill_viridis_c(name = "Intensidad\nesperada", limits = lims,
                         oob = scales::squish) +
    coord_equal(expand = FALSE) +
    labs(x = "x (km)", y = "y (km)", title = etiqueta) +
    theme_bw(base_size = 10) +
    theme(panel.grid = element_blank(),
          plot.title = element_text(hjust = 0.5, size = 10))
}

ggsave(out_f("intensidad.pdf"),
       mapa("lambda_cnn", quantile(mapa_df$lambda_cnn, c(0.02, 0.98))),
       width = 6, height = 7, device = cairo_pdf)

p_comp <- mapa("lambda_inla", lims_comun,
               sprintf("INLA-SPDE (sigma = %.2f, xi = %.0f km)", fit_inla$sigma, fit_inla$xi / 1000)) +
          mapa("lambda_cnn", lims_comun,
               sprintf("CNN (sigma = %.2f, xi = %.0f km)", fit_cnn$sigma, fit_cnn$xi / 1000)) +
          plot_layout(guides = "collect")
ggsave(out_f("comparado.pdf"), p_comp, width = 11, height = 6.5, device = cairo_pdf)

message("\nListo: salidas ", PREFIJO, "_* en ", fig_dir, "/")
