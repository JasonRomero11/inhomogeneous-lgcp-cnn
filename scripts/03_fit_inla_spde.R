# Joint INLA-SPDE fit of the 2020 seismic catalogue (Matern nu = 1).
#
# Betas and latent field are estimated jointly with "ccd" integration over the
# hyperparameters. This is the complementary Bayesian fit reported in the paper,
# not a benchmark: the simulation study evaluates the CNN only.
#
# Usage:  Rscript scripts/03_fit_inla_spde.R   (from the repository root)

suppressMessages({
  library(INLA)
  library(sp)
  library(sf)
  library(spatstat)
  library(Matrix)
  library(tidyverse)
})

set.seed(123)
stopifnot(dir.exists("data"), dir.exists("simulations"))

SIM_DIR <- "simulations"
OUT_DIR <- "outputs"
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

PREFIX <- "inla_catalog"
out_f  <- function(name) file.path(OUT_DIR, paste0(PREFIX, "_", name))

# ---- 1. Settings ------------------------------------------------------------

model_covariates <- c("inversa", "isostasia", "topografia", "volcanes")

# max.edge = extent / MESH_EDGE_DIV. At /240 the smallest resolvable xi is ~14 km.
MESH_CUTOFF   <- 5000
MESH_OFFSET   <- c(100, 20000)
MESH_EDGE_DIV <- 240

# Script 01 generates a Matern nu = 1 field, so alpha = nu + d/2 = 2 is an
# integer and the SPDE representation is exact. INLA's range is sqrt(8*nu)*xi.
NU_MATERN   <- 1
ALPHA_SPDE  <- NU_MATERN + 1
XI_TO_RANGE <- sqrt(8 * NU_MATERN)

# "ccd" integrates over ~9 hyperparameter configurations around the mode, so the
# credible intervals propagate that uncertainty.
INT_STRATEGY <- "ccd"

MAP_CELL <- 5000   # prediction cell size, m

# ---- 2. Design, window, catalogue and covariates ----------------------------

META <- readRDS(file.path(SIM_DIR, "design_meta.rds"))
stopifnot(identical(META$beta_names, c("(Intercept)", model_covariates)))
N_BETA <- length(META$beta_names)

if (!identical(META$field_model, "matern") || META$nu_matern != NU_MATERN)
  stop(sprintf("Incompatible field: the design is %s(nu = %g), this script assumes matern(nu = %g).",
               META$field_model, META$nu_matern, NU_MATERN))

OBS   <- readRDS(file.path(SIM_DIR, "observed_inputs.rds"))
X_OBS <- OBS$X
stopifnot(npoints(X_OBS) == META$n_obs)

zone_sf <- st_read("data/clip_zona_continental_simplificado.geojson", quiet = TRUE)
W <- Window(X_OBS)

message(sprintf("Catalogue: %d events | design: sigma U[%.1f, %.1f], xi U[%.0f, %.0f] km",
                META$n_obs, META$sigma_range[1], META$sigma_range[2],
                META$xi_range[1] / 1000, META$xi_range[2] / 1000))

# Covariates on a common grid, unsmoothed, as in script 01.
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

# ---- 3. Mesh and dual-Voronoi integration weights ---------------------------
# Mesh and weights depend only on W and are cached: rebuilding takes minutes.

book.mesh.dual <- function(mesh) {
  stopifnot(mesh$manifold == "R2")
  ce <- t(sapply(1:nrow(mesh$graph$tv), function(i)
    colMeans(mesh$loc[mesh$graph$tv[i, ], 1:2])))
  pls <- parallel::mclapply(1:mesh$n, function(i) {
    p <- unique(Reduce("rbind", lapply(1:3, function(k) {
      j <- which(mesh$graph$tv[, k] == i)
      if (length(j) > 0)
        return(rbind(ce[j, , drop = FALSE],
                     cbind(mesh$loc[mesh$graph$tv[j, k], 1] +
                             mesh$loc[mesh$graph$tv[j, c(2:3, 1)[k]], 1],
                           mesh$loc[mesh$graph$tv[j, k], 2] +
                             mesh$loc[mesh$graph$tv[j, c(2:3, 1)[k]], 2]) / 2))
      else return(ce[j, , drop = FALSE])
    })))
    j1 <- which(mesh$segm$bnd$idx[, 1] == i)
    j2 <- which(mesh$segm$bnd$idx[, 2] == i)
    if ((length(j1) > 0) | (length(j2) > 0)) {
      p <- unique(rbind(mesh$loc[i, 1:2], p,
                        mesh$loc[mesh$segm$bnd$idx[j1, 1], 1:2] / 2 +
                          mesh$loc[mesh$segm$bnd$idx[j1, 2], 1:2] / 2,
                        mesh$loc[mesh$segm$bnd$idx[j2, 1], 1:2] / 2 +
                          mesh$loc[mesh$segm$bnd$idx[j2, 2], 1:2] / 2))
      yy <- p[, 2] - mean(p[, 2]) / 2 - mesh$loc[i, 2] / 2
      xx <- p[, 1] - mean(p[, 1]) / 2 - mesh$loc[i, 1] / 2
    } else {
      yy <- p[, 2] - mesh$loc[i, 2]
      xx <- p[, 1] - mesh$loc[i, 1]
    }
    sp::Polygon(p[order(atan2(yy, xx)), ])
  })
  sp::SpatialPolygons(lapply(1:mesh$n, function(i) sp::Polygons(list(pls[[i]]), i)))
}

MAX_EDGE   <- max(diff(Frame(W)$xrange), diff(Frame(W)$yrange)) / MESH_EDGE_DIV
mesh_cache <- file.path(OUT_DIR,
                        sprintf("inla_mesh_d%d_c%d_o%d.rds",
                                MESH_EDGE_DIV, MESH_CUTOFF, MESH_OFFSET[2]))

if (file.exists(mesh_cache)) {
  .mc  <- readRDS(mesh_cache)
  mesh <- .mc$mesh
  w    <- .mc$w
  message(sprintf("Mesh from cache: %d vertices", mesh$n))
} else {
  zone_valid <- st_make_valid(zone_sf)
  mesh  <- inla.mesh.2d(boundary = zone_valid, max.edge = c(1, 3) * MAX_EDGE,
                        cutoff = MESH_CUTOFF, offset = MESH_OFFSET,
                        crs = st_crs(zone_valid))
  dmesh <- st_as_sf(book.mesh.dual(mesh))
  st_crs(dmesh) <- st_crs(zone_sf)
  ii <- st_intersects(dmesh, zone_sf, sparse = FALSE)[, 1]
  ar <- st_area(st_intersection(dmesh[ii, ], zone_sf))
  w <- numeric(nrow(dmesh)); w[ii] <- as.numeric(ar)
  saveRDS(list(mesh = mesh, w = w), mesh_cache)
  message(sprintf("Mesh built: %d vertices", mesh$n))
}

n_mesh <- mesh$n

# The quadrature must reproduce the area of the window.
cat(sprintf("Voronoi weights: %.0f km2 vs area %.0f km2 (ratio %.4f)\n",
            sum(w) / 1e6, area.owin(W) / 1e6, sum(w) / area.owin(W)))

Z_NODES <- eval_covar(mesh$loc[, 1:2])
stopifnot(!anyNA(Z_NODES))
IMA <- Diagonal(n_mesh, rep(1, n_mesh))

# ---- 4. Joint fit, with a prediction grid for the map -----------------------
# Standard LGCP quadrature: y = 0 at the mesh nodes with E = Voronoi area,
# y = 1 at the events with E = 0. The PC prior is anchored at the median of the
# sampling design, so it follows any change to the design.

spde <- inla.spde2.pcmatern(
  mesh        = mesh,
  alpha       = ALPHA_SPDE,
  prior.range = c(XI_TO_RANGE * mean(META$xi_range), 0.5),
  prior.sigma = c(mean(META$sigma_range), 0.5))
idx <- inla.spde.make.index("spatial.field", spde$n.spde)

pts <- cbind(X_OBS$x, X_OBS$y)
nX  <- nrow(pts)

A <- rbind(IMA, inla.spde.make.A(mesh, pts))
y <- rep(0:1, c(n_mesh, nX))
e <- c(w, rep(0, nX))

# b0 is an explicit column of ones: INLA's automatic "~ 1" does not project
# correctly through A and the estimate breaks silently.
covs <- as.data.frame(rbind(Z_NODES, eval_covar(pts)))
covs$b0 <- 1
stopifnot(nrow(covs) == n_mesh + nX, !anyNA(covs))

stk <- inla.stack(data = list(y = y, e = e), A = list(A, 1),
                  effects = list(idx, as.list(covs)), tag = "pp")

# Cell centres inside the continental window, with a 1 km inner buffer.
zone_inner <- st_buffer(zone_sf, -1000)
grid    <- st_make_grid(zone_inner, cellsize = MAP_CELL,
                        what = "centers", square = TRUE)
grid_in <- grid[st_within(grid, zone_inner, sparse = FALSE)]
pts_pred <- as.data.frame(st_coordinates(grid_in))
message(sprintf("Map grid: %d cells of %.0f km", length(grid_in), MAP_CELL / 1000))

covs_pred <- as.data.frame(eval_covar(pts_pred))
covs_pred$b0 <- 1
stopifnot(!anyNA(covs_pred))

# e = 0 must be explicit: otherwise inla.stack pads with NA when combining.
stk_pred <- inla.stack(
  data    = list(y = rep(NA_real_, nrow(pts_pred)), e = rep(0, nrow(pts_pred))),
  A       = list(inla.spde.make.A(mesh, as.matrix(pts_pred)), 1),
  effects = list(idx, as.list(covs_pred)), tag = "pred")
stk_full <- inla.stack(stk, stk_pred)

formula <- as.formula(paste("y ~ 0 + b0 +",
                            paste(model_covariates, collapse = " + "),
                            "+ f(spatial.field, model = spde)"))

# Gaussian prior on beta matching the mean and variance of the design prior
# U[lo, hi]. A standard N(0, 1) would shrink slopes of order -10 towards zero
# and the unexplained trend would inflate sigma and xi. b0 is weakly
# informative: it legitimately sits around -30.
beta_lo <- META$beta_range$lo
beta_hi <- META$beta_range$hi
ctrl_fixed <- list(mean = list(b0 = 0,     default = (beta_lo + beta_hi) / 2),
                   prec = list(b0 = 0.001, default = 12 / (beta_hi - beta_lo)^2))

message(sprintf("\n=== Fitting the catalogue (joint, int.strategy = %s) ===", INT_STRATEGY))
t_obs <- system.time({
  res <- inla(formula, family = "poisson",
              data = inla.stack.data(stk_full),
              E    = inla.stack.data(stk_full)$e,
              # cmin = 0 avoids the Poisson "GMRFLib_2order_approx: NAN/INF in
              # logl" warnings.
              control.inla      = list(int.strategy = INT_STRATEGY, cmin = 0),
              control.fixed     = ctrl_fixed,
              control.predictor = list(A = inla.stack.A(stk_full),
                                       compute = TRUE, link = 1),
              control.compute   = list(config = FALSE, cpo = FALSE,
                                       dic = FALSE, waic = FALSE))
})

# ---- 5. Outputs: parameters and intensity map -------------------------------

hp  <- res$summary.hyperpar
ir  <- grep("^Range", rownames(hp))[1]
isd <- grep("^Stdev", rownames(hp))[1]
stopifnot(is.finite(ir), is.finite(isd))

# INLA's "Range" is the practical range rho = sqrt(8)*xi for nu = 1, so the
# generator scale is xi = rho/sqrt(8).
OBS_OUT <- tibble(
  method   = sprintf("INLA-SPDE (%s)", INT_STRATEGY),
  N        = META$n_obs, time_s = t_obs[["elapsed"]],
  sigma    = hp[isd, "mean"],
  sigma_lo = hp[isd, "0.025quant"], sigma_hi = hp[isd, "0.975quant"],
  range_km    = hp[ir, "mean"] / 1000,
  range_lo_km = hp[ir, "0.025quant"] / 1000,
  range_hi_km = hp[ir, "0.975quant"] / 1000,
  xi_km    = hp[ir, "mean"] / XI_TO_RANGE / 1000,
  xi_lo_km = hp[ir, "0.025quant"] / XI_TO_RANGE / 1000,
  xi_hi_km = hp[ir, "0.975quant"] / XI_TO_RANGE / 1000,
  mode_status = res$mode$mode.status)

cat("\n--- Catalogue (hyperparameter posterior) ---\n")
print(OBS_OUT %>% mutate(across(where(is.numeric), ~ signif(.x, 4))))
write_csv(OBS_OUT, out_f("observed.csv"))

BETAS <- as_tibble(res$summary.fixed, rownames = "coef") %>%
  select(coef, mean, sd, `0.025quant`, `0.975quant`)
cat("\n--- Fixed effects (reference: beta_hat from ppm) ---\n")
print(BETAS %>% mutate(across(where(is.numeric), ~ signif(.x, 3))))
print(round(setNames(as.numeric(OBS$beta_hat), META$beta_names), 3))
write_csv(BETAS, out_f("betas.csv"))

idx_pred <- inla.stack.index(stk_full, "pred")$data
lam_pred <- exp(res$summary.linear.predictor[idx_pred, "mean"])
map_df   <- pts_pred %>% mutate(lambda = lam_pred)
write_csv(map_df, out_f("grid.csv"))

cat(sprintf("\nMap integral: %.0f expected events (catalogue: %d)\n",
            sum(map_df$lambda) * MAP_CELL^2, META$n_obs))

lims <- quantile(map_df$lambda, c(0.02, 0.98), na.rm = TRUE)
map_base <- ggplot(map_df, aes(X / 1000, Y / 1000, fill = lambda)) +
  geom_raster() +
  scale_fill_viridis_c(name = "Expected\nintensity\n(events/m²)",
                       limits = lims, oob = scales::squish) +
  coord_equal(expand = FALSE) +
  labs(x = "x (km)", y = "y (km)") +
  theme_bw(base_size = 10) +
  theme(panel.grid = element_blank())

p_map <- map_base +
  labs(title = "Expected intensity of the 2020 seismic catalogue (INLA-SPDE)",
       subtitle = sprintf("sigma = %.2f, practical range = %.0f km (xi = %.1f km)",
                          OBS_OUT$sigma, OBS_OUT$range_km, OBS_OUT$xi_km)) +
  theme(plot.title = element_text(hjust = 0.5, size = 10),
        plot.subtitle = element_text(hjust = 0.5, size = 8))
ggsave(out_f("intensity.pdf"), p_map, width = 6, height = 7, device = cairo_pdf)
ggsave(out_f("intensity_notitle.pdf"), map_base, width = 6, height = 6.5,
       device = cairo_pdf)

message(sprintf("\nDone. sigma = %.3f [%.3f, %.3f] | range = %.1f km [%.1f, %.1f] | %.1f min",
                OBS_OUT$sigma, OBS_OUT$sigma_lo, OBS_OUT$sigma_hi,
                OBS_OUT$range_km, OBS_OUT$range_lo_km, OBS_OUT$range_hi_km,
                t_obs[["elapsed"]] / 60))
message(sprintf("Outputs %s_* in %s/", PREFIX, OUT_DIR))
