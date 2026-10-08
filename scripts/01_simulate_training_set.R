# Training set for the inhomogeneous LGCP, Colombian seismicity 2020.
#
# Draws (sigma, xi, slopes, N) from the design prior, simulates one LGCP
# realisation per draw and stores the two-step inputs: the ppm first-order fit,
# the centred inhomogeneous L curve and the 43 multiscale descriptors.
#
# Usage:  Rscript scripts/01_simulate_training_set.R   (from the repository root)

suppressMessages({
  library(pbmcapply)
  library(spatstat)
  library(sf)
})

stopifnot(dir.exists("data"))

# ---- Configuration ----------------------------------------------------------

set.seed(123)
N_CORES <- 8

OUT_DIR <- "simulations"
FIG_DIR <- "outputs"

N_SIMS     <- 20000
CHUNK_SIZE <- 100
BASE_SEED  <- 42

# The window is 1410 x 1899 km, so dimyx = c(950, 706) gives square 2 km pixels,
# matching the resolution of the covariate rasters.
DIM     <- c(950, 706)
DIM_RES <- c(127, 94)      # coarse grid, count-residual figure only

RMAX  <- 200000            # m, largest r on the L curve
NRVAL <- 513
R_SEQ <- seq(0, RMAX, length.out = NRVAL)

# Latent field prior. The lower end of xi is near the mean nearest-neighbour
# distance (4.3 km).
SIGMA_MIN <- 0.5
SIGMA_MAX <- 3.5
XI_MIN    <- 10000         # m
XI_MAX    <- 150000        # m

# Slopes: all four covariates are scaled distance rasters and seismicity decays
# away from the structure, so the range is skewed negative.
BETA_MIN <- -11
BETA_MAX <-   3

# The intercept is not sampled: N is, and the intercept is solved for.
N_MIN <- 2000
N_MAX <- 35000

MIN_POINTS <- 100
MAX_POINTS <- 80000

NQUAD_PPM <- 256           # ppm() quadrature

# Floor on lambda_hat, relative to the mean intensity. Guards the 1/lambda_hat
# weight of the inhomogeneous L estimator; its activation is reported at the end.
LAMBDA_FLOOR <- 1e-3

L_WITH_TRUE_LAMBDA <- FALSE  # FALSE = two-step, lambda_hat from ppm
SAVE_LAMBDA        <- FALSE
PLOT_EXAMPLE       <- TRUE

dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(FIG_DIR, showWarnings = FALSE, recursive = TRUE)

# ---- Data, window and covariates --------------------------------------------

COV_NAMES <- c("inversa", "isostasia", "topografia", "volcanes")

ZONA <- st_read("data/clip_zona_continental_simplificado.geojson", quiet = TRUE)
W    <- as.owin(st_geometry(ZONA))
unitname(W) <- c("metre", "metres")
WMASK <- as.mask(W, dimyx = DIM)

GDF <- st_read("data/gdf_espacial_2020_paper.gpkg", quiet = TRUE)
XY  <- st_coordinates(GDF)
INW <- inside.owin(XY[, 1], XY[, 2], W)

X_OBS <- unique(ppp(XY[INW, 1], XY[INW, 2], window = W, check = FALSE))
N_DUP <- sum(INW) - npoints(X_OBS)

# Covariates arrive already scaled to [-1, 1] from preprocessing.
COVS <- lapply(COV_NAMES, function(n) readRDS(file.path("data", paste0(n, "_im_scaled.rds"))))
names(COVS) <- COV_NAMES
COVS <- lapply(COVS, function(z) as.im(z, xy = COVS[[1]]))
COVS <- lapply(COVS, function(z) z[W, drop = FALSE])
names(COVS) <- COV_NAMES

RHS <- paste(COV_NAMES, collapse = " + ")

# Stage 1: inhomogeneous Poisson fit.
fit_first_order <- function(pp, rhs = RHS) {
  fo <- as.formula(paste("pp ~", rhs))
  environment(fo) <- environment()
  ppm(fo, covariates = COVS, nd = NQUAD_PPM)
}

FIT_OBS <- fit_first_order(X_OBS)

# nearestValue() fills the ~500 border pixels the raster does not reach; without
# it the trend would be NA there.
COVS_SIM <- lapply(COVS, function(z) nearestValue(as.im(z, W = WMASK)))

BETA_NAMES <- names(coef(FIT_OBS))
N_BETA     <- length(BETA_NAMES)
PIXAREA    <- WMASK$xstep * WMASK$ystep

stopifnot(BETA_NAMES[1] == "(Intercept)",
          identical(BETA_NAMES[-1], COV_NAMES))

message(sprintf("Window: %.0f x %.0f km | area %.0f km^2 | grid %d x %d (pixel %.2f x %.2f km)",
                diff(Frame(W)$xrange) / 1000, diff(Frame(W)$yrange) / 1000,
                area.owin(W) / 1e6, DIM[1], DIM[2],
                diff(Frame(W)$xrange) / DIM[2] / 1000,
                diff(Frame(W)$yrange) / DIM[1] / 1000))
message(sprintf("Data: %d events (%d exact duplicates dropped, %d outside W) | intensity %.4f per km^2",
                npoints(X_OBS), N_DUP, sum(!INW), intensity(X_OBS) * 1e6))
message(sprintf("Coefficients: %d (%s)", N_BETA, paste(BETA_NAMES, collapse = ", ")))
message(sprintf("Observed beta_hat: %s",
                paste(sprintf("%s=%.3f", BETA_NAMES, coef(FIT_OBS)), collapse = ", ")))
message(sprintf("sigma ~ U[%.1f, %.1f] | xi ~ U[%.0f, %.0f] km | slopes ~ U[%d, %d] | N ~ U[%d, %d]",
                SIGMA_MIN, SIGMA_MAX, XI_MIN / 1000, XI_MAX / 1000,
                BETA_MIN, BETA_MAX, N_MIN, N_MAX))

# ---- Multiscale intensity descriptors ---------------------------------------

NQUAD_BANK     <- c(2, 4, 8, 16, 32, 64)          # nx = ny = k
KDE_BW_BANK    <- c(10, 20, 40, 80, 160) * 1000   # m
KDE_DIMYX_BANK <- c(512, 380)

QUAD_TESS <- lapply(NQUAD_BANK, function(k) {
  list(k = k, tess = quadrats(W, nx = k, ny = k))
})
names(QUAD_TESS) <- paste0("k", NQUAD_BANK)

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
  vals <- vals[!is.na(vals)]          # NA = outside W
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

# 3 x 6 quadrat descriptors plus 5 x 5 KDE descriptors: 43 in total.
extract_intensity_features <- function(pp) {
  quad_bank <- do.call(c, unname(lapply(QUAD_TESS, function(z) {
    qc <- tryCatch(as.vector(quadratcount(pp, tess = z$tess)),
                   error = function(e) NULL)
    st <- quad_stats(qc)
    setNames(as.list(st), paste0("quad_", names(st), "_k", z$k))
  })))

  kde_bank <- do.call(c, unname(lapply(KDE_BW_BANK, function(b) {
    st <- kde_stats(pp, b, KDE_DIMYX_BANK)
    setNames(as.list(st), paste0("kde_", names(st), "_bw", round(b / 1000)))
  })))

  c(quad_bank, kde_bank)
}

FEAT_NAMES <- names(extract_intensity_features(X_OBS))
FEAT_BANK  <- FEAT_NAMES
stopifnot(length(FEAT_NAMES) == 43)
message(sprintf("Intensity features: %d (%d quadrat sizes x 3 + %d bandwidths x 5)",
                length(FEAT_NAMES), length(NQUAD_BANK), length(KDE_BW_BANK)))

# ---- Generator --------------------------------------------------------------

NU_MATERN <- 1

sim_matern_field <- function(sigma, xi) {
  spatstat.random::rGRFmatern(W = W, mu = 0, var = sigma^2, scale = xi,
                              nu = NU_MATERN, dimyx = DIM)
}

trend_image <- function(beta) {
  tr <- as.im(beta[1], W = WMASK)
  for (j in seq_along(COV_NAMES)) {
    Zj <- COVS_SIM[[COV_NAMES[j]]]
    bj <- beta[j + 1]
    tr <- eval.im(tr + bj * Zj)
  }
  tr
}

# beta_0 = log(n_target) - log(int_W exp(Z beta_slopes) ds), with a log-sum-exp
# shift: the exponent reaches ~44 and exp(44) would overflow the sum.
solve_intercept <- function(slopes, n_target) {
  eta <- trend_image(c(0, slopes))
  v   <- eta$v[is.finite(eta$v)]
  m   <- max(v)
  log(n_target) - (m + log(sum(exp(v - m))) + log(PIXAREA))
}

# Centred form, so that E[Lambda(s)] = exp(Z(s) beta). eval.im() resolves images
# by symbol, so the trend has to be bound to a variable first.
lambda_image <- function(beta, Y, sigma) {
  tr  <- trend_image(beta)
  cst <- 0.5 * sigma^2
  eval.im(exp(tr + Y - cst))
}

floor_lambda <- function(lam, pp) {
  if (LAMBDA_FLOOR <= 0) return(list(lambda = lam, hit = 0))
  fl <- LAMBDA_FLOOR * npoints(pp) / area.owin(W)
  list(lambda = eval.im(pmax(lam, fl)), hit = mean(lam[pp] < fl))
}

# L_inhom(r) - r = 0 under an inhomogeneous Poisson process, so departures
# measure the clustering left by the latent field.
compute_L_curve <- function(X, lam) {
  L <- Linhom(X, lambda = lam, correction = "border", r = R_SEQ)$border
  if (anyNA(L)) return(NULL)
  L - R_SEQ
}

# One realisation; NULL if it degenerates.
simulate_lgcp <- function(sigma, xi, slopes, n_target, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)

  tryCatch({
    beta <- c(solve_intercept(slopes, n_target), slopes)
    Y <- sim_matern_field(sigma, xi)
    lam <- lambda_image(beta, Y, sigma)

    # rpoispp() materialises the whole pattern before MAX_POINTS could filter
    # it; the integral is cheap, so reject here instead.
    if (integral.im(lam) > MAX_POINTS) return(NULL)

    X <- rpoispp(lam)[W]
    n <- npoints(X)
    if (n < MIN_POINTS || n > MAX_POINTS) return(NULL)

    fit  <- fit_first_order(X)
    bhat <- unname(coef(fit))
    if (anyNA(bhat) || length(bhat) != N_BETA) return(NULL)

    fl <- floor_lambda(if (L_WITH_TRUE_LAMBDA) lam else
      predict(fit, type = "trend", locations = WMASK), X)

    Lc <- compute_L_curve(X, fl$lambda)
    if (is.null(Lc)) return(NULL)

    ft <- extract_intensity_features(X)

    out <- list(
      X         = X,
      N         = n,
      n_target  = n_target,
      sigma     = sigma,
      xi        = xi,
      beta      = beta,        # true, solved intercept included
      beta_hat  = bhat,        # stage 1, from ppm()
      floor_hit = fl$hit,
      r         = R_SEQ,
      L         = Lc           # stage 2, L(r) - r
    )
    out <- c(out, ft)
    if (SAVE_LAMBDA) out$lambda <- lam
    out
  }, error = function(e) NULL)
}

sample_params <- function(n, seed) {
  set.seed(seed)
  list(
    sigma    = runif(n, SIGMA_MIN, SIGMA_MAX),
    xi       = runif(n, XI_MIN,    XI_MAX),
    n_target = runif(n, N_MIN,     N_MAX),
    slopes   = matrix(runif(n * length(COV_NAMES), BETA_MIN, BETA_MAX),
                      nrow = n, ncol = length(COV_NAMES),
                      dimnames = list(NULL, COV_NAMES))
  )
}

# ---- Observed inputs --------------------------------------------------------

build_observed_inputs <- function() {
  fl <- floor_lambda(predict(FIT_OBS, type = "trend", locations = WMASK), X_OBS)
  Lc <- compute_L_curve(X_OBS, fl$lambda)
  ft <- extract_intensity_features(X_OBS)
  c(list(X = X_OBS, N = npoints(X_OBS),
         beta_hat = unname(coef(FIT_OBS)), beta_names = BETA_NAMES,
         floor_hit = fl$hit,
         # Under the centred form the fitted trend estimates exp(Z beta), so
         # scripts 02-04 can resimulate without rebuilding the covariate setup.
         lambda = predict(FIT_OBS, type = "trend", locations = WMASK),
         lambda_res = predict(FIT_OBS, type = "trend",
                              locations = as.mask(W, dimyx = DIM_RES)),
         dim_res = DIM_RES,
         r = R_SEQ, L = Lc),
    ft)
}

OBS <- build_observed_inputs()
saveRDS(OBS, file.path(OUT_DIR, "observed_inputs.rds"))
message(sprintf("Observed inputs saved (N = %d, floor active on %.1f%% of the points)",
                OBS$N, 100 * OBS$floor_hit))

# Classical reference: minimum contrast on the inhomogeneous K with the
# first-order intensity taken as given, the exact analogue of what the CNN and
# INLA do. kppm() is not used because it refits its own trend with an internal
# ppm() that diverges on these data. The covariance family must match the
# generator: the practical range is sqrt(8*nu)*scale.
MC_REF_K <- tryCatch({
  e <- suppressWarnings(lgcp.estK(
    X_OBS, lambda = OBS$lambda,
    covmodel = list(model = "matern", nu = NU_MATERN),
    startpar = c(var = mean(c(SIGMA_MIN, SIGMA_MAX))^2,
                 scale = mean(c(XI_MIN, XI_MAX))),
    rmax = 100000, correction = "border",
    lower = c(SIGMA_MIN^2, XI_MIN), upper = c(SIGMA_MAX^2, XI_MAX)))
  c(sigma = sqrt(unname(e$par[1])), xi = unname(e$par[2]))
}, error = function(e) {
  warning("Minimum-contrast reference failed: ", conditionMessage(e))
  c(sigma = NA_real_, xi = NA_real_)
})
message(sprintf("Minimum-contrast reference (two-step): sigma = %.3f, xi = %.1f km",
                MC_REF_K[["sigma"]], MC_REF_K[["xi"]] / 1000))

# design_meta.rds lets scripts 02-04 check the design instead of assuming it.
saveRDS(list(beta_range = list(lo = BETA_MIN, hi = BETA_MAX, names = COV_NAMES),
             cov_names = COV_NAMES, n_beta = N_BETA,
             n_obs = npoints(X_OBS), n_dup = N_DUP,
             n_range = c(N_MIN, N_MAX),
             sigma_range = c(SIGMA_MIN, SIGMA_MAX),
             xi_range = c(XI_MIN, XI_MAX),
             rhs = RHS, r = R_SEQ, dim = DIM, dim_res = DIM_RES,
             beta_names = BETA_NAMES,
             lambda_floor = LAMBDA_FLOOR, nquad_ppm = NQUAD_PPM,
             feat_names = FEAT_NAMES,
             feat_bank = FEAT_BANK,
             nquad_bank = NQUAD_BANK, kde_bw_bank = KDE_BW_BANK,
             kde_dimyx_bank = KDE_DIMYX_BANK,
             beta_hat_obs = coef(FIT_OBS),
             field_model = "matern", nu_matern = NU_MATERN,
             mc_reference = list(K = MC_REF_K, method = "lgcp.estK two-step",
                                 covmodel = "matern", nu = NU_MATERN,
                                 rmax = 100000)),
        file.path(OUT_DIR, "design_meta.rds"))

# ---- Example figure ---------------------------------------------------------

if (PLOT_EXAMPLE) {
  ex <- simulate_lgcp(MC_REF_K[["sigma"]], MC_REF_K[["xi"]],
                      unname(coef(FIT_OBS))[-1], npoints(X_OBS), seed = 1)
  png(file.path(FIG_DIR, "example_pattern.png"), width = 1000, height = 750, res = 120)
  op <- par(mfrow = c(1, 2), mar = c(1, 1, 3, 1))
  plot(X_OBS, main = sprintf("Observed: %d events", OBS$N), pch = 16, cex = 0.15)
  if (!is.null(ex)) {
    plot(ex$X, main = sprintf("Simulated: %d (sigma = %.2f, xi = %.0f km)",
                              ex$N, ex$sigma, ex$xi / 1000), pch = 16, cex = 0.15)
  } else {
    plot.new(); title("Example realisation degenerate")
  }
  par(op); dev.off()
  if (!is.null(ex))
    message(sprintf("Example: N = %d | true intercept %.3f vs estimated %.3f",
                    ex$N, ex$beta[1], ex$beta_hat[1]))
}

# ---- Main loop --------------------------------------------------------------

total_valid <- 0
floor_hits  <- numeric(0)

P <- sample_params(N_SIMS, seed = BASE_SEED)
n_chunks <- ceiling(N_SIMS / CHUNK_SIZE)

message(sprintf("\nSimulating %d realisations in %d chunks...", N_SIMS, n_chunks))
message(sprintf("Estimated: ~%.1f h on %d cores (4.8 s per realisation)\n",
                N_SIMS * 4.8 / N_CORES / 3600, N_CORES))

for (k in seq_len(n_chunks)) {
  idx <- ((k - 1) * CHUNK_SIZE + 1):min(k * CHUNK_SIZE, N_SIMS)

  sims_k <- pbmcmapply(
    # One seed per realisation, so the result does not depend on core count.
    FUN = function(i) simulate_lgcp(P$sigma[i], P$xi[i], P$slopes[i, ],
                                    P$n_target[i], seed = BASE_SEED + i),
    idx, SIMPLIFY = FALSE, mc.cores = N_CORES
  )
  sims_k <- sims_k[!sapply(sims_k, is.null)]

  if (length(sims_k) == 0) {
    message(sprintf("  chunk %d/%d: no valid realisation", k, n_chunks))
    next
  }
  # X is 96 % of the size on disk and is not a model input.
  sims_k <- lapply(sims_k, function(z) z[setdiff(names(z), "X")])

  total_valid <- total_valid + length(sims_k)
  floor_hits <- c(floor_hits, sapply(sims_k, `[[`, "floor_hit"))
  message(sprintf("  chunk %d/%d valid: %d/%d (total %d)",
                  k, n_chunks, length(sims_k), length(idx), total_valid))

  saveRDS(sims_k, file.path(OUT_DIR, sprintf("sim_%04d.rds", k)))
  gc()
}

message(sprintf("\nDone: %d realisations saved in %s/", total_valid, OUT_DIR))
# What matters is the fraction of POINTS at the floor, not how many realisations
# touch it at all.
message(sprintf("Realisations touching the floor: %d (%.1f%%)",
                sum(floor_hits > 0), 100 * mean(floor_hits > 0)))
message(sprintf("Fraction of points at the floor: mean %.4f%%, max %.4f%%",
                100 * mean(floor_hits), 100 * max(floor_hits)))
message("  -> the second line is the one that must stay small (<1%).")
