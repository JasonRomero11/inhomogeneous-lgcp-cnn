# =============================================================================
# Minimum contrast (inhomogeneous K) on the test subsample, with a corrected
# optimiser. Based on scripts/bk/05_minimo_contraste_sismosJason.R, which is
# left untouched.
#
# The bug in the original: 486 of the 1000 fits returned exactly the first
# starting value, (sigma, xi) = (2, 80 km). Cause: with rmin = NULL the contrast
# includes r = 0, where the border-corrected K_inhom of a sparse pattern can be
# slightly negative (about -0.008 m^2); mincontrast() treats it as a bad value
# and replaces the objective by a constant penalty (~3e66), so L-BFGS-B reports
# convergence at its starting point and the ladder of starts, which only moved
# on after an error, accepted it. Fix: rmin = RMIN (one pixel of the 2 km
# simulation grid, below which the pattern carries no information about the
# field), every start is run, a fit is kept only if optim converged and moved,
# and the fit with the smallest contrast is chosen.
#
# Same 1000 test realisations (split seed 123, subsample seed 20260803; the 200
# joint INLA-SPDE realisations are the first 200), same two-step protocol
# (trend from the stored ppm beta_hat), same Matern(nu = 1) family, rmax 100 km,
# border correction, bounds = training support.
#
# Outputs: outputs/mc_fixed_estimates.csv, outputs/mc_fixed_metrics.csv
# Usage:   Rscript scripts/09_minimum_contrast.R   (from the repository root)
#          MC_WORKERS=8 to change parallelism; MC_N_TEST=20 for a quick test.
# =============================================================================
suppressMessages({
  library(sf)
  library(spatstat)
  library(tidyverse)
  library(pbmcapply)
})

set.seed(123)
stopifnot(dir.exists("simulations"), dir.exists("data"))

SIM_DIR <- "simulations"
fig_dir <- "outputs"
PREFIJO <- "mc_fixed"

covariables_modelo <- c("inversa", "isostasia", "topografia", "volcanes")

# Deben ser LOS MISMOS del 03/04 para que la submuestra coincida gid a gid.
N_EVAL    <- 1000          # el estimador en dos pasos cuesta
                          # ~87 s por ajuste y no falla, asi que 500 es
                          # asequible. La submuestra es ANIDADA: sample()
                          # selecciona en secuencia, de modo que las 200
                          # anteriores son las primeras 200 de estas 500 y
                          # sus ajustes en cache se reutilizan.
SEED_EVAL <- 20260803

NU_MATERN      <- 1
GEN_N_SIMS     <- 20000
GEN_BASE_SEED  <- 42
GEN_MAX_POINTS <- 80000

N_WORKERS <- as.integer(Sys.getenv("MC_WORKERS", "11"))
PAR_COLS  <- c("sigma", "xi")

out_f <- function(nombre) file.path(fig_dir, paste0(PREFIJO, "_", nombre))

# =============================================================================
# DISEÑO, VENTANA Y COVARIABLES (idéntico al 03/04)
# =============================================================================

META <- readRDS(file.path(SIM_DIR, "design_meta.rds"))
stopifnot(identical(META$beta_names, c("(Intercept)", covariables_modelo)))
N_BETA <- length(META$beta_names)
if (!identical(META$field_model, "matern") || META$nu_matern != NU_MATERN)
  stop("El diseño no es matern(nu = 1); este script asume la cadena multiescala Matérn.")

OBS   <- readRDS(file.path(SIM_DIR, "observed_inputs.rds"))
X_OBS <- OBS$X
stopifnot(npoints(X_OBS) == META$n_obs)
W <- Window(X_OBS)

covar <- lapply(covariables_modelo,
                function(n) readRDS(file.path("data", paste0(n, "_im_scaled.rds"))))
names(covar) <- covariables_modelo
covar <- lapply(covar, function(z) as.im(z, xy = covar[[1]]))
covar_W <- lapply(covar, function(z) z[W, drop = FALSE])
names(covar_W) <- covariables_modelo

# =============================================================================
# RÉPLICA DEL SPLIT TRAIN/VAL/TEST DE LA CNN (idéntica al 03/04)
# =============================================================================

FEAT_COLS <- META$feat_bank
stopifnot(!is.null(FEAT_COLS), length(FEAT_COLS) > 0)

message("=== Réplica del split train/val/test de la CNN ===")
sim_files <- list.files(SIM_DIR, pattern = "^sim_[0-9]+\\.rds$", full.names = TRUE)
stopifnot(length(sim_files) > 0)

SC <- bind_rows(lapply(sim_files, function(f) {
  s <- readRDS(f); s <- s[!vapply(s, is.null, logical(1))]
  if (!length(s)) return(NULL)
  tibble(
    file  = f, pos = seq_along(s),
    sigma = vapply(s, `[[`, numeric(1), "sigma"),
    xi    = vapply(s, `[[`, numeric(1), "xi"),
    N     = vapply(s, `[[`, numeric(1), "N"),
    hit   = vapply(s, `[[`, numeric(1), "floor_hit"),
    ok    = vapply(s, function(z)
              all(is.finite(z$L)) &&
              length(z$beta_hat) == N_BETA && all(is.finite(z$beta_hat)) &&
              all(is.finite(unlist(z[FEAT_COLS]))), logical(1))
  )
}))
SC$gid <- seq_len(nrow(SC))
SC2 <- SC %>% filter(ok, is.finite(sigma), is.finite(xi), is.finite(N))
n <- nrow(SC2)
message(sprintf("Realizaciones utilizables: %d", n))

FRAC_TRAIN <- 0.70; FRAC_VAL <- 0.15
MIN_PER_STRATUM <- 20

qbin <- function(x, k) {
  if (k <= 1) return(rep(1L, length(x)))
  br <- unique(quantile(x, seq(0, 1, length.out = k + 1), na.rm = TRUE))
  if (length(br) < 2) return(rep(1L, length(x)))
  cut(x, breaks = br, labels = FALSE, include.lowest = TRUE)
}

STRAT_BINS <- 1L
for (k in 8:1) {
  if (n / (k^length(PAR_COLS) * 2) >= MIN_PER_STRATUM) { STRAT_BINS <- k; break }
}
strata <- do.call(interaction,
                  c(lapply(PAR_COLS, function(p) qbin(SC2[[p]], STRAT_BINS)),
                    list(SC2$hit > 0), list(drop = TRUE)))

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

grp    <- stratified_split(strata, c(FRAC_TRAIN, FRAC_VAL, 1 - FRAC_TRAIN - FRAC_VAL))
idx_te <- which(grp == 3L)
message(sprintf("Test replicado: %d realizaciones", length(idx_te)))

set.seed(SEED_EVAL)
eval_rows <- sort(sample(idx_te, min(N_EVAL, length(idx_te))))
N_TEST <- as.integer(Sys.getenv("MC_N_TEST", "0"))
if (N_TEST > 0) { eval_rows <- eval_rows[seq_len(N_TEST)]; PREFIJO <- "mc_fixed_test" }
EVAL <- SC2[eval_rows, ]
message(sprintf("Submuestra a evaluar: %d realizaciones", nrow(EVAL)))

# Contraste duro contra el script de INLA vigente (07, dos pasos): los gid de
# la submuestra deben ser los mismos, o la comparacion no seria sobre el mismo
# conjunto. Antes apuntaba a inla_conjunto_*, del modelo conjunto ya retirado.
inla_est_path <- file.path(fig_dir, "inla_eb_estimates_sims.csv")
if (file.exists(inla_est_path)) {
  gids_inla <- sort(as.integer(read_csv(inla_est_path, show_col_types = FALSE)$gid))
  if (N_TEST == 0 && !identical(sort(as.integer(EVAL$gid)), gids_inla))
    stop("La submuestra NO coincide con la del 07: la comparación no sería honesta.")
  message(sprintf("OK: los %d gid coinciden con los de inla_eb_estimates_sims.csv", nrow(EVAL)))
}

# beta_hat de ppm por realización (la etapa 1 de dos pasos)
BHAT <- new.env()
for (f in unique(EVAL$file)) {
  s <- readRDS(f); s <- s[!vapply(s, is.null, logical(1))]
  rows_f <- EVAL %>% filter(file == f)
  for (j in seq_len(nrow(rows_f)))
    assign(as.character(rows_f$gid[j]),
           as.numeric(s[[rows_f$pos[j]]]$beta_hat), envir = BHAT)
}
rm(SC); invisible(gc())

# =============================================================================
# REGENERACIÓN DETERMINISTA DE LOS PATRONES (idéntica al 03/04)
# =============================================================================

message("\n=== Regeneración determinista de los patrones ===")

sample_params_gen <- function(n, seed) {
  set.seed(seed)
  list(
    sigma    = runif(n, META$sigma_range[1], META$sigma_range[2]),
    xi       = runif(n, META$xi_range[1],    META$xi_range[2]),
    n_target = runif(n, META$n_range[1],     META$n_range[2]),
    slopes   = matrix(runif(n * length(covariables_modelo),
                            META$beta_range$lo, META$beta_range$hi),
                      nrow = n, ncol = length(covariables_modelo),
                      dimnames = list(NULL, covariables_modelo))
  )
}
P_GEN <- sample_params_gen(GEN_N_SIMS, GEN_BASE_SEED)

key_par <- function(s, x) paste0(sprintf("%.17g", s), "_", sprintf("%.17g", x))
P_KEY   <- key_par(P_GEN$sigma, P_GEN$xi)
stopifnot(!any(duplicated(P_KEY)))
EVAL$i_gen <- match(key_par(EVAL$sigma, EVAL$xi), P_KEY)
stopifnot(!anyNA(EVAL$i_gen))

WMASK_GEN   <- as.mask(W, dimyx = META$dim)
PIXAREA_GEN <- WMASK_GEN$xstep * WMASK_GEN$ystep
COVS_GEN    <- lapply(covar_W, function(z) nearestValue(as.im(z, W = WMASK_GEN)))
names(COVS_GEN) <- covariables_modelo

trend_image_gen <- function(beta) {
  tr <- as.im(beta[1], W = WMASK_GEN)
  for (j in seq_along(covariables_modelo)) {
    Zj <- COVS_GEN[[covariables_modelo[j]]]
    bj <- beta[j + 1]
    tr <- eval.im(tr + bj * Zj)
  }
  tr
}

solve_intercept_gen <- function(slopes, n_target) {
  eta <- trend_image_gen(c(0, slopes))
  v   <- eta$v[is.finite(eta$v)]
  m   <- max(v)
  log(n_target) - (m + log(sum(exp(v - m))) + log(PIXAREA_GEN))
}

regen_X <- function(i) {
  set.seed(GEN_BASE_SEED + i)
  slopes <- P_GEN$slopes[i, ]
  beta   <- c(solve_intercept_gen(slopes, P_GEN$n_target[i]), slopes)
  Y      <- spatstat.random::rGRFmatern(W = W, mu = 0, var = P_GEN$sigma[i]^2,
                                        scale = P_GEN$xi[i], nu = NU_MATERN,
                                        dimyx = META$dim)
  tr  <- trend_image_gen(beta)
  cst <- 0.5 * P_GEN$sigma[i]^2
  lam <- eval.im(exp(tr + Y - cst))
  if (integral.im(lam) > GEN_MAX_POINTS)
    stop("integral.im > MAX_POINTS en i = ", i)
  rpoispp(lam)[W]
}

t_regen <- system.time({
  PATS <- pbmclapply(seq_len(nrow(EVAL)), function(j) regen_X(EVAL$i_gen[j]),
                     mc.cores = min(N_WORKERS, nrow(EVAL)), mc.preschedule = FALSE)
})
n_regen <- vapply(PATS, function(z) if (inherits(z, "ppp")) npoints(z) else NA_real_, numeric(1))
stopifnot(!anyNA(n_regen), all(n_regen == EVAL$N))
message(sprintf("Regeneración verificada (%d/%d conteos idénticos) en %.0f s",
                nrow(EVAL), nrow(EVAL), t_regen["elapsed"]))

# =============================================================================
# AJUSTE POR CONTRASTE MÍNIMO (dos pasos: lambda de ppm dada)
# =============================================================================
# Misma configuración que META$mc_reference (script 01), aplicada por
# realización. sigma se reporta como sqrt(var) para compararlo con el diseño.

STARTPAR <- c(var = mean(META$sigma_range)^2, scale = mean(META$xi_range))
LOWER    <- c(META$sigma_range[1]^2, META$xi_range[1])
UPPER    <- c(META$sigma_range[2]^2, META$xi_range[2])

# Escalera de arranques para el optimizador: L-BFGS-B aborta a veces la
# búsqueda de línea (ABNORMAL_TERMINATION_IN_LNSRCH) según el punto inicial;
# se prueba desde la media del prior y, si falla, desde dos esquinas suaves.
# Lower limit of r in the contrast: one pixel of the simulation grid (2 km).
RMIN <- 2000
STARTS <- list(STARTPAR,
               c(var = 1,    scale = 40000),
               c(var = 6.25, scale = 120000))

fit_mc <- function(j) {
  X    <- PATS[[j]]
  bhat <- get(as.character(EVAL$gid[j]), envir = BHAT)
  # eval.im() no reconoce imágenes dentro de una llamada: la tendencia debe
  # estar en una variable antes de exponenciarla (mismo detalle que en el 01).
  tr   <- trend_image_gen(bhat)
  lam  <- eval.im(exp(tr))
  fl   <- META$lambda_floor * npoints(X) / area.owin(W)
  lam  <- eval.im(pmax(lam, fl))
  t0 <- proc.time()["elapsed"]
  err <- NA_character_
  # rmax y rejilla de r explícitos. Dos límites distintos pueden dejar la K
  # por debajo de los 100 km nominales y hacer abortar a mincontrast ("rmax
  # exceeds the range of available data"): (i) la corrección border sólo
  # existe hasta la máxima distancia de un punto a la frontera; (ii) la regla
  # por defecto de Kinhom corta su rejilla en sqrt(1000/(pi*lambda_media)),
  # que con N > ~40.000 cae en 83-99 km. Pasar r explícito anula (ii); el
  # recorte por bdist cubre (i).
  rmax_j <- min(1e5, 0.99 * max(bdist.points(X)))
  r_j    <- seq(0, rmax_j, length.out = 513)
  # La K se precalcula con la rejilla explícita y se pasa como objeto fv:
  # lgcp.estK() no acepta r por "..." (acaba en un if interno que espera un
  # escalar), pero sí acepta una funcion resumen ya calculada como primer
  # argumento. Además así la K se computa UNA vez por realización, no una por
  # arranque de la escalera.
  K_j <- tryCatch(
    suppressWarnings(Kinhom(X, lambda = lam, correction = "border", r = r_j)),
    error = function(e) { err <<- conditionMessage(e); NULL })
  # Every start is run; a fit counts only if optim converged and moved away
  # from its start, and the one with the smallest contrast is kept.
  fits <- list()
  if (!is.null(K_j)) {
    for (sp in STARTS) {
      f <- tryCatch(
        suppressWarnings(lgcp.estK(K_j,
                                   covmodel = list(model = "matern", nu = NU_MATERN),
                                   startpar = sp, rmin = RMIN, rmax = rmax_j,
                                   lower = LOWER, upper = UPPER,
                                   control = list(parscale = unname(sp), maxit = 500))),
        error = function(e) { err <<- conditionMessage(e); NULL })
      if (is.null(f)) next
      moved <- any(abs(f$par / sp - 1) > 1e-4)
      if (isTRUE(f$opt$convergence == 0) && moved) fits[[length(fits) + 1]] <- f
    }
  }
  t1 <- proc.time()["elapsed"]
  if (!length(fits))
    return(list(ok = FALSE, sigma = NA_real_, xi = NA_real_, t = t1 - t0,
                err = if (is.na(err)) "no start converged and moved" else err))
  fit <- fits[[which.min(vapply(fits, function(f) f$opt$value, numeric(1)))]]
  p <- fit$par
  list(ok = TRUE, sigma = sqrt(unname(p[1])), xi = unname(p[2]), t = t1 - t0,
       err = NA_character_)
}

message(sprintf("\n=== Ajustes de contraste mínimo (%d realizaciones) ===", nrow(EVAL)))
t_mc <- system.time({
  FITS <- pbmclapply(seq_len(nrow(EVAL)), fit_mc,
                     mc.cores = min(N_WORKERS, nrow(EVAL)), mc.preschedule = FALSE)
  # Reintento EN SERIE de los fallos: con 11 workers concurrentes, Kinhom sobre
  # los patrones de conteo alto (40-60k puntos) puede quedarse sin memoria; en
  # serie el mismo ajuste dispone de toda la RAM y suele completarse.
  malos <- which(!vapply(FITS, function(z) is.list(z) && isTRUE(z$ok), logical(1)))
  if (length(malos)) {
    message(sprintf("Reintentando en serie %d ajustes fallidos (N mediano %.0f)...",
                    length(malos), median(EVAL$N[malos])))
    for (j in malos) FITS[[j]] <- fit_mc(j)
  }
})

RES <- EVAL %>%
  transmute(gid, sigma_true = sigma, xi_true = xi, N,
            ok        = vapply(FITS, `[[`, logical(1), "ok"),
            sigma_hat = vapply(FITS, `[[`, numeric(1), "sigma"),
            xi_hat    = vapply(FITS, `[[`, numeric(1), "xi"),
            tiempo_s  = vapply(FITS, `[[`, numeric(1), "t"),
            error     = vapply(FITS, `[[`, character(1), "err"))
write_csv(RES, out_f("estimates.csv"))
message(sprintf("Ajustes válidos: %d/%d | mediana %.1f s | total %.1f min de reloj",
                sum(RES$ok), nrow(RES), median(RES$tiempo_s), t_mc["elapsed"] / 60))

# =============================================================================
# MÉTRICAS (las mismas de la CNN y de INLA)
# =============================================================================

metricas <- function(true, est) {
  e <- est - true
  tibble(Bias = mean(e), RMSE = sqrt(mean(e^2)),
         RMSE_rel = sqrt(mean(e^2)) / sd(true),
         MAPE = 100 * mean(abs(e / true)),
         R2 = 1 - sum(e^2) / sum((true - mean(true))^2))
}

OKR <- RES %>% filter(ok, is.finite(sigma_hat), is.finite(xi_hat))
MET <- bind_rows(
  metricas(OKR$sigma_true, OKR$sigma_hat) %>%
    mutate(metodo = "Contraste mínimo (K)", n = nrow(OKR), param = "sigma"),
  metricas(OKR$xi_true, OKR$xi_hat) %>%
    mutate(metodo = "Contraste mínimo (K)", n = nrow(OKR), param = "xi")
) %>% select(metodo, n, param, everything())
write_csv(MET, out_f("metrics.csv"))

write_csv(tibble(etapa = c("regeneracion", "ajustes_mc"),
                 n = nrow(EVAL),
                 media_s = c(t_regen["elapsed"] / nrow(EVAL), mean(RES$tiempo_s)),
                 mediana_s = c(NA, median(RES$tiempo_s)),
                 total_min_reloj = c(t_regen["elapsed"], t_mc["elapsed"]) / 60),
          out_f("tiempos.csv"))

print(as.data.frame(MET), digits = 4)
message("\nSalidas escritas con prefijo ", PREFIJO, "_* en ", fig_dir, "/")
