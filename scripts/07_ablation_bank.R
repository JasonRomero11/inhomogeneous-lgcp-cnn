# Ablation of the multiscale descriptor bank, and comparison with the
# INLA-SPDE fit on the same test realisations.
#
# Three versions of the estimator are compared on the test split of
# 02_train_cnn.R, with the same split, scalers, architecture, training schedule
# and seeds (1000+i); only the descriptor branch changes:
#
#   no_bank      D(r), beta_hat and N only (no descriptor branch)
#   single_scale 8 descriptors: the quadrat resolution and the KDE bandwidth
#                with the largest linear signal on log(xi) in the TRAINING set
#                (3 quadrat + 5 KDE statistics), dense branch c(32, 16)
#   multiscale   the 43-descriptor bank of the paper (predictions read from
#                outputs/cnn_pred_test.csv, written by 02_train_cnn.R)
#
# The INLA-SPDE fit (trend and field estimated together) was run on 200
# of those test realisations by scripts/bk/04_INLA_SPDE_conjunto_sismosJason.R;
# its estimates are read from outputs/inla_conjunto_estimates_sims.csv.
#
# Nothing written by scripts 01-06 is modified. Outputs carry the prefix
# "ablation_".
#
# Usage:  Rscript scripts/07_ablation_bank.R   (from the repository root)

stopifnot(file.exists("scripts/02_train_cnn.R"),
          file.exists("outputs/cnn_pred_test.csv"),
          file.exists("outputs/inla_conjunto_estimates_sims.csv"))

T0 <- Sys.time()

# ---- 1. Configuration, data, split and scalers of script 02 ----------------
# Evaluate script 02 up to (not including) the call that trains the paper
# model, so that every preprocessing step is literally the same.
local({
  lines   <- readLines("scripts/02_train_cnn.R")
  stop_at <- grep("^FIT\\s*<-\\s*train_seeds\\(", lines)[1]
  stopifnot(!is.na(stop_at))
  eval(parse(text = lines[seq_len(stop_at - 1)], keep.source = FALSE),
       envir = globalenv())
})

OUT_DIR <- "outputs"
PREFIX  <- "ablation_"

# ---- 2. Single scale chosen on the training set ----------------------------
# For each quadrat resolution and each KDE bandwidth, R2 of a linear model of
# log(xi) on that scale's descriptors, fitted on the training rows only.
lin_r2 <- function(cols) {
  y <- log(DAT$Y[idx_tr, "xi"])
  x <- DAT$F[idx_tr, cols, drop = FALSE]
  summary(lm(y ~ x))$r.squared
}
quad_scales <- paste0("k", META$nquad_bank)
kde_scales  <- paste0("bw", META$kde_bw_bank / 1000)
r2_quad <- sapply(quad_scales, function(s) lin_r2(grep(paste0("^quad_.*_", s, "$"), FEAT_COLS, value = TRUE)))
r2_kde  <- sapply(kde_scales,  function(s) lin_r2(grep(paste0("^kde_.*_",  s, "$"), FEAT_COLS, value = TRUE)))
message("\nLinear R2 of log(xi) per single scale (training set):")
print(round(r2_quad, 3)); print(round(r2_kde, 3))

best_quad <- names(which.max(r2_quad))
best_kde  <- names(which.max(r2_kde))
SINGLE_COLS <- c(grep(paste0("^quad_.*_", best_quad, "$"), FEAT_COLS, value = TRUE),
                 grep(paste0("^kde_.*_",  best_kde,  "$"), FEAT_COLS, value = TRUE))
stopifnot(length(SINGLE_COLS) == 8)
message(sprintf("Single-scale variant: quadrats %s + KDE %s km -> %s",
                best_quad, sub("bw", "", best_kde), paste(SINGLE_COLS, collapse = ", ")))

# ---- 3. Variants ------------------------------------------------------------
# Same layers, callbacks and head as train_once() in script 02; the descriptor
# branch is dropped (feat_idx = NULL) or fed a subset of the bank.
train_variant_once <- function(tag, feat_idx, feat_units) {
  in_S2 <- layer_input(shape = c(NR, S2_CHANNELS), name = paste0("S2", tag))
  in_B  <- layer_input(shape = c(ncol(DAT$B)),     name = paste0("S1", tag))
  in_N  <- layer_input(shape = c(1),               name = paste0("N",  tag))
  inputs   <- list(in_S2, in_B, in_N)
  branches <- list(build_S2_branch(in_S2), build_dense_branch(in_B, c(32, 16)), in_N)
  x_tr <- list(X_S2$tr, X_B$tr, X_N$tr)
  x_va <- list(X_S2$va, X_B$va, X_N$va)
  x_te <- list(X_S2$te, X_B$te, X_N$te)

  if (!is.null(feat_idx)) {
    in_F <- layer_input(shape = c(length(feat_idx)), name = paste0("Ifeat", tag))
    inputs   <- c(inputs, in_F)
    branches <- c(branches, build_dense_branch(in_F, feat_units))
    x_tr <- c(x_tr, list(X_F$tr[, feat_idx, drop = FALSE]))
    x_va <- c(x_va, list(X_F$va[, feat_idx, drop = FALSE]))
    x_te <- c(x_te, list(X_F$te[, feat_idx, drop = FALSE]))
  }

  output <- layer_concatenate(branches) %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dropout(rate = DROPOUT) %>%
    layer_dense(units = 32, activation = "relu") %>%
    layer_dropout(rate = DROPOUT) %>%
    layer_dense(units = length(PAR_COLS), activation = "linear")

  model <- keras_model(inputs = inputs, outputs = output)
  model %>% compile(loss = "mse", optimizer = optimizer_adam(learning_rate = LR),
                    metrics = list("mae"))
  model %>% fit(x = x_tr, y = Y_tr, epochs = EPOCHS, batch_size = BATCH_SIZE,
                validation_data = list(x_va, Y_va), callbacks = make_callbacks(),
                verbose = 0)
  list(pred = invert_target(predict(model, x_te, verbose = 0)),
       n_params = as.numeric(model$count_params()))
}

train_variant <- function(label, feat_idx, feat_units = c(32, 16)) {
  message(sprintf("\n=== %s (%d seeds) ===", label, N_SEEDS))
  t0 <- Sys.time()
  runs <- lapply(seq_len(N_SEEDS), function(i) {
    tensorflow::set_random_seed(1000 + i)
    r <- train_variant_once(sprintf("_%s_s%d", gsub("\\W", "", label), i), feat_idx, feat_units)
    message(sprintf("  seed %d done (%.1f min)", i, as.numeric(difftime(Sys.time(), t0, units = "mins"))))
    r
  })
  pred <- Reduce(`+`, lapply(runs, `[[`, "pred")) / N_SEEDS
  colnames(pred) <- PAR_COLS
  list(pred = pred, n_params = runs[[1]]$n_params)
}

FIT_NOBANK <- train_variant("no_bank", NULL)
FIT_SINGLE <- train_variant("single_scale", match(SINGLE_COLS, FEAT_COLS), c(32, 16))

# ---- 4. Multiscale predictions of the paper and INLA-SPDE estimates ---------
pred_ms <- readr::read_csv(file.path(OUT_DIR, "cnn_pred_test.csv"), show_col_types = FALSE)
stopifnot(identical(as.integer(pred_ms$gid), as.integer(idx_te)),
          max(abs(pred_ms$sigma_true - TRUE_PAR[, "sigma"])) < 1e-9)

PRED <- tibble(gid = idx_te,
               sigma_true = TRUE_PAR[, "sigma"], xi_true = TRUE_PAR[, "xi"],
               sigma_no_bank = FIT_NOBANK$pred[, "sigma"], xi_no_bank = FIT_NOBANK$pred[, "xi"],
               sigma_single  = FIT_SINGLE$pred[, "sigma"], xi_single  = FIT_SINGLE$pred[, "xi"],
               sigma_multi   = pred_ms$sigma_cnn,          xi_multi   = pred_ms$xi_cnn)
write_csv(PRED, file.path(OUT_DIR, paste0(PREFIX, "pred_test.csv")))

# Several rows per gid are possible in the INLA file (retries); keep the last.
INLA <- readr::read_csv(file.path(OUT_DIR, "inla_conjunto_estimates_sims.csv"),
                        show_col_types = FALSE) %>%
  group_by(gid) %>% slice_tail(n = 1) %>% ungroup() %>%
  transmute(gid, sigma_inla = sigma, xi_inla = xi, inla_domain = dominio,
            inla_time_s = tiempo_total_s)
stopifnot(nrow(INLA) == 200, all(INLA$gid %in% idx_te))
JOINT <- inner_join(PRED, INLA, by = "gid")

# ---- 5. Metrics -------------------------------------------------------------
metric_row <- function(tv, pv, method, set, param) {
  e <- pv - tv
  tibble(set = set, method = method, param = param, n = length(tv),
         Bias = mean(e), RMSE = sqrt(mean(e^2)), RMSE_rel = sqrt(mean(e^2)) / sd(tv),
         MAPE = 100 * mean(abs(e / tv)),
         R2 = 1 - sum(e^2) / sum((tv - mean(tv))^2))
}
cnn_methods <- c(no_bank = "CNN without descriptors",
                 single  = "CNN, single-scale descriptors (8)",
                 multi   = "CNN, multiscale bank (43)")

metrics_set <- function(d, set, with_inla) {
  out <- purrr::map_dfr(names(cnn_methods), function(m) purrr::map_dfr(PAR_COLS, function(p)
    metric_row(d[[paste0(p, "_true")]], d[[paste0(p, "_", m)]], cnn_methods[[m]], set, p)))
  if (with_inla) out <- bind_rows(out, purrr::map_dfr(PAR_COLS, function(p)
    metric_row(d[[paste0(p, "_true")]], d[[paste0(p, "_inla")]], "INLA-SPDE", set, p)))
  out
}

dom <- JOINT %>% filter(inla_domain, is.finite(sigma_inla), is.finite(xi_inla))
METRICS <- bind_rows(
  metrics_set(PRED, "test (all)", FALSE),
  metrics_set(JOINT, "INLA subset (200)", FALSE),
  metrics_set(dom, sprintf("INLA converged in domain (%d)", nrow(dom)), TRUE)
) %>% mutate(across(c(Bias, RMSE), ~ ifelse(param == "xi", .x / 1000, .x)))  # xi in km
write_csv(METRICS, file.path(OUT_DIR, paste0(PREFIX, "metrics.csv")))

message("\nMetrics (xi bias/RMSE in km):")
print(METRICS %>% mutate(across(Bias:R2, ~ round(.x, 3))), n = Inf, width = Inf)

INLA_SUMMARY <- tibble(
  n_fits = nrow(JOINT), n_converged_in_domain = nrow(dom),
  pct_failed = 100 * (1 - nrow(dom) / nrow(JOINT)),
  median_time_s = median(JOINT$inla_time_s, na.rm = TRUE),
  median_time_converged_s = median(dom$inla_time_s, na.rm = TRUE)
)
message("\nINLA-SPDE fit:"); print(INLA_SUMMARY)
write_csv(INLA_SUMMARY, file.path(OUT_DIR, paste0(PREFIX, "inla_summary.csv")))

write_csv(tibble(variant = c("no_bank", "single_scale", "multiscale"),
                 descriptors = c(0, 8, 43),
                 single_scale_columns = c(NA, paste(SINGLE_COLS, collapse = ";"), NA),
                 network_parameters = c(FIT_NOBANK$n_params, FIT_SINGLE$n_params, NA)),
          file.path(OUT_DIR, paste0(PREFIX, "variants.csv")))

message(sprintf("\nDone in %.1f min.", as.numeric(difftime(Sys.time(), T0, units = "mins"))))
