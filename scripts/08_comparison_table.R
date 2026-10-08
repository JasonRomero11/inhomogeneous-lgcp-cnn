# Comparison table of the estimators on common test realisations. No training:
# it only reads the predictions already written by
#   07_ablation_bank.R            outputs/ablation_pred_test.csv  (three CNNs, full test)
#   09_minimum_contrast.R         outputs/mc_fixed_estimates.csv  (minimum contrast, 1000 test realisations)
#   bk/04_INLA_SPDE_conjunto_*.R  outputs/inla_conjunto_estimates_sims.csv (joint INLA-SPDE, 200 of those)
# The INLA and minimum-contrast subsamples are nested random samples of the
# test split (seed 20260803): the 200 INLA realisations are the first 200 of
# the 1000 minimum-contrast ones.
#
# Outputs (prefix "comparison_"): metrics on (a) the 1000 minimum-contrast
# realisations and (b) the joint INLA-SPDE fits inside the design support.
#
# Usage:  Rscript scripts/08_comparison_table.R   (from the repository root)

suppressPackageStartupMessages({library(dplyr); library(readr); library(purrr)})
OUT <- "outputs"

pred <- read_csv(file.path(OUT, "ablation_pred_test.csv"), show_col_types = FALSE)
mc <- read_csv(file.path(OUT, "mc_fixed_estimates.csv"), show_col_types = FALSE) %>%
  transmute(gid, sigma_true_mc = sigma_true, sigma_mc = sigma_hat, xi_mc = xi_hat,
            mc_ok = ok, mc_time_s = tiempo_s)
inla <- read_csv(file.path(OUT, "inla_conjunto_estimates_sims.csv"), show_col_types = FALSE) %>%
  group_by(gid) %>% slice_tail(n = 1) %>% ungroup() %>%
  transmute(gid, sigma_inla = sigma, xi_inla = xi, inla_valid = dominio & is.finite(sigma) & is.finite(xi),
            inla_time_s = tiempo_total_s)

d_mc <- inner_join(pred, mc, by = "gid")
stopifnot(nrow(d_mc) == nrow(mc), max(abs(d_mc$sigma_true - d_mc$sigma_true_mc)) < 1e-9)
d_inla <- inner_join(d_mc, inla, by = "gid") %>% filter(inla_valid)

metric_row <- function(tv, pv) {
  e <- pv - tv
  tibble(n = length(tv), Bias = mean(e), RMSE = sqrt(mean(e^2)),
         MAPE = 100 * mean(abs(e / tv)), R2 = 1 - sum(e^2) / sum((tv - mean(tv))^2))
}
methods <- c(no_bank = "CNN without descriptors", single = "CNN, single scale (8)",
             multi = "CNN, multiscale bank (43)", mc = "Minimum contrast (K_inhom)",
             inla = "Joint INLA-SPDE")
table_on <- function(d, set, which) {
  map_dfr(which, function(m) map_dfr(c("sigma", "xi"), function(p) {
    tv <- d[[paste0(p, "_true")]]; pv <- d[[paste0(p, "_", m)]]
    ok <- is.finite(pv)
    metric_row(tv[ok], pv[ok]) %>%
      mutate(set = set, method = methods[[m]], param = p, n_missing = sum(!ok), .before = 1)
  }))
}

TAB <- bind_rows(
  table_on(d_mc, sprintf("minimum-contrast sample (%d)", nrow(d_mc)), c("no_bank", "single", "multi", "mc")),
  table_on(d_inla, sprintf("valid joint INLA-SPDE fits (%d)", nrow(d_inla)), names(methods))
) %>% mutate(across(c(Bias, RMSE), ~ ifelse(param == "xi", .x / 1000, .x)))   # xi in km
write_csv(TAB, file.path(OUT, "comparison_metrics.csv"))
print(TAB %>% mutate(across(Bias:R2, ~ round(.x, 3))), n = Inf, width = Inf)

TIMES <- tibble(
  method = c("Minimum contrast", "Joint INLA-SPDE"),
  n = c(sum(is.finite(mc$mc_time_s)), sum(is.finite(inla$inla_time_s))),
  median_s = c(median(mc$mc_time_s, na.rm = TRUE), median(inla$inla_time_s, na.rm = TRUE)),
  failed = c(sum(!mc$mc_ok), sum(!inla$inla_valid)))
write_csv(TIMES, file.path(OUT, "comparison_times.csv"))
print(TIMES)
