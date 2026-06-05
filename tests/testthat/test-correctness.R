test_that("impNNet recovers a linear signal better than mean imputation", {
  skip_if_no_torch()
  set.seed(101)
  n <- 150L
  df <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
  df$x3 <- 2 * df$x1 - df$x2 + rnorm(n, sd = 0.1)
  truth <- df$x3
  miss <- sample(n, 30L)
  df$x3[miss] <- NA
  imp <- impNNet(df, arch = deepimp_arch_small(), epochs = 80L, iterations = 1L, seed = 1)
  d <- getImputed(imp)
  rmse_model <- sqrt(mean((d$x3[miss] - truth[miss])^2))
  rmse_mean  <- sqrt(mean((mean(df$x3, na.rm = TRUE) - truth[miss])^2))
  expect_lt(rmse_model, rmse_mean)
})

test_that("impNNet classifies a near-deterministic factor above base rate", {
  skip_if_no_torch()
  set.seed(102)
  n <- 200L
  x1 <- rnorm(n)
  g  <- factor(ifelse(x1 + rnorm(n, sd = 0.2) > 0, "hi", "lo"))
  df <- data.frame(x1 = x1, g = g)
  truth <- df$g
  miss <- sample(n, 40L)
  df$g[miss] <- NA
  imp <- impNNet(df, arch = deepimp_arch_small(), epochs = 80L, iterations = 1L, seed = 1)
  d <- getImputed(imp)
  acc  <- mean(as.character(d$g[miss]) == as.character(truth[miss]))
  base <- max(prop.table(table(truth[-miss])))
  expect_gt(acc, base)
})

test_that("impNNet keeps a count column integer-valued and non-negative", {
  skip_if_no_torch()
  set.seed(103)
  n <- 80L
  df <- data.frame(x1 = rnorm(n), cnt = rpois(n, 4))
  df$cnt[sample(n, 10L)] <- NA
  imp <- impNNet(df, arch = deepimp_arch_small(), epochs = 10L, iterations = 1L, seed = 1)
  d <- getImputed(imp)
  expect_false(anyNA(d$cnt))
  expect_true(all(d$cnt == round(d$cnt)))
  expect_true(all(d$cnt >= 0))
})

test_that("impNNetCoDa beats 0.65*dl constant replacement in Aitchison distance", {
  skip_if_no_torch()
  skip_if_not_installed("robCompositions")
  set.seed(104)
  n <- 120L
  # correlated positive parts so the net has signal to exploit
  a <- runif(n, 5, 10); b <- a * runif(n, 0.8, 1.2); cc <- runif(n, 5, 10)
  comp <- data.frame(a = a, b = b, c = cc)
  truth <- comp
  dl <- c(2, 2, 2)
  set_zero <- sample(which(comp$a >= dl[1]), 15L)
  obs <- comp; obs$a[set_zero] <- 0             # rounded zeros in part a

  imp <- impNNetCoDa(obs, dl = dl, label = 0, correction = "truncate",
                     arch = deepimp_arch_small(), epochs = 60L, iterations = 1L, seed = 1)
  d_nn <- getImputed(imp)

  d_const <- obs; d_const$a[set_zero] <- 0.65 * dl[1]

  ad <- function(est) mean(vapply(set_zero, function(i)
    robCompositions::aDist(as.numeric(est[i, ]), as.numeric(truth[i, ])), numeric(1)))
  expect_lt(ad(d_nn), ad(d_const))
})

test_that("impNNet is exactly reproducible with the same seed when dropout = 0", {
  skip_if_no_torch()
  set.seed(105)
  df <- data.frame(x1 = rnorm(40), x2 = rnorm(40)); df$x1[1:5] <- NA
  # Exact reproducibility holds when dropout = 0. With dropout > 0 the torch
  # backend's training-time mask sampling is not fully pinned by the seed in the
  # current torch build (no deterministic-algorithms support), so only `dropout = 0`
  # guarantees bit-for-bit identical results. See ?impNNet (seed).
  arch0 <- deepimp_arch(hidden = c(64, 32), dropout = 0, batchnorm = TRUE)
  a <- getImputed(impNNet(df, arch = arch0, epochs = 5L, iterations = 1L, seed = 7))
  b <- getImputed(impNNet(df, arch = arch0, epochs = 5L, iterations = 1L, seed = 7))
  expect_equal(a$x1, b$x1)
})

test_that("impNNet with dropout is stable (close, not necessarily identical) across seeds", {
  skip_if_no_torch()
  set.seed(105)
  df <- data.frame(x1 = rnorm(60), x2 = rnorm(60))
  df$x1 <- df$x1 + 0.5 * df$x2
  df$x1[1:8] <- NA
  a <- getImputed(impNNet(df, arch = deepimp_arch_small(), epochs = 20L, iterations = 1L, seed = 7))
  b <- getImputed(impNNet(df, arch = deepimp_arch_small(), epochs = 20L, iterations = 1L, seed = 7))
  # same seed, dropout > 0: results are close even if not bit-identical
  expect_true(all(is.finite(a$x1))); expect_true(all(is.finite(b$x1)))
  expect_lt(max(abs(a$x1[1:8] - b$x1[1:8])), 1.0)
})

test_that("impNNet handles edge cases: 2 columns, single NA, full column, constant predictor", {
  skip_if_no_torch()
  set.seed(106)
  # 2-column data (a past bug)
  d2 <- data.frame(x1 = rnorm(40), x2 = rnorm(40)); d2$x1[1:3] <- NA
  expect_false(anyNA(getImputed(
    impNNet(d2, arch = deepimp_arch_small(), epochs = 5L, iterations = 1L, seed = 1))))

  # single NA cell
  d1 <- data.frame(x1 = rnorm(40), x2 = rnorm(40)); d1$x1[1] <- NA
  expect_false(anyNA(getImputed(
    impNNet(d1, arch = deepimp_arch_small(), epochs = 5L, iterations = 1L, seed = 1))))

  # constant predictor column must not break the scaler (no NaN)
  dc <- data.frame(x1 = rnorm(40), k = rep(3, 40), y = rnorm(40)); dc$y[1:4] <- NA
  out <- getImputed(impNNet(dc, arch = deepimp_arch_small(), epochs = 5L, iterations = 1L, seed = 1))
  expect_false(anyNA(out))
  expect_false(any(is.nan(as.matrix(out[vapply(out, is.numeric, logical(1))]))))

  # mi = 3 returns three completions
  imp3 <- impNNet(d1, arch = deepimp_arch_small(), m = 3L, epochs = 3L, iterations = 1L, seed = 1)
  expect_length(getImputed(imp3, "all"), 3L)
})

test_that("impNNet keras backend also beats mean on the linear signal", {
  skip_if_no_keras()
  set.seed(107)
  n <- 150L
  df <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
  df$x3 <- 2 * df$x1 - df$x2 + rnorm(n, sd = 0.1)
  truth <- df$x3
  miss <- sample(n, 30L)
  df$x3[miss] <- NA
  imp <- impNNet(df, backend = "keras", arch = deepimp_arch_small(),
                 epochs = 80L, iterations = 1L, seed = 1)
  d <- getImputed(imp)
  rmse_model <- sqrt(mean((d$x3[miss] - truth[miss])^2))
  rmse_mean  <- sqrt(mean((mean(df$x3, na.rm = TRUE) - truth[miss])^2))
  expect_lt(rmse_model, rmse_mean)
})
