test_that("torch backend builds, fits, and predicts for regression", {
  skip_if_no_torch()

  set.seed(1)
  torch::torch_manual_seed(1)

  n <- 80L
  x <- matrix(rnorm(n * 3L), n, 3L)
  y <- x[, 1] - 0.5 * x[, 2] + rnorm(n, sd = 0.1)

  backend <- .backend("torch")
  arch <- deepimp_arch_small()

  model <- build_model(backend, task = "regression",
                       input_dim = 3L, output_dim = 1L, arch = arch)
  fitted <- fit_model(backend, model,
                      train_x = x[1:60, ], train_y = y[1:60],
                      valid_x = x[61:80, ], valid_y = y[61:80],
                      control = list(epochs = 5L, batch_size = 16L,
                                     patience = 3L, verbose = FALSE))
  p <- predict_model(backend, fitted$model, x[71:80, ])

  expect_length(p, 10L)
  expect_true(all(is.finite(p)))
})

test_that("torch backend builds, fits, and predicts for classification", {
  skip_if_no_torch()

  set.seed(2)
  torch::torch_manual_seed(2)

  n <- 80L
  x <- matrix(rnorm(n * 3L), n, 3L)
  lp <- x[, 1] + x[, 2]
  y <- factor(ifelse(lp > 0, "b", "a"), levels = c("a", "b"))

  backend <- .backend("torch")
  arch <- deepimp_arch_small()

  model <- build_model(backend, task = "classification",
                       input_dim = 3L, output_dim = 2L, arch = arch)
  fitted <- fit_model(backend, model,
                      train_x = x[1:60, ], train_y = y[1:60],
                      valid_x = x[61:80, ], valid_y = y[61:80],
                      control = list(epochs = 5L, batch_size = 16L,
                                     patience = 3L, verbose = FALSE))
  probs <- predict_model(backend, fitted$model, x[71:80, ])

  expect_equal(dim(probs), c(10L, 2L))
  expect_true(all(abs(rowSums(probs) - 1) < 1e-5))
})

test_that("a singleton final training batch is skipped, not fed to batch norm", {
  skip_if_no_torch()
  set.seed(11); torch::torch_manual_seed(11)
  # n_train = 33, batch_size = 32 -> the last minibatch has one row, which would
  # break nn_batch_norm1d. The training loop skips it; validation is unaffected.
  x <- matrix(rnorm(33 * 3), ncol = 3); y <- x[, 1] - x[, 2]
  xv <- matrix(rnorm(10 * 3), ncol = 3); yv <- xv[, 1] - xv[, 2]
  b <- .backend("torch")
  m <- build_model(b, "regression", 3L, 1L, deepimp_arch_small())
  expect_no_error(
    fit <- fit_model(b, m, x, y, xv, yv,
                     control = list(epochs = 5L, patience = 3L, batch_size = 32L,
                                    verbose = FALSE))
  )
  expect_length(predict_model(b, fit$model, x), 33L)
})

test_that("impNNet imputes a small mixed-type frame end to end (beer-like, regression + binary)", {
  skip_if_no_torch()
  set.seed(12)
  n <- 86L
  df <- data.frame(
    a = rnorm(n), b = rnorm(n), c = rnorm(n),
    grp = factor(sample(c("x", "y"), n, replace = TRUE))
  )
  # sparse MCAR holes (1-3 per column) reproduce the n_train %% batch_size == 1 regime
  for (j in seq_len(ncol(df))) df[sample(n, sample(1:3, 1)), j] <- NA
  expect_no_error(
    imp <- impNNet(df, arch = deepimp_arch_small(), epochs = 20L, seed = 1)
  )
  expect_false(anyNA(getImputed(imp)))
})

test_that("torch fit reports its history and restores the best weights", {
  skip_if_no_torch()
  set.seed(3); torch::torch_manual_seed(3)
  x <- matrix(rnorm(120 * 2), ncol = 2); y <- x[, 1] + rnorm(120, sd = 0.3)
  b <- .backend("torch")
  m <- build_model(b, "regression", 2L, 1L, deepimp_arch_small(dropout = 0))
  fit <- fit_model(b, m, x[1:90, ], y[1:90], x[91:120, ], y[91:120],
                   control = list(epochs = 30L, patience = 5L, batch_size = 16L))
  h <- fit$history
  expect_true(h$epochs >= 1L && h$epochs <= 30L)
  expect_true(h$best_epoch >= 0L && h$best_epoch <= h$epochs)
  p <- predict_model(b, fit$model, x[91:120, ])
  # the restored weights are the ones that scored best_valid
  expect_equal(mean((p - y[91:120])^2), h$best_valid, tolerance = 1e-5)
})
