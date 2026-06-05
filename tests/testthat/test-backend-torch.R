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

test_that("singleton final training batch does not empty the validation loader (early-stopping safe)", {
  skip_if_no_torch()
  set.seed(11); torch::torch_manual_seed(11)
  # n_train = 33, batch_size = 32 -> 33 %% 32 == 1 would, under a shared drop_last,
  # also drop the single validation batch and crash luz early stopping. The backend
  # must instead shrink the batch size so training keeps a non-singleton last batch
  # and the validation loss is still computed.
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
