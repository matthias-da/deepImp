test_that("torch backend builds, fits, and predicts for regression", {
  skip_if_not_installed("torch")
  skip_if_not_installed("luz")

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
  skip_if_not_installed("torch")
  skip_if_not_installed("luz")

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
