test_that("keras regression backend trains, predicts finite, and beats the mean", {
  skip_if_no_keras()
  set.seed(1)
  n <- 120L
  x <- matrix(rnorm(n * 2), ncol = 2)
  y <- x[, 1] - x[, 2]                     # deterministic signal
  tr <- 1:90; te <- 91:120
  b <- .backend("keras")
  model <- build_model(b, "regression", input_dim = 2L, output_dim = 1L,
                       arch = deepimp_arch_small())
  fit <- fit_model(b, model, x[tr, ], y[tr], x[tr, ], y[tr],
                   control = list(epochs = 40L, patience = 10L, batch_size = 16L,
                                  verbose = FALSE))
  pred <- predict_model(b, fit$model, x[te, ])
  expect_length(pred, length(te))
  expect_true(all(is.finite(pred)))
  rmse_model <- sqrt(mean((pred - y[te])^2))
  rmse_mean  <- sqrt(mean((mean(y[tr]) - y[te])^2))
  expect_lt(rmse_model, rmse_mean)         # correctness, not just shape
})

test_that("keras classification returns a probability matrix and beats base rate", {
  skip_if_no_keras()
  set.seed(2)
  n <- 150L
  x <- matrix(rnorm(n * 2), ncol = 2)
  cls <- as.integer(x[, 1] > 0) + 1L       # 2 classes (1-based), near-deterministic
  tr <- 1:110; te <- 111:150
  b <- .backend("keras")
  model <- build_model(b, "classification", input_dim = 2L, output_dim = 2L,
                       arch = deepimp_arch_small())
  fit <- fit_model(b, model, x[tr, ], cls[tr], x[tr, ], cls[tr],
                   control = list(epochs = 40L, patience = 10L, batch_size = 16L,
                                  verbose = FALSE))
  probs <- predict_model(b, fit$model, x[te, ])
  expect_equal(dim(probs), c(length(te), 2L))
  expect_true(all(abs(rowSums(probs) - 1) < 1e-5))
  pred_cls <- max.col(probs, ties.method = "first")
  acc <- mean(pred_cls == cls[te])
  base <- max(prop.table(table(cls[tr])))
  expect_gt(acc, base)                      # catches the 0-based to_categorical offset bug
})

test_that("keras backend honours arch depth/width without batchnorm/dropout", {
  skip_if_no_keras()
  set.seed(3)
  n <- 40L
  x <- matrix(rnorm(n * 2), ncol = 2); y <- x[, 1]
  b <- .backend("keras")
  model <- build_model(b, "regression", 2L, 1L,
                       arch = deepimp_arch(hidden = c(8, 4), dropout = 0, batchnorm = FALSE))
  fit <- fit_model(b, model, x, y, x, y,
                   control = list(epochs = 5L, patience = 3L, batch_size = 16L, verbose = FALSE))
  expect_length(predict_model(b, fit$model, x), n)
})

test_that("keras offset head nests the GLM (parity with torch)", {
  skip_if_no_keras()
  set.seed(4)
  x <- matrix(rnorm(100 * 2), ncol = 2); y <- 0.5 + x[, 1] - 2 * x[, 2] + rnorm(100)
  b <- .backend("keras")
  arch <- deepimp_arch_small(dropout = 0)
  m <- build_model(b, "regression", 2L, 1L, arch)
  fit <- fit_model(b, m, x[1:80, ], y[1:80], x[81:100, ], y[81:100],
                   control = list(epochs = 0L))
  expect_equal(fit$history$best_epoch, 0L)
  g <- .glm_fit(x[1:80, ], y[1:80], "regression", ridge = arch$glm_ridge)
  expect_equal(predict_model(b, fit$model, x[81:100, ]),
               as.numeric(.glm_link(g, x[81:100, ])), tolerance = 1e-4)

  yc <- ifelse(x[, 1] + rnorm(100) > 0, 2L, 1L)
  mc <- build_model(b, "classification", 2L, 2L, arch)
  fc <- fit_model(b, mc, x[1:80, ], yc[1:80], x[81:100, ], yc[81:100],
                  control = list(epochs = 0L))
  gc <- .glm_fit(x[1:80, ], yc[1:80], "classification", K = 2L, ridge = arch$glm_ridge)
  eta <- .glm_link(gc, x[81:100, ])
  expect_equal(unname(predict_model(b, fc$model, x[81:100, ])),
               exp(eta) / rowSums(exp(eta)), tolerance = 1e-4)

  # after training, validation loss is never worse than the GLM's
  fit2 <- fit_model(b, build_model(b, "regression", 2L, 1L, arch),
                    x[1:80, ], y[1:80], x[81:100, ], y[81:100],
                    control = list(epochs = 20L, patience = 5L, batch_size = 16L))
  mse_glm <- mean((as.numeric(.glm_link(g, x[81:100, ])) - y[81:100])^2)
  mse_net <- mean((predict_model(b, fit2$model, x[81:100, ]) - y[81:100])^2)
  expect_lte(mse_net, mse_glm + 1e-4)
})
