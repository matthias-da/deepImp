test_that("deepimp_arch validates and stores the GLM-offset settings", {
  a <- deepimp_arch()
  expect_true(a$glm_offset)
  expect_equal(a$glm_ridge, 1e-3)
  expect_false(deepimp_arch(glm_offset = FALSE)$glm_offset)
  expect_true(deepimp_arch_small()$glm_offset)
  expect_error(deepimp_arch(glm_offset = NA), "glm_offset")
  expect_error(deepimp_arch(glm_ridge = -1), "glm_ridge")
  expect_output(print(a), "GLM offset")
})

test_that(".glm_fit regression with ridge = 0 reproduces lm()", {
  set.seed(1)
  x <- matrix(rnorm(100 * 3), ncol = 3)
  y <- 1 + x %*% c(2, -1, 0.5) + rnorm(100)
  g <- .glm_fit(x, y, "regression", ridge = 0)
  expect_equal(as.numeric(.glm_link(g, x)), as.numeric(fitted(lm(y ~ x))),
               tolerance = 1e-8)
})

test_that(".glm_fit drops aliased one-hot columns and stays finite", {
  set.seed(2)
  f <- factor(sample(c("a", "b", "c"), 80, replace = TRUE))
  x <- cbind(rnorm(80), stats::model.matrix(~ f - 1))   # full one-hot + intercept
  y <- x[, 1] + as.integer(f) + rnorm(80)
  g <- .glm_fit(x, y, "regression")
  expect_true(all(is.finite(.glm_link(g, x))))
  expect_equal(length(g$keep), 3L)                       # one dummy aliased
})

test_that(".glm_fit classification matches nnet::multinom and handles absent classes", {
  set.seed(3)
  x <- matrix(rnorm(150 * 2), ncol = 2)
  lp <- cbind(0, x[, 1], -x[, 2])
  y <- apply(lp, 1, function(e) sample(3, 1, prob = exp(e) / sum(exp(e))))
  g <- .glm_fit(x, y, "classification", K = 3L, ridge = 0)
  eta <- .glm_link(g, x)
  pr <- exp(eta) / rowSums(exp(eta))
  mn <- nnet::multinom(factor(y) ~ x, trace = FALSE)
  expect_equal(unname(pr), unname(fitted(mn)), tolerance = 1e-3)

  # class 2 absent from the training rows (e.g. a bootstrap resample)
  keep <- y != 2
  g2 <- .glm_fit(x[keep, ], y[keep], "classification", K = 3L)
  eta2 <- .glm_link(g2, x)
  expect_equal(dim(eta2), c(150L, 3L))
  expect_true(all(eta2[, 2] < pmin(eta2[, 1], eta2[, 3])))
})

test_that("the ridge guard keeps a separated multinomial fit finite", {
  x <- matrix(c(-3:-1, 1:3), ncol = 1)
  y <- c(1, 1, 1, 2, 2, 2)                               # perfectly separated
  g <- .glm_fit(x, y, "classification", K = 2L, ridge = 1e-3)
  expect_true(all(is.finite(g$B)))
  expect_lt(max(abs(g$B)), 50)
})

test_that("an untrained offset head reproduces its GLM (restore-best scores epoch 0)", {
  skip_if_no_torch()
  set.seed(4); torch::torch_manual_seed(4)
  x <- matrix(rnorm(100 * 2), ncol = 2); y <- 0.5 + x[, 1] - 2 * x[, 2] + rnorm(100)
  b <- .backend("torch")
  arch <- deepimp_arch_small(dropout = 0)
  m <- build_model(b, "regression", 2L, 1L, arch)
  fit <- fit_model(b, m, x[1:80, ], y[1:80], x[81:100, ], y[81:100],
                   control = list(epochs = 0L, patience = 5L))
  expect_equal(fit$history$best_epoch, 0L)
  g <- .glm_fit(x[1:80, ], y[1:80], "regression", ridge = arch$glm_ridge)
  expect_equal(predict_model(b, fit$model, x[81:100, ]),
               as.numeric(.glm_link(g, x[81:100, ])), tolerance = 1e-5)

  # classification: probabilities = softmax of the multinomial logits
  yc <- ifelse(x[, 1] + rnorm(100) > 0, 2L, 1L)
  mc <- build_model(b, "classification", 2L, 2L, arch)
  fc <- fit_model(b, mc, x[1:80, ], yc[1:80], x[81:100, ], yc[81:100],
                  control = list(epochs = 0L))
  gc <- .glm_fit(x[1:80, ], yc[1:80], "classification", K = 2L, ridge = arch$glm_ridge)
  eta <- .glm_link(gc, x[81:100, ])
  expect_equal(unname(predict_model(b, fc$model, x[81:100, ])),
               exp(eta) / rowSums(exp(eta)), tolerance = 1e-5)
})

test_that("the offset head never does worse on validation than its GLM", {
  skip_if_no_torch()
  set.seed(5); torch::torch_manual_seed(5)
  x <- matrix(rnorm(200 * 2), ncol = 2)
  y <- x[, 1] * x[, 2] + 0.5 * x[, 1] + rnorm(200, sd = 0.3)   # interaction
  b <- .backend("torch")
  arch <- deepimp_arch_small(dropout = 0)
  m <- build_model(b, "regression", 2L, 1L, arch)
  fit <- fit_model(b, m, x[1:150, ], y[1:150], x[151:200, ], y[151:200],
                   control = list(epochs = 60L, patience = 10L))
  g <- .glm_fit(x[1:150, ], y[1:150], "regression", ridge = arch$glm_ridge)
  mse_glm <- mean((as.numeric(.glm_link(g, x[151:200, ])) - y[151:200])^2)
  mse_net <- mean((predict_model(b, fit$model, x[151:200, ]) - y[151:200])^2)
  expect_lte(mse_net, mse_glm + 1e-6)
  expect_lt(mse_net, 0.8 * mse_glm)          # the correction learns the interaction
})
