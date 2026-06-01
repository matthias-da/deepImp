test_that(".task_for_type maps types to model tasks", {
  expect_equal(.task_for_type("numeric"), "regression")
  expect_equal(.task_for_type("count"), "regression")
  expect_equal(.task_for_type("mixed"), "regression")
  expect_equal(.task_for_type("binary"), "classification")
  expect_equal(.task_for_type("nominal"), "classification")
})

test_that(".fit_scaler / .apply_scaler standardise and round-trip", {
  m <- matrix(c(1, 2, 3, 10, 20, 30), ncol = 2)
  s <- .fit_scaler(m)
  z <- .apply_scaler(m, s)
  expect_equal(colMeans(z), c(0, 0), tolerance = 1e-8)
  expect_equal(apply(z, 2, sd), c(1, 1), tolerance = 1e-8)
  # zero-variance columns must not produce NaN
  s0 <- .fit_scaler(matrix(c(5, 5, 5), ncol = 1))
  expect_false(any(is.nan(.apply_scaler(matrix(c(5, 5, 5), ncol = 1), s0))))
})

test_that(".predictor_matrix scales numerics and one-hots factors with shared encoding", {
  df <- data.frame(a = c(1, 2, 3, 4), b = factor(c("x", "y", "x", "y")))
  types <- c("numeric", "nominal")
  fit <- .predictor_matrix(df, types)
  expect_true(is.matrix(fit$x))
  expect_equal(nrow(fit$x), 4L)
  # applying the fitted encoding to new rows keeps the same columns
  new <- .predictor_matrix(df[1:2, , drop = FALSE], types, encoding = fit$encoding)
  expect_equal(ncol(new$x), ncol(fit$x))
  expect_equal(colnames(new$x), colnames(fit$x))
})

test_that(".convergence is 0 for identical frames and scale-free", {
  old <- data.frame(n = c(1, 2, 3), f = factor(c("a", "b", "a")))
  na_mask <- matrix(c(TRUE, FALSE, TRUE, FALSE, TRUE, FALSE), ncol = 2)
  expect_equal(.convergence(old, old, na_mask, c("numeric", "nominal")), 0)
  new <- old; new$f <- factor(c("a", "a", "a"))   # changes the masked cell (row 2: b -> a)
  expect_gt(.convergence(old, new, na_mask, c("numeric", "nominal")), 0)
})
