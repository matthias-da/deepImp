test_that(".handle_deprecated_args maps dense and small with warnings", {
  # expect_warning() returns the condition, not the value; capture value separately.
  expect_warning(.handle_deprecated_args(list(dense = c(8, 4)), arch = NULL, normalize = TRUE),
                 "deprecated")
  res <- suppressWarnings(
    .handle_deprecated_args(list(dense = c(8, 4)), arch = NULL, normalize = TRUE))
  expect_s3_class(res$arch, "deepimp_arch")
  expect_equal(res$arch$hidden, c(8L, 4L))
  expect_length(res$dots, 0L)

  expect_warning(.handle_deprecated_args(list(small = TRUE), arch = NULL, normalize = TRUE),
                 "deprecated")
  res2 <- suppressWarnings(
    .handle_deprecated_args(list(small = TRUE), arch = NULL, normalize = TRUE))
  expect_equal(res2$arch$hidden, deepimp_arch_small()$hidden)
})

test_that(".handle_deprecated_args maps normalize_x and warns normalize_y ignored", {
  expect_warning(.handle_deprecated_args(list(normalize_x = FALSE), arch = NULL, normalize = TRUE),
                 "deprecated")
  res <- suppressWarnings(
    .handle_deprecated_args(list(normalize_x = FALSE), arch = NULL, normalize = TRUE))
  expect_false(res$normalize)
  expect_warning(
    .handle_deprecated_args(list(normalize_y = TRUE), arch = NULL, normalize = TRUE),
    "ignored")
})

test_that(".handle_deprecated_args errors on removed args", {
  expect_error(.handle_deprecated_args(list(model_numeric = 1), arch = NULL, normalize = TRUE),
               "removed")
  expect_error(.handle_deprecated_args(list(loss = "mse"), arch = NULL, normalize = TRUE),
               "removed")
})

test_that(".handle_deprecated_args passes genuine arch dots through untouched", {
  res <- .handle_deprecated_args(list(hidden = c(16, 8), dropout = 0.2),
                                 arch = NULL, normalize = TRUE)
  expect_null(res$arch)
  expect_equal(res$dots$hidden, c(16, 8))
  expect_equal(res$dots$dropout, 0.2)
})

test_that("impNNet end to end accepts deprecated dense with a warning", {
  skip_if_no_torch()
  set.seed(1)
  df <- data.frame(a = rnorm(40), b = rnorm(40)); df$a[1:4] <- NA
  expect_warning(
    impNNet(df, dense = c(8, 4), epochs = 2L, iterations = 1L, seed = 1),
    "deprecated")
  imp <- suppressWarnings(
    impNNet(df, dense = c(8, 4), epochs = 2L, iterations = 1L, seed = 1))
  expect_equal(imp$arch$hidden, c(8L, 4L))
})
