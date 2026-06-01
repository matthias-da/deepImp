test_that("impNNetCoDa errors when a correction needs dl but it is missing", {
  x <- data.frame(a = c(0, 2, 3), b = c(1, 2, 3), c = c(2, 1, 1))
  expect_error(impNNetCoDa(x, label = 0, correction = "truncate"), "dl")
})

test_that("impNNetCoDa imputes rounded zeros as strictly positive values <= dl", {
  skip_if_not_installed("torch"); skip_if_not_installed("luz")
  skip_if_not_installed("robCompositions")
  set.seed(1)
  n <- 60L
  x <- data.frame(a = runif(n, 5, 10), b = runif(n, 5, 10), c = runif(n, 5, 10))
  dl <- c(1, 1, 1)
  x$a[1:8] <- 0                      # rounded zeros (label = 0)
  imp <- impNNetCoDa(x, dl = dl, label = 0, correction = "truncate",
                     arch = deepimp_arch_small(), epochs = 3L, iterations = 1L, seed = 1)
  d1 <- getImputed(imp)
  expect_s3_class(imp, "deepimp")
  expect_false(anyNA(d1))
  expect_true(all(d1$a > 0))
  expect_true(all(d1$a[1:8] <= dl[1] + 1e-6))
})

test_that("impNNetCoDa records coda/dl/correction in info", {
  skip_if_not_installed("torch"); skip_if_not_installed("luz")
  set.seed(2)
  n <- 40L
  x <- data.frame(a = runif(n, 5, 10), b = runif(n, 5, 10), c = runif(n, 5, 10))
  x$b[1:5] <- 0
  imp <- impNNetCoDa(x, dl = c(1, 1, 1), label = 0, arch = deepimp_arch_small(),
                     epochs = 2L, iterations = 1L, seed = 2)
  expect_true(imp$info$coda)
  expect_equal(imp$info$correction, "truncate")
  expect_equal(imp$info$dl, c(1, 1, 1))
})
