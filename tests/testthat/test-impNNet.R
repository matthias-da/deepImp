test_that("impNNet returns a deepimp object and imputes all NAs", {
  skip_if_not_installed("torch")
  skip_if_not_installed("luz")
  set.seed(123)
  n <- 60L
  dat <- data.frame(
    x1 = rnorm(n),
    x2 = rnorm(n)
  )
  dat$x1[c(3, 7, 11)] <- NA
  dat$x2[c(5, 9)] <- NA

  imp <- impNNet(dat, m = 1L, epochs = 2L, arch = deepimp_arch_small())
  expect_s3_class(imp, "deepimp")
  d1 <- getImputed(imp, m = 1L)
  expect_false(anyNA(d1))
})

test_that("impNNet errors if both arch and ... arch-scalars are given", {
  skip_if_not_installed("torch")
  set.seed(1)
  dat <- data.frame(x1 = rnorm(20), x2 = rnorm(20))
  dat$x1[3] <- NA
  expect_error(
    impNNet(dat, arch = deepimp_arch_small(), hidden = c(16, 8)),
    "both `arch` and"
  )
})

test_that("impNNet builds arch from ... when arch is NULL", {
  skip_if_not_installed("torch")
  skip_if_not_installed("luz")
  set.seed(7)
  n <- 40L
  dat <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
  dat$x1[c(2, 4)] <- NA
  imp <- impNNet(dat, m = 1L, epochs = 2L, hidden = c(16, 8))
  expect_s3_class(imp, "deepimp")
  expect_equal(imp$arch$hidden, c(16L, 8L))
})
