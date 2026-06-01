test_that("torch and keras both impute and beat mean-imputation on a known DGP", {
  skip_if_not_installed("torch")
  skip_if_not_installed("luz")
  skip_if_not_installed("keras3")
  set.seed(11)
  n <- 120L
  df <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
  df$x3 <- df$x1 - df$x2 + rnorm(n, sd = 0.1)   # strong linear signal
  truth <- df$x3
  miss <- sample(n, 24L)
  df$x3[miss] <- NA

  rmse <- function(backend) {
    imp <- impNNet(df, backend = backend, arch = deepimp_arch_small(),
                   epochs = 60L, iterations = 1L, seed = 1)
    d <- getImputed(imp)
    expect_false(anyNA(d))
    sqrt(mean((d$x3[miss] - truth[miss])^2))
  }
  rmse_mean <- sqrt(mean((mean(df$x3, na.rm = TRUE) - truth[miss])^2))

  rmse_torch <- rmse("torch")
  rmse_keras <- rmse("keras")
  expect_lt(rmse_torch, rmse_mean)
  expect_lt(rmse_keras, rmse_mean)
  # comparable, NOT equal (different init/RNG/optimiser internals)
})
