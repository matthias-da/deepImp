test_that(".apply_correction truncates at phi", {
  pred <- c(1, -2, 5)
  phi  <- c(0, 0, 0)
  expect_equal(.apply_correction(pred, phi, "truncate", s = 1), c(0, -2, 0))
})

test_that(".apply_correction 'none' passes predictions through", {
  pred <- c(1, -2, 5)
  phi  <- c(0, 0, 0)
  expect_equal(.apply_correction(pred, phi, "none", s = 1), pred)
})

test_that(".apply_correction 'expectation' pulls below phi and is finite", {
  # pred=2, phi=0, s=1 -> alpha=-2, lambda=dnorm(-2)/pnorm(-2), out=2 - 1*lambda
  lambda <- dnorm(-2) / pnorm(-2)
  expect_equal(.apply_correction(2, 0, "expectation", s = 1), 2 - lambda)
  # deep tail must stay finite and <= phi (no Inf/NaN)
  out <- .apply_correction(50, 0, "expectation", s = 1)
  expect_true(is.finite(out))
  expect_lte(out, 0)
})

test_that(".apply_correction 'expectation' falls back to truncate when s invalid", {
  expect_equal(.apply_correction(c(2, 5), c(0, 0), "expectation", s = 0),
               c(0, 0))   # s<=0 -> truncate
  expect_equal(.apply_correction(c(2, 5), c(0, 0), "expectation", s = NA_real_),
               c(0, 0))
})

test_that("pivotCoord round-trips a known composition (closure)", {
  skip_if_not_installed("robCompositions")
  comp <- data.frame(a = c(1, 2, 3), b = c(2, 1, 4), c = c(3, 3, 1))
  back <- robCompositions::pivotCoordInv(robCompositions::pivotCoord(comp))
  # compare closed (row-normalised) compositions; pivotCoordInv renames columns,
  # so drop dimnames before comparing values.
  norm <- function(m) { z <- as.matrix(m) / rowSums(as.matrix(m)); dimnames(z) <- NULL; z }
  expect_equal(norm(back), norm(comp), tolerance = 1e-8)
})

test_that(".impute_column_coda fills rounded zeros below the detection limit", {
  skip_if_not_installed("torch"); skip_if_not_installed("luz")
  skip_if_not_installed("robCompositions")
  set.seed(1); torch::torch_manual_seed(1)
  n <- 50L
  comp <- data.frame(a = runif(n, 5, 10), b = runif(n, 5, 10), c = runif(n, 5, 10))
  dl <- c(1, 1, 1)
  comp$a[1:6] <- dl[1] * 0.5
  na_mask <- is.na(comp)
  na_mask[1:6, 1] <- TRUE
  filled <- comp; filled$a[1:6] <- dl[1] * 0.5
  control <- list(epochs = 3L, patience = 2L, batch_size = 16L, verbose = FALSE)
  vals <- .impute_column_coda(1L, filled, na_mask, rep("numeric", 3),
                              arch = deepimp_arch_small(), backend = .backend("torch"),
                              control = control, normalize = TRUE, validation_split = 0.2,
                              dl = dl, coda = TRUE, correction = "truncate")
  expect_length(vals, 6L)
  expect_true(all(is.finite(vals)))
  expect_true(all(vals > 0))
  expect_true(all(vals <= dl[1] + 1e-6))
})
