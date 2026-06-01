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
