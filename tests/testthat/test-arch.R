test_that("deepimp_arch returns a validated object with defaults", {
  a <- deepimp_arch()
  expect_s3_class(a, "deepimp_arch")
  expect_equal(a$hidden, c(256, 128, 64))
  expect_equal(a$dropout, 0.1)
  expect_equal(a$activation, "relu")
  expect_true(a$batchnorm)
  expect_equal(a$optimizer, "adam")
  expect_equal(a$learning_rate, 1e-3)
})

test_that("deepimp_arch validates its arguments", {
  expect_error(deepimp_arch(hidden = c(10, -1)), "positive")
  expect_error(deepimp_arch(hidden = c(10, 2.5)), "integer")
  expect_error(deepimp_arch(hidden = integer(0)), "at least one")
  expect_error(deepimp_arch(dropout = 1), "\\[0, 1\\)")
  expect_error(deepimp_arch(dropout = -0.1), "\\[0, 1\\)")
  expect_error(deepimp_arch(activation = "swish"), "activation")
  expect_error(deepimp_arch(optimizer = "lbfgs"), "optimizer")
  expect_error(deepimp_arch(learning_rate = 0), "positive")
})

test_that("deepimp_arch_small is a lighter preset", {
  s <- deepimp_arch_small()
  expect_s3_class(s, "deepimp_arch")
  expect_true(sum(s$hidden) < sum(deepimp_arch()$hidden))
})

test_that("print.deepimp_arch is invisible and returns its input", {
  a <- deepimp_arch()
  expect_output(print(a), "deepimp_arch")
  expect_identical(withVisible(print(a))$value, a)
})
