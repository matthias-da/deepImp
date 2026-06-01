test_that("guessType returns a stable contract", {
  df <- data.frame(num = rnorm(50), fac = factor(sample(letters[1:3], 50, TRUE)))
  g <- guessType(df)
  expect_named(g, c("indices", "type"))
  expect_true(is.matrix(g$indices))
  expect_equal(length(g$type), ncol(df))
  expect_equal(colnames(g$indices), colnames(df))
})

test_that("guessType identifies the basic types", {
  set.seed(1)
  df <- data.frame(
    numeric = rnorm(100),
    binary  = sample(0:1, 100, TRUE),
    nominal = factor(sample(letters[1:4], 100, TRUE)),
    count   = rpois(100, 3)
  )
  ty <- guessType(df)$type
  names(ty) <- colnames(df)
  expect_equal(unname(ty["numeric"]), "numeric")
  expect_equal(unname(ty["binary"]), "binary")
  expect_equal(unname(ty["nominal"]), "nominal")
})

test_that("guessType detects a semi-continuous (mixed) variable", {
  set.seed(2)
  x <- rlnorm(200)
  x[sample(200, 80)] <- 0          # ~40% structural zeros => semi-continuous
  ty <- guessType(data.frame(x = x))$type
  expect_equal(ty, "mixed")
})
