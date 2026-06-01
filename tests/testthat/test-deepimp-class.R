test_that("fake_deepimp builds a valid object and getImputed works", {
  fake <- fake_deepimp()
  expect_s3_class(fake, "deepimp")
  expect_true(is.list(fake$imputed))

  d1 <- getImputed(fake, m = 1L)
  expect_s3_class(d1, "data.frame")
  expect_equal(nrow(d1), nrow(fake$data))

  all_imp <- getImputed(fake, m = "all")
  expect_true(is.list(all_imp))
  expect_length(all_imp, length(fake$imputed))
})

test_that("as.data.frame.deepimp mirrors getImputed", {
  fake <- fake_deepimp()
  d1 <- as.data.frame(fake, m = 1L)
  expect_s3_class(d1, "data.frame")
  expect_equal(d1, getImputed(fake, m = 1L))
})

test_that("print.deepimp returns object invisibly and prints summary", {
  fake <- fake_deepimp()
  expect_output(print(fake), "deepimp")
  out <- withVisible(print(fake))
  expect_false(out$visible)
})
