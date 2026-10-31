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

test_that("getImputed(m = 'long') stacks the data in the mice long format", {
  fake <- fake_deepimp()
  long <- getImputed(fake, m = "long")
  n <- nrow(fake$data)
  expect_equal(nrow(long), 3L * n)
  expect_identical(names(long)[1:2], c(".imp", ".id"))
  expect_equal(long$.imp, rep(0:2, each = n))
  expect_equal(long$.id, rep(seq_len(n), 3L))
  expect_equal(long[long$.imp == 0, -(1:2)], fake$data, ignore_attr = TRUE)
  expect_equal(long[long$.imp == 2, -(1:2)], fake$imputed[[2]], ignore_attr = TRUE)
  expect_error(getImputed(fake, m = "wide"), "'long'")
})

test_that("as_mids() converts, and warns on point-mode replicates", {
  skip_if_not_installed("mice")
  fake <- fake_deepimp()
  expect_warning(md <- as_mids(fake), "point-mode")
  expect_s3_class(md, "mids")
  expect_equal(md$m, 2L)
  expect_equal(mice::complete(md, 2), fake$imputed[[2]], ignore_attr = TRUE)

  fake$info$draw <- "pmm"
  expect_no_warning(md2 <- as_mids(fake))
  fit <- with(md2, lm(a ~ b))
  expect_s3_class(mice::pool(fit), "mipo")

  one <- fake; one$imputed <- one$imputed[1]
  expect_warning(as_mids(one), "m >= 2")
  expect_error(as_mids(list()), "deepimp object")
})

test_that("plot(type = 'trace') draws the chain trace", {
  fake <- fake_deepimp()
  expect_message(plot(fake, type = "trace"), "No chain trace")
  fake$info$trace <- list(
    mean = array(rnorm(2 * 3 * 2), c(2, 3, 2), dimnames = list(c("a", "b"), NULL, NULL)),
    sd   = array(runif(2 * 3 * 2), c(2, 3, 2), dimnames = list(c("a", "b"), NULL, NULL)))
  pdf(NULL); on.exit(dev.off())
  expect_invisible(plot(fake, type = "trace"))
  expect_invisible(plot(fake, type = "trace", vars = "a"))
  expect_error(plot(fake, type = "trace", vars = "zz"), "No trace for: zz")
})
