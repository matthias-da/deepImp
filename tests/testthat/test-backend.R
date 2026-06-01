test_that("seam generics dispatch on the backend tag", {
  expect_silent(.backend("torch"))
  expect_s3_class(.backend("torch"), "deepimp_backend_torch")
  expect_error(.backend("nope"), "Unknown backend")
})

test_that("a backend with no build_model method errors with guidance", {
  # An unknown backend tag has no build_model method -> falls back to
  # build_model.default, which raises an actionable error. (torch and keras are
  # both implemented; this covers the not-available path generically.)
  fake <- structure(list(name = "madeup"),
                    class = c("deepimp_backend_madeup", "deepimp_backend"))
  expect_error(build_model(fake, "regression", 3L, 1L, deepimp_arch()),
               "not available")
})
