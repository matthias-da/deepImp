test_that("seam generics dispatch on the backend tag", {
  expect_silent(.backend("torch"))
  expect_s3_class(.backend("torch"), "deepimp_backend_torch")
  expect_error(.backend("nope"), "Unknown backend")
})

test_that("unavailable backend errors with guidance", {
  fake <- structure(list(), class = c("deepimp_backend_keras", "deepimp_backend"))
  # keras backend is not implemented in this plan -> build_model has no method
  expect_error(build_model(fake, "regression", 3L, 1L, deepimp_arch()))
})
