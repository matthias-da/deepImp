fake_deepimp <- function() {
  set.seed(1)
  n <- 20L
  d <- data.frame(
    a = c(NA, rnorm(n - 1)),
    b = factor(sample(c("x", "y"), n, replace = TRUE))
  )
  d$b[2] <- NA
  imputed <- list(
    within(d, { a[is.na(a)] <- 0; b[is.na(b)] <- "x" }),
    within(d, { a[is.na(a)] <- 1; b[is.na(b)] <- "y" })
  )
  new_deepimp(
    data = d,
    imputed = imputed,
    arch = deepimp_arch_small(),
    info = list(loss = c(1.0, 0.5), epochs = 2L, backend = "fake")
  )
}

# --- backend-availability skips -------------------------------------------------
# skip_if_not_installed() only checks that the *R package* is present. On CRAN /
# win-builder the torch and keras3 R packages are installed, but their native
# runtimes are not: install_torch() (libtorch / "Lantern") and
# keras3::install_keras() (Python keras / TensorFlow) are never run there, so any
# backend op errors ("Lantern is not loaded" / "No module named 'keras'"). These
# helpers skip unless the *runtime* backend is actually usable.

skip_if_no_torch <- function() {
  testthat::skip_if_not_installed("torch")
  testthat::skip_if_not_installed("luz")
  ok <- tryCatch(isTRUE(torch::torch_is_installed()), error = function(e) FALSE)
  if (!ok) testthat::skip("libtorch (Lantern) runtime not installed")
}

skip_if_no_keras <- function() {
  testthat::skip_if_not_installed("keras3")
  # keras3 (R) installed != Python keras / TensorFlow available. Probe the backend
  # exactly as build_model() does (build an empty Sequential model); skip if it
  # errors, e.g. "No module named 'keras'" on CRAN / win-builder.
  ok <- tryCatch({ keras3::keras_model_sequential(); TRUE },
                 error = function(e) FALSE)
  if (!ok) testthat::skip("Python keras backend (keras / TensorFlow) not available")
}
