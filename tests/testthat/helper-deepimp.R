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
