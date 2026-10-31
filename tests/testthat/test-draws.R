test_that(".split_rows: hold-out split and bootstrap with out-of-bag validation", {
  set.seed(1)
  sp <- .split_rows(50L, 0.2)
  expect_length(sp$valid, 10L)
  expect_length(intersect(sp$train, sp$valid), 0L)
  bs <- .split_rows(50L, bootstrap = TRUE)
  expect_length(bs$train, 50L)
  expect_true(anyDuplicated(bs$train) > 0L)
  expect_setequal(bs$valid, setdiff(1:50, bs$train))
  expect_length(.split_rows(1L, bootstrap = TRUE)$valid, 1L)
})

test_that(".pmm_draw copies observed values from the k nearest predictions", {
  set.seed(2)
  po <- sort(runif(200)); yo <- 1000 + seq_along(po)     # one distinct value per donor
  pm <- c(0.001, 0.5, 0.999, 2)
  out <- replicate(200, .pmm_draw(pm, po, yo, k = 5L))
  expect_true(all(out %in% yo))
  for (i in seq_along(pm)) {
    nearest <- yo[order(abs(po - pm[i]))[1:5]]
    expect_true(all(out[i, ] %in% nearest))
  }
  # all k donors get picked
  expect_length(unique(out[2, ]), 5L)
  # k larger than the number of donors
  expect_true(all(.pmm_draw(c(0.3, 0.7), c(0.2, 0.8), c(1, 2), k = 5L) %in% c(1, 2)))
  # agrees with a brute-force window on random input
  set.seed(3)
  po <- rnorm(300); pm <- rnorm(40)
  d <- .pmm_draw(pm, po, seq_along(po), k = 3L)
  for (i in seq_along(pm)) expect_true(d[i] %in% order(abs(po - pm[i]))[1:3])
})

test_that(".sample_classes follows the probabilities", {
  set.seed(4)
  p <- matrix(c(0.7, 0.2, 0.1), nrow = 20000, ncol = 3, byrow = TRUE)
  s <- .sample_classes(p)
  expect_equal(as.numeric(prop.table(table(factor(s, 1:3)))), c(0.7, 0.2, 0.1),
               tolerance = 0.02)
  expect_equal(.sample_classes(matrix(1, 3, 1)), rep(1L, 3))
  expect_equal(.sample_classes(rbind(c(0, 1), c(1, 0))), c(2L, 1L))
})

mixed_frame <- function(n = 160L, seed = 10) {
  set.seed(seed)
  x1 <- rnorm(n); x2 <- rnorm(n)
  d <- data.frame(
    x1 = x1, x2 = x2,
    y  = 1 + x1 - x2 + rnorm(n, sd = 0.5),
    cnt = rpois(n, exp(0.5 + 0.3 * x1)),
    g  = factor(ifelse(x1 + rnorm(n) > 0, "a", "b"))
  )
  d$y[sample(n, 30)] <- NA
  d$cnt[sample(n, 20)] <- NA
  d$g[sample(n, 25)] <- NA
  d
}

test_that("draw = 'pmm' imputes observed values, runs fixed sweeps, keeps a trace", {
  skip_if_no_torch()
  d <- mixed_frame()
  arch <- deepimp_arch(hidden = c(16, 8), dropout = 0)
  imp <- impNNet(d, arch = arch, m = 2L, draw = "pmm", iterations = 2L,
                 epochs = 5L, patience = 2L, seed = 1)
  expect_identical(imp$info$draw, "pmm")
  expect_true(imp$info$bootstrap)
  for (r in 1:2) {
    di <- getImputed(imp, r)
    expect_false(anyNA(di))
    expect_true(all(di$y[is.na(d$y)] %in% d$y[!is.na(d$y)]))
    expect_true(all(di$cnt[is.na(d$cnt)] %in% d$cnt[!is.na(d$cnt)]))
    expect_true(is.integer(di$cnt))
    expect_identical(levels(di$g), levels(d$g))
  }
  # fixed number of sweeps in draw mode (no eps stop)
  expect_true(all(lengths(imp$info$convergence) == 2L))
  tr <- imp$info$trace
  expect_equal(dim(tr$mean), c(3L, 2L, 2L))
  expect_identical(dimnames(tr$mean)[[1]], c("y", "cnt", "g"))
  expect_false(anyNA(tr$mean))
  # replicates differ
  expect_false(identical(getImputed(imp, 1)$y, getImputed(imp, 2)$y))
})

test_that("draw mode is exactly reproducible with a seed and dropout = 0", {
  skip_if_no_torch()
  d <- mixed_frame(n = 80L)
  arch <- deepimp_arch(hidden = c(8), dropout = 0)
  run <- function(draw) getImputed(impNNet(d, arch = arch, m = 1L, draw = draw,
                                           iterations = 1L, epochs = 3L, seed = 9))
  expect_identical(run("pmm"), run("pmm"))
  expect_identical(run("normal"), run("normal"))
})

test_that("draw = 'normal' adds noise with the held-out residual SD", {
  skip_if_no_torch()
  d <- mixed_frame()
  arch <- deepimp_arch(hidden = c(16), dropout = 0)
  # y is the first incomplete column, so with the same seed both runs fit the same
  # model for it; the difference of the imputations is the added noise.
  pt <- getImputed(impNNet(d, arch = arch, iterations = 1L, epochs = 20L, seed = 2))
  nd <- getImputed(impNNet(d, arch = arch, iterations = 1L, epochs = 20L, seed = 2,
                           draw = "normal", bootstrap = FALSE))
  miss <- is.na(d$y)
  noise <- nd$y[miss] - pt$y[miss]
  expect_gt(sd(noise), 0.3)                      # true residual SD 0.5
  expect_lt(sd(noise), 1.0)
  expect_true(all(nd$cnt >= 0) && all(nd$cnt == round(nd$cnt)))
})

test_that("draw mode keeps class shares of an imbalanced factor (no argmax collapse)", {
  skip_if_no_torch()
  set.seed(11)
  n <- 300L
  d <- data.frame(x1 = rnorm(n), x2 = rnorm(n),
                  g = factor(sample(c("rare", "common"), n, TRUE, prob = c(0.15, 0.85))))
  miss <- sample(n, 100L)
  d$g[miss] <- NA
  arch <- deepimp_arch(hidden = c(8), dropout = 0)
  pt <- getImputed(impNNet(d, arch = arch, iterations = 1L, epochs = 10L, seed = 3))
  dr <- getImputed(impNNet(d, arch = arch, iterations = 1L, epochs = 10L, seed = 3,
                           draw = "pmm"))
  share <- function(v) mean(v[miss] == "rare")
  expect_lt(share(pt$g), 0.05)                   # argmax: almost never the rare class
  expect_gt(share(dr$g), 0.05)                   # draws: near the observed 15 %
  expect_lt(share(dr$g), 0.30)
})

test_that("impNNet validates the draw arguments", {
  d <- data.frame(x1 = c(NA, rnorm(9)), x2 = rnorm(10))
  expect_error(impNNet(d, draw = "foo"), "should be one of")
  expect_error(impNNet(d, draw = "pmm", pmm_k = 0), "pmm_k")
  expect_error(impNNet(d, draw = "pmm", bootstrap = NA), "bootstrap")
})

test_that("point mode with an offset head still stops on eps and returns a trace", {
  skip_if_no_torch()
  d <- mixed_frame(n = 80L)
  imp <- impNNet(d, arch = deepimp_arch(hidden = c(8), dropout = 0), iterations = 4L,
                 eps = Inf, epochs = 3L, seed = 1)
  expect_identical(imp$info$draw, "none")
  expect_length(imp$info$convergence[[1]], 1L)          # eps = Inf stops after sweep 1
  expect_true(all(is.na(imp$info$trace$mean[, 2:4, 1])))
})
