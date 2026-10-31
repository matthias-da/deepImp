# GLM offset for the network heads (backend-agnostic helpers).
#
# Each head is a GLM link plus an MLP correction: a linear model for regression
# heads and a multinomial logit for classification heads, fitted to convergence
# on the training rows of that fit. The backends add the link values to the
# output of an MLP whose last layer starts at zero, and the untrained state
# competes in restore-best, so the head nests the GLM. A small ridge penalty
# (`arch$glm_ridge`) guards against separation, rare levels and p close to n.

#' @importFrom nnet multinom
NULL

# Does this head get a GLM offset? Never for a single-class classification head
# (its softmax is constant anyway).
.use_glm_offset <- function(arch, task, output_dim) {
  isTRUE(arch$glm_offset) && (task == "regression" || output_dim >= 2L)
}

# Fit the GLM on the encoded predictors `x`. Exactly aliased columns (e.g. a
# full one-hot block next to the intercept) are dropped via a pivoted QR first;
# the ridge then handles near-collinearity and separation.
#   regression     : ridge least squares, intercept unpenalised; `y` numeric.
#   classification : nnet::multinom with decay = ridge; `y` holds 1-based class
#                    codes in 1..K. Classes absent from the training rows (a
#                    bootstrap resample can miss a rare level) get a link value
#                    log(1 / (n + 1)) below the row's smallest present logit.
# Returns list(keep, B, absent, gap): link = cbind(1, x[, keep]) %*% B, with K
# columns for classification (class 1 or the first present class as reference).
.glm_fit <- function(x, y, task, K = 1L, ridge = 1e-3) {
  x <- as.matrix(x)
  n <- nrow(x)
  q <- qr(cbind(1, x))
  kept <- q$pivot[seq_len(q$rank)]
  keep <- sort(kept[kept > 1L]) - 1L
  X1 <- cbind(1, x[, keep, drop = FALSE])
  p1 <- ncol(X1)

  if (task == "regression") {
    pen <- diag(c(0, rep(ridge, p1 - 1L)), nrow = p1)
    yv <- as.numeric(y)
    b <- tryCatch(solve(crossprod(X1) + pen, crossprod(X1, yv)),
                  error = function(e) qr.solve(crossprod(X1) + pen + diag(1e-8, p1),
                                               crossprod(X1, yv)))
    B <- matrix(as.numeric(b), ncol = 1L)
    B[!is.finite(B)] <- 0
    return(list(keep = keep, B = B, absent = integer(0), gap = 0))
  }

  codes <- as.integer(y)
  present <- sort(unique(codes))
  absent <- setdiff(seq_len(K), present)
  B <- matrix(0, nrow = p1, ncol = K)
  if (length(present) >= 2L) {
    yf <- factor(codes, levels = present)
    n_wts <- (p1 + 1L) * length(present) + 10L
    mn <- nnet::multinom(yf ~ X1 - 1, decay = ridge, maxit = 500L, trace = FALSE,
                         MaxNWts = max(1000L, n_wts))
    cf <- matrix(stats::coef(mn), ncol = p1)       # (K_present - 1) x p1
    B[, present[-1L]] <- t(cf)
  }
  B[!is.finite(B)] <- 0
  list(keep = keep, B = B, absent = absent, gap = log(n + 1))
}

# GLM link values for new rows: a one-column matrix (regression) or an n x K
# matrix of logits (classification).
.glm_link <- function(g, x) {
  x <- as.matrix(x)
  eta <- cbind(1, x[, g$keep, drop = FALSE]) %*% g$B
  if (length(g$absent)) {
    present <- setdiff(seq_len(ncol(eta)), g$absent)
    floor_ <- apply(eta[, present, drop = FALSE], 1L, min) - g$gap
    eta[, g$absent] <- floor_
  }
  unname(eta)
}
