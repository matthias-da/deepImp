# CoDa-specific engine pieces: rounded-zero correction + per-column primitive.

# Censor a predicted (pivot) coordinate at the detection-limit coordinate phi.
#   "truncate"    : pred > phi -> phi              (Templ 2021, Algorithm 2)
#   "none"        : pred unchanged                 (ablation)
#   "expectation" : E[Z | Z < phi] for Z ~ N(pred, s^2)  (opt-in; beyond Templ 2021)
# `phi` is a scalar or a vector recycled to length(pred); `s` is a positive scalar.
.apply_correction <- function(pred, phi, correction = c("truncate", "expectation", "none"),
                              s = NA_real_) {
  correction <- match.arg(correction)
  phi <- rep_len(phi, length(pred))
  if (correction == "none") return(pred)
  if (correction == "truncate" || !is.finite(s) || s <= 0) {
    return(ifelse(pred > phi, phi, pred))
  }
  # expectation: left-truncated normal mean, computed with a log-space inverse Mills ratio.
  alpha  <- (phi - pred) / s
  lambda <- exp(stats::dnorm(alpha, log = TRUE) - stats::pnorm(alpha, log.p = TRUE))
  out <- pred - s * lambda
  out <- pmin(out, phi)                       # E[Z|Z<phi] < phi always
  bad <- !is.finite(out)
  out[bad] <- phi[bad]                        # tail underflow -> degrade to truncation
  out
}

#' @importFrom robCompositions pivotCoord pivotCoordInv adjustImputed
#' @importFrom stats sd
NULL

# CoDa per-column primitive: pivot part j to the front, regress the first pivot
# coordinate on the rest via the backend seam, censor at the detection limit, and
# back-transform. Returns the imputed (raw-scale) values for the missing cells of
# column j. Shares the base argument tuple with .impute_column; CoDa args
# (dl, coda, correction) arrive via column_args.
.impute_column_coda <- function(j, filled, na_mask, types, arch, backend, control,
                                normalize, validation_split,
                                dl, coda = TRUE,
                                correction = c("truncate", "expectation", "none")) {
  correction <- match.arg(correction)
  p <- ncol(filled)
  ord <- c(j, setdiff(seq_len(p), j))
  comp <- as.data.frame(filled[, ord, drop = FALSE])
  mask_ord <- na_mask[, ord, drop = FALSE]
  tiny <- 2 * .Machine$double.eps
  comp[comp < tiny] <- tiny

  obs <- !na_mask[, j]
  mis <-  na_mask[, j]
  n <- nrow(comp)

  if (coda) {
    Z <- as.matrix(robCompositions::pivotCoord(comp))
    phi <- robCompositions::pivotCoord(cbind(dl[j], comp[, -1, drop = FALSE]))[, 1]
    if (any(!is.finite(phi))) {
      stop("non-finite detection-limit coordinate (phi); check `dl`.", call. = FALSE)
    }
  } else {
    Z <- as.matrix(comp)
    phi <- rep(dl[j], n)
  }
  if (any(!is.finite(Z))) stop("non-finite log-ratio coordinates; check the data.", call. = FALSE)

  z1 <- Z[, 1]
  X  <- Z[, -1, drop = FALSE]

  # Predictor scaling (governed by `normalize`) + internal target standardisation,
  # mirroring .impute_column so predictions return on the z1 scale.
  enc <- .predictor_matrix(as.data.frame(X), rep("numeric", ncol(X)), scale = normalize)
  Xobs <- enc$x[obs, , drop = FALSE]
  Xmis <- .predictor_matrix(as.data.frame(X[mis, , drop = FALSE]),
                            rep("numeric", ncol(X)), encoding = enc$encoding)$x

  yobs <- z1[obs]
  yc <- mean(yobs); ys <- stats::sd(yobs); if (!is.finite(ys) || ys == 0) ys <- 1
  ystd <- (yobs - yc) / ys

  n_obs <- nrow(Xobs)
  n_val <- max(1L, floor(validation_split * n_obs))
  if (n_val >= n_obs) n_val <- max(1L, n_obs - 1L)
  val_idx <- sample.int(n_obs, n_val)
  tr_idx  <- setdiff(seq_len(n_obs), val_idx)
  if (length(tr_idx) < 1L) { tr_idx <- seq_len(n_obs); val_idx <- seq_len(n_obs) }

  model <- build_model(backend, "regression", ncol(Xobs), 1L, arch)
  fit <- fit_model(backend, model,
                   Xobs[tr_idx, , drop = FALSE], ystd[tr_idx],
                   Xobs[val_idx, , drop = FALSE], ystd[val_idx], control)
  pred <- predict_model(backend, fit$model, Xmis) * ys + yc

  s <- NA_real_
  if (correction == "expectation") {
    resid <- yobs - (predict_model(backend, fit$model, Xobs) * ys + yc)
    s <- stats::sd(resid)
  }
  z1_imp <- .apply_correction(pred, phi[mis], correction, s = s)

  Z[mis, 1] <- z1_imp
  if (coda) {
    back <- robCompositions::pivotCoordInv(Z)
    back <- robCompositions::adjustImputed(back, comp, mask_ord)
    out <- as.matrix(back)[mis, 1]
  } else {
    out <- pmax(Z[mis, 1], tiny)
  }
  as.numeric(out)
}
