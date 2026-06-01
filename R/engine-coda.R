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
