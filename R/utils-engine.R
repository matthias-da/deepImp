# Backend-agnostic helpers for the imputation engine. Not exported.

.task_for_type <- function(type) {
  if (type %in% c("numeric", "count", "mixed", "compositional")) "regression"
  else if (type %in% c("binary", "nominal")) "classification"
  else stop("Unsupported variable type: ", type, call. = FALSE)
}

.fit_scaler <- function(m) {
  center <- colMeans(m, na.rm = TRUE)
  scale <- apply(m, 2, stats::sd, na.rm = TRUE)
  scale[!is.finite(scale) | scale == 0] <- 1   # guard constant columns
  list(center = center, scale = scale)
}

.apply_scaler <- function(m, scaler) {
  sweep(sweep(m, 2, scaler$center, "-"), 2, scaler$scale, "/")
}

# Build a numeric model matrix from predictor columns: numeric/count/mixed are
# standardised, factor/binary/nominal are one-hot encoded. `encoding` lets a test
# set reuse the encoding fitted on the training set.
.predictor_matrix <- function(df, types, encoding = NULL, scale = TRUE) {
  num_cols <- which(types %in% c("numeric", "count", "mixed", "compositional"))
  fac_cols <- which(types %in% c("binary", "nominal"))

  if (is.null(encoding)) {
    scaler <- NULL
    if (length(num_cols)) {
      scaler <- if (scale) .fit_scaler(as.matrix(df[, num_cols, drop = FALSE]))
                else list(center = rep(0, length(num_cols)), scale = rep(1, length(num_cols)))
    }
    encoding <- list(
      num_cols = num_cols, fac_cols = fac_cols, scaler = scaler,
      levels = lapply(fac_cols, function(j) levels(as.factor(df[[j]])))
    )
    names(encoding$levels) <- names(df)[fac_cols]
  }

  parts <- list()
  if (length(encoding$num_cols)) {
    parts[["num"]] <- .apply_scaler(as.matrix(df[, encoding$num_cols, drop = FALSE]),
                                    encoding$scaler)
  }
  if (length(encoding$fac_cols)) {
    for (k in seq_along(encoding$fac_cols)) {
      j <- encoding$fac_cols[k]
      f <- factor(df[[j]], levels = encoding$levels[[k]])
      mm <- stats::model.matrix(~ f - 1)
      colnames(mm) <- paste0(names(df)[j], "_", encoding$levels[[k]])
      parts[[paste0("fac", k)]] <- mm
    }
  }
  x <- if (length(parts)) do.call(cbind, parts) else matrix(numeric(0), nrow = nrow(df))
  storage.mode(x) <- "double"
  list(x = x, encoding = encoding)
}

# Scale-free convergence: standardised mean change on numeric cells, mismatch
# rate on categorical cells, averaged over imputed cells.
.convergence <- function(old, new, na_mask, types) {
  n_imp <- sum(na_mask)
  if (n_imp == 0) return(0)
  total <- 0
  for (j in seq_along(types)) {
    miss <- na_mask[, j]
    if (!any(miss)) next
    if (.task_for_type(types[j]) == "regression") {
      ov <- as.numeric(old[[j]]); nv <- as.numeric(new[[j]])
      s <- stats::sd(ov, na.rm = TRUE); if (!is.finite(s) || s == 0) s <- 1
      total <- total + sum(abs(ov[miss] - nv[miss]) / s)
    } else {
      total <- total + sum(as.character(old[[j]])[miss] != as.character(new[[j]])[miss])
    }
  }
  total / n_imp
}

# Rows of the observed part used to train and to validate one fit.
#   bootstrap = FALSE : a random hold-out of `validation_split` (the 1.1.0 split).
#   bootstrap = TRUE  : a bootstrap resample trains, the out-of-bag rows validate
#                       (early stopping, restore-best, and the normal-draw sigma).
.split_rows <- function(n_obs, validation_split = 0.2, bootstrap = FALSE) {
  if (bootstrap) {
    tr <- sample.int(n_obs, n_obs, replace = TRUE)
    va <- setdiff(seq_len(n_obs), tr)
    if (!length(va)) va <- unique(tr)        # only when n_obs is tiny
    return(list(train = tr, valid = va))
  }
  n_val <- max(1L, floor(validation_split * n_obs))
  if (n_val >= n_obs) n_val <- max(1L, n_obs - 1L)
  va <- sample.int(n_obs, n_val)
  tr <- setdiff(seq_len(n_obs), va)
  if (length(tr) < 1L) { tr <- seq_len(n_obs); va <- seq_len(n_obs) }
  list(train = tr, valid = va)
}

# Predictive mean matching: for each missing cell, take the `k` observed rows
# whose predictions are closest to its prediction and copy the observed value of
# one of them, chosen at random. Searches a window of 2k candidates around each
# prediction in the sorted observed predictions, so it costs O((n_mis + n_obs)
# log n_obs) instead of an n_mis x n_obs distance matrix.
.pmm_draw <- function(pred_mis, pred_obs, y_obs, k = 5L) {
  n_obs <- length(pred_obs); n_mis <- length(pred_mis)
  k <- max(1L, min(as.integer(k), n_obs))
  o <- order(pred_obs); ps <- pred_obs[o]; ys <- y_obs[o]
  pos <- findInterval(pred_mis, ps)
  cand <- outer(pos, seq.int(-k + 1L, k), "+")
  valid <- cand >= 1L & cand <= n_obs
  cand <- pmin(pmax(cand, 1L), n_obs)
  d <- abs(matrix(ps[cand], nrow = n_mis) - pred_mis)
  d[!valid] <- Inf
  ranks <- matrix(t(apply(d, 1L, order)), nrow = n_mis)
  pick <- sample.int(k, n_mis, replace = TRUE)
  ys[cand[cbind(seq_len(n_mis), ranks[cbind(seq_len(n_mis), pick)])]]
}

# Sample one class index per row from an n x K probability matrix.
.sample_classes <- function(probs) {
  probs <- as.matrix(probs); K <- ncol(probs)
  if (K == 1L) return(rep(1L, nrow(probs)))
  cp <- probs %*% (upper.tri(diag(K), diag = TRUE) * 1)   # row-wise cumulative sums
  u <- stats::runif(nrow(probs)) * cp[, K]
  pmin(rowSums(u > cp) + 1L, K)
}

# Chain statistics of the imputed cells (mice-style trace): mean and SD per
# incomplete variable. Factors enter by their integer codes, as in mice.
.chain_stats <- function(filled, na_mask, cols) {
  vals <- lapply(cols, function(j) as.numeric(filled[[j]][na_mask[, j]]))
  list(mean = vapply(vals, mean, numeric(1)),
       sd   = vapply(vals, function(v) if (length(v) > 1L) stats::sd(v) else NA_real_,
                     numeric(1)))
}
