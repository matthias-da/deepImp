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
