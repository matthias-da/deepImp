# The backend-agnostic imputation engine: per-column primitive + chained loop.

#' @importFrom VIM kNN hotdeck
#' @importFrom robCompositions impKNNa
.initialize_na <- function(x, method = "knn") {
  switch(method,
    knn = , kNN = VIM::kNN(x, imp_var = FALSE),
    kNNa = , knna = tryCatch(
      as.data.frame(robCompositions::impKNNa(x)$xImp),
      error = function(e) {
        warning("kNNa initialisation (robCompositions::impKNNa) failed (",
                conditionMessage(e), "); falling back to knn.", call. = FALSE)
        VIM::kNN(x, imp_var = FALSE)
      }),
    hotdeck = VIM::hotdeck(x, imp_var = FALSE),
    mean = {
      for (j in seq_along(x)) {
        if (is.numeric(x[[j]])) {
          x[[j]][is.na(x[[j]])] <- mean(x[[j]], na.rm = TRUE)
        } else {
          lev <- names(which.max(table(x[[j]])))
          x[[j]][is.na(x[[j]])] <- lev
        }
      }
      x
    },
    stop("Unsupported initialize: ", method, call. = FALSE)
  )
}

# Write imputed values back into a column, preserving its type and factor levels.
.write_back <- function(col, miss, values) {
  if (is.factor(col)) {
    col[miss] <- factor(values, levels = levels(col))
  } else if (is.integer(col)) {
    col[miss] <- as.integer(round(as.numeric(values)))
  } else if (is.numeric(col)) {
    col[miss] <- as.numeric(values)
  } else {
    col[miss] <- values
  }
  col
}

# Impute the missing cells of column j given the currently filled frame.
.impute_column <- function(j, filled, na_mask, types, arch, backend, control,
                           normalize, validation_split) {
  task <- .task_for_type(types[j])
  pred_types <- types[-j]
  obs  <- !na_mask[, j]
  miss <-  na_mask[, j]

  fit_enc  <- .predictor_matrix(filled[obs, -j, drop = FALSE], pred_types, scale = normalize)
  test_enc <- .predictor_matrix(filled[miss, -j, drop = FALSE], pred_types,
                                encoding = fit_enc$encoding)

  # Seeded validation split (engine-side, so held-out rows are backend-independent).
  n_obs <- nrow(fit_enc$x)
  n_val <- max(1L, floor(validation_split * n_obs))
  if (n_val >= n_obs) n_val <- max(1L, n_obs - 1L)
  val_idx <- sample.int(n_obs, n_val)
  tr_idx  <- setdiff(seq_len(n_obs), val_idx)
  if (length(tr_idx) < 1L) { tr_idx <- seq_len(n_obs); val_idx <- seq_len(n_obs) }

  y <- filled[obs, j]
  if (task == "regression") {
    yv <- as.numeric(y); yc <- mean(yv); ys <- stats::sd(yv)
    if (!is.finite(ys) || ys == 0) ys <- 1
    ystd <- (yv - yc) / ys
    model <- build_model(backend, task, ncol(fit_enc$x), 1L, arch)
    fit <- fit_model(backend, model,
                     fit_enc$x[tr_idx, , drop = FALSE], ystd[tr_idx],
                     fit_enc$x[val_idx, , drop = FALSE], ystd[val_idx], control)
    pred <- predict_model(backend, fit$model, test_enc$x) * ys + yc
    if (types[j] == "count") pred <- pmax(0, round(pred))
    pred
  } else {
    yf <- factor(y); levs <- levels(yf); codes <- as.integer(yf)
    model <- build_model(backend, task, ncol(fit_enc$x), length(levs), arch)
    fit <- fit_model(backend, model,
                     fit_enc$x[tr_idx, , drop = FALSE], codes[tr_idx],
                     fit_enc$x[val_idx, , drop = FALSE], codes[val_idx], control)
    probs <- predict_model(backend, fit$model, test_enc$x)
    levs[max.col(probs, ties.method = "first")]
  }
}

# Chained imputation: m replicates -> iterations -> columns -> convergence check.
.run_chain <- function(x, types, arch, backend, initialize = "knn", iterations = 3L,
                       m = 1L, eps = 1e-2, normalize = TRUE, control = list(),
                       validation_split = 0.2, seed = NULL, verbose = FALSE,
                       column_fn = .impute_column, column_args = list()) {
  na_mask <- is.na(x)
  start <- .initialize_na(x, initialize)
  cols <- which(colSums(na_mask) > 0)

  imputed <- vector("list", m)
  conv    <- vector("list", m)
  seeds   <- integer(m)

  for (rep_i in seq_len(m)) {
    seeds[rep_i] <- if (is.null(seed)) NA_integer_ else as.integer(seed + rep_i - 1L)
    if (!is.na(seeds[rep_i])) {
      set.seed(seeds[rep_i])
      # Seed torch only for the torch backend, and only when its native runtime is
      # present. The torch R package is a hard dependency (so requireNamespace is
      # always TRUE), but on CRAN/win-builder libtorch ("Lantern") is not installed;
      # seeding it on the keras path then errors ("Lantern is not loaded").
      if (inherits(backend, "deepimp_backend_torch") && torch::torch_is_installed()) {
        torch::torch_manual_seed(seeds[rep_i])
      }
    }
    filled <- start
    changes <- numeric(0)
    for (it in seq_len(iterations)) {
      old <- filled
      for (j in cols) {
        if (verbose) {
          message(sprintf("  imputation %d/%d  iter %d/%d  var %d/%d",
                          rep_i, m, it, iterations, j, ncol(x)))
        }
        vals <- do.call(column_fn,
                        c(list(j, filled, na_mask, types, arch, backend, control,
                               normalize, validation_split), column_args))
        filled[[j]] <- .write_back(filled[[j]], na_mask[, j], vals)
      }
      ch <- .convergence(old, filled, na_mask, types)
      changes <- c(changes, ch)
      if (ch < eps) break
    }
    imputed[[rep_i]] <- filled
    conv[[rep_i]] <- changes
  }

  list(imputed = imputed, convergence = conv, seeds = seeds, na_mask = na_mask)
}
