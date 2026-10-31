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
#   draw = "none"   : conditional mean (regression) / most probable class.
#   draw = "pmm"    : predictive mean matching on the regression heads.
#   draw = "normal" : prediction + N(0, sigma^2), sigma from held-out residuals.
#   Any draw mode samples the classification heads from their softmax.
#   bootstrap = TRUE: fit on a bootstrap resample, validate on out-of-bag rows.
.impute_column <- function(j, filled, na_mask, types, arch, backend, control,
                           normalize, validation_split, draw = "none",
                           bootstrap = FALSE, pmm_k = 5L) {
  task <- .task_for_type(types[j])
  pred_types <- types[-j]
  obs  <- !na_mask[, j]
  miss <-  na_mask[, j]

  fit_enc  <- .predictor_matrix(filled[obs, -j, drop = FALSE], pred_types, scale = normalize)
  test_enc <- .predictor_matrix(filled[miss, -j, drop = FALSE], pred_types,
                                encoding = fit_enc$encoding)

  # Seeded split (engine-side, so held-out rows are backend-independent).
  sp <- .split_rows(nrow(fit_enc$x), validation_split, bootstrap)
  tr_idx <- sp$train; val_idx <- sp$valid
  xo <- fit_enc$x

  y <- filled[obs, j]
  if (task == "regression") {
    yv <- as.numeric(y); yc <- mean(yv); ys <- stats::sd(yv)
    if (!is.finite(ys) || ys == 0) ys <- 1
    ystd <- (yv - yc) / ys
    model <- build_model(backend, task, ncol(xo), 1L, arch)
    fit <- fit_model(backend, model,
                     xo[tr_idx, , drop = FALSE], ystd[tr_idx],
                     xo[val_idx, , drop = FALSE], ystd[val_idx], control)
    pred <- predict_model(backend, fit$model, test_enc$x) * ys + yc
    if (draw == "pmm") {
      po <- predict_model(backend, fit$model, xo) * ys + yc
      return(.pmm_draw(pred, po, yv, pmm_k))
    }
    if (draw == "normal") {
      pv <- predict_model(backend, fit$model, xo[val_idx, , drop = FALSE]) * ys + yc
      sigma <- sqrt(mean((yv[val_idx] - pv)^2))
      pred <- pred + stats::rnorm(length(pred), 0, sigma)
    }
    if (types[j] == "count") pred <- pmax(0, round(pred))
    pred
  } else {
    yf <- factor(y); levs <- levels(yf); codes <- as.integer(yf)
    model <- build_model(backend, task, ncol(xo), length(levs), arch)
    fit <- fit_model(backend, model,
                     xo[tr_idx, , drop = FALSE], codes[tr_idx],
                     xo[val_idx, , drop = FALSE], codes[val_idx], control)
    probs <- predict_model(backend, fit$model, test_enc$x)
    if (draw == "none") return(levs[max.col(probs, ties.method = "first")])
    levs[.sample_classes(probs)]
  }
}

# Chained imputation: m replicates -> sweeps -> columns. With early_stop = TRUE
# (point mode) a replicate stops once the pooled change falls below `eps`; in
# draw mode every replicate runs all `iterations` sweeps. Per sweep the mean and
# SD of the imputed cells of every incomplete variable are kept (the trace).
.run_chain <- function(x, types, arch, backend, initialize = "knn", iterations = 3L,
                       m = 1L, eps = 1e-2, normalize = TRUE, control = list(),
                       validation_split = 0.2, seed = NULL, verbose = FALSE,
                       column_fn = .impute_column, column_args = list(),
                       early_stop = TRUE) {
  na_mask <- is.na(x)
  start <- .initialize_na(x, initialize)
  cols <- which(colSums(na_mask) > 0)

  imputed <- vector("list", m)
  conv    <- vector("list", m)
  seeds   <- integer(m)
  tr_dim  <- list(names(x)[cols], NULL, NULL)
  tr_mean <- array(NA_real_, c(length(cols), iterations, m), dimnames = tr_dim)
  tr_sd   <- tr_mean

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
      if (length(cols)) {
        st <- .chain_stats(filled, na_mask, cols)
        tr_mean[, it, rep_i] <- st$mean
        tr_sd[, it, rep_i]   <- st$sd
      }
      ch <- .convergence(old, filled, na_mask, types)
      changes <- c(changes, ch)
      if (early_stop && ch < eps) break
    }
    imputed[[rep_i]] <- filled
    conv[[rep_i]] <- changes
  }

  list(imputed = imputed, convergence = conv, seeds = seeds, na_mask = na_mask,
       trace = list(mean = tr_mean, sd = tr_sd))
}
