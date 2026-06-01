#' deepImp: Imputation with Deep Learning Methods
#'
#' Imputes missing values with configurable neural networks, for both mixed-type
#' data ([impNNet()]) and compositional data with rounded zeros
#' ([impNNetCoDa()]). The network architecture (depth, width, dropout, activation,
#' optimiser) is described by a [deepimp_arch()] object. Networks run on a native
#' \pkg{torch} backend by default (no Python required) or, optionally, on
#' \pkg{keras3}. Results are returned as a `"deepimp"` object; read the completed
#' data with [getImputed()].
#'
#' See `vignette("deepImp")` for a walkthrough.
#'
#' @references
#' Templ, M. (2021). Imputation of rounded zeros for compositional data using
#' neural networks. In: Advances in Compositional Data Analysis (Festschrift).
#' Springer. \doi{10.1007/978-3-030-71175-7}
#' @keywords internal
"_PACKAGE"
