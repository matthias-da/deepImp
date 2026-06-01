# Pre-compute the deepImp vignette.
#
# The vignette trains neural networks, which needs a working torch backend that is
# not available on CRAN's check machines. To show real outputs while keeping the
# package check fast and dependency-free, we knit the live source
# `deepImp.Rmd.orig` (eval = TRUE) ONCE here into the shipped `deepImp.Rmd`, with
# all outputs and figures baked in as static content. CRAN then only re-renders
# static markdown.
#
# WORKFLOW: edit `deepImp.Rmd.orig` (not `deepImp.Rmd`), then run this script from
# the package root with a torch backend installed:
#
#   Rscript -e 'devtools::load_all(); knitr::knit("vignettes/deepImp.Rmd.orig", output = "vignettes/deepImp.Rmd")'
#
# or simply: Rscript vignettes/precompute.R

if (!requireNamespace("knitr", quietly = TRUE)) stop("knitr is required")

# Knit within the vignettes/ directory so figure paths resolve relative to it.
old <- setwd("vignettes")
on.exit(setwd(old), add = TRUE)

knitr::knit("deepImp.Rmd.orig", output = "deepImp.Rmd")
message("Pre-computed vignettes/deepImp.Rmd (and figures under vignettes/figure/).")
