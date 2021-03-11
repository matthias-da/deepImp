rm(list = ls())
set.seed(123)

l <- list.files("inst/doc/prederror/R/")
grep(".rda", l)

rda2list <- function(file) {
  e <- new.env()
  load(file, envir = e)
  as.list(e)
}

folder <- "inst/doc/prederror/R/"
files <- list.files(folder, pattern = ".rda$")

results <- Map(rda2list, file.path(folder, files))
names(results) <- tools::file_path_sans_ext(files)
library(data.table)
res <- lapply(results, rbindlist)
results <- rbindlist(results)

results[[1]]$df_cforest_gjovik
results[[1]]$df_lda_gjovik
results[[1]]$df_ctree_gjovik
results[[1]]$df_qda_gjovik
results[[1]]$df_glm_gjovik
results[[1]]$df_naive_bayes_gjovik
results[[1]]$df_knn_gjovik
results[[1]]$df_rpart_gjovik
rbindlist(results[[2]])
