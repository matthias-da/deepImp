# deepImp

Imputation of missing values with configurable neural networks, for mixed-type data
(`impNNet()`) and compositional data with rounded zeros (`impNNetCoDa()`). The network
architecture is described by a `deepimp_arch()` object. Models run on a native
[torch](https://torch.mlverse.org/) backend by default (no Python required), or optionally
on [keras3](https://keras3.posit.co/).

## Installation

```r
# from the repository
remotes::install_bitbucket("matthias-da/deepimp")

# the default backend (native libtorch; no Python)
torch::install_torch()

# optional: the keras backend (Python + TensorFlow)
# install.packages("keras3"); keras3::install_keras()
```

## Quick start

Mixed-type imputation:

```r
library(deepImp)
data(sleep, package = "VIM")
imp <- impNNet(sleep, arch = deepimp_arch_small(), epochs = 50, seed = 1)
completed <- getImputed(imp)
```

Compositional data (rounded zeros below a detection limit):

```r
x <- data.frame(a = runif(50, 5, 10), b = runif(50, 5, 10), c = runif(50, 5, 10))
x$a[1:5] <- 0
imp <- impNNetCoDa(x, dl = c(1, 1, 1), label = 0, arch = deepimp_arch_small(), seed = 1)
getImputed(imp)
```

See `vignette("deepImp")` for a full walkthrough.

## Reference

Templ, M. (2021). Imputation of rounded zeros for compositional data using neural networks.
In: Advances in Compositional Data Analysis (Festschrift). Springer.
