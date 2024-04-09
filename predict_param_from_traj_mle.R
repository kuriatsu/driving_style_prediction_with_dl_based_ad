
tnorm.mle <- function(x, a = -Inf, b = Inf) {
  # parameter estimation for truncated normal data with known extents
  
  n <- length(x)
  ab <- c(a, b)
  init <- c(x[1], sd(x))
  # init <- c(x[1], log(sd(x)))
  
  if (is.finite(a)) {
    if (is.finite(b)) {
      # two-sided truncation
      f <- function(params) {
        mu <- params[1]
        sigma <- params[2]
        # sigma <- exp(params[2])
        n*log(diff(pnorm(ab, mu, sigma))) - sum(dnorm(x, mu, sigma, TRUE))
      }
    } else {
      # left-truncated
      f <- function(params) {
        mu <- params[1]
        sigma <- exp(params[2])
        n*pnorm(a, mu, sigma, FALSE, TRUE) - sum(dnorm(x, mu, sigma, TRUE))
      }
    }
  } else {
    if (is.finite(b)) {
      # right-truncated
      f <- function(params) {
        mu <- params[1]
        sigma <- exp(params[2])
        n*pnorm(b, mu, sigma, TRUE, TRUE) - sum(dnorm(x, mu, sigma, TRUE))
      }
    } else {
      # non-truncated normal
      return(c(mu = mean(x), sigma = sd(x)))
    }
  }
  
  # solve for mu and sigma
  params <- optim(init, f)$par
  c(mu = params[1], sigma = params[2])
  # c(mu = params[1], sigma = exp(params[2]))
}

require("reticulate")
library("stats4")
source_python("pickle_reader.py")
pickle_data <- read_pickle_file("data.pickle")

results <- list() 
mu_list <- list(40, 45, 50, 55, 60)
sd_list <- list(1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0)
for (mu in mu_list) {
    for (sd in sd_list) {
        for (sample_num in 1:100) {
            for (iter in 0:100) {
                target_data <- pickup_data(pickle_data, mu, sd, sample_num)

                # trunc <- 0.1 * (40.0 * pi / 180.0) / 50.0
                # result <- tnorm.mle(unlist(target_data), -trunc, trunc)
                result <- tnorm.mle(unlist(target_data), 0, 100)
                result <- c(mu=mu, sd=sd, pred_mu=result[["mu"]], pred_sd=result[["sigma"]], iter=iter, sample_num=sample_num) 
                results[[length(results)+1]] <- result
                print(result) 
            }
        }
    }
}

save_data(results)
