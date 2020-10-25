## Linear Model

# y = beta_0 + beta_1 * x + epsilon

# epsilon ~ N(0,4)
# x ~ N(0,1)
# beta_0 = .5
# beta_1 = 2

set.seed(20)

x <- rnorm(100)
e <- rnorm(100, 0, 2)

y <- 0.5 + 2*x + e

summary(y)

plot(x,y)

b <- rbinom(100, 1, 0.5)
plot(b,y)

# suppose we want to simulate from a Poisson model where
# y ~ Poisson(mu)
# log(mu) = beta_0 + beta_1*x

set.seed(1)

x <- rnorm(100)

log.mu <- 0.5 + 0.3 * x

y <- rpois(100, exp(log.mu))

summary(y)

plot(x,y)

