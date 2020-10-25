## Random Sampling
set.seed(1)

sample(1:10, 4)

sample(letters, 5)

# permutation
# [1]  1  2  9  5  3  4  8  6  7 10
sample(1:10)

# sampling with replacement
# [1]  8 10  3  7  2  3  4  1  4  9
sample(1:10, replace = TRUE)


