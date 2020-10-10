paste(c(1:3), c("X", "Y", "Z"), sep ="")
## [1] "1X" "2Y" "3Z"
paste(LETTERS, 1:4, sep = "-")

rndm <- sample( rnorm(1000), rep(NA, 1000), 100)

nums <- 1:10
# only the 2nd and 10th element
nums[c(2,10)]

# every element except 2nd and 10th
nums[c(-2,-10)]
nums[-c(2,10)]

# matrix
m <- matrix(1:6, nrow = 2, ncol = 3)

a = 1:10
dim(a) <- c(2,5)

# factors: integer vector where each integer has a label
f <- factor(c("yes", "yes", "yes", "no", "no", "yes", "no" ))

table(f)
unclass(f)
attr(f,"levels")

# set levels order
f <- factor(c("yes", "yes", "yes", "no", "no", "yes", "no" ), levels = c("yes", "no"))

# NA and NaN
na <- c(1, 2, NA, 4)
nan <- c(1, 2, NaN, 4)

is.na(na)
is.nan(na)
is.na(nan)
is.nan(nan)

# dataframe
df <- data.frame(index=1:4, value = c(T, T, F, F))

nrow(df)
ncol(df)

# named data structures
namedarray <- 1:3
names(namedarray) <- c("Name", "Surname", "Age")
names(namedarray)

# check identical names: identical(first, second)

namedlist <- list(first = 1, second = 2)
namedlist$first

namedmatrix <- matrix(1.4, nrow = 2, ncol = 2)
dimnames(namedmatrix) <- list(c("A", "B"), c("C", "D"))

# reading tabular data
# read.table or read.csv, readLines, source, dget, load, unserialize

# data <- read.table("file.txt")

# reading large datasets is useful to specify the class of each column

# initial <- read.table("file.txt", nrows=100)
# classes <- sapply(initial, class)
# data <- read.table("file.txt", colClasses = classes)

# calculate data dimension: 1.500.000 rows and 120 columns
# 1.500.000 x 120 x 8 bytes/numeric
# 1440000000 bytes
# with 2^20 bytes in each MB
# 1.373,29 MB 
# 1.34 GB

# subsetting
# array
subs <- c("a", "b", "c", "d", "e", "f")
subs[1]
subs[1:4]
subs[subs > "a"]
condition <- subs > "a"
subs[condition]

# lists
l <- list(numbers = 1:4, float = 0.6)
l[1] 
l["numbers"]

## result:  $numbers
##          [1] 1 2 3 4

l[[1]]
l[["numbers"]]
l$numbers

## result:  [1] 1 2 3 4

l[(c(1,2))]
l[[c(1,2)]]

l[[1]][[3]]

## result:  [1] 3

# matrices

# returns ad array or value
m[1,2]
m[1,]

# returns a submatrix
m[1,2, drop = FALSE]
m[1,, drop = FALSE]

# partial matching

l$num # instead of l$numbers
l[["num"]] # NULL
l[["num", exact = FALSE]] # [1] 1 2 3 4

# removing NA values
a_with_na <- c(1, 2, NA, 4, NA, 5)
a_with_na[!is.na(a_with_na)]

# elements not NA in both arrays
b_with_na <- c("a", "b", NA, NA, NA, "f")
good <- complete.cases(a_with_na, b_with_na)
a_with_na[good]
b_with_na[good]

# removing NA values in a dataframe
# good <- complete.cases(df) returns the indexes of the rows without NA
# df[good,][1:6,] filters the rows without NA and then takes the first 6

# matrix multplication: %*% operator


# questions
data = read.csv("hw1_data.csv")
data[1:2,]
dim(data)
data[(dim(data)[1]-1):dim(data)[1],]
data[47,]

# how many missing values are in the ozone column?
length(data[,1][is.na(data[,1])])

# what is the mean of the Ozone column in this dataset?
# excluding missing values (NA)
mean(data[,1][!is.na(data[,1])])

# extract the subset of rows of the data frame where Ozone values are above 31 and Temp values are above 90. 
# what is the mean of Solar.R in this subset?
solar <- data[,2] 
temp <- data[,4]
ozone <- data[,1]
indexes <- ozone > 30 & temp > 90
indexes[is.na(indexes)] <- FALSE
mean(solar[indexes])

# mean of the temp when month is 6
mean(data[,4][data[,5] == 6])

# maximum ozone in the month of may
max(data[,1][!is.na(data[,1])][data[,5] == 5])


