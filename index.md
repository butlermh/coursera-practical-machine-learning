# Classifying barbell lifts

## Practical Machine Learning Peer Assessment

### July 2014

## Introduction

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. Here we investigate a dataset where 6 participants were asked to perform barbell lifts correctly and incorrectly in 5 different ways, with accelerometers on the belt, forearm, arm, and dumbell. We build a classification model that can classify which way the participant is lifting the barbell, and then check this classification model against a test data set. 

More information about this dataset is available from the [website (1)](#ref1) specifically the section on the Weight Lifting Exercise Dataset. The [training data set(2)](#ref2) and the [testing data set (3)](#ref3) are also available.

## Methods

### Getting and preparing the data

First we load the training and testing data sets. Here it was necessary to pay attention to the fact that missing values could be represented in several ways, either by an `NA`, a totally empty value or `#DIV/0!` indicating a divide by zero error. In order to avoid numerical variables being erroneously converted to factor variables it was necessary to convert these values explicitly to NAs:


```r
train <- read.csv("data/pml-training.csv", na.strings = c("NA", "", "#DIV/0!"))
test <- read.csv("data/pml-testing.csv", na.strings = c("NA", "", "#DIV/0!"))
summary(train$classe)
```

```
##    A    B    C    D    E 
## 5580 3797 3422 3216 3607
```


Second, examining the dataset, it is apparent that the different values of `classe` are grouped in sequential order so a model could appear to do very well just classifying based on the index. Therefore it is important to remove the id column, `x`, and any other column which could be used to place the values is sequential order such as timestamps. Also the names of the people who supplied the data are also grouped in sequential order so they need to be removed as well: 


```r
drops <- c("user_name", "raw_timestamp_part_1", "raw_timestamp_part_2", "cvtd_timestamp", 
    "X", "new_window")
train.k <- train[, !(names(train) %in% drops)]
test.k <- test[, !(names(test) %in% drops)]
```


Third, examining the dataset it was apparent that some columns had a lot of missing values, so we calculate how many missing values each column has and remove any columns that have too many missing values:


```r
n <- dim(train.k)[2]
naCols <- vector(length = n)
for (i in 1:n) {
    naCols[i] <- sum(is.na(train.k[, i]))
}
train.s <- train.k[, which(naCols < 10)]
test.s <- test.k[, which(naCols < 10)]
```


Fourth, we we subdivide the training set to create a cross validation set. We allocate 70% of the original training set to the new training set, and the other 30% to the cross validation set:


```r
library(caret)
inTrain <- createDataPartition(y = train.s$classe, p = 0.7, list = FALSE)
training <- train.s[inTrain, ]
train.cv <- train.s[-inTrain, ]
```


### Classification and regression trees

First we will investigate using classification and regression trees as proposed by [Breiman (4)](#ref4) and [Ripley (5)](#ref5):


```r
library(tree)
fit1 <- tree(classe ~ ., method = "tree", data = training)
p1 <- predict(fit1, type = "class")
table(training$classe, p1)
```

```
##    p1
##        A    B    C    D    E
##   A 3556  236   44   59   11
##   B  295 1605  424  161  173
##   C   41  164 2100   86    5
##   D  165  191  716  788  392
##   E   21   93   70   70 2271
```

```r
fit1.prune <- prune.misclass(fit1, best = 10)
```


Figure 2 shows a pruned version of the generated tree in order to make the diagram legible. The full tree has 22 nodes so it is rather complex.


```r
plot(fit1.prune)
title(main = "Figure 1: Tree created using tree function")
text(fit1.prune, cex = 1.2)
```

![plot of chunk unnamed-chunk-6](figure/unnamed-chunk-6.png) 

Does this complexity mean the tree is overfitting? We will investigate this using pruning. First though, we will estimate the in-sample error:


```r
nright = table(p1 == training$classe)
tree_in_error = as.vector(100 * (1 - nright["TRUE"]/sum(nright)))
```


We estimate the in-sample error to be **24.8744 %**. Next, how does the tree perform on cross-validation set?


```r
p2 <- predict(fit1, newdata = train.cv, type = "class")
table(train.cv$classe, p2)
```

```
##    p2
##        A    B    C    D    E
##   A 1506  118   24   22    4
##   B  123  675  192   80   69
##   C   13   70  872   70    1
##   D   70   73  313  370  138
##   E   14   35   27   29  977
```

```r
nright = table(p2 == train.cv$classe)
tree_out_error = as.vector(100 * (1 - nright["TRUE"]/sum(nright)))
```


We estimate the out-of-sample error to be **25.2336 %**. Can we improve performance on cross-validation set by pruning?


```r
error.cv <- {
    Inf
}
for (i in 2:19) {
    prune.data <- prune.misclass(fit1, best = i)
    pred.cv <- predict(prune.data, newdata = train.cv, type = "class")
    nright = table(pred.cv == train.cv$classe)
    error = as.vector(100 * (1 - nright["TRUE"]/sum(nright)))
    error.cv <- c(error.cv, error)
}
error.cv
```

```
##  [1]   Inf 63.40 47.19 47.19 47.19 47.19 42.06 42.06 39.97 36.65 36.65
## [12] 35.19 33.70 32.64 29.97 29.97 28.85 28.22 27.37
```

```r
plot(error.cv, type = "l", xlab = "Size of tree (number of nodes)", ylab = "Out of sample error(%)", 
    main = "Figure 2: Relationship between tree size and out of sample error")
```

![plot of chunk unnamed-chunk-9](figure/unnamed-chunk-9.png) 


Despite the complexity of the tree in Figure 1, Figure 2 does not indicate overfitting as the out of sample error does not increase as more nodes are added to the tree. 

### RPart trees

Next we will use a different type of classification and regression tree implemented by rpart, also as proposed by as proposed by [Breiman (4)](#ref4):


```r
library(rpart)
fit2 <- rpart(classe ~ ., data = training)
p3 <- predict(fit2, type = "class")
table(training$classe, p3)
```

```
##    p3
##        A    B    C    D    E
##   A 3562  130   42  114   58
##   B  479 1640  176  285   78
##   C   56  166 1946  167   61
##   D  121  204  329 1488  110
##   E  117  277  183  303 1645
```

```r
library(rpart.plot)
prp(fit2, cex = 1.3)
title(main = "Figure 3: Tree constructed using the rpart function")
```

![plot of chunk unnamed-chunk-10](figure/unnamed-chunk-10.png) 


Looking at the trees in Figure 1 and Figure 3 although the diagrams are different because they come from different packages there are similarities in the decisions they are using.


```r
cp <- fit2$cp
plot(cp[, 2], cp[, 3], type = "l", xlab = "Size of tree (number of nodes", ylab = "Out of sample error (%)", 
    main = "Figure 4: Relationship between tree size and out of sample error")
```

![plot of chunk unnamed-chunk-11](figure/unnamed-chunk-11.png) 

Next we estimate the in-sample error:


```r
nright = table(p3 == training$classe)
rpart_in_error = as.vector(100 * (1 - nright["TRUE"]/sum(nright)))
```


We estimate the in-sample error to be **25.1583 %** so it does slightly better than the `tree` package. Next, how does the tree perform on cross-validation set?


```r
p4 <- predict(fit2, newdata = train.cv, type = "class")
table(train.cv$classe, p4)
```

```
##    p4
##        A    B    C    D    E
##   A 1521   50   17   69   17
##   B  209  702   67  120   41
##   C   22   76  816   84   28
##   D   43   83  144  659   35
##   E   53  122  113  126  668
```

```r
nright = table(p4 == train.cv$classe)
rpart_out_error = as.vector(100 * (1 - nright["TRUE"]/sum(nright)))
```


We estimate the out-of-sample error to be **25.8114 %**. Again this is an improvement on `tree`. 

### Random forests

Next we apply random forests also proposed by [Breiman (6)](#ref6) and [Breiman (7)](#ref7):


```r
library(randomForest)
fit3 <- randomForest(classe ~ ., data = training, method = "class")
p5 <- predict(fit3, type = "class")
table(training$classe, p5)
```

```
##    p5
##        A    B    C    D    E
##   A 3905    0    0    0    1
##   B    5 2651    2    0    0
##   C    0    5 2391    0    0
##   D    0    0   11 2241    0
##   E    0    0    0    6 2519
```

```r
nright = table(p5 == training$classe)
forest_in_error = as.vector(100 * (1 - nright["TRUE"]/sum(nright)))
```


The in-sample error for the random forest is **0.2184 %**. This is a big improvement over the two previous tree based methods. How does the forest perform on cross-validation set?


```r
p6 <- predict(fit3, newdata = train.cv, type = "class")
table(train.cv$classe, p6)
```

```
##    p6
##        A    B    C    D    E
##   A 1674    0    0    0    0
##   B    3 1136    0    0    0
##   C    0    1 1024    1    0
##   D    0    0    8  956    0
##   E    0    0    0    5 1077
```

```r
nright = table(p6 == train.cv$classe)
forest_out_error = as.vector(100 * (1 - nright["TRUE"]/sum(nright)))
```


The out-of-sample error for the random forest is **0.3059 %**. Again this is much better than the previous tree based methods. 

## Results

The random forest clearly performs better, approaching 99% accuracy for in-sample and out-of-sample error so we will select this model and apply it to the test data set. We use the provided function to classify 20 data points from the test set by the type of lift. We then upload these classifications to Coursera to confirm that the model is working correctly.


```r
destDir = "./output"
pml_write_files = function(x) {
    n = length(x)
    for (i in 1:n) {
        filename = paste0(destDir, "/", "problem_id_", i, ".txt")
        write.table(x[i], file = filename, quote = FALSE, row.names = FALSE, 
            col.names = FALSE)
    }
}
if (!file.exists(destDir)) {
    dir.create(destDir)
}
p7 <- predict(fit3, newdata = test.s, type = "class")
pml_write_files(p7)
```


With the random forest model, all twenty classifications correctly matched the test set values at Coursera.

## Conclusions

In this report we created three different classification models, using `tree`, `rpart` and `randomForest` for a data set of barbell lifts. We found the `randomForest` model performed the best, approaching accuracy of 99% for the out-of-sample data set. Using this model, we were able to correctly classify all twenty values from the test data set.

## References

<a name="ref1"></a>[1] Dataset [http://groupware.les.inf.puc-rio.br/har](http://groupware.les.inf.puc-rio.br/har)

<a name="ref2"></a>[2] Training data set [https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv)

<a name="ref3"></a>[3] Testing data set [https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv)

<a name="ref4"></a>[4]  Breiman L., Friedman J. H., Olshen R. A., and Stone, C. J. (1984), Classification and Regression Trees. Wadsworth.

<a name="ref5"></a>[5] Ripley, B. D. (1996), Pattern Recognition and Neural Networks., Cambridge University Press, Cambridge. Chapter 7.

<a name="ref6"></a>[6] Breiman, L. (2001), Random Forests, Machine Learning 45(1), 5-32.

<a name="ref7"></a>[7] Breiman, L (2002), Manual On Setting Up, Using, And Understanding Random Forests V3.1, [ http://oz.berkeley.edu/users/breiman/Using_random_forests_V3.1.pdf]( http://oz.berkeley.edu/users/breiman/Using_random_forests_V3.1.pdf)
