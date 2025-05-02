#CLEANING

#load librarys
library(dplyr)
library(caret)

#Download df
dfpor <- read.csv("https://raw.githubusercontent.com/katieluong33/BUAN-381/refs/heads/main/student-mat.csv", header = TRUE, sep = ";")
dfmath <- read.csv("https://raw.githubusercontent.com/katieluong33/BUAN-381/refs/heads/main/student-por.csv", header = TRUE, sep = ";")

#add a column to each df called "class" to identify which course 
#the data is from, and convert it to a factor
dfpor$class <- "p"
dfpor$class <- as.factor(dfpor$class)
class(dfpor$class)

dfmath$class <- "m"
dfmath$class <- as.factor(dfmath$class)
class(dfmath$class)

#merge the two df together
mergeddf <- rbind(dfpor,dfmath)

#clasification tasks
#split the data into three sets. 
#70% for training, and 15 per each hold out partition

set.seed(314)
train_index <- createDataPartition(mergeddf$class, p = 0.7, list = FALSE)
trainset <- mergeddf[train_index, ]
holdout <- mergeddf[-train_index, ]

set.seed(314)
val_index <- createDataPartition(holdout$class, p = 0.5, list = FALSE)
val_set <- holdout[val_index, ]
test_set <- holdout[-val_index, ]

