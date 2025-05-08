
# BUAN 381 Final Project
# Brianna Floyd, Katie Luong, Anastasia Lomtadze, Sophia Saenger

### Load libraries
library(psych)
library(polycor)
library(tidyverse)
library(mgcv)
library(e1071) #SVM LIBRARY
library(caret) #FOR confusionMatrix()
library(parsnip)
library(rpart)
library(rpart.plot)
library(randomForest)
library(Metrics)
library(pROC)
library(glmnet)

set.seed(314)  # for reproducibility

### Import data
df_p <- read.table('https://raw.githubusercontent.com/katieluong33/BUAN-381/refs/heads/main/student-por.csv', header = TRUE, sep = ";")
df_m <- read.table("https://raw.githubusercontent.com/katieluong33/BUAN-381/refs/heads/main/student-mat.csv", header = TRUE, sep = ";")

### Data cleaning
df_p$class <- "p"
df_p$class <- as.factor(df_p$class)
df_m$class <- "m"
df_m$class <- as.factor(df_m$class)
df <- rbind(df_p,df_m)

df <- na.omit(df)

df[df == "yes"] <- 1
df[df == "no"] <- 0
df$activities <- as.numeric(df$activities)
df$schoolsup <- as.numeric(df$schoolsup)

View(df)

print(dim(df))
print(colnames(df))

numeric_data <- df[sapply(df, is.numeric)]

### Correlation matrix
cor_matrix <- cor(numeric_data, use = "complete.obs")
print(cor_matrix)
print(nrow(df))

### Split the data - training and remaining
train_index <- createDataPartition(df$G3, p = 0.70, list = FALSE)
train_data <- df[train_index, ]
remaining <- df[-train_index, ]

### Split the data - validation and test
valid_index <- createDataPartition(remaining$G3, p = 0.5, list = FALSE)
valid_data <- remaining[valid_index, ]
test_data <- remaining[-valid_index, ]

########## REGRESSION ##########
### 4a - Bivariate model
bivariate_model <- lm(G3 ~ G2, data = train_data)
summary(bivariate_model)

bivariate_pred_train <- predict(bivariate_model, train_data)

bivariate_rmse_train <- rmse(train_data$G3, bivariate_pred_train)

### 4b - Bivariate with nonlinearity model
bivariate_model_2 <- lm(G3 ~ I(G2^2), data = train_data)
summary(bivariate_model_2)

bivariate_2_pred_train <- predict(bivariate_model_2, train_data)

bivariate_2_rmse_train <- rmse(train_data$G3, bivariate_2_pred_train)

### 4c - Bivariate with nonlinearity and lasso regularization model
train_data$G2_squared <- train_data$G2^2

cv_model <- cv.glmnet(
  x = cbind(train_data$G2, train_data$G2_squared),  # Input matrix with G2 and G2_squared
  y = train_data$G3,                              # Outcome variable G3
  alpha = 1,                                      # Lasso regularization
  nfolds = 10                                     # 10-fold cross-validation (you can change this)
)

best_lambda <- cv_model$lambda.min
cat("Best lambda: ", best_lambda, "\n")

bivariate_model_3 <- glmnet(
  x = cbind(train_data$G2, train_data$G2_squared),  # Input matrix with G2 and G2_squared
  y = train_data$G3,                              # Outcome variable G3
  alpha = 1,                                      # Lasso regularization
  lambda = best_lambda                            # Best lambda from cross-validation
)

bivariate_3_pred_train <- predict(bivariate_model_3, newx = cbind(train_data$G2, train_data$G2_squared))

bivariate_3_rmse_train <- rmse(train_data$G3, bivariate_3_pred_train)

train_data$G2_squared <- NULL

### 4d - Spline model
spline_model <- gam(G3 ~ s(G2), data = train_data, family = gaussian)
summary(spline_model)

spline_pred_train <- predict(spline_model, train_data)

spline_rmse_train <- rmse(train_data$G3, spline_pred_train)

### 4e - Bivariate plot
bivariate_pred_valid <- predict(bivariate_model, valid_data)
bivariate_rmse_valid <- rmse(valid_data$G3, bivariate_pred_valid)

bivariate_2_pred_valid <- predict(bivariate_model_2, valid_data)
bivariate_2_rmse_valid <- rmse(valid_data$G3, bivariate_2_pred_valid)

valid_data$G2_squared <- valid_data$G2^2
bivariate_3_pred_valid <- predict(bivariate_model_3, newx = cbind(valid_data$G2, valid_data$G2_squared))
bivariate_3_rmse_valid <- rmse(valid_data$G3, bivariate_3_pred_valid)

spline_pred_valid <- predict(spline_model, valid_data)
spline_rmse_valid <- rmse(valid_data$G3, spline_pred_valid)

ggplot() +
  # Actual data points (train and valid sets)
  geom_point(aes(x = train_data$G2, y = train_data$G3), color = "pink", alpha = 0.5) +  # Training data
  geom_point(aes(x = valid_data$G2, y = valid_data$G3), color = "red", alpha = 0.5) +   # Validation data
  
  # Model 1 (Bivariate model): solid for training, dashed for validation
  geom_line(aes(x = train_data$G2, y = bivariate_pred_train), color = "blue", linetype = "solid") +
  geom_line(aes(x = valid_data$G2, y = bivariate_pred_valid), color = "blue", linetype = "dashed") +
  
  # Model 2 (Bivariate with nonlinearity): solid for training, dashed for validation
  geom_line(aes(x = train_data$G2, y = bivariate_2_pred_train), color = "green", linetype = "solid") +
  geom_line(aes(x = valid_data$G2, y = bivariate_2_pred_valid), color = "green", linetype = "dashed") +
  
  # Model 3 (Bivariate with nonlinearity and lasso): solid for training, dashed for validation
  geom_line(aes(x = train_data$G2, y = bivariate_3_pred_train), color = "purple", linetype = "solid") +
  geom_line(aes(x = valid_data$G2, y = bivariate_3_pred_valid), color = "purple", linetype = "dashed") +
  
  # Model 4 (Spline model): solid for training, dashed for validation
  geom_line(aes(x = train_data$G2, y = spline_pred_train), color = "orange", linetype = "solid") +
  geom_line(aes(x = valid_data$G2, y = spline_pred_valid), color = "orange", linetype = "dashed") +
  
  # Labels and theme
  labs(
    title = "Model Predictions vs Actual Data",
    x = "G2",
    y = "G3",
    caption = "Solid lines: Training data, Dashed lines: Validation data"
  ) +
  theme_minimal() +
  theme(legend.position = "none")

valid_data$G2_squared <- NULL

### 4f - Bivariate validation table
bivariate_pred_valid <- predict(bivariate_model, valid_data)
bivariate_rmse_valid <- rmse(valid_data$G3, bivariate_pred_valid)

bivariate_2_pred_valid <- predict(bivariate_model_2, valid_data)
bivariate_2_rmse_valid <- rmse(valid_data$G3, bivariate_2_pred_valid)

valid_data$G2_squared <- valid_data$G2^2
bivariate_3_pred_valid <- predict(bivariate_model_3, newx = cbind(valid_data$G2, valid_data$G2_squared))
bivariate_3_rmse_valid <- rmse(valid_data$G3, bivariate_3_pred_valid)

spline_pred_valid <- predict(spline_model, valid_data)
spline_rmse_valid <- rmse(valid_data$G3, spline_pred_valid)

Validation_Table <- as.table(matrix(c(bivariate_rmse_train, bivariate_2_rmse_train, bivariate_3_rmse_train, spline_rmse_train, 
                                      bivariate_rmse_valid, bivariate_2_rmse_valid, bivariate_3_rmse_valid, spline_rmse_valid), 
                                    ncol=4, byrow=TRUE))

colnames(Validation_Table) <- c('Bivariate', 'Nonlinearity', 'Lasso', 'Spline')
rownames(Validation_Table) <- c('RMSE_IN', 'RMSE_OUT')
Validation_Table

spline_pred_test <- predict(spline_model, test_data)
spline_rmse_test <- rmse(test_data$G3, spline_pred_test)
print(spline_rmse_test)

### 5a - Multivariate model
multivariate_model <- lm(G3 ~ ., data = train_data)
summary(multivariate_model)

multivariate_pred_train <- predict(multivariate_model, train_data)
multivariate_pred_valid <- predict(multivariate_model, valid_data)

multivariate_rmse_train <- rmse(train_data$G3, multivariate_pred_train)
multivariate_rmse_valid <- rmse(valid_data$G3, multivariate_pred_valid)

### 5b - Multivariate with lasso regularization model
terms_x <- terms(G3 ~ ., data = train_data)

x <- model.matrix(terms_x, data = train_data)[, -1]
x_valid <- model.matrix(terms_x, data = valid_data)[, -1]

y <- train_data$G3

cv_model <- cv.glmnet(
  x = x,                        # All predictors
  y = y,                        # Outcome variable
  alpha = 1,                    # Lasso regularization
  nfolds = 10                   # 10-fold CV
)

best_lambda <- cv_model$lambda.min
cat("Best lambda: ", best_lambda, "\n")

multivariate_2_model <- glmnet(
  x = x,
  y = y,
  alpha = 1,
  lambda = best_lambda
)

y_valid <- valid_data$G3

multivariate_2_pred_train <- predict(multivariate_2_model, newx = x)
multivariate_2_pred_valid <- predict(multivariate_2_model, newx = x_valid)

multivariate_2_rmse_train <- rmse(train_data$G3, multivariate_2_pred_train)
multivariate_2_rmse_valid <- rmse(y_valid, multivariate_2_pred_valid)


### 5c - Multivariate with nonlinearity model
multivariate_3_model <- lm(G3 ~ activities + romantic + I(G2^2), data = train_data)
summary(multivariate_3_model)

x_valid <- model.matrix(G3 ~ activities + romantic + I(G2^2), data = valid_data)

multivariate_3_pred_train <- predict(multivariate_3_model, train_data)
multivariate_3_pred_valid <- predict(multivariate_3_model, valid_data)

multivariate_3_rmse_train <- rmse(train_data$G3, multivariate_3_pred_train)
multivariate_3_rmse_valid <- rmse(valid_data$G3, multivariate_3_pred_valid)

### 5d - SVM model
svm_model<- svm(G3 ~ ., 
                data = train_data, 
                type = "eps-regression",
                kernel = "radial",
                cost=1,                   #REGULARIZATION PARAMETER
                gamma = 1/(ncol(df)-1), #DEFAULT KERNEL PARAMETER
                coef0 = 0,                    #DEFAULT KERNEL PARAMETER
                degree=2,                     #POLYNOMIAL KERNEL PARAMETER
                scale = FALSE)                #RESCALE DATA? (SET TO TRUE TO NORMALIZE)
summary(svm_model)

svm_pred_train <- predict(svm_model, train_data)
svm_pred_valid <- predict(svm_model, valid_data)

svm_rmse_train <- rmse(train_data$G3, svm_pred_train)
svm_rmse_valid <- rmse(valid_data$G3, svm_pred_valid)

### 5e - Decision Tree model
tree_model <- train(G3 ~ ., data = train_data, method = "rpart")
print(tree_model)

tree_pred_train <- predict(tree_model, train_data)
tree_pred_valid <- predict(tree_model, valid_data)

tree_rmse_train <- rmse(train_data$G3, tree_pred_train)
tree_rmse_valid <- rmse(valid_data$G3, tree_pred_valid)

rpart_model <- tree_model$finalModel

rpart.plot(rpart_model,
            type = 2,             # labels all nodes
            extra = 101,          # show predicted value and % observations
            fallen.leaves = TRUE,
            main = "Decision Tree for Predicting G3")

### 5f - Random Forest model
train_data$G2_squared <- NULL
valid_data$G2_squared <- NULL
test_data$G2_squared <- NULL

forest_model <- train(
  G3 ~ .,                    # Target variable (G3) and predictors
  data = train_data,                 # Data used for training
  method = "rf",             # Random Forest method
  importance = TRUE          # Keep track of variable importance
)
print(forest_model)

forest_pred_train <- predict(forest_model, train_data)
forest_pred_valid <- predict(forest_model, valid_data)

forest_rmse_train <- rmse(train_data$G3, forest_pred_train)
forest_rmse_valid <- rmse(valid_data$G3, forest_pred_valid)

### 2nd Random Forest model
forest_model_2 <- randomForest(
  G3 ~ ., 
  data = train_data, 
  ntree = 500, 
  mtry = 4, 
  maxnodes = 20,   # restrict tree size
  importance = TRUE
)
print(forest_model_2)

forest_2_pred_train <- predict(forest_model_2, train_data)
forest_2_pred_valid <- predict(forest_model_2, valid_data)

forest_2_rmse_train <- rmse(train_data$G3, forest_2_pred_train)
forest_2_rmse_valid <- rmse(valid_data$G3, forest_2_pred_valid)

### 5g - Table to compare the models
Validation_Table_2 <- as.table(matrix(c(multivariate_rmse_train, multivariate_2_rmse_train, multivariate_3_rmse_train, 
                                      svm_rmse_train, tree_rmse_train, forest_rmse_train, forest_2_rmse_train, 
                                      multivariate_rmse_valid, multivariate_2_rmse_valid, multivariate_3_rmse_valid, 
                                      svm_rmse_valid, tree_rmse_valid, forest_rmse_valid, forest_2_rmse_valid), ncol=7, byrow=TRUE))

colnames(Validation_Table_2) <- c('Multivariate', 'Lasso', 'Nonlinearity', 'SVM','Decision Tree', 'Random Forest', 'Random Forest 2')
rownames(Validation_Table_2) <- c('RMSE_IN', 'RMSE_OUT')
Validation_Table_2

# Test on best model
forest_pred_test <- predict(forest_model, test_data)
forest_rmse_test <- rmse(test_data$G3, forest_pred_test)
print(forest_rmse_test)

########## CLASSIFICATION ##########
### 8a - Logistic regression model
logit_model <- glm(activities ~ ., data = train_data, family = binomial(link="logit"), )
summary(logit_model)

# Accuracy
logit_prob_train <- predict(logit_model, newdata = train_data, type = "response")
logit_pred_train <- ifelse(logit_prob_train > 0.5, 1, 0)
logit_acc_train <- mean(logit_pred_train == train_data$activities)

logit_prob_valid <- predict(logit_model, newdata = valid_data, type = "response")
logit_pred_valid <- ifelse(logit_prob_valid > 0.5, 1, 0)
logit_acc_valid <- mean(logit_pred_valid == valid_data$activities)

# ROC and AUC
logit_roc_train <- roc(train_data$activities, logit_prob_train)
plot(logit_roc_train)
logit_auc_train <- auc(logit_roc_train)

logit_roc_valid <- roc(valid_data$activities, logit_prob_valid)
plot(logit_roc_valid)
logit_auc_valid <- auc(logit_roc_valid)

### 8b - Probit model
probit_model <- glm(activities ~ ., data = train_data, family = binomial(link="probit"), )
summary(probit_model)

# Accuracy
probit_prob_train <- predict(probit_model, newdata = train_data, type = "response")
probit_pred_train <- ifelse(probit_prob_train > 0.5, 1, 0)
probit_acc_train <- mean(probit_pred_train == train_data$activities)

probit_prob_valid <- predict(probit_model, newdata = valid_data, type = "response")
probit_pred_valid <- ifelse(probit_prob_valid > 0.5, 1, 0)
probit_acc_valid <- mean(probit_pred_valid == valid_data$activities)

# ROC and AUC
probit_roc_train <- roc(train_data$activities, probit_prob_train)
plot(probit_roc_train)
probit_auc_train <- auc(probit_roc_train)

probit_roc_valid <- roc(valid_data$activities, probit_prob_valid)
plot(probit_roc_valid)
probit_auc_valid <- auc(probit_roc_valid)

### 9a - SVM model
df$activities <- as.numeric(df$activities)

svm_model_2 <- svm(activities ~ ., 
                   data = train_data, 
                   type = "C-classification", 
                   kernel = "radial", 
                   cost = 1,
                   gamma = 1/(ncol(df) - 1), 
                   scale = FALSE,
                   probability = TRUE)  # Enable probability estimation

# Accuracy
svm_2_pred_train <- predict(svm_model_2, newdata = train_data, probability = TRUE)
svm_2_prob_train <- attr(svm_2_pred_train, "probabilities")[, "1"]
svm_2_acc_train <- mean(svm_2_pred_train == train_data$activities)

svm_2_pred_valid <- predict(svm_model_2, newdata = valid_data, probability = TRUE)
svm_2_prob_valid <- attr(svm_2_pred_valid, "probabilities")[, "1"]
svm_2_acc_valid <- mean(svm_2_pred_valid == valid_data$activities)

# ROC and AUC
svm_2_roc_train <- roc(train_data$activities, svm_2_prob_train)
plot(svm_2_roc_train)
svm_2_auc_train <- auc(svm_2_roc_train)

svm_2_roc_valid <- roc(valid_data$activities, svm_2_prob_valid)
plot(svm_2_roc_valid)
svm_2_auc_valid <- auc(svm_2_roc_valid)

### 9b - Decision Tree model
train_data$activities <- factor(train_data$activities, levels = c(0, 1), labels = c("class0", "class1"))
valid_data$activities <- factor(valid_data$activities, levels = c(0, 1), labels = c("class0", "class1"))

tree_model_2 <- train(
  activities ~ ., 
  data = train_data, 
  method = "rpart",
  trControl = trainControl(classProbs = TRUE)
)

print(tree_model_2)

tree_2_pred_train <- predict(tree_model_2, newdata = train_data)
tree_2_pred_valid <- predict(tree_model_2, newdata = valid_data)

tree_2_pred_train <- as.factor(tree_2_pred_train)
tree_2_pred_valid <- as.factor(tree_2_pred_valid)
train_true <- as.factor(train_data$activities)
valid_true <- as.factor(valid_data$activities)

# Accuracy
tree_2_pred_train <- factor(tree_2_pred_train, levels = levels(train_true))
tree_2_pred_valid <- factor(tree_2_pred_valid, levels = levels(valid_true))

tree_2_acc_train <- mean(tree_2_pred_train == train_true)
tree_2_acc_valid <- mean(tree_2_pred_valid == valid_true)
tree_2_prob_train <- predict(tree_model_2, newdata = train_data, type = "prob")[, "class1"]
tree_2_prob_valid <- predict(tree_model_2, newdata = valid_data, type = "prob")[, "class1"]

# ROC and AUC
tree_2_roc_train <- roc(train_true, tree_2_prob_train)
plot(tree_2_roc_train, main = "ROC Curve - Train")
tree_2_auc_train <- auc(tree_2_roc_train)

tree_2_roc_valid <- roc(valid_true, tree_2_prob_valid)
plot(tree_2_roc_valid, main = "ROC Curve - Validation")
tree_2_auc_valid <- auc(tree_2_roc_valid)

### 9c - Random Forest model
train_data$G2_squared <- NULL
valid_data$G2_squared <- NULL
test_data$G2_squared <- NULL

train_data$activities <- as.factor(train_data$activities)
valid_data$activities <- as.factor(valid_data$activities)

forest_model_3 <- train(
  activities ~ .,                    # Target variable (activities) and predictors
  data = train_data,                 # Data used for training
  method = "rf",             # Random Forest method
  importance = TRUE          # Keep track of variable importance
)
print(forest_model_3)

# Accuracy
forest_3_pred_train <- predict(forest_model_3, train_data)
forest_3_acc_train <- mean(forest_3_pred_train == train_data$activities)

forest_3_pred_valid <- predict(forest_model_3, valid_data)
forest_3_acc_valid <- mean(forest_3_pred_valid == valid_data$activities)

# ROC and AUC
forest_3_prob_train <- predict(forest_model_3, newdata = train_data, type = "prob")[, "class1"]
forest_3_roc_train <- roc(train_true, forest_3_prob_train)
plot(forest_3_roc_train, main = "ROC Curve - Train")
forest_3_auc_train <- auc(forest_3_roc_train)

forest_3_prob_valid <- predict(forest_model_3, newdata = valid_data, type = "prob")[, "class1"]
forest_3_roc_valid <- roc(valid_true, forest_3_prob_valid)
plot(forest_3_roc_valid, main = "ROC Curve - Validation")
forest_3_auc_valid <- auc(forest_3_roc_valid)

### 2nd Random Forest model
forest_model_4 <- randomForest(
  activities ~ ., 
  data = train_data, 
  ntree = 500, 
  mtry = 4, 
  maxnodes = 20,   # restrict tree size
  importance = TRUE
)
print(forest_model_4)

# Accuracy
forest_4_pred_train <- predict(forest_model_4, train_data)
forest_4_acc_train <- mean(forest_4_pred_train == train_data$activities)

forest_4_pred_valid <- predict(forest_model_4, valid_data)
forest_4_acc_valid <- mean(forest_4_pred_valid == valid_data$activities)

# ROC and AUC
forest_4_prob_train <- predict(forest_model_4, newdata = train_data, type = "prob")[, "class1"]
forest_4_roc_train <- roc(train_true, forest_4_prob_train)
plot(forest_4_roc_train, main = "ROC Curve - Train")
forest_4_auc_train <- auc(forest_4_roc_train)

forest_4_prob_valid <- predict(forest_model_4, newdata = valid_data, type = "prob")[, "class1"]
forest_4_roc_valid <- roc(valid_true, forest_4_prob_valid)
plot(forest_4_roc_valid, main = "ROC Curve - Validation")
forest_4_auc_valid <- auc(forest_4_roc_valid)

### Table to compare the models
Validation_Table_2 <- as.table(matrix(c(logit_acc_train, probit_acc_train, svm_2_acc_train, 
                                      tree_2_acc_train, forest_3_acc_train, forest_4_acc_train,
                                      logit_acc_valid, probit_acc_valid, svm_2_acc_valid, 
                                      tree_2_acc_valid, forest_3_acc_valid, forest_4_acc_valid), ncol=6, byrow=TRUE))

colnames(Validation_Table_2) <- c('Logit', 'Probit', 'SVM', 'Decision Tree', 'Random Forest', 'Random Forest 2')
rownames(Validation_Table_2) <- c('ACC_IN', 'ACC_OUT')
Validation_Table_2

# Test on best model
forest_3_pred_test <- predict(forest_model_3, test_data)
forest_3_acc_test <- mean(forest_3_pred_test == test_data$activities)
print(forest_3_acc_test)
