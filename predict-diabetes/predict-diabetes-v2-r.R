
install.packages("readxl")
install.packages("caret")
install.packages("yardstick")
install.packages("rpart")
install.packages("e1071")
install.packages("pROC")

# Importing libraries
library(readxl)
library(caret)
library(yardstick)
library(rpart)
library(e1071)
library(pROC)

# Loading the dataset
setwd("~/RProject/predict-diabetes")

train_data <- read.csv("train.csv")
# Converting the dataset into a dataframe
train_data <- as.data.frame(train_data)

# Ensuring RISK is a factor
train_data$type <- as.factor(train_data$type)
levels(train_data$type) <- c("1", "0")  # Ensure correct factor levels
train_data$type <- relevel(train_data$type, ref = "0")  # Set "0" as the reference level

head(train_data)
summary(train_data)
str(train_data)

set.seed(123)

cat("Missing value: \n")
print(colSums(is.na(train_data)))

col_num <- names(Filter(is.numeric, train_data))
cat_num <- names(Filter(is.character, train_data))


# Replace NA with median
for(col in col_num){
  media_val <- median(train_data[[col]], na.rm = TRUE)
  train_data[[col]][is.na(train_data[[col]])] <- media_val
}

# Check
print(colSums(is.na(train_data)))

# Scale numeric feactures(optional)
#train_data_scaled <- scale(train_data[])

# Creating the matrix 
train_index <- createDataPartition(train_data$type, p = .8, list = FALSE)

# Way-01: Cross-validation on TRAIN set (e.g., 10-fold CV)
ctrl <- trainControl(method = 'cv', number = 10) # 10-folds CV

# Train a model (e.g., logistic regression) with CV
model <- train(
  type ~ ., 
  data = train_data, 
  method = "glm", 
  family = "binomial",
  trControl = ctrl
)

# OR

# Way-02: Cross-validation on TRAIN set (e.g., 10-fold CV)
folde_indexes <- createFolds(train_data$type, k = 10)

# diaplay structure of dataset
str(train_data)


accuracy_list <- c()
accuracy_score_vec <- c()
precision_score_vec <- c()
recall_score_vec <- c()
f1_score_vec <- c()

plot(NULL, xlim = c(0, 1),ylim = c(0, 1), xlab = "False positive rate",
     ylab = "Tru positive rate", main = "ROC Curves for 10-Fold cross-validation")
abline(a = 0, b =1, lty = 2, col = "gray")

"
for(i in 1:10){
  # createing training and test sets
  test_indexes <- folde_indexes[[i]]
  train_set <- train_data[-test_indexes,]
  test_set <- train_data[test_indexes,]
  
  print(paste("Length of test_set$type:", length(test_set$type)))
  print(paste("Length of predicted_classes:", length(predicted_classes)))
  
  # train Naive Bayes
  model <- naiveBayes(type ~., data = train_set)
  
  # 
  probs <- predict(model, test_set, type = "raw")
  
  # Predict classes 
  predicted_classes <- predict(model, test_set)
  
  # =============== Calculateing metrics ==============
  # Calculate accuracy
  accuracy_score <- mean(predicted_classes == test_set$type)
  accuracy_list <- c(accuracy_list, accuracy_score)
  
  if('1' %in% colnames(probs)){
    prob_predicted_class <- probs[, '1']
  }else{
    stop('Positive class '1' not found.')
  }
  
  accuracy_score_vec[i] <- accuracy_vec(truth = test_set$type, estimate = predict_classes)
  
  
  
}"


# =============================================

for(i in 1:10){
  # creating training and test sets
  test_indexes <- folde_indexes[[i]]
  train_set <- train_data[-test_indexes,]
  test_set <- train_data[test_indexes,]
  
  # train Naive Bayes
  model <- naiveBayes(type ~., data = train_set)
  
  # Predict classes
  predicted_classes <- predict(model, test_set)
  
  # =============== Calculateing metrics ==============
  # Calculate accuracy
  accuracy_score <- mean(predicted_classes == test_set$type)
  accuracy_list <- c(accuracy_list, accuracy_score)
  
  if("1" %in% colnames(predict(model, test_set, type = "raw"))){ # Moved predict here for consistency
    prob_predicted_class <- predict(model, test_set, type = "raw")[, "1"]
  } else {
    stop("Positive class '1' not found.")
  }
  
  print(paste("Fold:", i))
  print(paste("Length of test_set$type:", length(test_set$type)))
  print(paste("Length of predicted_classes:", length(predicted_classes)))
  
  accuracy_score_vec[i] <- accuracy_vec(truth = test_set$type, estimate = predicted_classes)
  
  # calculating precision
  precision_score_vec[i] <- precision_vec(test_set$type, predicted_classes)
  
  # calculating recall
  recall_score_vec[i] <- recall_vec(test_set$type, predicted_classes)
  f1_score_vec[i] <- f_meas_vec(test_set$type, predicted_classes, beta = 1)
  
  # Computing the ROC cureve 
  roc_curve <- roc(test_set$type, prob_predicted_class, levels = c("1", "0"), direction = "<")
  
  # Plot roc curve for this fold
  plot.roc(roc_curve, col = rgb(0.2, 0.6, 0.8, alpha = 0.5), add = TRUE)
  
}

# Compute meac


# calc decision-tree
d_tree_model <- rpart(type ~., data = train_data, method = "class")

# Accuracy
cat("Average accuracy accross 10 folds: ", mean(accuracy_list), "\n")
cat("Accuracy score: ", mean(accuracy_score_vec), "\n")

cat("Precision score: ", mean(precision_score_vec), "\n")
cat("Recall score: ", mean(precision_score_vec), "\n")
cat("F-1 score: ", mean(f1_score_vec), "\n")

# plot
plot(d_tree_model)
text(d_tree_model, use.n = TRUE, cex = 0.8)



