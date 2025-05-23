install.packages("caret")
install.packages("yardstick")
install.packages("e1071")
install.packages("pRoc")
install.packages("FNN")
install.packages("dplyr")
install.packages("ggplot2")
install.packages("lattice")
install.packages("splitTools")
install.packages("randomForest")


library(caret)
library(yardstick)
library(e1071)
library(pROC)
library(ggplot2)
library(lattice)
library(splitTools)
library(FNN)
library(dplyr)
library(randomForest)

setwd("~/Documents/RProjects/kaggle-competitions/predict-if-is-sick")

datatest <- read.csv("test.csv")
datatrain <- read.csv("train.csv")

# ======================== PREPROCESSING ===============================
## Data train set
str(datatrain)
summary(datatrain)
head(datatrain)

# Defining categorical variables
catvariables <- c("person_sex", "has_hypertension", "has_heart_disease", "marital_status", "employment_type", "residence_category", "smoking_status")
#catvariables_test <- c("person_sex", "has_hypertension", "has_heart_disease", "marital_status", "employment_type", "residence_category", "smoking_status")
numvariables <- c("person_age", "glucose_level_avg", "body_mass_index")

datatrain[, catvariables] <- lapply(datatrain[, catvariables], as.factor)
datatest[, catvariables] <- lapply(datatest[, catvariables], as.factor)

datatrain$sick <- as.factor(datatrain$sick)

# remove rows with missing values
datatrain <- na.omit(datatrain)
datatest <- na.omit(datatest)


summary(datatrain)
str(datatrain)

# check fo missing values
colSums(is.na(datatrain))

set.seed(123)

# Make sure all categorical columns have the same levels
for (col in catvariables) {
  datatrain[[col]] <- factor(datatrain[[col]])
  datatest[[col]] <- factor(datatest[[col]], levels = levels(datatrain[[col]]))
}

#for(col in catvariables){
#  levels(datatest[[col]]) <- levels(datatrain[[col]])
#}

# since we already have two data files, for training an test, 
# we do not need create 10 folds cross-validation

#==========================================

# Setup a training control
ctrl <- trainControl(method = "none")

# Define models
models <- list()

# 1. Logistic regression
models$logistic_reg <- train(sick ~. - id, data = datatrain, method = "glm", family = "binomial", trControl = ctrl)

# 2. Decision tree
models$tree <- train(sick ~. - id, data = datatrain, method = "rpart", trControl = ctrl)

# 3. K-NN
models$knn <- train(sick ~. - id, data = datatrain, method = "knn", trControl = ctrl)

# 4. SVM 
models$svm <- train(sick ~. - id, data = datatrain, method = "svmRadial", trControl = ctrl)

# 5. Random forest
models$rf <- train(sick ~. - id, data = datatrain, method = "rf", trControl = ctrl)

# 6. XGBoost (requires converting to numeric matrix)
xgb_data <- model.matrix(sick ~ . - id, data = datatrain)
xgb_data_labels <- datatrain$sick
models$xgb <- train(
  x = xgb_data, 
  y = xgb_data_labels, 
  method = "xgbTree",
  trControl = ctrl,
  tuneGrid = data.frame(nrounds = 100, max_depth = 6, eta = 0.3, gamma = 0,
                        colsample_bytree = 1, min_child_weight = 1, subsample = 1)
)

# Prepare datatest for XGBoost (same features)
xgb_test <- model.matrix(~ . - id, data = datatest)

sum(is.na(datatest))  # Should be 0
which(!complete.cases(datatest))  # Identify the row with NA

# You can remove rows from datatest with invalid factor levels
datatest <- droplevels(datatest)

# Remove incomplete rows
datatest_clean <- datatest[complete.cases(datatest), ]


sum(is.na(datatest_clean))  # Should be 0

# Predict on dataset with all models
results <- data.frame(id = datatest_clean$id)

results$logistic_reg <- predict(models$logistic_reg, newdata = datatest_clean)
results$tree     <- predict(models$tree, newdata = datatest_clean)
results$knn      <- predict(models$knn, newdata = datatest_clean)
results$svm      <- predict(models$svm, newdata = datatest_clean)
results$rf       <- predict(models$rf, newdata = datatest_clean)
results$xgb      <- predict(models$xgb, newdata = model.matrix(~ . - id, data = datatest_clean))

# Save results
write.csv(results, "multiple_model_predictions.csv", row.names = FALSE)


# =======================================================
# ================== VISUALIZE THE DATA

print("Predictions from each model:")
print(head(results))  # Show first few predictions


library(ggplot2)
library(reshape2)

# Melt the results for ggplot
results_melted <- melt(results, id.vars = "id", variable.name = "Model", value.name = "Prediction")

# Convert to factor
results_melted$Prediction <- as.factor(results_melted$Prediction)

# Bar plot: number of predicted "sick" (1) vs "not sick" (0) per model
ggplot(results_melted, aes(x = Model, fill = Prediction)) +
  geom_bar(position = "dodge") +
  labs(title = "Predicted Sick vs Not Sick by Model", x = "Model", y = "Count") +
  theme_minimal()

# Save the Plot as PNG
ggsave("prediction_summary_plot.png", width = 10, height = 6)






# predict and get probabilities
pred_probs <- predict(model, newdata = datatest, type = "prob")

# saving the predictions
pred_results <- data.frame(id = datatest$id, sick_predicted = predictions_class_label, row.names = NULL)
write.csv(pred_results, "predictions_class_labels.csv", row.names = FALSE)

# Saving the probabilities 
prob_results <- data.frame(id = datatest$id, prob_0 = pred_probs[, 1], prob_1 = pred_probs[, 2])
write.csv(prob_results, "probabilities_of_sick", row.names = FALSE)

predicted <- read.csv("predictions_class_labels.csv")
str(predicted)

# ====================================================

