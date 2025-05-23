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

# Check class distribution
table(datatrain$sick)
prop.table(table(datatrain$sick)) * 100

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

# Handle missing values
datatrain <- na.omit(datatrain)
datatest <- na.omit(datatest)
datatest <- datatest[complete.cases(datatest), ]


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

# Normalize/standardize features for K-NN and SVM models
prepProc <- preProcess(datatrain[, - which(names(datatrain) %in% c("sick", "id"))], method = c("center", "scale"))
train_scaled <- predict(prepProc, datatrain)
test_scaled <- predict(prepProc, datatest)
#==========================================

# Using crsoss validation and hiperparameters
ctrl <- trainControl(method = "cv", number = 10)

# Define models
models <- list()

# 1. Logistic regression
models$logistic_reg <- train(sick ~. - id, data = datatrain, method = "glm", family = "binomial", trControl = ctrl)

# 2. Decision tree
models$tree <- train(sick ~. - id, data = datatrain, method = "rpart", 
                     trControl = ctrl, tuneLength = 10)

# 3. K-NN
models$knn <- train(sick ~. - id, data = train_scaled, method = "knn", 
                    trControl = ctrl, tuneLength = 10)

# 4. SVM 
models$svm <- train(sick ~. - id, data = train_scaled, method = "svmRadial", 
                    trControl = ctrl, , tuneLength = 10)

# 5. Random forest
models$rf <- train(sick ~. - id, data = datatrain, method = "rf", 
                   trControl = ctrl, , tuneLength = 5, ntree = 200)

# 6. XGBoost (requires converting to numeric matrix)
xgb_data <- model.matrix(sick ~ . - id, data = datatrain)

xgb_labels <- datatrain$sick

models$xgb <- train(
  x = xgb_data, 
  y = xgb_labels, 
  method = "xgbTree",
  trControl = ctrl,
  tuneLength = 10,
  tuneGrid = data.frame(nrounds = 100, max_depth = 6, eta = 0.3, gamma = 0,
                        colsample_bytree = 1, min_child_weight = 1, subsample = 1)
)


# get the variable importance of the training model
importance_rf <- varImp(models$rf)
print(importance_rf)
plot(importance_rf)

# Prepare datatest for XGBoost (same features)
xgb_test <- model.matrix(~ . - id, data = datatest)

sum(is.na(datatest))  # Should be 0
which(!complete.cases(datatest))  # Identify the row with NA

# You can remove rows from datatest with invalid factor levels
datatest <- droplevels(datatest)
test_scaled <- droplevels(test_scaled)

# Remove incomplete rows
datatest_clean <- datatest[complete.cases(datatest), ]
test_scaled <- test_scaled[complete.cases(test_scaled), ]

sum(is.na(datatest_clean))  # Should be 0


# =============================   Evaluation Process
# Predict on dataset with all models
results <- data.frame(id = datatest_clean$id)
results_svm_knn <- data.frame(id = test_scaled$id)

results$logistic_reg <- predict(models$logistic_reg, newdata = datatest_clean)
results$tree     <- predict(models$tree, newdata = datatest_clean)

# predict SVM and KNN normalized and standardiseds
results_svm_knn$knn      <- predict(models$knn, newdata = test_scaled)
results$knn <- results_svm_knn$knn
results_svm_knn$svm      <- predict(models$svm, newdata = test_scaled)
results$svm <- results_svm_knn$svm

results$rf       <- predict(models$rf, newdata = datatest_clean)
results$xgb      <- predict(models$xgb, newdata = model.matrix(~ . - id, data = datatest_clean))

# Feature Importance (Optional but Powerful)
varImp(results$xgb)

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

