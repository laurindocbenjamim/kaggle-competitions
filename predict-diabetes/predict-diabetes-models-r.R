
install.packages("readxl")
install.packages("caret")
install.packages("yardstick")
install.packages("rpart")
install.packages("e1071")
install.packages("pROC")
install.packages("tidyverse")
install.packages("xgboost")
install.packages("randomForest")

# Importing libraries
library(readxl)
library(caret)
library(yardstick)
library(rpart)
library(e1071)
library(pROC)
# Load required libraries
library(tidyverse)

library(randomForest)
library(xgboost)

# Loading the dataset
setwd("~/RProject/predict-diabetes")



# 1. Load the datasets
train_data <- read.csv("train.csv")
test_data <- read.csv("test.csv")

# 2. Preprocess the data (handle missing values if any)
train_data <- train_data %>%
  mutate(across(where(is.numeric), ~ifelse(is.na(.), median(., na.rm = TRUE), .)))

test_data <- test_data %>%
  mutate(across(where(is.numeric), ~ifelse(is.na(.), median(., na.rm = TRUE), .)))

# 3. Prepare training data (remove 'id' column as it's not a feature)
train_features <- train_data %>% select(-id, -type)
train_target <- as.factor(train_data$type)

# 4. Prepare test data (keep 'id' for final output)
test_features <- test_data %>% select(-id)
test_ids <- test_data$id

# 5. Define models to train
models <- list(
  logistic = list(method = "glm", family = "binomial"),
  random_forest = list(method = "rf"),
  xgboost = list(method = "xgbTree")
)

# 6. Train models and make predictions
predictions <- list()

for (model_name in names(models)) {
  # Train model
  set.seed(123)
  model <- train(
    x = train_features,
    y = train_target,
    method = models[[model_name]]$method,
    family = if (!is.null(models[[model_name]]$family)) models[[model_name]]$family else NULL,
    trControl = trainControl(method = "cv", number = 5)
  )
  
  # Make predictions
  preds <- predict(model, newdata = test_features)
  
  # Store predictions with IDs
  predictions[[model_name]] <- data.frame(
    id = test_ids,
    type = as.numeric(as.character(preds))  # Convert factor to numeric (0/1)
  )
  
  # Save predictions to CSV
  write.csv(predictions[[model_name]], 
            file = paste0(model_name, "_predictions.csv"), 
            row.names = FALSE)
}

# 7. (Optional) Ensemble prediction (average probabilities)
if ("logistic" %in% names(predictions) && 
    "random_forest" %in% names(predictions) && 
    "xgboost" %in% names(predictions)) {
  
  ensemble_preds <- data.frame(
    id = test_ids,
    type = round((predictions$logistic$type + 
                    predictions$random_forest$type + 
                    predictions$xgboost$type) / 3)
  )
  
  write.csv(ensemble_preds, "ensemble_predictions.csv", row.names = FALSE)
}

# Load final predictions
final_preds <- read.csv("ensemble_predictions.csv")



# Create distribution plot of test predictions
ggplot(final_preds, aes(x = factor(type), fill = factor(type))) +
  geom_bar(fill = c("blue", "red"), alpha = 0.7) +
  labs(title = "Distribution of Predicted Diabetes Cases in Test Set",
       x = "Diabetes Prediction (0 = No, 1 = Yes)",
       y = "Count") +
  theme_minimal()

