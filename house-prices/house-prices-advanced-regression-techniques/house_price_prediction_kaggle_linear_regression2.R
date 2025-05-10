
# Installing libraries
install.packages("systemfonts")

# Install basic system dependencies first
install.packages(c("httr", "xml2", "curl", "gargle"), dependencies = TRUE)

install.packages(c("tidyverse", "caret", "readr", "Metrics", "glmnet", "randomForest", "xgboost"))
install.packages("randomForest") 
install.packages("xgboost")
install.packages("data.table") 
install.packages("glmnet")
install.packages("Metrics")
install.packages("tidyverse")
install.packages("textshaping")
install.packages("ragg")

install.packages("corrplot")
#install.packages("nloptr")
install.packages("nloptr", configure.args = "--with-nlopt=/usr")
install.packages("mice")

# to save plots in pdf or PNG
install.packages("patchwork")
install.packages("gridExtra")
#install.packages("reshape2")
install.packages("dplyr")

library(reshape2)

library(data.table)
library(ggplot2)
library(systemfonts)
library(textshaping)
library(mice)
library(ragg)
library(tidyverse)
library(caret)
library(readr)
library(Metrics)
library(glmnet)
library(randomForest)
library(data.table)

library(patchwork)
library(gridExtra)
library(corrplot)
library(dplyr)
library(caret)
library(glmnet)
library(randomForest)
library(xgboost)
library(Metrics)
library(ggplot2)



# Loading the dataset
setwd("~/Documents/RProjects/kaggle competitions/house-prices/house-prices-advanced-regression-techniques")
#sample_submission <- read_csv("sample_submission.csv")
#train_data <- read_csv("train.csv")
#test_data <- read_csv("test_set.csv")

# Load the data
data <- fread("preprocessed_cleaned_data.csv", data.table = FALSE)


# =============================================
# COMPREHENSIVE HOUSING PRICE PREDICTION ANALYSIS
# =============================================



# ----------------------
# 1. DATA PREPARATION
# ----------------------

# Load dataset
#data <- read.csv("preprocessed_cleaned_data.csv")

# Check data structure
cat("Data dimensions:", dim(data), "\n")
cat("First few rows:\n")
print(head(data))

# Split data into training and test sets
set.seed(123)
train_index <- createDataPartition(data$SalePrice, p = 0.8, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

cat("Training samples:", nrow(train_data), "\n")
cat("Test samples:", nrow(test_data), "\n")

# ----------------------
# 2. EVALUATION METRICS FUNCTION
# ----------------------

calculate_metrics <- function(predictions, actual) {
  metrics <- list(
    RMSE = rmse(actual, predictions),
    MAE = mae(actual, predictions),
    MAPE = mape(actual, predictions),
    R2 = cor(actual, predictions)^2,
    Adjusted_R2 = 1 - ((1 - cor(actual, predictions)^2) * (length(actual) - 1) / 
                         (length(actual) - ncol(train_data) - 1)),
                       RMSLE = rmsle(actual, predictions)
    )
    return(metrics)
}

# ----------------------
# 3. MODEL TRAINING WITH CROSS-VALIDATION
# ----------------------

# Set up 10-fold cross-validation
ctrl <- trainControl(method = "cv", number = 10, savePredictions = TRUE)

# 3.1 Linear Regression
cat("\nTraining Linear Regression...\n")
lm_model <- train(SalePrice ~ ., 
                  data = train_data, 
                  method = "lm",
                  trControl = ctrl)

# 3.2 Ridge Regression
cat("\nTraining Ridge Regression...\n")
ridge_model <- train(SalePrice ~ .,
                     data = train_data,
                     method = "glmnet",
                     tuneGrid = expand.grid(alpha = 0, lambda = seq(0.001, 1, length = 10)),
                     trControl = ctrl)

# 3.3 Lasso Regression
cat("\nTraining Lasso Regression...\n")
lasso_model <- train(SalePrice ~ .,
                     data = train_data,
                     method = "glmnet",
                     tuneGrid = expand.grid(alpha = 1, lambda = seq(0.001, 0.1, length = 10)),
                     trControl = ctrl)

# 3.4 Random Forest
cat("\nTraining Random Forest...\n")
rf_model <- train(SalePrice ~ .,
                  data = train_data,
                  method = "rf",
                  trControl = ctrl,
                  tuneLength = 3)

# 3.5 XGBoost
cat("\nTraining XGBoost...\n")
xgb_model <- train(SalePrice ~ .,
                   data = train_data,
                   method = "xgbTree",
                   trControl = ctrl,
                   tuneLength = 3)

  # ----------------------
# 4. MODEL EVALUATION
# ----------------------

# 4.1 Cross-Validation Results
cat("\nCross-Validation Results:\n")
cv_results <- resamples(list(
  Linear = lm_model,
  Ridge = ridge_model,
  Lasso = lasso_model,
  RandomForest = rf_model,
  XGBoost = xgb_model
))

print(summary(cv_results))

# Visualize CV results
dotplot(cv_results, metric = "RMSE")
dotplot(cv_results, metric = "Rsquared")

# 4.2 Test Set Evaluation
models <- list(
  "Linear Regression" = lm_model,
  "Ridge Regression" = ridge_model,
  "Lasso Regression" = lasso_model,
  "Random Forest" = rf_model,
  "XGBoost" = xgb_model
)

test_metrics <- data.frame(matrix(ncol = 6, nrow = 0))
colnames(test_metrics) <- c("Model", "RMSE", "MAE", "MAPE", "R2", "RMSLE")

for (model_name in names(models)) {
  model <- models[[model_name]]
  
  # Make predictions
  if (inherits(model, "train")) {
    pred <- predict(model, test_data)
  } else {
    pred <- predict(model, newx = as.matrix(test_data[-which(names(test_data) == "SalePrice")]))
  }
  
  # Calculate metrics
  metrics <- calculate_metrics(pred, test_data$SalePrice)
  
  # Store results
  test_metrics <- rbind(test_metrics, data.frame(
    Model = model_name,
    RMSE = metrics$RMSE,
    MAE = metrics$MAE,
    MAPE = metrics$MAPE,
    R2 = metrics$R2,
    RMSLE = metrics$RMSLE
  ))
}

cat("\nTest Set Performance:\n")
print(test_metrics)
  
  # 4.3 Residual Analysis for Best Model
best_model_name <- test_metrics$Model[which.min(test_metrics$RMSE)]
best_model <- models[[best_model_name]]
  
cat("\nBest Model:", best_model_name, "\n")
  
if (inherits(best_model, "train")) {
  predictions <- predict(best_model, test_data)
  residuals <- test_data$SalePrice - predictions
    
    # Residual plots
  par(mfrow = c(2, 2))
  plot(predictions, residuals, main = "Residuals vs Fitted")
  abline(h = 0, col = "red")
  qqnorm(residuals)
  qqline(residuals, col = "red")
  hist(residuals, main = "Histogram of Residuals")
  plot(density(residuals), main = "Density of Residuals")
}
  
  # ----------------------
  # 5. FEATURE IMPORTANCE
  # ----------------------
  
cat("\nFeature Importance Analysis:\n")
if (inherits(best_model, "train")) {
    # For tree-based models
    if (best_model_name %in% c("Random Forest", "XGBoost")) {
      imp <- varImp(best_model)
      print(imp)
      
      # Plot feature importance
      imp_df <- data.frame(Feature = rownames(imp$importance), Importance = imp$importance$Overall)
      imp_df <- imp_df[order(-imp_df$Importance), ]
      imp_df <- head(imp_df, 20)
      
      ggplot(imp_df, aes(x = reorder(Feature, Importance), y = Importance)) +
        geom_bar(stat = "identity") +
        coord_flip() +
        ggtitle(paste("Top 20 Important Features -", best_model_name)) +
        xlab("Features") +
        ylab("Importance")
    }
    
    # For regularized regression
if (best_model_name %in% c("Ridge Regression", "Lasso Regression")) {
  coefs <- coef(best_model$finalModel, s = best_model$bestTune$lambda)
  coefs <- data.frame(Feature = rownames(coefs), Coefficient = coefs[,1])
      coefs <- coefs[order(-abs(coefs$Coefficient)), ]
      coefs <- coefs[coefs$Coefficient != 0 & coefs$Feature != "(Intercept)", ]
      coefs <- head(coefs, 20)
      
      print(coefs)
      
      ggplot(coefs, aes(x = reorder(Feature, Coefficient), y = Coefficient)) +
        geom_bar(stat = "identity") +
        coord_flip() +
        ggtitle(paste("Top 20 Important Features -", best_model_name)) +
        xlab("Features") +
        ylab("Coefficient")
    }
  }
  
  # ----------------------
  # 6. MODEL DEPLOYMENT
  # ----------------------
  
# Save best model
saveRDS(best_model, "best_housing_model.rds")
  
cat("\nAnalysis complete. Best model saved as 'best_housing_model.rds'\n")
