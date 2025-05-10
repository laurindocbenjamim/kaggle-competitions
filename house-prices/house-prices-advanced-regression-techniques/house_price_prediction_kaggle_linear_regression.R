
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



# Loading the dataset
setwd("~/Documents/RProjects/kaggle competitions/house-prices/house-prices-advanced-regression-techniques")
#sample_submission <- read_csv("sample_submission.csv")
#train_data <- read_csv("train.csv")
#test_data <- read_csv("test_set.csv")

# Load the data
dataset <- fread("preprocessed_cleaned_data.csv", data.table = FALSE)

# Examine data structure
str(dataset)
summary(dataset)
head(dataset)

# check for missing values
sum(is.na(dataset))

# Since the data is already preprocessed, 
# we can proceed with building our linear regression model

# set seed for reproducibility
set.seed(123)

# Create indices for train/test split
train_indices <- sample(1:nrow(dataset), 0.8 * nrow(dataset))

# Split the data
train_data <- dataset[train_indices, ]
test_data <- dataset[-train_indices, ]

# Build the linear regression model
lm_model <- lm(SalePrice ~., data = train_data)

# view model summary
summary(lm_model)

# Now we can evaluate the performance in both train an test sets

# Predictions on training set
train_prediction <- predict(lm_model, train_data)

# Predictions on testing set
test_prediction <- predict(lm_model, test_data)

# Calculate RMSE
train_rmse <- sqrt(mean((train_data$SalePrice - train_prediction)^2))
test_rmse <- sqrt(mean((test_data$SalePrice - test_prediction)^2))

# Calculate R-squared
train_rsq <- summary(lm_model)$r.squared
test_rsq <- 1 - (sum((test_data$SalePrice - test_prediction)^2) / 
                   sum((test_data$SalePrice - mean(test_data$SalePrice))^2))

cat("Training RMSE:", train_rmse, "\n")
cat("Test RMSE:", test_rmse, "\n")
cat("Training R-squared:", train_rsq, "\n")
cat("Test R-squared:", test_rsq, "\n")


# check for multicollinearity using Variance Inflation Factors (VIF)

# Install and load car package if needed
if (!require(car)) install.packages("car")
library(car)

# calculate VIF
vif_values <- vif(lm_model)

# View VIF values
print(vif_values)

# Diagnostic plots
par(mfrow = c(2,2))
print(lm_model)

# 

