
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



# Loading the dataset
setwd("~/Documents/RProjects/kaggle competitions/house-prices/house-prices-advanced-regression-techniques")
#sample_submission <- read_csv("sample_submission.csv")
#train_data <- read_csv("train.csv")
#test_data <- read_csv("test_set.csv")

sample_submission <- fread("sample_submission.csv")
train_data <- fread("train.csv")
test_data <- fread("test.csv")

# converting the the dataset to Dataframe
train_df <- as.data.frame(train_data)
test_df <- as.data.frame(test_data)

# Explore the data structure for train_data
str(train_df)
summary(train_df)
head(train_df)

# Explore the data structure for test_data
str(test_df)
summary(test_df)
head(test_df)


# Preprocessing of train_data
# Checking for missing values in entire dataset
check_missing <- function(data){
  missing_sum <- colSums(is.na(data) | data == "" | data == "NA" | data == "N/A")
  if(sum(missing_sum) > 0){
    cat("Missing values fiund:\n")
    print(missing_sum[missing_sum > 0])
    return(TRUE)
  }else{
    cat("No missing values found:\n")
    return(FALSE)
  }
}

# Handle missing values based on type
handle_missing_values <- function(data){
  # Fo numeric values: impute with median
  num_cols <- sapply(data, is.numeric)
  if(any(num_cols)){
    for(col in names(data)[num_cols]){
      if(any(is.na(data[[col]]))){
        data[[col]][is.na(data[[col]])] <- median(data[[col]], na.rm = TRUE)
      }
    }
  }
  
  # For factor columns: impute with mode
  factor_cols <- sapply(data, is.factor)
  if(any(factor_cols)){
    for(col in names(data)[factor_cols]){
      if(any(is.na(data[[col]]))){
        model_val <- names(sort(table(data[[col]]), decrising = TRUE))[1]
        data[[col]][is.na(data[[col]])] <- model_val
      }
    }
  }
  
  # For character column: Impute with "unknow"
  char_cols <- sapply(data, is.character)
  if(any(char_cols)){
    for(col in names(data)[char_cols]){
      if(any(is.na(data[[col]]) | data[[col]] =="")){
        data[[col]][is.na(data[[col]]) | data[[col]] ==""] <- "Unknown"
      }
    }
  }
  return(data)
}

train_data <- NULL
test_data <- NULL
# Handling missin values: train_data
if (check_missing(train_df)) {
  train_data <- handle_missing_values(train_df)
  cat("Missing values handled successfully\n")
}

# Handling missin values: test_data
if (check_missing(test_df)) {
  test_data <- handle_missing_values(test_df)
  cat("Missing values handled successfully\n")
}

replace_nan <- function(df) {
  for (col in names(df)) {
    if (is.numeric(df[[col]])) {
      df[[col]][is.nan(df[[col]])] <- median(df[[col]], na.rm = TRUE)
    }
  }
  return(df)
}

# Replace NaNa
train_data <- replace_nan(train_data)

print(train_data)

# 
# Check missing value
if (check_missing(train_data)) {
  cat("In train data\n")
}else if (check_missing(test_data)) {
  cat("In test data\n")
}

# Hadling Outliers
# Detect and handle outliers using IQR
handle_outliers <- function(data, cols = NULL, remove = FALSE, cap = TRUE){
  if (is.null(cols)) {
    cols <- names(data)[sapply(data, is.numeric)]
  }
  
  for (col in cols) {
    if (is.numeric(data[[col]])) {
      # Calculate bounds using a more robust method
      qnt <- quantile(data[[col]], probs = c(0.25, 0.75), na.rm = TRUE)
      iqr <- IQR(data[[col]], na.rm = TRUE)
      lower <- qnt[1] - 1.5 * iqr
      upper <- qnt[2] + 1.5 * iqr
      
      # Safely identify outliers
      outliers <- which(data[[col]] < lower | data[[col]] > upper)
      
      if (length(outliers) > 0) {
        if (remove) {
          # Remove outliers completely
          data <- data[-outliers, ]
        } else if (cap) {
          # Cap outliers at boundaries - SAFE VERSION
          data[outliers, col] <- ifelse(data[outliers, col] < lower, lower, 
                                        ifelse(data[outliers, col] > upper, upper, 
                                               data[outliers, col]))
        } else {
          cat("Outliers detected in", col, ":", length(outliers), "\n")
        }
      }
    }
  }
  return(data)
}

# Handle outliers: robust way
handle_outliers_v2 <- function(data, cols = NULL, method = c("cap", "remove", "none")) {
  method <- match.arg(method)
  if (is.null(cols)) {
    cols <- names(data)[sapply(data, is.numeric)]
  }
  
  for (col in cols) {
    if (is.numeric(data[[col]])) {
      x <- data[[col]]
      qnt <- quantile(x, probs = c(0.25, 0.75), na.rm = TRUE)
      iqr <- IQR(x, na.rm = TRUE)
      lower <- qnt[1] - 1.5 * iqr
      upper <- qnt[2] + 1.5 * iqr
      
      if (method == "cap") {
        x[x < lower & !is.na(x)] <- lower
        x[x > upper & !is.na(x)] <- upper
        data[[col]] <- x
      } else if (method == "remove") {
        data <- data[!(x < lower | x > upper) | is.na(x), ]
      }
    }
  }
  return(data)
}

# Handle outliers in numeric columns
#numeric_cols <- sapply(train_data[, setdiff(names(train_data), "Id")], is.numeric)
#numeric_df <- train_data[, names(train_data) %in% names(numeric_cols[numeric_cols])]
#numeric_col_names <- names(train_data)[-1][sapply(train_data, is.numeric)]
numeric_col_names_train <- names(train_data)[sapply(train_data, is.numeric) & names(train_data) != "Id"]
numeric_col_names_test <- names(test_data)[sapply(test_data, is.numeric) & names(test_data) != "Id"]

missing_num <- colSums(is.na(train_data[, numeric_col_names_train]))
print(missing_num[missing_num > 0])

total_na <- sum(is.na(train_data))
total_nan <- sum(sapply(train_data, function(x) sum(is.nan(as.numeric(x)))))
print(total_na)
print(total_nan)

#print(numeric_col_names)

# Option 1: Cap outliers (recommended)
train_data_clean <- handle_outliers(train_data, cols = numeric_col_names_train, cap = TRUE)

test_data_clean <- handle_outliers(test_data, cols = numeric_col_names_test, cap = TRUE)

# Option 2: Remove outliers
#clean_data <- handle_outliers(train_data, cols = numeric_col_names, remove = TRUE)

# Option 3: Just detect outliers
#clean_data <- handle_outliers(train_data, cols = numeric_col_names, cap = FALSE, remove = FALSE)





# Normalize numeric features
train_data_clean[numeric_col_names_train] <- scale(train_data_clean[numeric_col_names_train])
test_data_clean[numeric_col_names_test] <- scale(test_data_clean[numeric_col_names_test])
print(clean_df)
print(test_data_clean)


# Visualize SalePrice distribution (often right-skewed)
ggplot(train_data_clean, aes(x = SalePrice)) + geom_histogram(bins = 50) + theme_minimal()

train_data_clean$SalesPriceLog <- log(train_data_clean$SalesPrice + 1)
train_data_clean$SalePrice_Transform <- sqrt(train_data_clean$SalePrice)

# Check for NA
sum(is.na(train_data_clean$SalePrice))
# Check for non-positive values
sum(train_data_clean$SalePrice <= 0, na.rm = TRUE)
# Summary statistics
summary(train_data_clean$SalePrice)



#Since SalePrice is skewed, consider a log transformation for modeling
# Basic 
signed_log <- function(x) {
  sign(x) * log(abs(x) + 1)
}

# Apply to a specific column
#train_data_clean$SalesPrice_log <- safe_log(train_data_clean$SalePrice)

#  Or apply to all numeric columns
numeric_cols <- sapply(train_data_clean, is.numeric)
train_data_clean[numeric_cols] <- lapply(train_data_clean[numeric_cols], safe_log)

train_data_clean$SalesPriceLog <- log(train_data_clean$SalesPrice + 1)


# Square Ttransformation

# Square root transformation
train_data_clean$SalePrice_Transform <- sqrt(train_data_clean$SalePrice)

# Cube root transformation (handles negatives)
train_data_clean$SalePrice_Negative <- sign(train_data_clean$SalePrice) * abs(train_data_clean$SalePrice)^(1/3)


# create 
set.seed(123)
train_index <- sample(1:nrow(train_data_clean), 0.8 * nrow(train_data_clean))
print(train_index)



