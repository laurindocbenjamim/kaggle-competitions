
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



# Loading the dataset
setwd("~/Documents/RProjects/kaggle competitions/house-prices/house-prices-advanced-regression-techniques")
#sample_submission <- read_csv("sample_submission.csv")
#train_data <- read_csv("train.csv")
#test_data <- read_csv("test_set.csv")

sample_submission <- fread("sample_submission.csv")

# Load the data
train <- fread("train.csv", data.table = FALSE)
test <- fread("test.csv", data.table = FALSE)

print(train$SalePrice)

# Combine train and test for consistent processing
test$SalePrice <- NA
full_data <- rbind(train, test)

print(full_data$SalePrice)
# Identify categorical and numeric columns
cat_cols <- names(Filter(is.character, full_data))
num_cols <- names(Filter(is.numeric, full_data))

# Handle missing values
# For categorical columns: replace NA with mode
for (col in cat_cols) {
  mode_val <- names(sort(table(full_data[[col]]), decreasing = TRUE))[1]
  full_data[[col]][is.na(full_data[[col]])] <- mode_val
}

# For numeric columns: replace NA with median
for (col in num_cols) {
  median_val <- median(full_data[[col]], na.rm = TRUE)
  full_data[[col]][is.na(full_data[[col]])] <- median_val
}
print(full_data$SalePrice)
# Outlier handling (IQR method)
handle_outliers <- function(data, exclude_cols = "SalePrice"){
  num_cols <- setdiff(num_cols, exclude_cols)  # Remove excluded columns
  for (col in num_cols) {
    Q1 <- quantile(data[[col]], 0.25, na.rm = TRUE)
    Q3 <- quantile(data[[col]], 0.75, na.rm = TRUE)
    IQR_val <- Q3 - Q1
    lower <- Q1 - 1.5 * IQR_val
    upper <- Q3 + 1.5 * IQR_val
    
    median_val <- median(data[[col]], na.rm = TRUE)
    data[[col]][data[[col]] < lower | data[[col]] > upper] <- median_val
  }
  return(data)
}

print(full_data$SalePrice)
# Process features
features_data <- handle_outliers(full_data[, !names(full_data) %in% "SalePrice"])
print(features_data)

# Keep original SalePrice
data_cleaned <- cbind(features_data, SalePrice = full_data$SalePrice)

print(data_cleaned$SalePrice)

# Re-encode categorical variables
n_unique <- sapply(data_cleaned[cat_cols], function(x) length(unique(x)))
low_card <- names(n_unique[n_unique <= 10])
high_card <- names(n_unique[n_unique > 10])

# One-hot encoding for low-cardinality
dummies <- dummyVars(~ ., data = data_cleaned[low_card], fullRank = TRUE)
onehot_encoded <- predict(dummies, newdata = full_data[low_card])
onehot_encoded <- as.data.frame(onehot_encoded)

# Frequency encoding for high-cardinality
freq_encode <- function(col) {
  freq <- table(col)
  return(as.numeric(freq[col]))
}
freq_encoded <- lapply(data_cleaned[high_card], freq_encode)
freq_encoded <- as.data.frame(freq_encoded)
colnames(freq_encoded) <- paste0("freq_", high_card)

# Drop original categorical columns
data_cleaned_cat_drop <- data_cleaned[, !(names(data_cleaned) %in% cat_cols)]

# Combine all features
data_cleaned_cat_drop <- cbind(data_cleaned_cat_drop, onehot_encoded, freq_encoded)

print(data_cleaned_cat_drop$SalePrice)
# Split back into train and test
#final_train <- data_cleaned_cat_drop[!is.na(data_cleaned_cat_drop$SalePrice), ]
#final_test <- data_cleaned_cat_drop[is.na(data_cleaned_cat_drop$SalePrice), ]

print(data_cleaned_cat_drop$SalePrice)

# Find constant attributs or with zero standard deviation 
constant_cols <- sapply(data_cleaned_cat_drop, function(x) sd(x, na.rm = TRUE) == 0)

# Save the processed datasets
write.csv(data_cleaned_cat_drop, "train_processed.csv", row.names = FALSE)
#write.csv(final_test, "test_processed.csv", row.names = FALSE)


###

# Load required libraries
library(e1071)

# Load data
#data <- fread("train_processed.csv", data.table = FALSE)

# Load the data
#house_data <- read.csv("train_processed.csv")
house_data <- data_cleaned_cat_drop

# Check structure
str(house_data)
dim(house_data)
summary(house_data)

# 1. First ensure SalePrice exists and is numeric
if(!"SalePrice" %in% names(house_data)) {
  stop("SalePrice column not found in dataset")
}else{
  print(house_data$SalePrice)
}

house_data$SalePrice <- as.numeric(house_data$SalePrice)

# 2. Remove rows with NA SalePrice
house_data <- house_data[!is.na(house_data$SalePrice), ]

# 3. Remove ID column if exists
if("Id" %in% names(house_data)) house_data$Id <- NULL

# 4. Separate target and predictors
target <- house_data$SalePrice
predictors <- house_data[, !names(house_data) %in% "SalePrice"]

# 5. Near-zero variance filtering (only on predictors)
library(caret)
nzv <- nearZeroVar(predictors)
if(length(nzv) > 0) {
  predictors <- predictors[, -nzv]
}

# 6. Highly correlated features (only on numeric predictors)
numeric_predictors <- predictors[, sapply(predictors, is.numeric)]
if(ncol(numeric_predictors) > 1) {
  cor_matrix <- cor(numeric_predictors, use = "complete.obs")
  high_cor <- findCorrelation(cor_matrix, cutoff = 0.9)
  if(length(high_cor) > 0) {
    numeric_predictors <- numeric_predictors[, -high_cor]
  }
  # Update predictors with filtered numeric columns
  predictors[, names(numeric_predictors)] <- numeric_predictors
}

# 7. Normalization and transformation
numeric_cols <- sapply(predictors, is.numeric)
binary_cols <- sapply(predictors, function(x) all(x %in% c(0, 1, NA)))
numeric_cols <- numeric_cols & !binary_cols

# Apply transformations to numeric predictors
for(col in names(predictors)[numeric_cols]) {
  skew <- e1071::skewness(predictors[[col]], na.rm = TRUE)
  if(abs(skew) > 0.75) {
    predictors[[col]] <- log1p(predictors[[col]])
  }
}

# Standardize numeric predictors
if(sum(numeric_cols) > 0) {
  preprocess_params <- preProcess(predictors[, numeric_cols, drop = FALSE], 
                                  method = c("center", "scale"))
  predictors[, numeric_cols] <- predict(preprocess_params, predictors[, numeric_cols, drop = FALSE])
}

# 8. Combine back with target
house_data_clean <- cbind(predictors, SalePrice = target)

# Final check
str(house_data_clean)
summary(house_data_clean$SalePrice)



#====================================================]

# 1. Compute correlation matrix (numeric variables only)
# 1. Separate target and predictors SAFELY
target <- house_data_clean$SalePrice
predictors <- house_data_clean[, !names(house_data_clean) %in% "SalePrice", drop = FALSE]

# 2. Filter numeric predictors and remove zero-variance columns
prepare_predictors <- function(df) {
  # Select numeric columns
  numeric_data <- df[, sapply(df, is.numeric), drop = FALSE]
  
  # Calculate standard deviations
  col_sds <- apply(numeric_data, 2, sd, na.rm = TRUE)
  
  # Identify problematic columns (zero or NA SD)
  bad_cols <- is.na(col_sds) | col_sds == 0
  
  if(any(bad_cols)) {
    message("Removing columns with zero/NA variance: ", 
            paste(names(numeric_data)[bad_cols], collapse = ", "))
    numeric_data <- numeric_data[, !bad_cols, drop = FALSE]
  }
  
  return(numeric_data)
}

# 3. Prepare predictors
good_predictors <- prepare_predictors(predictors)

# 4. Check if we have predictors left
if(ncol(good_predictors) == 0) {
  stop("No valid predictors remaining after filtering")
}

# 5. Combine with target
analysis_data <- cbind(good_predictors, SalePrice = target)

# Save the the prepocessed and clean data into a file
write.csv(analysis_data, "preprocessed_cleaned_data.csv", row.names = FALSE)

print(analysis_data$SalePrice)

# Find columns with zero standard deviation
constant_cols <- sapply(analysis_data, function(x) sd(x, na.rm = TRUE) == 0)

print(constant_cols)

# Remove these columns from your analysis
analysis_data_w_n_zero <- analysis_data[, !constant_cols]

print(analysis_data_w_n_zero)

if(any(final_check == 0)) {
  warning("Constant column still present: ", names(which(final_check == 0)))
}

# 7. Compute correlations (now safe)
#cor_matrix <- cor(analysis_data_w_n_zero, use = "pairwise.complete.obs")


cat("SalePrice summary:\n")
summary(target)
cat("Standard deviation:", sd(target, na.rm = TRUE), "\n")

cat("\nFinal data structure:\n")
str(analysis_data_w_n_zero)
cat("\nStandard deviations:\n")
print(apply(analysis_data_w_n_zero, 2, sd, na.rm = TRUE))

# More robust correlation calculation
cor_matrix <- suppressWarnings(
  cor(analysis_data_w_n_zero, 
      use = "pairwise.complete.obs",
      method = "pearson")
)

# 7.1. Get SalePrice correlations
sale_price_cor <- cor_matrix["SalePrice", ]
sale_price_cor <- sale_price_cor[names(sale_price_cor) != "SalePrice"]  # Remove self-correlation

# 7.2. Get top 10 features (absolute correlation)
top_features <- names(sort(abs(sale_price_cor), decreasing = TRUE))[1:min(10, length(sale_price_cor))]

top_cor <- cor_matrix[c(top_features, "SalePrice"), c(top_features, "SalePrice")]

# https://rpubs.com/laurindocbenjam/predict-house-sale-price-in-R
# 8. Plot heatmap
corrplot(top_cor,
         method = "color",
         type = "upper",
         tl.col = "black",
         tl.srt = 45,
         addCoef.col = "black",
         number.cex = 0.7,
         title = "Top Features Correlated with SalePrice",
         mar = c(0,0,1,0))
