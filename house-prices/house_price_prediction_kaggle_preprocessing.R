
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

# Load the data
train <- fread("train.csv", data.table = FALSE)
test <- fread("test.csv", data.table = FALSE)

# Combine train and test for consistent processing
test$SalePrice <- NA
full_data <- rbind(train, test)

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

# Outlier handling (IQR method)
for (col in num_cols) {
  Q1 <- quantile(full_data[[col]], 0.25, na.rm = TRUE)
  Q3 <- quantile(full_data[[col]], 0.75, na.rm = TRUE)
  IQR_val <- Q3 - Q1
  lower <- Q1 - 1.5 * IQR_val
  upper <- Q3 + 1.5 * IQR_val
  
  median_val <- median(full_data[[col]], na.rm = TRUE)
  full_data[[col]][full_data[[col]] < lower | full_data[[col]] > upper] <- median_val
}

# Re-encode categorical variables
n_unique <- sapply(full_data[cat_cols], function(x) length(unique(x)))
low_card <- names(n_unique[n_unique <= 10])
high_card <- names(n_unique[n_unique > 10])

# One-hot encoding for low-cardinality
dummies <- dummyVars(~ ., data = full_data[low_card], fullRank = TRUE)
onehot_encoded <- predict(dummies, newdata = full_data[low_card])
onehot_encoded <- as.data.frame(onehot_encoded)

# Frequency encoding for high-cardinality
freq_encode <- function(col) {
  freq <- table(col)
  return(as.numeric(freq[col]))
}
freq_encoded <- lapply(full_data[high_card], freq_encode)
freq_encoded <- as.data.frame(freq_encoded)
colnames(freq_encoded) <- paste0("freq_", high_card)

# Drop original categorical columns
full_data <- full_data[, !(names(full_data) %in% cat_cols)]

# Combine all features
final_data <- cbind(full_data, onehot_encoded, freq_encoded)

# Split back into train and test
final_train <- final_data[!is.na(final_data$SalePrice), ]
final_test <- final_data[is.na(final_data$SalePrice), ]

# Save the processed datasets
write.csv(final_train, "train_processed.csv", row.names = FALSE)
write.csv(final_test, "test_processed.csv", row.names = FALSE)


###

# Load required libraries
library(e1071)

# Load data
#data <- fread("train_processed.csv", data.table = FALSE)

# Load the data
house_data <- read.csv("train_processed.csv")

# Check structure
str(house_data)
dim(house_data)
summary(house_data)

# 1. First ensure SalePrice exists and is numeric
if(!"SalePrice" %in% names(house_data)) {
  stop("SalePrice column not found in dataset")
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
#install.packages("reshape2")
library(reshape2)
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

# 6. Verify no zero-variance columns remain
final_check <- apply(analysis_data, 2, sd, na.rm = TRUE)
if(any(final_check == 0)) {
  warning("Constant column still present: ", names(which(final_check == 0)))
}

# 7. Compute correlations (now safe)
cor_matrix <- cor(analysis_data, use = "pairwise.complete.obs")


cat("SalePrice summary:\n")
summary(target)
cat("Standard deviation:", sd(target, na.rm = TRUE), "\n")

cat("\nFinal data structure:\n")
str(analysis_data)
cat("\nStandard deviations:\n")
print(apply(analysis_data, 2, sd, na.rm = TRUE))

# More robust correlation calculation
cor_matrix <- suppressWarnings(
  cor(analysis_data, 
      use = "pairwise.complete.obs",
      method = "pearson")
)

# 6. Get SalePrice correlations
sale_price_cor <- cor_matrix["SalePrice", ]
sale_price_cor <- sale_price_cor[names(sale_price_cor) != "SalePrice"]  # Remove self-correlation

# 7. Get top 10 features (absolute correlation)
top_features <- names(sort(abs(sale_price_cor), decreasing = TRUE))[1:min(10, length(sale_price_cor))]

top_cor <- cor_matrix[c(top_features, "SalePrice"), c(top_features, "SalePrice")]

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
