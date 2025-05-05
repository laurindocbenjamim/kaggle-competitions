
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

# Check and Describe the data structure
str(sample_submission)
str(train_df)
str(test_df)

summary(sample_submission)
summary(train_df)
summary(test_df)

# Selecting all columns except the Id which is not necessary
train_df <- train_df[, !names(train_df) %in% "Id"]
test_df <- test_df[, !names(test_df) %in% "Id"]

# Or if the Id is the first column
#train_df <- train_df[, -1]
#test_df <- test_df[, -1]
str(train_df)
summary(train_df)

# Visualizar
ggplot(train_df, aes(x = SalePrice)) + geom_histogram(bins = 50) + theme_minimal()

# Get the data types for each columns
data_types <- sapply(train_df, class)

# Create a data frame for better readability
data_types_df <- data.frame(Column = names(train_df), DataType = data_types)
print(data_types_df)

# Identifying the numerical attributes/columns
num_cols <- names(train_df)[sapply(train_df, is.numeric)]
# Select numerical columns
train_num_cols <- train_df[, num_cols]

# checking for missing values
missing_values_count <- colSums(is.na(train_num_cols))
cat("Missing values: ", "\n")
print(missing_values_count[missing_values_count > 0])

# Input missing values for numerical columns
imp <- mice(train_num_cols, method = "pmm", m = 5, maxit = 5, seed = 123)
train_numeric_complete <- complete(imp)

# View logged events
imp$loggedEvents

# Create a list to store plots
plot_list <- list()

# Loop through numerical columns
for (col in num_cols){
  p <- ggplot(train_num_cols, aes(x = .data[[col]])) +
    geom_histogram(bins = 30, fill = "blue", alpha = 0.7) +
    theme_minimal() + 
    labs(title = paste("Distribution of ", col), x = col, y = "Count")
  
  plot_list[[col]] <- p
}

# Arrange plots in a grid (4 columns)
n_plots <- length(plot_list)

columns <- length(names(train_num_cols))
print(columns)
grid.arrange(grobs = plot_list, ncol = 37)

# Combine plots
wrap_plots(plot_list, ncol = 37)

