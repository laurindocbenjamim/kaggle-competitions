
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

# index for spliting data in training and testing 
train_index <- createDataPartition(train_data$type, p = .8, list = FALSE, times = 1)

# Way-01: Cross-validation on TRAIN set (e.g., 10-fold CV)
ctrl <- trainControl(method = 'cv', number = 10) # 10-folds CV

# Train a model (e.g., logistic regression) with CV
model <- train(
  type ~ ., 
  data = train_data[train_index,], 
  method = "glm", 
  family = "binomial",
  trControl = ctrl
)

# OR

# Way-02: Cross-validation on TRAIN set (e.g., 10-fold CV)
#folde_indexes <- createFolds(train_data$type, k = 10)

# diaplay structure of dataset
str(train_data)



# =======================  Evaluate the test data =========================

test_data <- read.csv("test.csv")
# Converting the dataset into a dataframe
test_data <- as.data.frame(test_data)

# Ensuring RISK is a factor
#test_data <- as.factor(test_data)

head(test_data)
summary(test_data)
str(test_data)

set.seed(123)

cat("Missing value: \n")
print(colSums(is.na(test_data)))

col_num <- names(Filter(is.numeric, test_data))
cat_num <- names(Filter(is.character, test_data))


# Replace NA with median

for(col in col_num){
    media_val <- median(test_data[[col]], na.rm = TRUE)
    test_data[[col]][is.na(test_data[[col]])] <- media_val
}


# Check
print(colSums(is.na(test_data)))



install.packages("tibble")
library(tibble)

# Step 3: Evaluate on TEST set
sumission_predictions <- predict(model, test_data)

# format csv as especificed in competition description 
prediction_file <- tibble(Id = test_data$id, type = sumission_predictions)

print(prediction_file)


# save file 
write.csv(prediction_file, "submission.csv")


submited_data <- read.csv("submission.csv")

str(submited_data)
summary(submited_data)
