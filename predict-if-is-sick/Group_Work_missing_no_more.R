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
#====== Modeling with no 10-folds cross-validation
# Train the model prediction 
model <- randomForest(sick ~. - id, data = datatrain, ntree = 100)

# predict class labels (0 or 1) on test model
predictions_class_label <- predict(model, newdata = datatest)

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

