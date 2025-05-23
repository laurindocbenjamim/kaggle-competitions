install.packages("caret")
install.packages("yardstick")
install.packages("e1071")
install.packages("pROC")
install.packages("ggplot2")
install.packages("lattice")
install.packages("splitTools")


library(yardstick)
library(caret)
library(e1071)
library(pROC)
library(ggplot2)
library(lattice)
library(splitTools)

setwd("~/Documents/RProjects/kaggle-competitions/predict-if-is-sick")

train_data <- read.csv("train.csv")
test_data <- read.csv("test.csv")

summary(train_data)
summary(test_data)

str(train_data)
str(test_data)


train_data$person_sex<-as.factor(train_data$person_sex)
train_data$marital_status<-as.factor(train_data$marital_status)
train_data$employment_type<-as.factor(train_data$employment_type)
train_data$residence_category<-as.factor(train_data$residence_category)
train_data$smoking_status<-as.factor(train_data$smoking_status)
train_data$person_age<-as.integer(train_data$person_age)
train_data$sick<-as.factor(train_data$sick)
levels(train_data$sick)<-c(0,1)
train_data$sick<-relevel(train_data$sick, ref="0")
str(train_data)

print(colSums(is.na(train_data)))

set.seed(123)
fold_indexes <- create_folds(train_data$sick,k = 10,type="stratified",invert=TRUE)
print(fold_indexes)

f1<-c()

library(OneR)

for(i in 1:10){
  train_data <- train_data[-fold_indexes[[i]], ]
  test_data <- train_data[fold_indexes[[i]], ]

  Model_OneR<-OneR(train_data[,-c(1)])


  f1[i] <- f_meas_vec(train_data$sick, as.factor(train_data$sick), event_level = "second")
}



submission<-data.frame(ID=train_data$id, sick=train_data$sick)
write.csv(submission, "submissiOn.csv", row names = FALSE)
  