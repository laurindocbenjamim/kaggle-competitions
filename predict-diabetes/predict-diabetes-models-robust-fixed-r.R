
install.packages("readxl")
install.packages("caret")
install.packages("yardstick")
install.packages("rpart")
install.packages("e1071")
install.packages("pROC")
install.packages("tidyverse")
install.packages("xgboost")
install.packages("randomForest")
install.packages("glmnet")
install.packages("kernlab")
install.packages("MLmetrics")
install.packages("recipes")
install.packages("tidymodels")
install.packages("DALEX")
install.packages("ggplot2")
install.packages("patchwork")
install.packages("ggthemes")

# ----------------------------
# 1. SETUP & DATA PREPARATION
# ----------------------------

# Load required libraries
library(tidyverse)
library(caret)
library(randomForest)
library(xgboost)
library(glmnet)
library(kernlab)
library(pROC)
library(MLmetrics)
library(recipes)
library(tidymodels)
library(DALEX)
library(ggplot2)
library(patchwork)

# Set random seed for reproducibility
set.seed(123)

# Load datasets
train_data <- read.csv("train.csv") %>% 
  mutate(type = as.factor(ifelse(type == 1, "Diabetic", "Healthy")))

test_data <- read.csv("test.csv")

# ----------------------------
# 2. DATA PREPROCESSING
# ----------------------------

# Create recipe for preprocessing
prep_recipe <- recipe(type ~ ., data = train_data) %>%
  update_role(id, new_role = "ID") %>%
  step_rm(id) %>%
  step_impute_median(all_numeric()) %>%
  step_normalize(all_numeric())

# Apply preprocessing
preprocessed_data <- prep(prep_recipe) %>%
  bake(new_data = NULL)

# Split training data for validation
train_split <- initial_split(preprocessed_data, prop = 0.8, strata = "type")
train_set <- training(train_split)
val_set <- testing(train_split)

# Prepare test set
test_preprocessed <- prep(prep_recipe) %>%
  bake(new_data = test_data)
test_ids <- test_data$id

# ----------------------------
# 3. MODEL TRAINING
# ----------------------------

# Define control parameters
ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = TRUE
)

# Train models
models <- list(
  logistic = train(type ~ ., data = train_set, method = "glmnet",
                   trControl = ctrl, metric = "ROC"),
  rf = train(type ~ ., data = train_set, method = "ranger",
             trControl = ctrl, metric = "ROC", importance = "impurity"),
  xgb = train(type ~ ., data = train_set, method = "xgbTree",
              trControl = ctrl, metric = "ROC")
)

# ----------------------------
# 4. MODEL EVALUATION (FIXED METRIC EXTRACTION)
# ----------------------------

# Create empty list to store metrics
model_metrics_list <- list()

for (model_name in names(models)) {
  # Get predictions
  preds <- predict(models[[model_name]], val_set)
  probs <- predict(models[[model_name]], val_set, type = "prob")[, "Diabetic"]
  actual <- val_set$type
  
  # Calculate metrics
  cm <- confusionMatrix(preds, actual)
  roc_obj <- roc(response = actual, predictor = probs)
  
  # Store metrics as numeric values (not S3 objects)
  model_metrics_list[[model_name]] <- data.frame(
    model = model_name,
    AUC = as.numeric(auc(roc_obj)),  # Convert to numeric
    Sensitivity = cm$byClass["Sensitivity"],
    Specificity = cm$byClass["Specificity"],
    stringsAsFactors = FALSE
  )
}

# Combine metrics safely
model_metrics <- bind_rows(model_metrics_list)

# ----------------------------
# 5. VISUALIZATION FUNCTIONS
# ----------------------------

# Confusion Matrix Plot
conf_matrix_plot <- function(model, data, title) {
  preds <- predict(model, data)
  cm <- confusionMatrix(preds, data$type)
  
  ggplot(as.data.frame(cm$table), 
         aes(x = Reference, y = Prediction, fill = Freq)) +
    geom_tile(color = "white") +
    geom_text(aes(label = Freq), color = "black", size = 6) +
    scale_fill_gradient(low = "#F5F5F5", high = "#3F7FBF") +
    labs(title = title, x = "Actual", y = "Predicted") +
    theme_minimal()
}

# ROC Curve Plot
roc_plot <- function(model, data) {
  probs <- predict(model, data, type = "prob")[, "Diabetic"]
  roc_obj <- roc(response = data$type, predictor = probs)
  
  ggroc(roc_obj, color = "#E41A1C") +
    geom_abline(slope = 1, intercept = 1, linetype = "dashed") +
    annotate("text", x = 0.7, y = 0.3, 
             label = paste("AUC =", round(auc(roc_obj), 3))) +
    labs(title = "ROC Curve") +
    theme_minimal()
}

# ----------------------------
# 6. GENERATE VISUALIZATIONS
# ----------------------------

# Create plots directory
dir.create("plots", showWarnings = FALSE)

# Generate plots for each model
for (model_name in names(models)) {
  p1 <- conf_matrix_plot(models[[model_name]], val_set, 
                         paste(model_name, "Confusion Matrix"))
  p2 <- roc_plot(models[[model_name]], val_set)
  
  # Combine and save
  combined <- p1 + p2 + plot_layout(ncol = 2)
  ggsave(paste0("plots/", model_name, "_eval.png"), combined, 
         width = 12, height = 5)
}

# Model comparison plot
model_comparison <- model_metrics %>%
  pivot_longer(-model) %>%
  ggplot(aes(x = model, y = value, fill = name)) +
  geom_col(position = "dodge") +
  labs(title = "Model Performance Comparison", y = "Score") +
  theme_minimal()

ggsave("plots/model_comparison.png", model_comparison, 
       width = 8, height = 5)

# ----------------------------
# 7. FINAL PREDICTIONS
# ----------------------------

# Select best model
best_model_name <- model_metrics$model[which.max(model_metrics$AUC)]
best_model <- models[[best_model_name]]

# Make predictions
final_preds <- data.frame(
  id = test_ids,
  type = ifelse(predict(best_model, test_preprocessed) == "Diabetic", 1, 0),
  probability = predict(best_model, test_preprocessed, type = "prob")[, "Diabetic"]
)

# Save predictions
write.csv(final_preds, "final_predictions.csv", row.names = FALSE)

# ----------------------------
# 8. OUTPUT SUMMARY
# ----------------------------

cat("\n=== MODELING RESULTS ===\n")
cat("Best model:", best_model_name, "\n")
cat("Validation AUC:", round(max(model_metrics$AUC), 3), "\n")
cat("Predictions saved to: final_predictions.csv\n")
cat("Visualizations saved to: plots/ directory\n")

# Open the plots directory
if (Sys.info()["sysname"] == "Windows") {
  shell.exec("plots")
} else {
  system("open plots")
}
