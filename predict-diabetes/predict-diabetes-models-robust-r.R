
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
library(patchwork)  # For combining plots
library(ggthemes)  # Additional plot themes
# ----------------------------
# 1. SETUP & DATA PREPARATION
# ----------------------------



# Set random seed for reproducibility
set.seed(123)

# Load datasets
train_data <- read.csv("train.csv") %>% 
  mutate(type = as.factor(ifelse(type == 1, "Diabetic", "Healthy")))

test_data <- read.csv("test.csv")

# ----------------------------
# 2. DATA PREPROCESSING (UPDATED)
# ----------------------------

# Create recipe for preprocessing with corrected imputation step
prep_recipe <- recipe(type ~ ., data = train_data) %>%
  update_role(id, new_role = "ID") %>%
  step_rm(id) %>%
  step_impute_median(all_numeric()) %>%  # Updated from step_medianimpute()
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
# 4. MODEL EVALUATION VISUALIZATIONS
# ----------------------------

## 4.1 Enhanced Confusion Matrix Plot
conf_matrix_plot <- function(model, data, title) {
  preds <- predict(model, data)
  cm <- confusionMatrix(preds, data$type)
  
  plot_data <- as.data.frame(cm$table) %>%
    mutate(
      color_strength = ifelse(Reference == Prediction, Freq/max(Freq), 0.1),
      label_color = ifelse(Reference == Prediction, "white", "black")
    )
  
  ggplot(plot_data, aes(x = Reference, y = Prediction, fill = color_strength)) +
    geom_tile(color = "white", alpha = 0.8) +
    geom_text(aes(label = Freq, color = label_color), size = 6, fontface = "bold") +
    scale_fill_gradient(low = "#F5F5F5", high = "#3F7FBF", guide = "none") +
    scale_color_identity() +
    labs(title = title,
         x = "Actual Status",
         y = "Predicted Status") +
    theme_fivethirtyeight() +
    theme(axis.title = element_text(),
          plot.title = element_text(hjust = 0.5, face = "bold"))
}

## 4.2 Enhanced ROC Curve Plot
roc_plot <- function(model, data) {
  probs <- predict(model, data, type = "prob")
  roc_obj <- roc(ifelse(data$type == "Diabetic", 1, 0), probs[, "Diabetic"])
  
  ggroc(roc_obj, color = "#E41A1C", size = 1.5) +
    geom_abline(intercept = 1, slope = 1, linetype = "dashed", color = "gray50") +
    annotate("text", x = 0.7, y = 0.3, 
             label = paste("AUC =", round(auc(roc_obj), 3)),
             size = 5, fontface = "bold") +
    labs(title = "ROC Curve",
         x = "False Positive Rate (1 - Specificity)",
         y = "True Positive Rate (Sensitivity)") +
    theme_economist() +
    theme(plot.title = element_text(hjust = 0.5, face = "bold"))
}

## 4.3 Enhanced Probability Distribution Plot
prob_dist_plot <- function(model, data) {
  probs <- predict(model, data, type = "prob")
  
  ggplot(data.frame(Probability = probs[, "Diabetic"], 
                    Actual = data$type),
         aes(x = Probability, fill = Actual)) +
    geom_density(alpha = 0.7, color = NA) +
    geom_rug(aes(color = Actual), sides = "b", alpha = 0.5) +
    scale_fill_manual(values = c("#377EB8", "#E41A1C")) +
    scale_color_manual(values = c("#377EB8", "#E41A1C"), guide = "none") +
    labs(title = "Predicted Probability Distribution",
         x = "Predicted Probability of Diabetes",
         y = "Density") +
    theme_stata() +
    theme(plot.title = element_text(hjust = 0.5, face = "bold"),
          legend.position = "top")
}

## 4.4 Enhanced Feature Importance Plot
feature_importance_plot <- function(model) {
  imp <- varImp(model)$importance
  imp$Feature <- rownames(imp)
  
  ggplot(imp %>% arrange(Overall) %>% tail(10), 
         aes(x = Overall, y = reorder(Feature, Overall), fill = Overall)) +
    geom_col(width = 0.7) +
    geom_text(aes(label = round(Overall, 2)), hjust = -0.1, size = 3.5) +
    scale_fill_gradient(low = "#4DAF4A", high = "#1B7837") +
    labs(title = "Top 10 Important Features",
         x = "Importance Score",
         y = "",
         fill = "Importance") +
    expand_limits(x = max(imp$Overall) * 1.1) +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, face = "bold"),
          legend.position = "none")
}

# Generate all evaluation plots for each model
dir.create("plots", showWarnings = FALSE)

for (model_name in names(models)) {
  cat("\nGenerating evaluation plots for:", model_name, "\n")
  
  # Confusion Matrix
  p1 <- conf_matrix_plot(models[[model_name]], val_set, 
                         paste(toupper(model_name), "Confusion Matrix"))
  
  # ROC Curve
  p2 <- roc_plot(models[[model_name]], val_set)
  
  # Probability Distribution
  p3 <- prob_dist_plot(models[[model_name]], val_set)
  
  # Feature Importance (if available)
  p4 <- tryCatch({
    feature_importance_plot(models[[model_name]])
  }, error = function(e) {
    ggplot() + 
      labs(title = "Feature Importance Not Available") +
      theme_void()
  })
  
  # Combine and save plots
  combined_plot <- (p1 + p2) / (p3 + p4) + 
    plot_annotation(
      title = paste("Model Evaluation:", str_to_title(model_name)),
      theme = theme(plot.title = element_text(size = 16, face = "bold", hjust = 0.5))
    )
  
  ggsave(
    paste0("plots/", model_name, "_evaluation.png"), 
    combined_plot, 
    width = 14, 
    height = 10,
    dpi = 300
  )
}

# ----------------------------
# 5. FINAL PREDICTIONS & VISUALIZATION
# ----------------------------

# Select best model (highest AUC)
model_metrics <- map_dfr(models, ~.$results[which.max(.$results$ROC),], .id = "model")
best_model_name <- model_metrics$model[which.max(model_metrics$ROC)]
best_model <- models[[best_model_name]]

# Make final predictions
final_preds <- data.frame(
  id = test_ids,
  type = ifelse(predict(best_model, test_preprocessed) == "Diabetic", 1, 0),
  probability = predict(best_model, test_preprocessed, type = "prob")[, "Diabetic"]
)

# Save predictions
write.csv(final_preds, "final_predictions.csv", row.names = FALSE)

# Enhanced Test set predictions visualization
pred_plot <- ggplot(final_preds, aes(x = probability, fill = factor(type))) +
  geom_histogram(binwidth = 0.05, position = "identity", alpha = 0.7) +
  geom_vline(xintercept = 0.5, linetype = "dashed", color = "red") +
  scale_fill_manual(
    values = c("#377EB8", "#E41A1C"),
    labels = c("Healthy", "Diabetic"),
    name = "Prediction"
  ) +
  labs(
    title = "Distribution of Predicted Diabetes Probabilities",
    subtitle = paste("Best Model:", best_model_name),
    x = "Predicted Probability",
    y = "Count"
  ) +
  theme_economist() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5),
    legend.position = "top"
  )

ggsave("plots/test_predictions_distribution.png", pred_plot, width = 10, height = 6, dpi = 300)

# ----------------------------
# 6. MODEL COMPARISON VISUALIZATION
# ----------------------------

# Enhanced model comparison plot
model_comparison <- model_metrics %>%
  select(model, ROC, Sensitivity, Specificity) %>%
  pivot_longer(-model) %>%
  mutate(
    name = factor(name, 
                  levels = c("ROC", "Sensitivity", "Specificity"),
                  labels = c("AUC-ROC", "Sensitivity", "Specificity"))
  ) %>%
  ggplot(aes(x = model, y = value, fill = name)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  geom_text(
    aes(label = round(value, 3)), 
    position = position_dodge(width = 0.8),
    vjust = -0.5,
    size = 3.5
  ) +
  scale_fill_manual(values = c("#E41A1C", "#377EB8", "#4DAF4A")) +
  labs(
    title = "Model Performance Comparison",
    subtitle = "Validation Set Metrics",
    y = "Score",
    x = "Model",
    fill = "Metric"
  ) +
  ylim(0, 1) +
  theme_fivethirtyeight() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5),
    legend.position = "top",
    axis.title = element_text()
  )

ggsave("plots/model_comparison.png", model_comparison, width = 10, height = 6, dpi = 300)

# ----------------------------
# 7. OUTPUT SUMMARY
# ----------------------------

cat("\n\n=== MODELING COMPLETE ===\n")
cat("Best model:", best_model_name, "\n")
cat("Validation AUC:", round(max(model_metrics$ROC), 3), "\n")
cat("Sensitivity:", round(model_metrics$Sensitivity[model_metrics$model == best_model_name], 3), "\n")
cat("Specificity:", round(model_metrics$Specificity[model_metrics$model == best_model_name], 3), "\n")
cat("Predictions saved to: final_predictions.csv\n")
cat("Visualizations saved to: plots/ directory\n")

# Open the plots directory
if (Sys.info()["sysname"] == "Windows") {
  shell.exec("plots")
} else {
  system("open plots")
}