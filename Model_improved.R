# R version of the RedemptionSalesImprovedModel using functional programming

# Required libraries
library(randomForest)
library(lightgbm)
library(xgboost)
library(Metrics)
library(ggplot2)
library(dplyr)
library(lubridate)


# Convert Timestamp to Date format
df$Timestamp <- as.Date(df$Timestamp)

# ---- Core functions ----

calculate_metrics <- function(y_true, y_pred) {
  mask <- y_true != 0
  mape <- mean(abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
  list(
    MSE = mse(y_true, y_pred),
    RMSE = rmse(y_true, y_pred),
    MAE = mae(y_true, y_pred),
    MAPE = mape,
    R2 = 1 - sum((y_true - y_pred)^2) / sum((y_true - mean(y_true))^2)
  )
}

run_cv_model <- function(data, target_col, n_splits = 4, test_size = 365) {
  data <- data %>% arrange(Timestamp)
  n <- nrow(data)
  split_size <- floor((n - test_size) / n_splits)
  
  results <- list()
  preds <- list()
  feature_importances <- list()
  test_indices <- list()
  
  for (i in seq_len(n_splits)) {
    train_end <- split_size * i
    test_start <- train_end + 1
    test_end <- test_start + test_size - 1
    if (test_end > n) break
    
    train <- data[1:train_end, ]
    test <- data[test_start:test_end, ]
    
    y_train <- train[[target_col]]
    y_test <- test[[target_col]]
    X_train <- train %>% select(-all_of(target_col), -Timestamp)
    X_test <- test %>% select(-all_of(target_col), -Timestamp)
    
    row_idx <- test$Timestamp
    test_indices[[i]] <- row_idx
    
    # Base Model (Day-of-Year mean)
    doy <- yday(train$Timestamp)
    doy_means <- tapply(y_train, doy, mean, na.rm = TRUE)
    base_pred <- doy_means[yday(row_idx)]
    results[["Base"]][[i]] <- calculate_metrics(y_test, base_pred)
    preds[["Base"]][[i]] <- data.frame(index = row_idx, prediction = base_pred)
    
    # Random Forest
    rf <- randomForest(X_train, y_train, ntree = 200, nodesize = 20)
    rf_pred <- predict(rf, X_test)
    results[["RandomForest"]][[i]] <- calculate_metrics(y_test, rf_pred)
    preds[["RandomForest"]][[i]] <- data.frame(index = row_idx, prediction = rf_pred)
    feature_importances[["RandomForest"]] <- setNames(importance(rf)[, 1], rownames(importance(rf)))
    
    
    # LightGBM
    dtrain_lgb <- lgb.Dataset(data = as.matrix(X_train), label = y_train)
    lgb_model <- lgb.train(list(objective = "regression", learning_rate = 0.05,
                                max_depth = 70, num_leaves = 64, subsample = 0.6),
                           dtrain_lgb, nrounds = 600)
    lgb_pred <- predict(lgb_model, as.matrix(X_test))
    results[["LightGBM"]][[i]] <- calculate_metrics(y_test, lgb_pred)
    preds[["LightGBM"]][[i]] <- data.frame(index = row_idx, prediction = lgb_pred)
    lgb_imp <- lgb.importance(lgb_model)
    feature_importances[["LightGBM"]] <- setNames(lgb_imp$Gain, lgb_imp$Feature)
    
    # XGBoost
    dtrain_xgb <- xgb.DMatrix(data = as.matrix(X_train), label = y_train)
    xgb_model <- xgb.train(params = list(objective = "reg:squarederror",
                                         learning_rate = 0.05, max_depth = 6,
                                         subsample = 0.8, colsample_bytree = 0.8),
                           data = dtrain_xgb, nrounds = 600)
    xgb_pred <- predict(xgb_model, xgb.DMatrix(as.matrix(X_test)))
    results[["XGBoost"]][[i]] <- calculate_metrics(y_test, xgb_pred)
    preds[["XGBoost"]][[i]] <- data.frame(index = row_idx, prediction = xgb_pred)
    xgb_imp <- xgb.importance(model = xgb_model)
    feature_importances[["XGBoost"]] <- setNames(xgb_imp$Gain, xgb_imp$Feature)
    
    # Ensemble
    ensemble_pred <- (rf_pred + lgb_pred + xgb_pred) / 3
    results[["Ensemble"]][[i]] <- calculate_metrics(y_test, ensemble_pred)
    preds[["Ensemble"]][[i]] <- data.frame(index = row_idx, prediction = ensemble_pred)
  }
  
  list(
    results = results,
    preds = preds,
    importances = feature_importances,
    summary = summarise_results(results),
    test_indices = test_indices
  )
}



summarise_results <- function(results) {
  m1 <- data.frame(lapply(do.call(rbind, results$Base)%>%
                            as.data.frame(),function(col) unlist(col)))%>%
    colMeans()
  m2 <- data.frame(lapply(do.call(rbind, results$RandomForest)%>%
                            as.data.frame(),function(col) unlist(col)))%>%
    colMeans()
  m3 <- data.frame(lapply(do.call(rbind, results$LightGBM)%>%
                            as.data.frame(),function(col) unlist(col)))%>%
    colMeans()
  m4 <- data.frame(lapply(do.call(rbind, results$XGBoost)%>%
                            as.data.frame(),function(col) unlist(col)))%>%
    colMeans()
  m5 <- data.frame(lapply(do.call(rbind, results$Ensemble)%>%
                            as.data.frame(),function(col) unlist(col)))%>%
    colMeans()
  performance_results <- data.frame(Base = m1, RandomForest = m2, LightGBM = m3,
                                    XGBoost = m4, Ensemble = m5)%>%round(2)
  return(performance_results)
}

plot_predictions <- function(data, preds, target_col) {
  
  for (model_name in names(preds)) {
    for (i in 1:length(preds[[model_name]])) {
      pred_df <- preds[[model_name]][[i]]
      dt <- data.frame(
        index = as.Date(pred_df$index),
        observed = data %>% filter(Timestamp %in% pred_df$index) %>% pull(!!sym(target_col)),
        predicted = pred_df$prediction
      )%>%tibble()
      
      dt_long <- tidyr::pivot_longer(dt, cols = c("observed", "predicted"), names_to = "type", values_to = "value")
      p <- ggplot(dt_long, aes(x = index, y = value, color = type)) +
        geom_point(size = 2.5) +
        #geom_line(lwd = 1.5) +
        scale_color_manual(values = c("observed" = "grey", "predicted" = "red")) +
        ggtitle(paste(model_name, "- Split", i)) +
        theme_bw() +
        theme(legend.title = element_blank(),
            legend.key.size =  unit(1.5, 'cm'),
            legend.text = element_text(size=12),
            axis.title = element_text(size = 15, face = 'bold'),
            axis.text = element_text(size = 13),
            title = element_text(size = 16, face='bold')
            )
        
        print(p)
    }
  }
}



plot_feature_importance <- function(importances) {
  #par(mfrow = c(3, 1))
  for (model in c("RandomForest", "LightGBM", "XGBoost")) {
    if (!is.null(importances[[model]])) {
      imp <- sort(importances[[model]], decreasing = F)[1:10]
      imp_df <- data.frame(Feature = names(imp), Importance = imp)

      #barplot(imp, horiz = TRUE, main = paste("Top 10 -", model), col = "skyblue")
      #barplot(imp, names.arg = names(imp), horiz = TRUE, las = 1,
       #       main = paste("Top 10 -", model), col = "skyblue")

        p <- ggplot(imp_df, aes(x = reorder(Feature, Importance), y = Importance)) +
        geom_bar(stat = "identity", fill = "skyblue") +
        coord_flip() +
        labs(title = paste("Top 10 -", model), x = "Feature", y = "Importance") +
        theme_bw()+
        theme(
            axis.title = element_text(size = 15, face = 'bold'),
            axis.text = element_text(size = 13),
            title = element_text(size = 16, face='bold')
        )
        print(p)

    
    }
  }
  #par(mfrow = c(1, 1))
}