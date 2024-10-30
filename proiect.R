# Încărcarea librăriilor necesare
library(tidyverse)
library(caret)
library(ggplot2)
library(rpart)
library(rsample)
library(rpart.plot)
library(pROC)
library(ipred)
library(randomForest)
library(ranger)

# Încărcarea setului de date
data <- read.csv("C:/Users/Dragos/Documents/Big Data/Proiect Big Data 2.0/Churn_Modelling.csv")

# Explorarea setului de date
head(data)
str(data)
summary(data)
View(data)

# Curățarea datelor
# Verificarea valorilor lipsă
sum(is.na(data))

# Codificarea variabilelor categorice
data$Geography <- as.factor(data$Geography)
data$Gender <- as.factor(data$Gender)

# Eliminarea coloanelor irelevante
data <- data %>% select(-RowNumber, -CustomerId, -Surname)

# Verificarea valorilor unice pentru variabilele categorice
unique(data$Geography)
unique(data$Gender)

# Verificarea valorilor unice pentru variabilele binare
unique(data$HasCrCard)
unique(data$IsActiveMember)
unique(data$Exited)

# Verificarea structurii după preprocesare
head(data)
str(data)

# Vizualizarea prin metode grafice a datelor
# Distribuția variabilei țintă
ggplot(data, aes(x = as.factor(Exited), fill = as.factor(Exited))) +
  geom_bar() +
  labs(title = "Distribuția variabilei țintă", x = "Exited", y = "Count") +
  scale_fill_manual(values = c("blue", "red"), name = "Exited")

# Distribuția variabilei CreditScore pentru fiecare categorie a variabilei țintă
ggplot(data, aes(x = CreditScore, fill = as.factor(Exited))) +
  geom_histogram(binwidth = 50, position = "dodge") +
  labs(title = "Distribuția CreditScore în funcție de variabila țintă", x = "CreditScore", fill = "Exited")

# Distribuția variabilei Age pentru fiecare categorie a variabilei țintă
ggplot(data, aes(x = Age, fill = as.factor(Exited))) +
  geom_histogram(binwidth = 5, position = "dodge") +
  labs(title = "Distribuția Age în funcție de variabila țintă", x = "Age", fill = "Exited")

# Distribuția variabilei țintă în funcție de Geography
ggplot(data, aes(x = Geography, fill = as.factor(Exited))) +
  geom_bar(position = "dodge") +
  labs(title = "Distribuția variabilei țintă în funcție de Țară", x = "Țară", fill = "Exited")

# Distribuția variabilei țintă în funcție de Gender
ggplot(data, aes(x = Gender, fill = as.factor(Exited))) +
  geom_bar(position = "dodge") +
  labs(title = "Distribuția variabilei țintă în funcție de Sex", x = "Sex", fill = "Exited")

# Distribuția variabilei CreditScore în funcție de Geography
ggplot(data, aes(x = CreditScore, fill = as.factor(Exited))) +
  geom_histogram(binwidth = 50, position = "dodge") +
  facet_wrap(~Geography) +
  labs(title = "Distribuția CreditScore în funcție de Țară și variabila țintă", x = "CreditScore", fill = "Exited")

# Distribuția variabilei Age în funcție de Gender
ggplot(data, aes(x = Age, fill = as.factor(Exited))) +
  geom_histogram(binwidth = 5, position = "dodge") +
  facet_wrap(~Gender) +
  labs(title = "Distribuția Age în funcție de Sex și variabila țintă", x = "Age", fill = "Exited")

# Distribuția variabilei EstimatedSalary în funcție de variabila țintă
ggplot(data, aes(x = EstimatedSalary, fill = as.factor(Exited))) +
  geom_histogram(binwidth = 10000, position = "dodge") +
  labs(title = "Distribuția EstimatedSalary în funcție de variabila țintă", x = "EstimatedSalary", fill = "Exited")

# Distribuția variabilei Tenure în funcție de variabila țintă
ggplot(data, aes(x = Tenure, fill = as.factor(Exited))) +
  geom_bar(position = "dodge") +
  labs(title = "Distribuția Tenure în funcție de variabila țintă", x = "Tenure", fill = "Exited")

# Distribuția variabilei Balance în funcție de variabila țintă
ggplot(data, aes(x = Balance, fill = as.factor(Exited))) +
  geom_histogram(binwidth = 50000, position = "dodge") +
  labs(title = "Distribuția Balance în funcție de variabila țintă", x = "Balance", fill = "Exited")


# Împărțirea setului de date
set.seed(123)
split <- initial_split(data, prop = 0.7, strata = Exited)
train <- training(split)
test <- testing(split)

# Verificarea dimensiunilor seturilor de date
print(dim(train))
print(dim(test))

# Verificarea structurii după împărțire
head(train)
head(test)
view(train)
view(test)

# Model complex pentru a observa care variabile sunt semnificative pentru variabila dependenta aleasa
model1 <- glm(Exited ~ ., data = data)
summary(model1)

# Convertirea coloanei 'Exited' în factor
train$Exited <- factor(train$Exited, levels = c(0, 1), labels = c("No", "Yes"))
test$Exited <- factor(test$Exited, levels = c(0, 1), labels = c("No", "Yes"))

# Verificarea conversiei
str(train$Exited)
str(test$Exited)

# regresie logistica
# Antrenarea modelului de regresie logistică
logistic_model <- glm(Exited ~ ., data = train, family = binomial)

# Prezicerea pe setul de test
logistic_prob <- predict(logistic_model, newdata = test, type = "response")

# Convertirea la factor cu nivelurile corecte
logistic_pred <- factor(ifelse(logistic_prob > 0.5, "Yes", "No"), levels = c("No", "Yes"))

# Crearea matricei de confuzie folosind etichetele de clasă prezise și cele reale
conf_matrix_logistic <- confusionMatrix(logistic_pred, test$Exited)
print(conf_matrix_logistic)

# Calcularea ROC și AUC
roc_result_logistic <- roc(test$Exited, logistic_prob)
auc_value_logistic <- auc(roc_result_logistic)
print(paste("AUC:", auc_value_logistic))

# Plotarea curbei ROC
plot(roc_result_logistic, main = "ROC Curve for Logistic Regression Model", col = "#1c61b6")

# naive bayes
library(e1071)

# Antrenarea modelului Naive Bayes
nb_model <- naiveBayes(Exited ~ ., data = train)

# Prezicerea pe setul de test
nb_prob <- predict(nb_model, newdata = test, type = "raw")
nb_pred <- ifelse(nb_prob[, "Yes"] > 0.5, "Yes", "No")

# Convertirea la factor cu nivelurile corecte
nb_pred <- factor(nb_pred, levels = c("No", "Yes"))

# Crearea matricei de confuzie folosind etichetele de clasă prezise și cele reale
conf_matrix_nb <- confusionMatrix(nb_pred, test$Exited)
print(conf_matrix_nb)

# Calcularea ROC și AUC
roc_result_nb <- roc(test$Exited, nb_prob[, "Yes"])
auc_value_nb <- auc(roc_result_nb)
print(paste("AUC:", auc_value_nb))

# Plotarea curbei ROC
plot(roc_result_nb, main = "ROC Curve for Naive Bayes Model", col = "#1c61b6")


# Construirea modelului de arbori de decizie (m1)
set.seed(123)
m1 <- rpart(
  formula = Exited ~ .,
  data = train,
  method = "class"
)
m1
summary(m1)

# Plotarea arborelui de decizie
rpart.plot(m1)

# Prezicerea probabilităților pe setul de test (m1)
m1_prob_predictions <- predict(m1, newdata = test, type = "prob")
range(m1_prob_predictions)

# Prezicerea etichetelor de clasă pe setul de test (m1)
m1_predictions <- predict(m1, newdata = test, type = "class")

# Matricea de confuzie pentru evaluarea modelului (m1)
m1_conf_matrix <- confusionMatrix(m1_predictions, test$Exited)
print(m1_conf_matrix)

# Calcularea AUC pentru modelul de arbori de decizie (m1)
m1_roc_obj <- roc(response = test$Exited, predictor = m1_prob_predictions[, "Yes"])
m1_auc_value <- auc(m1_roc_obj)
print(paste("AUC pentru m1:", m1_auc_value))

# Plotarea curbei ROC (m1)
plot(m1_roc_obj, main = "ROC Curve for Decision Tree Model (m1)", col = "#1c61b6")

set.seed(123)
# Incercam sa facem prunning pe arborele construit mai sus cu cp = 0.01
m1_pruned <- prune(m1, cp = 0.01)
m1_pruned
summary(m1_pruned)

# Plotarea arborelui pruned
rpart.plot(m1_pruned)

# Prezicerea etichetelor de clasă pe setul de test pentru arborele pruned
pred_m1_pruned <- predict(m1_pruned, newdata = test, type = "class")

# Matricea de confuzie pentru evaluarea arborelui pruned
conf_matrix_m1_pruned <- confusionMatrix(pred_m1_pruned, test$Exited)
print(conf_matrix_m1_pruned)

# Prezicerea probabilităților pe setul de test (m1_prunned)
m1_pruned_prob_predictions <- predict(m1_pruned, newdata = test, type = "prob")

m1_pruned_roc_obj <- roc(response = test$Exited, predictor = m1_pruned_prob_predictions[, "Yes"])
m1_pruned_auc_value <- auc(m1_pruned_roc_obj)
print(paste("AUC pentru m1 pruned:", m1_pruned_auc_value))


# Construirea unui nou arbore de decizie fără restricții (m2)
set.seed(123)
m2 <- rpart(
  formula = Exited ~ .,
  data = train,
  method = "class",
  control = list(cp = 0)
)
m2
summary(m2)
rpart.plot(m2)

# Prezicerea etichetelor de clasă pe setul de test (m2)
m2_predictions <- predict(m2, newdata = test, type = "class")
m2_conf_matrix <- confusionMatrix(m2_predictions, test$Exited)
print(m2_conf_matrix)

# Prezicerea probabilităților pe setul de test (m2)
m2_prob_predictions <- predict(m2, newdata = test, type = "prob")

# Calcularea AUC pentru modelul de arbori de decizie (m2)
m2_roc_obj <- roc(response = test$Exited, predictor = m2_prob_predictions[, "Yes"])
m2_auc_value <- auc(m2_roc_obj)
print(paste("AUC pentru m2:", m2_auc_value))

# Plotarea curbei ROC (m2)
plot(m2_roc_obj, main = "ROC Curve for Decision Tree Model (m2)", col = "#1c61b6")

# Prunarea arborelui m2 cu cp = 0.02
m2_pruned <- prune(m2, cp = 0.02)
m2_pruned
summary(m2_pruned)

# Plotarea arborelui pruned
rpart.plot(m2_pruned)

# Prezicerea etichetelor de clasă pe setul de test pentru arborele pruned
pred_m2_pruned <- predict(m2_pruned, newdata = test, type = "class")
conf_matrix_m2_pruned <- confusionMatrix(pred_m2_pruned, test$Exited)
print(conf_matrix_m2_pruned)

# Prezicerea probabilităților pe setul de test (m2_prunned)
m2_pruned_prob_predictions <- predict(m2_pruned, newdata = test, type = "prob")

m2_pruned_roc_obj <- roc(response = test$Exited, predictor = m2_pruned_prob_predictions[, "Yes"])
m2_pruned_auc_value <- auc(m2_pruned_roc_obj)
print(paste("AUC pentru m2 pruned:", m2_pruned_auc_value))


library(tree)
# Construirea modelului de arbori de decizie folosind 'tree'
set.seed(123)
m1_tree <- tree(Exited ~ ., data = train)
m1_tree
summary(m1_tree)

# Plotarea arborelui de decizie
plot(m1_tree)
text(m1_tree, pretty = 0)

# Prezicerea etichetelor de clasă pe setul de test
pred_m1_tree <- predict(m1_tree, newdata = test, type = "class")

# Matricea de confuzie pentru evaluarea modelului
conf_matrix_m1_tree <- confusionMatrix(pred_m1_tree, test$Exited)
print(conf_matrix_m1_tree)

# Prezicerea probabilităților pe setul de test
prob_predictions_tree <- predict(m1_tree, newdata = test, type = "vector")

# Calcularea AUC pentru modelul de arbori de decizie
roc_obj_tree <- roc(response = test$Exited, predictor = prob_predictions_tree[, "Yes"])
auc_value_tree <- auc(roc_obj_tree)
print(paste("AUC:", auc_value_tree))

# Plotarea curbei ROC
plot(roc_obj_tree, main = "ROC Curve for Decision Tree Model (tree package)", col = "#1c61b6")


# Construirea arborelui de decizie folosind indexul Gini
set.seed(123)
m1_tree_gini <- tree(
  formula = Exited ~ .,
  data = train,
  split = "gini"
)
m1_tree_gini
summary(m1_tree_gini)

# Prezicerea etichetelor de clasă pe setul de test
pred_m1_tree_gini <- predict(m1_tree_gini, newdata = test, type = "class")

# Matricea de confuzie pentru evaluarea modelului
conf_matrix_m1_tree_gini <- confusionMatrix(pred_m1_tree_gini, test$Exited)
print(conf_matrix_m1_tree_gini)

# Prezicerea probabilităților pe setul de test
prob_predictions_tree_gini <- predict(m1_tree_gini, newdata = test, type = "vector")

# Calcularea AUC pentru modelul de arbori de decizie
roc_obj_tree_gini <- roc(response = test$Exited, predictor = prob_predictions_tree_gini[, "Yes"])
auc_value_tree_gini <- auc(roc_obj_tree_gini)
print(paste("AUC:", auc_value_tree_gini))

# Plotarea curbei ROC
plot(roc_obj_tree_gini, main = "ROC Curve for Decision Tree Model (Gini index)", col = "#1c61b6")


# Construirea modelului de bagging
set.seed(123)
bagged_exited <- bagging(Exited ~ ., data = train, coob = TRUE)
bagged_exited
summary(bagged_exited)

# Prezicerea etichetelor de clasă pe setul de test
pred_bagged_exited <- predict(bagged_exited, newdata = test, type = "class")
conf_matrix_bagged_exited <- confusionMatrix(pred_bagged_exited, test$Exited)
print(conf_matrix_bagged_exited)

# Prezicerea probabilităților pe setul de test
pred_probs_bagged_exited <- predict(bagged_exited, newdata = test, type = "prob")

# Calcularea AUC pentru modelul de bagging
roc_obj_bagged_exited <- roc(test$Exited, pred_probs_bagged_exited[, "Yes"])
auc_value_bagged_exited <- auc(roc_obj_bagged_exited)
print(paste("AUC:", auc_value_bagged_exited))

# Plotarea curbei ROC
plot(roc_obj_bagged_exited, main = "ROC Curve for Bagging Model", col = "#1c61b6")

ntree <- seq(10, 50, by = 1)
misclassification <- vector(mode = "numeric", length = length(ntree))

for (i in seq_along(ntree)) {
  set.seed(123)
  model <- bagging(
    Exited ~ .,
    data = train,
    coob = TRUE,
    nbagg = ntree[i]
  )
  misclassification[i] <- model$err
}

plot(ntree, misclassification, type = "l", lwd = "2", main = "Misclassification Rate vs. Number of Trees")

# Construirea modelului de bagging cu numărul optim de arbori
set.seed(123)
bagged_exited_optimal <- bagging(Exited ~ ., data = train, coob = TRUE, nbagg = 32)

summary(bagged_exited_optimal)

# Prezicerea etichetelor de clasă pe setul de test
pred_bagged_exited_optimal <- predict(bagged_exited_optimal, newdata = test, type = "class")
conf_matrix_bagged_exited_optimal <- confusionMatrix(pred_bagged_exited_optimal, test$Exited)
print(conf_matrix_bagged_exited_optimal)

# Prezicerea probabilităților pe setul de test
pred_probs_bagged_exited_optimal <- predict(bagged_exited_optimal, newdata = test, type = "prob")

# Calcularea AUC pentru modelul de bagging optim
roc_obj_bagged_exited_optimal <- roc(test$Exited, pred_probs_bagged_exited_optimal[, "Yes"])
auc_value_bagged_exited_optimal <- auc(roc_obj_bagged_exited_optimal)
print(paste("AUC:", auc_value_bagged_exited_optimal))

# Plotarea curbei ROC
plot(roc_obj_bagged_exited_optimal, main = "ROC Curve for Optimal Bagging Model", col = "#1c61b6")


# Construirea modelului de Random Forest
set.seed(123)
rf_model <- randomForest(
  formula = Exited ~ .,
  data = train,
  importance = TRUE
)

# Afișarea modelului
print(rf_model)
plot(rf_model)

# Prezicerea etichetelor de clasă pe setul de test
rf_predictions <- predict(rf_model, newdata = test, type = "class")
conf_matrix_rf <- confusionMatrix(rf_predictions, test$Exited)
print(conf_matrix_rf)

# Prezicerea probabilităților pe setul de test
rf_prob_predictions <- predict(rf_model, newdata = test, type = "prob")

# Calcularea AUC pentru modelul de Random Forest
rf_roc_obj <- roc(response = test$Exited, predictor = rf_prob_predictions[, "Yes"])
rf_auc_value <- auc(rf_roc_obj)
print(paste("AUC:", rf_auc_value))

# Plotarea curbei ROC
plot(rf_roc_obj, main = "ROC Curve for Random Forest Model", col = "#1c61b6")


# tunning random forest
# Definirea grilei de hiperparametri
hyper_grid <- expand.grid(
  mtry = seq(2, 8, by = 1), # Numărul de variabile selectate aleatoriu la fiecare împărțire
  min.node.size = seq(3, 9, by = 2), # Dimensiunea minimă a nodurilor terminale
  sample.fraction = c(0.55, 0.632, 0.7, 0.8), # Proporția de observații prelevate
  AUC = 0 # Placeholder pentru valorile AUC
)

# Realizarea căutării în grilă
for (i in 1:nrow(hyper_grid)) {
  model <- ranger(
    formula = Exited ~ .,
    data = train,
    num.trees = 500,
    mtry = hyper_grid$mtry[i],
    min.node.size = hyper_grid$min.node.size[i],
    sample.fraction = hyper_grid$sample.fraction[i],
    seed = 123,
    probability = TRUE
  )

  # Prezicerea pe setul de antrenament folosind predicția OOB
  oob_pred <- predict(model, data = train, type = "response")$predictions
  roc_result <- roc(train$Exited, oob_pred[, "Yes"], levels = rev(levels(train$Exited)))
  hyper_grid$AUC[i] <- auc(roc_result)
}

# Găsirea celor mai buni parametri
best_params <- hyper_grid[which.max(hyper_grid$AUC), ]
print(best_params)

# Antrenarea modelului optim
optimal_ranger <- ranger(
  formula = Exited ~ .,
  data = train,
  mtry = best_params$mtry,
  min.node.size = best_params$min.node.size,
  sample.fraction = best_params$sample.fraction,
  num.trees = 500,
  importance = "impurity",
  probability = TRUE,
  seed = 123
)

# Afișarea sumarului modelului
print(optimal_ranger)

# Prezicerea etichetelor de clasă pe setul de test
prob_predictions <- predict(optimal_ranger, data = test, type = "response")$predictions

# Selectarea corectă a clasei "Yes"
predicted_classes <- ifelse(prob_predictions[, "Yes"] > 0.5, "Yes", "No")

# Convertirea la factor cu nivelurile corecte
predicted_classes <- factor(predicted_classes, levels = c("No", "Yes"))

# Crearea matricei de confuzie folosind etichetele de clasă prezise și cele reale
conf_matrix <- confusionMatrix(predicted_classes, test$Exited)
print(conf_matrix)

# Calcularea ROC și AUC
roc_result <- roc(test$Exited, prob_predictions[, "Yes"])
auc_value <- auc(roc_result)
print(paste("AUC:", auc_value))

# Plotarea curbei ROC
plot(roc_result, main = "ROC Curve for Optimized Ranger Model", col = "#1c61b6")


library(ROSE)
# Am incercat sa aplic metodele de mai sus, dar dupa reechilibrarea setului de antrenament
train_balanced <- ROSE(Exited ~ ., data = train, seed = 123)$data

# Verificarea distribuției
table(train_balanced$Exited)

# Construirea modelului de regresie logistică pe setul de antrenament balansat
logistic_model_balaced <- glm(Exited ~ ., data = train_balanced, family = binomial)

# Prezicerea pe setul de test
logistic_prob_balanced <- predict(logistic_model_balaced, newdata = test, type = "response")

# Convertirea la factor cu nivelurile corecte
logistic_pred_balanced <- factor(ifelse(logistic_prob_balanced > 0.5, "Yes", "No"), levels = c("No", "Yes"))

# Crearea matricei de confuzie folosind etichetele de clasă prezise și cele reale
conf_matrix_logistic_balanced <- confusionMatrix(logistic_pred_balanced, test$Exited)
print(conf_matrix_logistic_balanced)

# Calcularea ROC și AUC
roc_result_logistic_balanced <- roc(test$Exited, logistic_prob_balanced)
auc_value_logistic_balanced <- auc(roc_result_logistic_balanced)
print(paste("AUC:", auc_value_logistic_balanced))

# Plotarea curbei ROC
plot(roc_result_logistic_balanced, main = "ROC Curve for Balanced Logistic Regression Model", col = "#1c61b6")

# Construirea modelului Naive Bayes pe setul de antrenament balansat
nb_model_balanced <- naiveBayes(Exited ~ ., data = train_balanced)

# Prezicerea pe setul de test
nb_prob_balanced <- predict(nb_model_balanced, newdata = test, type = "raw")
nb_pred_balanced <- ifelse(nb_prob_balanced[, "Yes"] > 0.5, "Yes", "No")

# Convertirea la factor cu nivelurile corecte
nb_pred_balanced <- factor(nb_pred_balanced, levels = c("No", "Yes"))

# Crearea matricei de confuzie folosind etichetele de clasă prezise și cele reale
conf_matrix_nb_balanced <- confusionMatrix(nb_pred_balanced, test$Exited)
print(conf_matrix_nb_balanced)

# Calcularea ROC și AUC
roc_result_nb_balanced <- roc(test$Exited, nb_prob_balanced[, "Yes"])
auc_value_nb_balanced <- auc(roc_result_nb_balanced)
print(paste("AUC:", auc_value_nb_balanced))

# Plotarea curbei ROC
plot(roc_result_nb_balanced, main = "ROC Curve for Balanced Naive Bayes Model", col = "#1c61b6")


# Construirea modelului de arbori de decizie pe setul de antrenament balansat
set.seed(123)
m1_balanced <- rpart(
  formula = Exited ~ .,
  data = train_balanced,
  method = "class"
)
m1_balanced
summary(m1_balanced)

# Plotarea arborelui de decizie
rpart.plot(m1_balanced)

# Prezicerea pe setul de test
prob_predictions_m1_balanced <- predict(m1_balanced, newdata = test, type = "prob")
predictions_m1_balanced <- predict(m1_balanced, newdata = test, type = "class")

# Crearea matricei de confuzie și calcularea AUC
conf_matrix_m1_balanced <- confusionMatrix(predictions_m1_balanced, test$Exited)
print(conf_matrix_m1_balanced)

roc_obj_m1_balanced <- roc(response = test$Exited, predictor = prob_predictions_m1_balanced[, "Yes"])
auc_value_m1_balanced <- auc(roc_obj_m1_balanced)
print(paste("AUC:", auc_value_m1_balanced))

# Plotarea curbei ROC
plot(roc_obj_m1_balanced, main = "ROC Curve for Balanced Decision Tree Model (m1)", col = "#1c61b6")


# Incercam sa facem prunning pe arborele construit mai sus cu cp = 0.01
m1_pruned_balanced <- prune(m1_balanced, cp = 0.01)
m1_pruned_balanced
summary(m1_pruned_balanced)

# Plotarea arborelui pruned
rpart.plot(m1_pruned_balanced)

# Prezicerea etichetelor de clasă pe setul de test pentru arborele pruned
pred_m1_pruned_balanced <- predict(m1_pruned_balanced, newdata = test, type = "class")

# Matricea de confuzie pentru evaluarea arborelui pruned
conf_matrix_m1_pruned_balanced <- confusionMatrix(pred_m1_pruned_balanced, test$Exited)
print(conf_matrix_m1_pruned_balanced)

# Construirea unui arborelui de decizie fără restricții pe setul de antrenament balansat
set.seed(123)
m2_balanced <- rpart(
  formula = Exited ~ .,
  data = train_balanced,
  method = "class",
  control = list(cp = 0)
)
m2_balanced
summary(m2_balanced)
rpart.plot(m2_balanced)

# Prezicerea etichetelor de clasă pe setul de test (m2)
predictions_m2_balanced <- predict(m2_balanced, newdata = test, type = "class")
conf_matrix_m2_balanced <- confusionMatrix(predictions_m2_balanced, test$Exited)
print(conf_matrix_m2_balanced)

# Prezicerea probabilităților pe setul de test (m2)
prob_predictions_m2_balanced <- predict(m2_balanced, newdata = test, type = "prob")

# Calcularea AUC pentru modelul de arbori de decizie (m2)
roc_obj_m2_balanced <- roc(response = test$Exited, predictor = prob_predictions_m2_balanced[, "Yes"])
auc_value_m2_balanced <- auc(m2_roc_obj)
print(paste("AUC pentru m2_balanced:", auc_value_m2_balanced))

# Plotarea curbei ROC (m2)
plot(roc_obj_m2_balanced, main = "ROC Curve for Decision Tree Model (m2_balanced)", col = "#1c61b6")

# Prunarea arborelui m2 cu cp = 0.02
m2_pruned_balanced <- prune(m2_balanced, cp = 0.02)
m2_pruned_balanced
summary(m2_pruned_balanced)

# Plotarea arborelui pruned
rpart.plot(m2_pruned_balanced)

# Prezicerea etichetelor de clasă pe setul de test pentru arborele pruned
pred_m2_pruned_balanced <- predict(m2_pruned_balanced, newdata = test, type = "class")
conf_matrix_m2_pruned_balanced <- confusionMatrix(pred_m2_pruned_balanced, test$Exited)
print(conf_matrix_m2_pruned_balanced)

# Prezicerea probabilităților pe setul de test (m2_prunned)
prob_predictions_m2_pruned_balanced <- predict(m2_pruned_balanced, newdata = test, type = "prob")

roc_obj_m2_pruned_balanced <- roc(response = test$Exited, predictor = prob_predictions_m2_pruned_balanced[, "Yes"])
auc_value_m2_pruned_balanced <- auc(roc_obj_m2_pruned_balanced)
print(paste("AUC pentru m2 pruned balanced:", auc_value_m2_pruned_balanced))


# Construirea modelului de bagging pe setul de antrenament balansat
set.seed(123)
bagged_model_balanced <- bagging(Exited ~ ., data = train_balanced, coob = TRUE)

# Prezicerea pe setul de test
prob_predictions_bagging_balanced <- predict(bagged_model_balanced, newdata = test, type = "prob")
predictions_bagging_balanced <- predict(bagged_model_balanced, newdata = test, type = "class")

# Crearea matricei de confuzie și calcularea AUC pentru bagging
conf_matrix_bagging_balanced <- confusionMatrix(predictions_bagging_balanced, test$Exited)
print(conf_matrix_bagging_balanced)

roc_obj_bagging_balanced <- roc(response = test$Exited, predictor = prob_predictions_bagging_balanced[, "Yes"])
auc_value_bagging_balanced <- auc(roc_obj_bagging_balanced)
print(paste("AUC:", auc_value_bagging_balanced))

# Plotarea curbei ROC pentru bagging
plot(roc_obj_bagging_balanced, main = "ROC Curve for Bagging Model Balanced", col = "#1c61b6")


# Construirea modelului Random Forest
set.seed(123)
rf_model_balanced <- randomForest(Exited ~ ., data = train_balanced, importance = TRUE, ntree = 500)

# Prezicerea pe setul de test
prob_predictions_rf_balanced <- predict(rf_model_balanced, newdata = test, type = "prob")
predictions_rf_balanced <- predict(rf_model_balanced, newdata = test, type = "class")

# Crearea matricei de confuzie și calcularea AUC pentru Random Forest
conf_matrix_rf_balanced <- confusionMatrix(predictions_rf_balanced, test$Exited)
print(conf_matrix_rf_balanced)

roc_obj_rf_balanced <- roc(response = test$Exited, predictor = prob_predictions_rf_balanced[, "Yes"])
auc_value_rf_balanced <- auc(roc_obj_rf_balanced)
print(paste("AUC:", auc_value_rf_balanced))

# Plotarea curbei ROC pentru Random Forest
plot(roc_obj_rf_balanced, main = "ROC Curve for Random Forest Model", col = "#1c61b6")
