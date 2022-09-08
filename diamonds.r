library('tidyverse')
library('GGally')
install.packages("tidymodels")

data(diamonds)

# Exploratory 

summary(diamonds)
diamonds_raw <- diamonds
ggpairs(diamonds)

qq_diamonds <- qqnorm((diamonds$price),main="Normal Q-Q Plot of Price");qqline((diamonds$price))
# Meh

qq_log_diamonds <- qqnorm(log(diamonds$price),main="Normal Q-Q Plot of log Price");qqline(log(diamonds$price))
# Ooh this is a much better fit

hist_norm <- ggplot(diamonds, aes(log(price)))  + 
  geom_histogram(aes(y = ..density..), colour = "black", fill = 'lightblue', bins = 75) + 
  stat_function(fun = dnorm, args = list(mean = mean(log(diamonds$price)), sd = sd(log(diamonds$price))))
hist_norm

price <- ggplot(data = diamonds, aes(x = price))
price + geom_histogram(binwidth = 250, colour = 'black', fill = 'mediumorchid') + labs(title = 'Cost of Diamonds in USD')

log_price <- ggplot(data = diamonds, aes(x = log(price)))
log_price + geom_histogram(binwidth = 0.1, colour = 'black', fill = 'mediumorchid') + labs(title = 'Log Price of Diamonds in USD') 

cut <- ggplot(data = diamonds, aes(x = cut, y = price, fill = cut)) 
cut + geom_boxplot()

carat <- ggplot(data = diamonds, aes(x = carat, y = price, colour = color)) 
carat + geom_point(size = 0.5)

carat + stat_ecdf() 

clarity <- ggplot(data = diamonds, aes(x = clarity, y = price, fill = clarity)) 
clarity + geom_boxplot()

diamonds %>% summarise_if(is.numeric, list(mean = mean, var = var)) %>%
  t()


mean(diamonds$carat)
sd(diamonds$table)

mean(diamonds$table)
sd(diamonds$table)

mean(diamonds$depth)
sd(diamonds$depth)

diamonds <- diamonds %>%
  mutate(price = price * 1.1055) %>%
  mutate(log_price = log(price)) %>%
  select(-price, -x, -y, -z) %>%
  mutate(cut = factor(cut, levels = c('Fair', 'Good', 'Very Good', 'Premium', 'Ideal'), ordered = TRUE),
         color = factor(color, levels = c('J', 'I', 'H', 'G', 'F', 'E', 'D'), ordered = TRUE),
         clarity = factor(clarity, levels = c('I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'), ordered = TRUE)) #%>%
#  mutate_at(c('carat', 'table', 'depth'), ~(scale(.) %>% as.vector))


mean(diamonds$carat)
sd(diamonds$table)

mean(diamonds$table)
sd(diamonds$table)

mean(diamonds$depth)
sd(diamonds$depth)

#install.packages("tictoc")
library(caTools)
library(tictoc)
set.seed(42)
split = sample.split(diamonds$log_price, SplitRatio = 0.8)
diamonds_train = subset(diamonds, split == TRUE)
diamonds_test = subset(diamonds, split == FALSE)

diamonds_train <- diamonds_train %>% 
  mutate_at(c('table', 'depth'), ~(scale(.) %>% as.vector))
diamonds_test <- diamonds_test %>% 
  mutate_at(c('table', 'depth'), ~(scale(.) %>% as.vector))  

mean(diamonds_train$carat)
sd(diamonds_train$carat)

tic.clearlog()

tic('mlm')
mlm <- lm(log_price ~ carat + color + cut + clarity + table + depth, diamonds_train)
mlm
summary(mlm)
toc(log = TRUE, quiet = FALSE)

time_log <- tic.log(format = TRUE)
time_log

poly <- lm(log_price ~ poly(carat,3) + color + cut + clarity + poly(table,3) + poly(depth,3), diamonds_train)
poly
summary(poly)


library(xgboost)
diamonds_train_xgb <- diamonds_train %>%
  mutate_if(is.factor, as.numeric)
diamonds_test_xgb <- diamonds_test %>%
  mutate_if(is.factor, as.numeric)

xgb <- xgboost(data = as.matrix(diamonds_train_xgb[-7]), label = diamonds_train_xgb$log_price, nrounds = 6166, verbose = 0)
# the rmse stopped decreasing after 6166 rounds 

xgb_pred = predict(xgb, as.matrix(diamonds_test_xgb[-7]))
xgb_pred


y_actual <- diamonds_test_xgb$log_price
y_predicted <- xgb_pred

test <- data.frame(cbind(y_actual, y_predicted))

xgb_scatter <- ggplot(test, aes(10**y_actual, 10**y_predicted)) + geom_point(colour = 'black', alpha = 0.2) + geom_smooth(method = lm)
xgb_scatter

library(e1071)
svr <- svm(formula = log_price ~ .,
                data = diamonds_train,
                type = 'eps-regression',
                kernel = 'radial')
# use the radial kernel if it is not a linear relationship between the independent and dependent variables.  
# I think this is the case because the linear model is performing the worst so far

# Decision Tree

library(rpart)
tree <- rpart(formula = log_price ~ .,
                  data = diamonds_train,
                  method = 'anova',
                  model = TRUE)
tree
# Random Forest

library(randomForest)
rf <- randomForest(log_price ~ .,
                   data = diamonds_train,
                   ntree = 500)
rf
# Model Performance
library(Metrics)

# Make predictions and compare model performance
mlm_pred <- predict(mlm, diamonds_test)
poly_pred <- predict(poly, diamonds_test)
svr_pred <- predict(svr, diamonds_test)
tree_pred <- predict(tree, diamonds_test)
rf_pred <- predict(rf, diamonds_test)
xgb_pred <- predict(xgb, as.matrix(diamonds_test_xgb[-7]))

# Calculate residuals (i.e. how different the predictions are from the log_price of the test data set)

xgb_resid <- diamonds_test_xgb$log_price - xgb_pred
library(modelr)
resid <- diamonds_test %>%  
  spread_residuals(mlm, poly, svr, tree, rf) %>%
  select(mlm, poly, svr, tree, rf) %>%
  rename_with( ~ paste0(.x, '_resid')) %>%
  cbind(xgb_resid)

predictions <- diamonds_test %>%
  select(log_price) %>%
  cbind(mlm_pred) %>%
  cbind(poly_pred) %>%
  cbind(svr_pred) %>%
  cbind(tree_pred) %>%
  cbind(rf_pred) %>%
  cbind(xgb_pred) %>%
  cbind(resid)           # This will be useful for plotting later

# Calculate R-squared - this describes how much of the variability is explained by the model - the closer to 1, the better

mean_log_price <- mean(diamonds_test$log_price)
tss =  sum((diamonds_test_xgb$log_price - mean_log_price)^2 )

square <- function(x) {x**2}
r2 <- function(x) {1 - x/tss}

r2_df <- resid %>%
  mutate_all(square) %>%
  summarize_all(sum) %>%
  mutate_all(r2) %>%
  gather(key = 'model', value = 'r2') %>%
  mutate(model = str_replace(model, '_resid', ''))
r2_df

xgb_rmse = sqrt(mean(residuals^2))

# Plot Performance

r2_plot <- ggplot(r2_df, aes(x = model, y = r2, colour = model, fill = model)) + geom_bar(stat = 'identity')
r2_plot + ggtitle('R-squared Values for each Model') + coord_cartesian(ylim = c(0.75, 1))

#varImpPlot(rf)
rf_imp <- importance(rf) 
rf_imp <- rf_imp %>%
  as.data.frame() %>%
  mutate(variable = row.names(rf_imp))
varImpPlot(rf)

# Training & predicting times
time_log <- tic.log(format = TRUE)
time_log


diamonds_test_sample <- diamonds_test %>%
  left_join(predictions, by = 'log_price') %>%
  slice_sample(n = 1000) 
ggplot(diamonds_test_sample, aes(x = exp(log_price), y = exp(rf_pred), size = abs(rf_resid))) +
  geom_point(alpha = 0.1) + labs(title = 'Predicted vs Actual Cost of Diamonds in USD', x = 'Price', y = 'Predicted Price', size = 'Residuals')


# Real Diamonds

# Build a data frame with the two diamonds
carat <- c(0.36, 0.71, 1.5)
clarity <- c('I1', 'VS2', 'VS1')
color <- c('I', 'I', 'F')
cut <- c('Premium', 'Premium', 'Premium')

impute_036 <- diamonds %>%
  filter(carat == 0.36) %>%
  select(depth, table) %>%
  summarize_all(mean)

impute_071 <- diamonds %>%
  filter(carat == 0.71) %>%
  select(depth, table) %>%
  summarize_all(mean)

impute_15 <- diamonds %>%
  filter(carat == 1.5) %>%
  select(depth, table) %>%
  summarize_all(mean)

table_depth <- rbind(impute_036, impute_071, impute_15)

real_diamonds <- data.frame(carat, clarity, color, cut) %>%
  cbind(table_depth) %>%
  mutate(cut = factor(cut, levels = c('Fair', 'Good', 'Very Good', 'Premium', 'Ideal'), ordered = TRUE),
         color = factor(color, levels = c('J', 'I', 'H', 'G', 'F', 'E', 'D'), ordered = TRUE),
         clarity = factor(clarity, levels = c('I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'), ordered = TRUE)) %>%
  mutate_at(c('table', 'depth'), ~(scale(.) %>% as.vector))

predicted_values <- predict(rf, real_diamonds) 
predicted_price <- exp(predicted_values)
predicted_price

test15 <- print(filter(diamonds_raw, carat == 1.5), n = 50)
