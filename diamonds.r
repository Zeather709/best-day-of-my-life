library('tidyverse')
library('GGally')

data(diamonds)

# Exploratory 

summary(diamonds)

ggpairs(diamonds)

qq_diamonds <- qqnorm((diamonds$price),main="Normal Q-Q Plot of Price");qqline((diamonds$price))
# Meh

qq_log_diamonds <- qqnorm(log(diamonds$price),main="Normal Q-Q Plot of log Price");qqline(log(diamonds$price))
# Ooh this is a much better fit

hist_norm <- ggplot(diamonds, aes(log(price)))  + 
  geom_histogram(aes(y = ..density..), colour = "black", fill = 'lightblue', bins = 75) + 
  stat_function(fun = dnorm, args = list(mean = mean(log(diamonds$price)), sd = sd(log(diamonds$price))))
hist_norm

ggplot(diamonds, aes(x=carat, color=color)) + geom_histogram(stat = 'count') + facet_grid(cut ~ clarity) + scale_colour_viridis_b(palette = 2)

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
         clarity = factor(clarity, levels = c('I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'), ordered = TRUE)) %>%
  mutate_at(c('carat', 'table', 'depth'), ~(scale(.) %>% as.vector))


mean(diamonds$carat)
sd(diamonds$table)

mean(diamonds$table)
sd(diamonds$table)

mean(diamonds$depth)
sd(diamonds$depth)


library(caTools)
set.seed(42)
split = sample.split(diamonds$log_price, SplitRatio = 0.8)
diamonds_train = subset(diamonds, split == TRUE)
diamonds_test = subset(diamonds, split == FALSE)

mlm <- lm(log_price ~ carat + color + cut + clarity + table + depth, diamonds_train)
mlm
summary(mlm)

poly <- lm(log_price ~ poly(carat,3) + color + cut + clarity + poly(table,3) + poly(depth,3), diamonds_train)
poly
summary(poly)


library(xgboost)
diamonds_train_xgb <- diamonds_train %>%
  mutate_if(is.factor, as.numeric)
diamonds_test_xgb <- diamonds_test %>%
  mutate_if(is.factor, as.numeric)

xgb <- xgboost(data = as.matrix(diamonds_train_xgb[-7]), label = diamonds_train_xgb$log_price, nrounds = 6166)
# the rmse stopped decreasing after 6166 rounds 

xgb_pred = predict(xgb, as.matrix(diamonds_test_xgb[-7]))
xgb_pred


y_actual <- diamonds_test_xgb$log_price
y_predicted <- xgb_pred

test <- data.frame(cbind(y_actual, y_predicted))

xgb_scatter <- ggplot(test, aes(10**y_actual, 10**y_predicted)) + geom_point(colour = 'black', alpha = 0.2) + geom_smooth(method = lm)
xgb_scatter

ggplot(diamonds_test_xgb[x, ], aes(x = carat, y = log_price)) + geom_point(colour = 'deepskyblue') + geom_line()

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
                   ntree = 1000)
rf
# Model Performance
install.packages("Metrics")
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

r2 <- resid %>%
  mutate_all(square) %>%
  summarize_all(sum) %>%
  mutate_all(r2)
r2

xgb_rmse = sqrt(mean(residuals^2))


diamonds_test_sample 
ggplot(diamonds_test, aes(x = log_price, y = predictions$poly_pred, size = abs(resid$poly_resid))) +
  geom_point(alpha = 0.1)

# Feature Importance - maybe do this later
library(caret)
library(rminer)

imp_poly <- varImp(poly) %>%
  arrange(desc(Overall)) %>%
  cbind(feature = rownames(.))
imp_mlm <- varImp(mlm) %>%
  cbind(feature = rownames(.))

imp <- full_join(imp_poly, imp_mlm, by = 'feature') %>%
  rename(imp_mlm = Overall.y, imp_poly = Overall.x) %>%
  select(feature, imp_mlm, imp_poly)

install.packages("caret")
library(caret)

imp_poly <- varImp(poly) %>%
  arrange(desc(Overall)) %>%
  cbind(feature = rownames(.))
imp_mlm <- varImp(mlm) %>%
  cbind(feature = rownames(.))

imp <- full_join(imp_poly, imp_mlm, by = 'feature') %>%
  rename(imp_mlm = Overall.y, imp_poly = Overall.x) %>%
  select(feature, imp_mlm, imp_poly)
