#install.packages("tidyverse")
#install.packages("GGally")
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
  geom_histogram(aes(y = ..density..), colour = "black", fill = 'lightblue') + 
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
  mutate(price = price * 1.1055,
         table = table/100,
         depth = depth/100) %>%
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


install.packages("caTools")
library(caTools)
set.seed(42)
split = sample.split(diamonds$log_price, SplitRatio = 0.8)
diamonds_train = subset(diamonds, split == TRUE)
diamonds_test = subset(diamonds, split == FALSE)

mlm <- lm(log_price ~ carat + color + cut + clarity + table + depth, diamonds_train)
mlm
summary(mlm)

poly <- lm(log_price ~ poly(carat,3) + color + cut + clarity + poly(table,3) + poly(depth,3), train)
poly
summary(poly)

install.packages('xgboost')
library(xgboost)

install.packages("e1071")
install.packages("rminer")

library(e1071)
library(rminer)
svm <- svm(formula = log_price ~ .,
                data = diamonds_train,
                type = 'eps-regression',
                kernel = 'radial')
varImp(svm)

library(caret)

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
